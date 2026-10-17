"""Deterministic fakes for offline tests. No network, no real providers."""

import hashlib
import json

import numpy as np
from langchain_core.messages import AIMessage

DIM = 1536
PARSE_FAIL = object()  # structured-output item meaning "model ignored the schema"


def fake_vector(text: str, dim: int = DIM) -> list:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).standard_normal(dim).tolist()


def ai(content: str, model: str = "fake-model") -> AIMessage:
    return AIMessage(
        content=content,
        response_metadata={"model_name": model, "id": "req-test"},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


class ScriptedChat:
    """Stands in for ChatOpenAI. The last script item repeats forever."""

    def __init__(self, model: str, script=None, structured_script=None):
        self.model = model
        self.script = list(script or ["ok"])
        self.structured_script = list(structured_script or [])
        self.calls = 0
        self.structured_methods = []
        self.prompts = []

    @staticmethod
    def _next(items):
        return items.pop(0) if len(items) > 1 else items[0]

    def invoke(self, prompt_value):
        self.calls += 1
        item = self._next(self.script)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(prompt_value)
        return ai(item, self.model) if isinstance(item, str) else item

    def with_structured_output(self, schema, method="json_schema", include_raw=False):
        parent = self

        class _Runnable:
            def invoke(self, prompt_value):
                parent.calls += 1
                parent.structured_methods.append(method)
                parent.prompts.append(str(prompt_value))
                item = parent._next(parent.structured_script)
                if isinstance(item, BaseException):
                    raise item
                if item is PARSE_FAIL:
                    return {"raw": ai("plain text, not JSON", parent.model), "parsed": None,
                            "parsing_error": ValueError("not json")}
                parsed = schema(**item) if isinstance(item, dict) else item
                return {"raw": ai(json.dumps(parsed.model_dump()), parent.model), "parsed": parsed,
                        "parsing_error": None}

        return _Runnable()


class FakeEmbeddingsClient:
    def __init__(self, dim: int = DIM, script=None):
        self.dim = dim
        self.calls = 0
        self.batches = []
        self.script = list(script or [])

    def _maybe_raise(self):
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item

    def embed_documents(self, texts):
        self.calls += 1
        self.batches.append(len(texts))
        self._maybe_raise()
        return [fake_vector(t, self.dim) for t in texts]

    def embed_query(self, text):
        self.calls += 1
        self._maybe_raise()
        return fake_vector(text, self.dim)


class _Hit(dict):
    def __init__(self, distance, entity, id=None):
        super().__init__(distance=distance, id=id)
        self.entity = entity


class FakeMilvusClient:
    """Read-only fake of pymilvus.MilvusClient.search."""

    def __init__(self, docs):
        self.docs = docs  # list of {"text", "source", "page"}
        self.search_calls = 0

    def load_collection(self, collection_name):
        pass

    def search(self, collection_name, data, anns_field, search_params, limit, output_fields):
        self.search_calls += 1
        hits = [_Hit(float(i), {**d, "embedding": fake_vector(d["text"])}, id=d.get("id", 1000 + i))
                for i, d in enumerate(self.docs[:limit])]
        return [hits]


class FakeTavilyClient:
    """Stands in for tavily.TavilyClient. The last script item repeats."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.kwargs = []

    def search(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item


class _EmbItem:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _EmbResponse:
    def __init__(self, vectors, prompt_tokens, model):
        self.data = [_EmbItem(i, v) for i, v in reversed(list(enumerate(vectors)))]  # out of order on purpose
        self.usage = None if prompt_tokens is None else type("U", (), {"prompt_tokens": prompt_tokens,
                                                                      "total_tokens": prompt_tokens})()
        self.model = model


class FakeRawEmbeddingsEndpoint:
    """Mimics openai.resources.Embeddings.create as used by the gateway."""

    def __init__(self, dim=DIM, tokens_per_text=7, usage=True, model="text-embedding-3-small"):
        self.dim, self.tokens_per_text, self.usage, self.model = dim, tokens_per_text, usage, model
        self.calls = []

    def create(self, input, model, encoding_format):
        self.calls.append({"n": len(input), "model": model, "encoding_format": encoding_format})
        vectors = [fake_vector(t, self.dim) for t in input]
        return _EmbResponse(vectors, self.tokens_per_text * len(input) if self.usage else None, self.model)


class FakeOpenAIEmbeddingsWithRaw:
    """Shape of langchain_openai.OpenAIEmbeddings: exposes `.client` (the raw endpoint)."""

    def __init__(self, **kw):
        self.client = FakeRawEmbeddingsEndpoint(**kw)


# ------------------------------------------------ query-aware retrieval fake
_VEC_TEXT = {}


def tracked_vector(text: str) -> list:
    v = fake_vector(text)
    _VEC_TEXT[tuple(round(x, 6) for x in v[:6])] = text
    return v


class TrackingEmbeddingsClient(FakeEmbeddingsClient):
    """Fake embeddings whose query vectors can be mapped back to their text by KeywordMilvusClient."""

    def embed_documents(self, texts):
        self.calls += 1
        self.batches.append(len(texts))
        return [tracked_vector(t) for t in texts]

    def embed_query(self, text):
        self.calls += 1
        return tracked_vector(text)


def _words(text):
    import re
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower())}


class KeywordMilvusClient(FakeMilvusClient):
    """Ranks documents by keyword overlap with the (recovered) query text. Read-only."""

    def search(self, collection_name, data, anns_field, search_params, limit, output_fields):
        self.search_calls += 1
        q = _VEC_TEXT.get(tuple(round(x, 6) for x in data[0][:6]), "")
        scored = sorted(((len(_words(q) & _words(d["text"])), i, d) for i, d in enumerate(self.docs)),
                        key=lambda t: (-t[0], t[1]))
        hits = [_Hit(float(-s), {**d, "embedding": fake_vector(d["text"])}, id=d.get("id", 2000 + i))
                for s, i, d in scored if s > 0][:limit]
        return [hits]
