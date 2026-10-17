"""
Mesh API model gateway — the ONLY module that constructs model clients.

Every LLM, structured-output, vision and embedding call in DeepDive goes through
Mesh API here (Agent Arena 2026 requirement). There is no fallback to any
other provider. Client-level retries are disabled; this module runs its own
retry loop so that every attempt is checked against, and recorded in, the
run's BudgetManager.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from budget.manager import BudgetExceeded, BudgetManager
from config.settings import Settings, get_settings
from gateway.errors import EmbeddingDimensionError, MeshError, StructuredOutputError, classify

logger = logging.getLogger("ai_research_assistant")

T = TypeVar("T")

# Patchable for tests (no real sleeping).
_sleep = time.sleep


def _backoff(attempt: int) -> float:
    return min(0.5 * (2 ** (attempt - 1)), 8.0)


@dataclass
class CallUsage:
    model_served: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    tokens_estimated: bool = False
    reported_cost_usd: Optional[float] = None
    request_id: Optional[str] = None
    cached_input_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached: bool = False                     # gateway response-cache hit
    cache_marker: Optional[str] = None       # "x-cache-header" (documented) or "id-prefix" (observed, undocumented)


# --------------------------------------------------------------- client factory
def _mesh_headers(settings: Settings) -> Optional[dict]:
    # Documented per-request cache bypass; only sent when explicitly configured.
    return {"X-Mesh-Cache": "no-store"} if settings.mesh_cache_mode == "no-store" else None


def _make_chat_client(model: str, settings: Settings, temperature: Optional[float] = None) -> Any:
    kwargs = dict(
        model=model,
        api_key=settings.mesh_api_key,
        base_url=settings.mesh_base_url,
        timeout=settings.mesh_call_timeout_s,
        max_retries=0,
        include_response_headers=True,   # needed to read Mesh's documented "X-Cache: HIT" marker
    )
    if _mesh_headers(settings):
        kwargs["default_headers"] = _mesh_headers(settings)
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


def _make_embeddings_client(settings: Settings) -> Any:
    return OpenAIEmbeddings(
        model=settings.mesh_embedding_model,
        api_key=settings.mesh_api_key,
        base_url=settings.mesh_base_url,
        check_embedding_ctx_length=False,  # send raw strings, not OpenAI token IDs
        chunk_size=max(1, settings.embedding_batch_size),
        max_retries=0,
        timeout=settings.mesh_call_timeout_s,
        default_headers=_mesh_headers(settings),
    )


def _settings(settings: Optional[Settings]) -> Settings:
    s = settings or get_settings()
    s.require("mesh_api_key")
    return s


# ------------------------------------------------------------------ retry loop
def _attempt_loop(
    budget: BudgetManager,
    kind: str,
    operation: str,
    model: str,
    call: Callable[[], T],
    extract_usage: Callable[[T], CallUsage],
    max_attempts: int,
) -> T:
    for attempt in range(1, max_attempts + 1):
        budget.check(kind)  # raises BudgetExceeded before anything is sent
        started = time.monotonic()
        try:
            result = call()
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - classified below
            latency = time.monotonic() - started
            decision = classify(exc)
            budget.record_mesh_attempt(operation, model, "error", latency_s=latency, error=type(exc).__name__)
            logger.warning(
                "MeshGateway | %s | model=%s | attempt %d/%d failed: %s (%s)",
                operation, model, attempt, max_attempts, type(exc).__name__, decision,
            )
            if decision == "spend_limit":
                raise BudgetExceeded("mesh_spend_limit") from exc
            if decision == "fatal" or attempt == max_attempts:
                raise MeshError(f"Mesh {operation} call failed: {type(exc).__name__}") from exc
            _sleep(_backoff(attempt))
            continue
        latency = time.monotonic() - started
        usage = extract_usage(result)
        budget.record_mesh_attempt(
            operation, model, "ok",
            model_served=usage.model_served,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            tokens_estimated=usage.tokens_estimated,
            reported_cost_usd=usage.reported_cost_usd,
            cached_input_tokens=usage.cached_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cached=usage.cached,
            cache_marker=usage.cache_marker,
            latency_s=latency,
            request_id=usage.request_id,
        )
        logger.info(
            "MeshGateway | %s | requested=%s served=%s | tokens in=%s out=%s | %.2fs",
            operation, model, usage.model_served, usage.input_tokens, usage.output_tokens, latency,
        )
        return result
    raise MeshError(f"Mesh {operation} call made no attempt")  # max_attempts < 1


def _usage_from_message(msg: Any) -> CallUsage:
    if not isinstance(msg, AIMessage):
        return CallUsage()
    meta = msg.response_metadata or {}
    usage = msg.usage_metadata or {}
    token_usage = meta.get("token_usage") or {}
    cost = token_usage.get("cost") if isinstance(token_usage, dict) else None
    in_details = usage.get("input_token_details") or {}
    out_details = usage.get("output_token_details") or {}
    headers = {str(k).lower(): str(v) for k, v in (meta.get("headers") or {}).items()}
    request_id = str(meta.get("id") or "")
    cached, marker = False, None
    if headers.get("x-cache", "").strip().upper() == "HIT":          # documented by Mesh
        cached, marker = True, "x-cache-header"
    elif request_id.startswith("chatcmpl-cached-"):                   # observed in live runs; undocumented
        cached, marker = True, "id-prefix"
    return CallUsage(
        cached=cached, cache_marker=marker,
        model_served=meta.get("model_name"),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        reported_cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
        request_id=meta.get("id"),
        cached_input_tokens=in_details.get("cache_read"),
        reasoning_tokens=out_details.get("reasoning"),
    )


# ------------------------------------------------------------------ public API
_ROLE_TEMPERATURE = {"supervisor": 0.0, "verifier": 0.0, "generator": None}


def chat_model(role: str, budget: BudgetManager, settings: Optional[Settings] = None) -> Runnable:
    """A Runnable chat model for `role`, usable in `prompt | model | parser` chains."""
    s = _settings(settings)
    model = s.model_for_role(role)
    client = _make_chat_client(model, s, _ROLE_TEMPERATURE.get(role))

    def _invoke(prompt_value):
        return _attempt_loop(
            budget, "llm", "chat", model,
            lambda: client.invoke(prompt_value),
            _usage_from_message, s.mesh_max_attempts,
        )

    return RunnableLambda(_invoke, name=f"mesh_chat_{role}")


def structured(role: str, schema: type, budget: BudgetManager, settings: Optional[Settings] = None) -> Runnable:
    """A Runnable returning a validated `schema` instance via Mesh structured output.

    Tries `json_schema`, then `function_calling`. Every attempt is counted. Some
    models silently ignore `response_format`, so output is always validated.
    """
    s = _settings(settings)
    model = s.model_for_role(role)
    client = _make_chat_client(model, s, _ROLE_TEMPERATURE.get(role))

    def _invoke(prompt_value):
        last_problem = "no method attempted"
        for method in ("json_schema", "function_calling"):
            runnable = client.with_structured_output(schema, method=method, include_raw=True)
            try:
                out = _attempt_loop(
                    budget, "llm", "structured", model,
                    lambda: runnable.invoke(prompt_value),
                    lambda r: _usage_from_message(r.get("raw") if isinstance(r, dict) else None),
                    s.mesh_max_attempts,
                )
            except MeshError as exc:
                if exc.__cause__ is not None and classify(exc.__cause__) == "retry":
                    raise  # transient failures already exhausted their attempts
                last_problem = f"{method}: {exc}"
                logger.warning("MeshGateway | structured | %s unavailable, trying next method", method)
                continue
            parsed = out.get("parsed") if isinstance(out, dict) else out
            if isinstance(parsed, dict):
                try:
                    parsed = schema.model_validate(parsed)
                except Exception:  # noqa: BLE001
                    parsed = None
            if isinstance(parsed, schema):
                return parsed
            last_problem = f"{method}: output did not match {schema.__name__}"
            logger.warning("MeshGateway | structured | %s returned invalid output, trying next method", method)
        raise StructuredOutputError(f"Structured output failed for role {role!r} ({last_problem})")

    return RunnableLambda(_invoke, name=f"mesh_structured_{role}")


class MeshEmbeddings(Embeddings):
    """Budgeted Mesh embeddings. Each batch of `embedding_batch_size` texts is one counted request."""

    def __init__(self, budget: BudgetManager, settings: Optional[Settings] = None):
        self.settings = _settings(settings)
        self.budget = budget
        self.model = self.settings.mesh_embedding_model
        self.batch_size = max(1, self.settings.embedding_batch_size)
        self._client = _make_embeddings_client(self.settings)

    def _request(self, texts: list):
        """One Mesh embeddings request -> (vectors, CallUsage).

        Uses the OpenAI-compatible client held by LangChain's OpenAIEmbeddings (same
        Mesh base URL and key, retries disabled) so the response's token usage is
        available. If usage is absent, input_tokens stays None and cost is "unknown".
        """
        raw = getattr(self._client, "client", None)
        if raw is not None and hasattr(raw, "create"):
            resp = raw.create(input=texts, model=self.model, encoding_format="float")
            data = sorted(resp.data, key=lambda d: d.index)
            vectors = [list(d.embedding) for d in data]
            usage = getattr(resp, "usage", None)
            tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
            if tokens is None and usage is not None:
                tokens = getattr(usage, "total_tokens", None)
            return vectors, CallUsage(model_served=getattr(resp, "model", None),
                                      input_tokens=tokens if isinstance(tokens, int) else None, output_tokens=0)
        # Clients without a raw endpoint (e.g. test doubles): vectors only, usage unknown.
        return self._client.embed_documents(list(texts)), CallUsage()

    def _call(self, texts: list) -> list:
        holder = {}

        def fn():
            vectors, usage = self._request(texts)
            holder["usage"] = usage
            return vectors

        vectors = _attempt_loop(
            self.budget, "embedding", "embedding", self.model, fn,
            lambda _r: holder.get("usage") or CallUsage(),
            self.settings.mesh_max_attempts,
        )
        if len(vectors) != len(texts):
            raise MeshError(f"Mesh returned {len(vectors)} embeddings for {len(texts)} inputs")
        for v in vectors:
            if len(v) != self.settings.embedding_dim:
                raise EmbeddingDimensionError(
                    f"Expected {self.settings.embedding_dim}-dimensional embeddings, got {len(v)}"
                )
        return vectors

    def embed_documents(self, texts: list) -> list:
        out = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._call(list(texts[i:i + self.batch_size])))
        return out

    def embed_query(self, text: str) -> list:
        return self._call([text])[0]


def embeddings(budget: BudgetManager, settings: Optional[Settings] = None) -> MeshEmbeddings:
    return MeshEmbeddings(budget, settings)


def list_models(settings: Optional[Settings] = None) -> list:
    """GET /v1/models — catalogue metadata only, no inference (used by the G1 probe)."""
    s = _settings(settings)
    resp = httpx.get(
        s.mesh_base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {s.mesh_api_key}"},
        timeout=s.mesh_call_timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", data) if isinstance(data, dict) else data
