"""T10–T12: gateway retry, structured output and embeddings, all metered."""

import httpx
import openai
import pytest
from pydantic import BaseModel

from budget import BudgetExceeded, BudgetLimits, BudgetManager
from gateway import EmbeddingDimensionError, MeshError, StructuredOutputError, chat_model, embeddings, structured
from tests.fakes import PARSE_FAIL, FakeEmbeddingsClient

REQ = httpx.Request("POST", "https://api.meshapi.ai/v1/chat/completions")


def status_error(cls, code):
    return cls(f"http {code}", response=httpx.Response(code, request=REQ), body=None)


class Answer(BaseModel):
    is_valid: str
    reasoning: str


def test_retryable_errors_are_retried_and_every_attempt_counted(fake_mesh):
    fake_mesh.generator.script = [status_error(openai.RateLimitError, 429),
                                  status_error(openai.RateLimitError, 429), "done"]
    b = BudgetManager(BudgetLimits())
    out = chat_model("generator", b).invoke("hi")
    assert out.content == "done"
    snap = b.snapshot()
    assert snap.mesh_chat_calls == 3 and snap.mesh_failed_attempts == 2
    assert b.events[-1]["model_served"] == "test/generator"


def test_auth_error_is_not_retried(fake_mesh):
    fake_mesh.generator.script = [status_error(openai.AuthenticationError, 401)]
    b = BudgetManager(BudgetLimits())
    with pytest.raises(MeshError):
        chat_model("generator", b).invoke("hi")
    assert fake_mesh.generator.calls == 1


def test_spend_limit_becomes_budget_exceeded(fake_mesh):
    fake_mesh.generator.script = [status_error(openai.APIStatusError, 402)]
    with pytest.raises(BudgetExceeded) as exc:
        chat_model("generator", BudgetManager(BudgetLimits())).invoke("hi")
    assert exc.value.reason == "mesh_spend_limit"


def test_timeouts_exhaust_attempts(fake_mesh):
    fake_mesh.generator.script = [openai.APITimeoutError(request=REQ)]
    b = BudgetManager(BudgetLimits())
    with pytest.raises(MeshError):
        chat_model("generator", b).invoke("hi")
    assert fake_mesh.generator.calls == 3 and b.snapshot().mesh_chat_calls == 3


def test_budget_reached_mid_retry_stops_further_attempts(fake_mesh):
    fake_mesh.generator.script = [status_error(openai.InternalServerError, 503)]
    b = BudgetManager(BudgetLimits(max_llm_calls=2))
    with pytest.raises(BudgetExceeded, match="max_llm_calls"):
        chat_model("generator", b).invoke("hi")
    assert fake_mesh.generator.calls == 2


def test_structured_json_schema_path(fake_mesh):
    fake_mesh.verifier.structured_script = [{"is_valid": "yes", "reasoning": "grounded"}]
    b = BudgetManager(BudgetLimits())
    out = structured("verifier", Answer, b).invoke("check")
    assert isinstance(out, Answer) and out.is_valid == "yes"
    assert fake_mesh.verifier.structured_methods == ["json_schema"]


def test_structured_falls_back_when_schema_ignored(fake_mesh):
    fake_mesh.verifier.structured_script = [PARSE_FAIL, {"is_valid": "no", "reasoning": "r"}]
    b = BudgetManager(BudgetLimits())
    out = structured("verifier", Answer, b).invoke("check")
    assert out.is_valid == "no"
    assert fake_mesh.verifier.structured_methods == ["json_schema", "function_calling"]
    assert b.snapshot().mesh_chat_calls == 2


def test_structured_falls_back_on_permanent_rejection(fake_mesh):
    fake_mesh.verifier.structured_script = [status_error(openai.BadRequestError, 400),
                                            {"is_valid": "yes", "reasoning": "r"}]
    out = structured("verifier", Answer, BudgetManager(BudgetLimits())).invoke("check")
    assert out.is_valid == "yes"


def test_structured_fails_loudly_when_no_method_works(fake_mesh):
    fake_mesh.verifier.structured_script = [PARSE_FAIL]
    b = BudgetManager(BudgetLimits())
    with pytest.raises(StructuredOutputError):
        structured("verifier", Answer, b).invoke("check")
    assert b.snapshot().mesh_chat_calls == 2


def test_embeddings_are_batched_and_each_batch_counted(fake_mesh):
    b = BudgetManager(BudgetLimits(max_embedding_calls=None))
    vecs = embeddings(b).embed_documents([f"text {i}" for i in range(130)])
    assert len(vecs) == 130 and len(vecs[0]) == 1536
    assert fake_mesh.embeddings.batches == [64, 64, 2]
    assert b.snapshot().mesh_embedding_calls == 3


def test_query_embedding_counted_and_limit_enforced(fake_mesh):
    b = BudgetManager(BudgetLimits(max_embedding_calls=1))
    emb = embeddings(b)
    emb.embed_query("q")
    with pytest.raises(BudgetExceeded, match="max_embedding_calls"):
        emb.embed_query("q2")


def test_wrong_embedding_dimension_is_rejected(monkeypatch, settings):
    import gateway.mesh

    monkeypatch.setattr(gateway.mesh, "_make_embeddings_client", lambda s: FakeEmbeddingsClient(dim=768))
    with pytest.raises(EmbeddingDimensionError):
        embeddings(BudgetManager(BudgetLimits())).embed_query("q")
