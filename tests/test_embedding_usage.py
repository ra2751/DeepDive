"""Gateway captures Mesh embedding token usage where available; otherwise cost is unknown."""

import pytest

from budget import BudgetLimits, BudgetManager, PriceTable, format_cost
from gateway import EmbeddingDimensionError, embeddings
from tests.fakes import FakeOpenAIEmbeddingsWithRaw, fake_vector

P = {"source": "test", "retrieved": "2026-10-16"}


def install(monkeypatch, **kw):
    import gateway.mesh

    client = FakeOpenAIEmbeddingsWithRaw(**kw)
    monkeypatch.setattr(gateway.mesh, "_make_embeddings_client", lambda s: client)
    return client.client


def priced_budget(**limits):
    table = PriceTable({"openai/text-embedding-3-small": {"input_per_1m_usd": 0.02, **P}})
    return BudgetManager(BudgetLimits(**limits), pricing=table)


def test_usage_from_mesh_response_is_recorded_and_priced(monkeypatch, settings):
    raw = install(monkeypatch, tokens_per_text=7)
    b = priced_budget()
    vec = embeddings(b, settings).embed_query("hello")
    assert vec == fake_vector("hello")                       # vectors returned in input order
    assert raw.calls == [{"n": 1, "model": "openai/text-embedding-3-small", "encoding_format": "float"}]
    snap = b.snapshot()
    assert snap.mesh_embedding_calls == 1 and snap.mesh_input_tokens == 7
    assert snap.cost_basis == "estimated" and snap.mesh_cost_usd == pytest.approx(7 * 0.02 / 1e6)
    assert b.events[-1]["model_served"] == "text-embedding-3-small"
    assert b.events[-1]["priced_as"] == "openai/text-embedding-3-small"


def test_batches_keep_order_and_count_one_request_each(monkeypatch, settings):
    raw = install(monkeypatch)
    b = priced_budget(max_embedding_calls=None)
    texts = [f"t{i}" for i in range(130)]
    vecs = embeddings(b, settings).embed_documents(texts)
    assert vecs == [fake_vector(t) for t in texts]
    assert [c["n"] for c in raw.calls] == [64, 64, 2]
    assert b.snapshot().mesh_embedding_calls == 3 and b.snapshot().mesh_input_tokens == 7 * 130


def test_missing_usage_means_unknown_cost(monkeypatch, settings):
    install(monkeypatch, usage=False)
    b = priced_budget()
    embeddings(b, settings).embed_query("hello")
    snap = b.snapshot()
    assert snap.cost_basis == "unknown" and snap.mesh_cost_usd is None and snap.mesh_input_tokens is None
    assert format_cost(snap) == "Cost unknown; limited by request caps"


def test_caps_and_dimension_check_still_apply(monkeypatch, settings):
    install(monkeypatch)
    b = priced_budget(max_embedding_calls=1)
    emb = embeddings(b, settings)
    emb.embed_query("a")
    from budget import BudgetExceeded
    with pytest.raises(BudgetExceeded, match="max_embedding_calls"):
        emb.embed_query("b")
    install(monkeypatch, dim=768)
    with pytest.raises(EmbeddingDimensionError):
        embeddings(priced_budget(), settings).embed_query("c")
