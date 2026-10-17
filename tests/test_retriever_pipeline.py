"""T18: retriever counts searches, handles empty results, refuses incompatible indexes."""

import pytest

from budget import BudgetLimits, BudgetManager
from retriever.retriever_pipeline import EmbeddingCompatibilityError, RetrieverPipeline
from tests.fakes import FakeMilvusClient
from tests.graph_helpers import DOCS


def test_empty_result_returns_empty_list(fake_mesh, settings):
    b = BudgetManager(BudgetLimits())
    r = RetrieverPipeline("c", budget=b, settings=settings, client=FakeMilvusClient([]))
    assert r.search("q") == []
    snap = b.snapshot()
    assert snap.tool_calls == 1 and snap.mesh_embedding_calls == 1


def test_search_returns_reranked_docs_with_pages(fake_mesh, settings):
    r = RetrieverPipeline("c", budget=BudgetManager(BudgetLimits()), settings=settings,
                          client=FakeMilvusClient(DOCS))
    out = r.search("attention", top_k=5)
    assert {d["page"] for d in out} == {3, 4}


def test_incompatible_index_is_refused(fake_mesh, use_settings):
    s = use_settings(MILVUS_EMBEDDING_COMPAT="incompatible")
    with pytest.raises(EmbeddingCompatibilityError):
        RetrieverPipeline("c", budget=BudgetManager(BudgetLimits()), settings=s, client=FakeMilvusClient(DOCS))
