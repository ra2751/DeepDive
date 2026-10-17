"""
LIVE G3 tests (LT-1–LT-6). They call Mesh, Tavily and Milvus for real and are
excluded from the default run and CI. Budget shares are enforced by the G3 ledger
(tests/integration/conftest.py). Run only with approval:

    pytest -m live tests/integration -v -s
"""

import math
from pathlib import Path

import pytest

from config.settings import ConfigError

pytestmark = pytest.mark.live

SYNTHETIC_DOCS = [{"content": "The Eiffel Tower was completed in 1889 for the Exposition Universelle in Paris.",
                   "source": "synthetic-fact.txt"}]


def _milvus(settings):
    from pymilvus import MilvusClient

    settings.require("milvus_endpoint", "milvus_api_key")
    return MilvusClient(uri=settings.milvus_endpoint, token=settings.milvus_api_key)


def test_lt1_chat_via_mesh(live_settings, g3, once):
    from gateway import chat_model

    b = g3.budget("LT-1", llm=1)
    try:
        msg = chat_model("generator", b, live_settings).invoke("Reply with the single word: ok")
        ok = bool(msg.content)
    finally:
        g3.commit(once, b, answer=getattr(locals().get("msg"), "content", None))
    assert ok and b.snapshot().mesh_llm_calls == 1


def test_lt2_validator_node_structured_output(live_settings, g3, once):
    """The real ValidationAgent node (application path) with a synthetic state."""
    from agents.validation_node import ValidationAgent

    b = g3.budget("LT-2", llm=1)
    state = {"query": "When was the Eiffel Tower completed?",
             "draft_answer": "The Eiffel Tower was completed in 1889.", "retrieved_docs": SYNTHETIC_DOCS}
    try:
        out = ValidationAgent(state, {"configurable": {"budget": b}})
    finally:
        g3.commit(once, b, validation=locals().get("out"))
    vm = out["validation_metadata"]
    assert vm["performed"] is True and vm["is_valid"] in ("yes", "no")


def test_lt3_lt4_embedding_dimension_and_milvus_compatibility(live_settings, g3, once):
    """One Mesh embedding covers LT-3 (dimension) and LT-4 (sampled compatibility). Milvus read-only."""
    from gateway import embeddings

    client = _milvus(live_settings)
    coll = live_settings.milvus_collection
    rows_before = client.get_collection_stats(coll).get("row_count")
    rows = client.query(collection_name=coll, filter="id >= 0", output_fields=["text", "source", "page", "embedding"],
                        limit=20)
    import json
    import re

    reviewed = json.loads((Path("data/probe/g2_milvus_sample.json")).read_text()).get("id") \
        if Path("data/probe/g2_milvus_sample.json").exists() else None
    suitable = [r for r in rows if isinstance(r.get("text"), str) and len(r["text"].strip()) >= 200]
    sample = next((r for r in suitable if r.get("id") == reviewed), None)
    if sample is None:  # not the G2-reviewed chunk: only proceed if an automated scan is clean
        risky = re.compile(r"(?i)confidential|proprietary|internal use only|[\w.+-]+@[\w-]+\.[\w.]+|api[_-]?key|password")
        sample = next((r for r in suitable if not risky.search(r["text"])), None)
    if sample is None:
        pytest.skip("no safe sample available; nothing sent to Mesh")
    b = g3.budget("LT-3/4", embedding=1)
    try:
        fresh = embeddings(b, live_settings).embed_query(sample["text"])
        stored = sample["embedding"]
        cos = sum(x * y for x, y in zip(stored, fresh)) / (
            math.sqrt(sum(x * x for x in stored)) * math.sqrt(sum(y * y for y in fresh)))
    finally:
        g3.commit(once, b, rows_before=rows_before, sample_source=sample.get("source"), sample_page=sample.get("page"),
                  sample_chars=len(sample["text"]), dim=len(locals().get("fresh") or []),
                  cosine=round(locals().get("cos", float("nan")), 6))
    assert len(fresh) == 1536
    assert cos >= 0.99


def test_lt5_tavily_search(live_settings, g3, once):
    from retriever.web_search import TavilyWebSearch

    try:
        live_settings.require("tavily_api_key")
    except ConfigError as e:
        pytest.skip(str(e))
    b = g3.budget("LT-5", web=1)
    try:
        results = TavilyWebSearch(b, live_settings).search("scaled dot-product attention transformer", max_results=2)
    finally:
        g3.commit(once, b, results=[{"url": r.url, "chars": len(r.content)} for r in (locals().get("results") or [])])
    assert results and b.snapshot().web_searches == 1


def test_lt6_end_to_end_web_question(live_settings, g3, once):
    """Full graph, web route. Bounded by its ledger share (3 Mesh LLM calls, 1 Tavily search)."""
    from graph_builder.graph_builder import build_graph

    b = g3.budget("LT-6", llm=3, web=1)
    q = "Find a recent PyTorch code implementation example of scaled dot-product attention."
    try:
        out = build_graph().invoke(
            {"query": q, "messages": [], "loop_count": 0,
             "validation_metadata": {"is_valid": "pending", "performed": False}, "draft_answer": None,
             "out_of_scope": False, "termination_reason": None, "notices": []},
            config={"configurable": {"budget": b}, "recursion_limit": 20})
    finally:
        o = locals().get("out") or {}
        g3.commit(once, b, query=q, answer_status=o.get("answer_status"), termination_reason=o.get("termination_reason"),
                  sources=[d.get("source") for d in o.get("retrieved_docs", [])], notices=o.get("notices"),
                  answer_excerpt=(o.get("final_response") or "")[:800],
                  validation=o.get("validation_metadata"))
    assert out["answer_status"] in {"answer_check_passed", "ungrounded", "not_validated", "incomplete_budget"}
    snap = b.snapshot()
    assert snap.mesh_llm_calls <= 3 and snap.web_searches <= 1
