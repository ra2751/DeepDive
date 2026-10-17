"""T14–T16: bounded termination; limits end runs without upgrading their status."""

import tavily.errors as te

from tests.fakes import FakeTavilyClient
from tests.graph_helpers import budget, patch_retriever, route, run_graph, verdict


def test_failing_validation_stops_after_max_retry_rounds(fake_mesh, monkeypatch):
    patch_retriever(monkeypatch)
    fake_mesh.supervisor.script = [route("rag_agent")]
    fake_mesh.generator.script = ["An answer that never passes the check."]
    fake_mesh.verifier.structured_script = [verdict("no")]
    b = budget()

    out = run_graph(b)

    assert fake_mesh.supervisor.calls == 3          # first pass + 2 retry rounds, then stop
    assert out["answer_status"] == "not_validated"
    assert out["termination_reason"] == "max_retry_rounds"
    assert out["usage"]["retry_rounds"] == 2
    assert out["validation_metadata"]["is_valid"] == "no"


def test_llm_budget_ends_run_as_incomplete_without_extra_attempt(fake_mesh, monkeypatch):
    patch_retriever(monkeypatch)
    fake_mesh.supervisor.script = [route("rag_agent")]
    fake_mesh.generator.script = ["draft"]
    fake_mesh.verifier.structured_script = [verdict("no")]
    b = budget(max_retry_rounds=50)                 # only MAX_LLM_CALLS=25 can stop this run

    out = run_graph(b)

    assert out["answer_status"] == "incomplete_budget"
    assert out["termination_reason"] == "max_llm_calls"
    assert out["usage"]["mesh_chat_calls"] == 25
    assert fake_mesh.total_chat_calls() == 25       # no 26th Mesh attempt was sent


def test_embedding_budget_stop_is_incomplete(fake_mesh, monkeypatch):
    patch_retriever(monkeypatch)
    fake_mesh.supervisor.script = [route("rag_agent")]
    fake_mesh.verifier.structured_script = [verdict("no")]
    b = budget(max_embedding_calls=1)

    out = run_graph(b)

    assert out["answer_status"] == "incomplete_budget"
    assert out["termination_reason"] == "max_embedding_calls"
    assert fake_mesh.embeddings.calls == 1


def test_web_search_failure_degrades_without_crash_or_upgrade(fake_mesh, monkeypatch, settings):
    import agents.web_node
    from retriever.web_search import TavilyWebSearch

    tavily = FakeTavilyClient([te.TimeoutError(15)])
    monkeypatch.setattr(agents.web_node, "get_web_search_provider",
                        lambda b: TavilyWebSearch(b, settings, client=tavily))
    fake_mesh.supervisor.script = [route("web_search_agent")]
    fake_mesh.verifier.structured_script = [verdict("yes")]
    b = budget()

    out = run_graph(b, "latest benchmark results")

    assert out["answer_status"] == "ungrounded"     # no sources, so never "passed"
    assert any("Web search unavailable" in n for n in out["notices"])
    assert tavily.calls == 2 and out["usage"]["web_searches"] == 2
    assert fake_mesh.generator.calls == 0           # no model call spent on an empty context
