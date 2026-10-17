"""T17: honest answer labels — a pass needs sources AND a performed, explicit 'yes'."""

from agents.final_response_node import derive_answer_status
from budget import BudgetLimits, BudgetManager
from tests.graph_helpers import budget, patch_retriever, route, run_graph, verdict


def test_rag_answer_with_sources_and_yes_passes(fake_mesh, monkeypatch):
    patch_retriever(monkeypatch)
    fake_mesh.supervisor.script = [route("rag_agent")]
    fake_mesh.generator.script = ["Grounded answer."]
    fake_mesh.verifier.structured_script = [verdict("yes")]

    out = run_graph(budget())

    assert out["answer_status"] == "answer_check_passed"
    assert out["termination_reason"] is None
    assert out["retrieved_docs"][0]["page"] == 3
    assert out["final_response"] == "Grounded answer."


def test_model_knowledge_answer_is_ungrounded_even_if_check_says_yes(fake_mesh):
    fake_mesh.supervisor.script = [route("general_llm_agent", topic="AI Non-Transformer Related")]
    fake_mesh.generator.script = ["Answer from model knowledge."]
    fake_mesh.verifier.structured_script = [verdict("yes")]

    out = run_graph(budget(), "what is a transformer")

    assert out["answer_status"] == "ungrounded"


def test_unavailable_validator_is_never_a_pass(fake_mesh, monkeypatch):
    from tests.fakes import PARSE_FAIL

    patch_retriever(monkeypatch)
    fake_mesh.supervisor.script = [route("rag_agent")]
    fake_mesh.verifier.structured_script = [PARSE_FAIL]

    out = run_graph(budget())

    assert out["answer_status"] == "not_validated"
    assert out["termination_reason"] == "validation_unavailable"


def test_out_of_scope_keyword_short_circuits_without_model_calls(fake_mesh):
    out = run_graph(budget(), "who will win the football match")
    assert out["answer_status"] == "out_of_scope"
    assert fake_mesh.total_chat_calls() == 0


def test_stale_or_unperformed_yes_is_not_a_pass():
    b = BudgetManager(BudgetLimits())
    state = {"draft_answer": "x", "retrieved_docs": [{"content": "c"}],
             "validation_metadata": {"is_valid": "yes", "performed": False}}
    assert derive_answer_status(state, b)[0] != "answer_check_passed"


def test_limit_reached_after_a_failed_check_is_never_a_pass():
    b = BudgetManager(BudgetLimits(max_retry_rounds=0))
    state = {"draft_answer": "x", "retrieved_docs": [{"content": "c"}],
             "validation_metadata": {"is_valid": "no", "performed": True}}
    assert derive_answer_status(state, b) == ("not_validated", "max_retry_rounds")
    state["termination_reason"] = "max_run_seconds"
    assert derive_answer_status(state, b) == ("incomplete_budget", "max_run_seconds")


def test_router_has_no_undefined_branch():
    import inspect

    import agents.validation_router as vr

    assert "end_note" not in inspect.getsource(vr)
