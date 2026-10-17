"""End-to-end offline tests of the Phase 2 research graph (scripted Mesh fakes, network blocked).

These tests prove the workflow logic. They are NOT measurements of live model accuracy.
"""

import pytest

from budget import BudgetLimits, BudgetManager
from graph_builder.research_graph import build_research_graph, run_research
from tests.research_fixtures import AUDIT, CONTRACT, PROCUREMENT_DOCS, SCORECARD, TRIAL_A, TRIAL_B, eid, judgment

from tests.research_world import QUESTION  # noqa: E402


@pytest.fixture
def world(monkeypatch, settings):
    from tests.research_world import install_world

    return install_world(monkeypatch, settings)


from tests.research_world import plan, script_procurement  # noqa: E402,F401


def run(budget=None, **kw):
    budget = budget or BudgetManager(BudgetLimits())
    return run_research(QUESTION, budget, domain_id=kw.pop("domain_id", "procurement"),
                        source_scope=kw.pop("source_scope", ["documents"]), **kw), budget


def test_full_workflow_keeps_supported_and_excludes_unsupported(world):
    script_procurement(world)
    out, budget = run()
    v, report = out["verifications"], out["report"]

    # 1. supported claim stays in the report, with a page-level citation
    assert v["C1"].status == "supported"
    assert [s.claim_ids for s in report.verified_sentences] == [["C1"]]
    assert [(c.marker, c.source, c.page) for c in report.citations] == [("C1", "contract.pdf", 1)]
    # 2. contradicted claim is excluded from verified findings and flagged for review (compliance = high risk)
    assert v["C2"].status == "contradicted" and "C2" in report.human_review
    assert all("C2" not in s.claim_ids for s in report.verified_sentences)
    # 3. insufficient claim was retried and ended unresolved
    assert v["C3"].status == "unresolved" and out["retry_round"] >= 1
    # 4. fabricated quote rejected -> claim never supported
    assert v["C4"].status == "unresolved"
    assert [r.reason for r in v["C4"].rejected] == ["quote_not_in_evidence"]
    # synthesizer misbehaviour was caught by the citation check
    assert {r["reason"].split(":")[0] for r in report.removed_sentences} == {
        "cites_unsupported_or_unknown_claims", "sentence_numbers_not_in_cited_claims"}
    assert report.status == "completed" and out["citation_issues"] == []
    # explicit hand-offs between distinct agents
    nodes = [t["node"] for t in out["trace"]]
    assert nodes[:5] == ["intake", "plan", "research", "extract", "verify"]
    assert "retry" in nodes and nodes[-3:] == ["synthesize", "cite", "report"]
    # 7. within budget
    snap = budget.snapshot()
    assert snap.mesh_llm_calls <= 25 and snap.mesh_embedding_calls <= 10 and snap.retry_rounds <= 2


def test_unsupported_claim_rejection_is_traceable(world):
    script_procurement(world)
    out, _ = run()
    c4 = next(u for u in out["report"].unverified if u["claim_id"] == "C4")
    assert c4["status"] == "unresolved"
    assert c4["rule"].endswith("still insufficient after the final retry round")
    assert c4["rejected_judgments"][0]["reason"] == "quote_not_in_evidence"
    assert c4["rejected_judgments"][0]["evidence_id"] == eid(SCORECARD)
    assert "C4" in out["report"].markdown and "excluded from the findings" in out["report"].markdown


@pytest.mark.parametrize("domain_id,question", [
    ("procurement", "How does the supplier's finance team report invoice disputes and payment terms?"),
    ("insurance", "What do insurers' finance disclosures say about claims reserves for flood coverage?"),
])
def test_finance_questions_are_not_blocked_by_old_ai_only_filter(world, domain_id, question):
    world.planner.structured_script = [plan("What do the documents say about payment terms?")]
    world.generator.structured_script = [{"claims": []}]
    out = run_research(question, BudgetManager(BudgetLimits()), domain_id=domain_id, source_scope=["documents"])
    assert out["status"] != "out_of_scope"
    assert "AI, Generative AI" not in out["report"].markdown
    assert [t["node"] for t in out["trace"]][:3] == ["intake", "plan", "research"]


def test_planner_can_reject_out_of_scope_without_research(world):
    world.planner.structured_script = [plan(in_scope=False)]
    out = run_research("What dose of Drug X should I take tonight?", BudgetManager(BudgetLimits()),
                       domain_id="pharma", source_scope=["documents"])
    assert out["report"].status == "out_of_scope" and world.embeddings.calls == 0


def test_unknown_domain_is_rejected_at_intake(world):
    out = run_research("anything", BudgetManager(BudgetLimits()), domain_id="astrology")
    assert out["report"].status == "out_of_scope" and world.planner.calls == 0


def test_tiny_llm_budget_terminates_without_false_support(world):
    script_procurement(world)
    out, budget = run(BudgetManager(BudgetLimits(max_llm_calls=2)))
    assert budget.snapshot().mesh_llm_calls <= 2
    assert all(v.status != "supported" for v in out["verifications"].values())
    assert out["report"].status == "no_verified_findings" and out["report"].verified_sentences == []


def test_budget_stop_during_extraction_still_reports(world):
    script_procurement(world)
    out, budget = run(BudgetManager(BudgetLimits(max_llm_calls=1)))
    assert out["termination_reason"] == "max_llm_calls"
    assert out["report"].status in {"no_verified_findings", "partial"}
    assert budget.snapshot().mesh_llm_calls == 1


def test_pharma_fact_check_requires_two_sources(world):
    world.docs = [TRIAL_A]
    world.verifier.structured_script = [{"judgments": [
        judgment("C1", TRIAL_A, "supports", "Drug X reduced HbA1c by 1.2 percent versus placebo at 26 weeks")]}]
    claim = "Drug X reduced HbA1c by 1.2 percent at 26 weeks."
    out = run_research("Is Drug X effective?", BudgetManager(BudgetLimits()), domain_id="pharma",
                       source_scope=["documents"], claims_to_check=[claim])
    v = out["verifications"]["C1"]
    assert v.status == "unresolved" and v.required_sources == 2 and v.needs_human_review
    assert out["report"].verified_sentences == []


def test_pharma_fact_check_two_independent_sources_supported(world):
    world.docs = [TRIAL_A, TRIAL_B]
    world.verifier.structured_script = [{"judgments": [
        judgment("C1", TRIAL_A, "supports", "Drug X reduced HbA1c by 1.2 percent versus placebo at 26 weeks"),
        judgment("C1", TRIAL_B, "supports", "found Drug X reduced HbA1c by 1.2 percent at 26 weeks")]}]
    world.generator.structured_script = [{"sentences": [
        {"text": "Drug X reduced HbA1c by 1.2 percent at 26 weeks.", "claim_ids": ["C1"]}]}]
    out = run_research("Is Drug X effective?", BudgetManager(BudgetLimits()), domain_id="pharma",
                       source_scope=["documents"], claims_to_check=["Drug X reduced HbA1c by 1.2 percent at 26 weeks."])
    v = out["verifications"]["C1"]
    assert v.status == "supported" and v.independent_sources == 2 and "C1" in out["report"].human_review
    assert {c.source for c in out["report"].citations} == {"trial_a.pdf", "trial_b.pdf"}


def test_web_evidence_through_budgeted_replay_provider(world, monkeypatch):
    import agents.research_agent
    from retriever.web_search import ReplayWebSearch

    url = "https://www.example.org/acme-news"
    text = "Acme Metals opened a new rolling mill in Ohio in 2025, according to the company."
    monkeypatch.setattr(agents.research_agent, "make_web",
                        lambda b, s: ReplayWebSearch(b, recordings={"Acme Metals news": [{"url": url, "content": text}]}))
    world.planner.structured_script = [plan("Acme Metals news")]
    from agents.research_common import evidence_id
    web_id = evidence_id(url, None, text)
    world.generator.structured_script = [
        {"claims": [{"text": "Acme Metals opened a new rolling mill in Ohio in 2025.", "sub_question_index": 1,
                     "claim_type": "fact", "evidence_ids": [web_id]}]},
        {"sentences": [{"text": "Acme Metals opened a new rolling mill in Ohio in 2025.", "claim_ids": ["C1"]}]}]
    world.verifier.structured_script = [{"judgments": [
        {"claim_id": "C1", "evidence_id": web_id, "stance": "supports",
         "quote": "Acme Metals opened a new rolling mill in Ohio in 2025"}]}]
    budget = BudgetManager(BudgetLimits())
    out = run_research(QUESTION, budget, domain_id="general", source_scope=["web"])
    assert out["verifications"]["C1"].status == "supported"
    assert out["report"].citations[0].url == url and out["report"].citations[0].page is None
    assert budget.snapshot().web_searches == 1


def test_every_model_call_goes_through_the_mesh_gateway(world):
    """No unauthorized provider calls: each recorded Mesh attempt maps to a gateway fake call,
    and the network is blocked for the whole test (conftest)."""
    script_procurement(world)
    out, budget = run()
    mesh_events = [e for e in budget.events if e.get("provider") == "mesh"]
    chat_calls = world.planner.calls + world.generator.calls + world.verifier.calls
    assert len([e for e in mesh_events if e["operation"] != "embedding"]) == chat_calls
    assert len([e for e in mesh_events if e["operation"] == "embedding"]) == world.embeddings.calls
    assert {e["provider"] for e in budget.events} <= {"mesh", "local", "replay", "tavily"}


def test_graph_compiles_with_distinct_agents():
    g = build_research_graph()
    assert [n for n in g.nodes if not n.startswith("__")] == [
        "intake", "plan", "research", "extract", "verify", "retry", "synthesize", "cite", "report"]
