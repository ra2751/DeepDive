"""
Self-verifying research graph:

  intake -> plan -> research -> extract -> verify ⇄ retry -> synthesize -> cite -> report

The legacy answer-level graph (graph_builder.build_graph) is kept unchanged as the rollback path.
"""

from langgraph.graph import END, START, StateGraph

from agents.citation_checker_agent import CitationCheckerAgent
from agents.claim_extractor_agent import ClaimExtractorAgent
from agents.intake_agent import IntakeAgent
from agents.planner_agent import PlannerAgent
from agents.report_agent import ReportAgent
from agents.research_agent import ResearchAgent
from agents.research_common import synthesis_reserve
from agents.research_state import ResearchState
from agents.retry_agent import RetryAgent
from agents.synthesizer_agent import SynthesizerAgent
from agents.verifier_agent import VerifierAgent
from budget import budget_from_config

RECURSION_LIMIT = 40


def after_intake(state):
    return "report" if state.get("status") == "out_of_scope" else "plan"


def after_plan(state):
    if state.get("status") == "out_of_scope":
        return "report"
    return "synthesize" if state.get("termination_reason") else "research"


def after_extract(state):
    if state.get("termination_reason") or not state.get("pending_claim_ids"):
        return "synthesize"
    return "verify"


def after_verify(state, config):
    """Targeted retry only for insufficient claims, while rounds and budget remain."""
    if state.get("termination_reason"):
        return "synthesize"
    if not any(v.status == "insufficient_evidence" for v in (state.get("verifications") or {}).values()):
        return "synthesize"
    if state.get("retry_round", 0) > 0 and not state.get("retry_found_new_evidence", True):
        return "synthesize"
    budget = budget_from_config(config)
    remaining = budget.llm_calls_remaining()
    if budget.retry_rounds_remaining() == 0 or budget.exhausted() or (
            remaining is not None and remaining <= synthesis_reserve(state)):
        return "synthesize"
    return "retry"


def after_retry(state):
    return "verify" if state.get("pending_claim_ids") else "synthesize"


def build_research_graph(checkpointer=None):
    g = StateGraph(ResearchState)
    g.add_node("intake", IntakeAgent)
    g.add_node("plan", PlannerAgent)
    g.add_node("research", ResearchAgent)
    g.add_node("extract", ClaimExtractorAgent)
    g.add_node("verify", VerifierAgent)
    g.add_node("retry", RetryAgent)
    g.add_node("synthesize", SynthesizerAgent)
    g.add_node("cite", CitationCheckerAgent)
    g.add_node("report", ReportAgent)

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", after_intake, {"plan": "plan", "report": "report"})
    g.add_conditional_edges("plan", after_plan, {"research": "research", "synthesize": "synthesize", "report": "report"})
    g.add_edge("research", "extract")
    g.add_conditional_edges("extract", after_extract, {"verify": "verify", "synthesize": "synthesize"})
    g.add_conditional_edges("verify", after_verify, {"retry": "retry", "synthesize": "synthesize"})
    g.add_conditional_edges("retry", after_retry, {"verify": "verify", "synthesize": "synthesize"})
    g.add_edge("synthesize", "cite")
    g.add_edge("cite", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


MAX_REUSED_EVIDENCE = 20


def prior_context_from(out: dict) -> dict:
    """Follow-up context from a finished run (kept only in the caller's session).

    Carries the question, the verified findings and only the evidence that was ACCEPTED as
    supporting or contradicting some claim, strongest sources first. Previous statuses are not
    carried over: the follow-up re-verifies everything it concludes.
    """
    verifs, evidence = out.get("verifications") or {}, out.get("evidence") or {}
    accepted = [e for v in verifs.values() for e in v.supporting_evidence_ids + v.contradicting_evidence_ids]
    accepted = [e for e in dict.fromkeys(accepted) if e in evidence]
    accepted.sort(key=lambda e: -(evidence[e].quality.score if evidence[e].quality else 0.0))
    plan, report = out.get("plan"), out.get("report")
    return {"question": (plan.resolved_question if plan and plan.resolved_question else out.get("query", "")),
            "domain_id": out["domain"].id if out.get("domain") else out.get("domain_id"),
            "findings": [s.text for s in (report.verified_sentences if report else [])],
            "evidence": {e: evidence[e] for e in accepted[:MAX_REUSED_EVIDENCE]}}


def run_research(query, budget, domain_id="general", source_scope=None, claims_to_check=None, graph=None,
                 prior_context=None):
    """Run one research question under a run-scoped BudgetManager; returns the final state."""
    state = {"query": query, "domain_id": domain_id, "notices": [], "trace": []}
    if prior_context:
        state["prior_context"] = prior_context
    if source_scope is not None:
        state["source_scope"] = source_scope
    if claims_to_check:
        state["claims_to_check"] = claims_to_check
    graph = graph or build_research_graph()
    return graph.invoke(state, config={"configurable": {"budget": budget}, "recursion_limit": RECURSION_LIMIT})
