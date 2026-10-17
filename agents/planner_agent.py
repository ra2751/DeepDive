"""Planner: scope verdict + up to 4 research sub-questions (1 Mesh call, supervisor model)."""

from langchain_core.runnables import RunnableConfig

from agents.research_common import MAX_SUBQUESTIONS, UNTRUSTED_NOTE, step
from budget import BudgetExceeded, budget_from_config
from gateway import MeshError, structured
from schemas.research import PlanOutput, ResearchPlan, SubQuestion

PROMPT = """You are the research planner for a {domain_name} research assistant.
Domain scope: {scope}
Out of scope examples: {oos}
Planning guidance: {guidance}
{untrusted}

<question>{question}</question>

Decide whether the question is in scope. If it is, write at most {max_sq} focused sub-questions that
together answer it, each answerable from documents or public web sources."""

FOLLOW_UP = """
This question is a FOLLOW-UP to earlier research in the same session.
<question>Previous question: {prev}</question>
<evidence id="previous-verified-findings">
{findings}
</evidence>
First rewrite the new question as a self-contained question (resolved_question), resolving references
such as "it", "they" or "that supplier" using the previous research. Then plan sub-questions for the
resolved question. Previous findings are context, not proof: new conclusions will be verified again."""


def PlannerAgent(state, config: RunnableConfig):
    domain, query = state["domain"], state["query"]
    scope = state.get("source_scope") or []
    if state.get("claims_to_check"):
        plan = ResearchPlan(in_scope=True, scope_reason="fact-check mode: user-supplied claims",
                            sub_questions=[SubQuestion(id="Q1", text=query, sources=scope)])
        return {"plan": plan, "trace": [step("plan", agent="Planner", mode="fact-check", sub_questions=1, mesh_calls=0)]}
    budget = budget_from_config(config)
    prompt = PROMPT.format(domain_name=domain.name, scope=domain.scope_guidance.strip(),
                           oos="; ".join(domain.out_of_scope_examples) or "none",
                           guidance=domain.planning_guidance or "none", untrusted=UNTRUSTED_NOTE,
                           question=query, max_sq=MAX_SUBQUESTIONS)
    prior = state.get("prior_context")
    if prior:
        prompt += FOLLOW_UP.format(prev=prior.get("question", ""),
                                   findings="\n".join(f"- {f}" for f in prior.get("findings") or []) or "(none)")
    try:
        out: PlanOutput = structured("supervisor", PlanOutput, budget).invoke(prompt)
    except BudgetExceeded as e:
        return {"termination_reason": e.reason, "trace": [step("plan", result="budget stop", reason=e.reason)]}
    except MeshError as e:
        return {"termination_reason": "mesh_error", "notices": [f"Planning failed: {type(e).__name__}"],
                "trace": [step("plan", result="mesh error")]}
    if not out.in_scope:
        return {"plan": ResearchPlan(in_scope=False, scope_reason=out.scope_reason), "status": "out_of_scope",
                "notices": [f"Out of scope for {domain.name}: {out.scope_reason}"],
                "trace": [step("plan", result="out of scope", reason=out.scope_reason)]}
    resolved = (out.resolved_question or "").strip() if prior else ""
    subs = [s.strip() for s in out.sub_questions if s and s.strip()][:MAX_SUBQUESTIONS] or [resolved or query]
    plan = ResearchPlan(in_scope=True, scope_reason=out.scope_reason, resolved_question=resolved,
                        sub_questions=[SubQuestion(id=f"Q{i}", text=t, sources=scope) for i, t in enumerate(subs, 1)])
    return {"plan": plan, "trace": [step("plan", agent="Planner", sub_questions=len(subs),
                                         resolved_question=resolved or None)]}
