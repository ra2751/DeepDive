"""Claim extractor: atomic, checkable claims tied to evidence IDs (1 Mesh call, generator model).

In fact-check mode the user's claims are used verbatim and no model call is made.
"""

from langchain_core.runnables import RunnableConfig

from agents.research_common import MAX_CLAIMS, UNTRUSTED_NOTE, evidence_block, step
from budget import BudgetExceeded, budget_from_config
from config.settings import get_settings
from gateway import MeshError, structured
from schemas.research import Claim, ClaimExtractionOutput
from verification.meta_claims import is_meta_claim
from verification.text import normalize

PROMPT = """You extract research claims for a {domain_name} research task.
{untrusted}

<question>{question}</question>
Sub-questions:
{subs}

Evidence passages:
{evidence}

Write at most {max_claims} ATOMIC claims (one checkable fact each, with exact numbers, dates and names)
that answer the question and are stated in the evidence. For each claim give the 1-based sub-question
index, a claim_type from {types}, and the IDs of the evidence passages it comes from. A projection,
estimate or expectation about the future is a forecast: keep who made it (e.g. "according to X") and never
state it as an outcome. Keep the evidence's own dates; do not add today's date or facts
that are not in the evidence. Never write claims about the evidence itself (for example "X is not
mentioned in the evidence"); if the evidence does not answer a sub-question, write no claim for it."""


def _risk(claim_type: str, domain) -> str:
    return "high" if claim_type in domain.high_risk_claim_types else "low"


def select_round_robin(items: list, limit: int) -> list:
    """Keep at most `limit` claims, taking them round-robin across sub-questions (in extraction order)
    so every sub-question stays represented where possible."""
    groups = {}
    for it in items:
        groups.setdefault(it.sub_question_index, []).append(it)
    out, queues = [], list(groups.values())
    while len(out) < limit and any(queues):
        for q in queues:
            if q and len(out) < limit:
                out.append(q.pop(0))
    order = {id(it): i for i, it in enumerate(items)}
    return sorted(out, key=lambda it: order[id(it)])


def ClaimExtractorAgent(state, config: RunnableConfig):
    domain, evidence, plan = state["domain"], state.get("evidence") or {}, state["plan"]
    if state.get("termination_reason"):
        return {"trace": [step("extract", skipped="run stopped")]}

    settings = get_settings()
    if state.get("claims_to_check"):
        # User-supplied claims have no declared type; in domains with high-risk claim types they are
        # treated as high risk (stricter source requirement and human review).
        risk = "high" if domain.high_risk_claim_types else "low"
        supplied = [c.strip() for c in state["claims_to_check"] if c and c.strip()]
        limit = settings.max_fact_check_claims
        claims = {f"C{i}": Claim(claim_id=f"C{i}", text=t, sub_question_id="Q1", risk=risk, user_supplied=True)
                  for i, t in enumerate(supplied[:limit], 1)}
        notices = []
        if len(supplied) > limit:   # never truncate silently
            notices.append(f"Only the first {limit} of {len(supplied)} supplied claims were checked "
                           f"(MAX_FACT_CHECK_CLAIMS). Not checked: " + " | ".join(supplied[limit:]))
        return {"claims": claims, "pending_claim_ids": list(claims), "notices": notices,
                "trace": [step("extract", agent="Claim extractor", mode="fact-check", claims=len(claims),
                               not_checked=max(0, len(supplied) - limit), mesh_calls=0)]}

    if not evidence:
        return {"pending_claim_ids": [], "trace": [step("extract", claims=0, reason="no evidence collected")]}

    budget = budget_from_config(config)
    prompt = PROMPT.format(domain_name=domain.name, untrusted=UNTRUSTED_NOTE,
                           question=plan.resolved_question or state["query"],
                           subs="\n".join(f"{i}. {s.text}" for i, s in enumerate(plan.sub_questions, 1)),
                           evidence="\n\n".join(evidence_block(e) for e in evidence.values()),
                           max_claims=settings.max_research_claims, types=domain.claim_types)
    try:
        out: ClaimExtractionOutput = structured("generator", ClaimExtractionOutput, budget).invoke(prompt)
    except BudgetExceeded as e:
        return {"termination_reason": e.reason, "trace": [step("extract", result="budget stop", reason=e.reason)]}
    except MeshError as e:
        return {"termination_reason": "mesh_error", "notices": [f"Claim extraction failed: {type(e).__name__}"],
                "trace": [step("extract", result="mesh error")]}

    unique, seen, meta = [], set(), []
    for ec in out.claims:
        key = normalize(ec.text)
        if key and is_meta_claim(ec.text):
            meta.append(ec.text)          # evidence-absence statements are never research findings
            continue
        if key and key not in seen:
            seen.add(key)
            unique.append(ec)
    selected = select_round_robin(unique, settings.max_research_claims)
    claims = {}
    for ec in selected:
        ctype = ec.claim_type if ec.claim_type in domain.claim_types else "fact"
        subs = plan.sub_questions
        sq = subs[ec.sub_question_index - 1].id if 1 <= ec.sub_question_index <= len(subs) else None
        cid = f"C{len(claims) + 1}"
        claims[cid] = Claim(claim_id=cid, text=ec.text.strip(), sub_question_id=sq, claim_type=ctype,
                            risk=_risk(ctype, domain),
                            candidate_evidence_ids=[e for e in ec.evidence_ids if e in evidence])
    notices = [f"{len(meta)} statement(s) about missing evidence were not treated as findings."] if meta else []
    return {"claims": claims, "pending_claim_ids": list(claims), "notices": notices,
            "trace": [step("extract", agent="Claim extractor", claims=len(claims), proposed=len(out.claims),
                           dropped=len(out.claims) - len(claims), meta_claims_filtered=len(meta),
                           limit=settings.max_research_claims)]}
