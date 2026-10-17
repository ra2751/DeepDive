"""Verifier: independent evidence retrieval + batched judgments by the verifier model (Gemini via Mesh).

The model only judges (claim, evidence) pairs and copies quotes. Status is decided by
verification.aggregation, which rejects judgments that fail the integrity checks.
"""

from langchain_core.runnables import RunnableConfig

from agents.research_agent import make_retriever
from agents.research_common import (MAX_EVIDENCE_PER_CLAIM, UNTRUSTED_NOTE, VERIFY_BATCH_SIZE, doc_evidence,
                                    evidence_block, rank_for_claim, step, synthesis_reserve)
from budget import BudgetExceeded, budget_from_config
from config.settings import get_settings
from gateway import MeshError, structured
from monitoring.logger import setup_logger
from schemas.research import ClaimVerification, VerificationBatchOutput
from verification.aggregation import verify_claim

logger = setup_logger()
INDEPENDENT_TOP_K = 3

PROMPT = """You are an independent fact verifier. For EACH claim, judge EACH evidence passage listed for it.
{untrusted}

Rules:
- stance "supports" only if the passage itself states the claim (same facts, numbers, dates, names).
- stance "contradicts" only if the passage states something incompatible with the claim.
- otherwise stance "irrelevant".
- quote: copy the exact sentence or phrase from the passage that justifies the stance, character for
  character. Never paraphrase, never combine passages, never invent text.
- Judge only the evidence IDs listed for that claim.

{claims}

Evidence passages:
{evidence}"""


def _not_verified(claim_id, round_no, reason):
    return ClaimVerification(claim_id=claim_id, status="insufficient_evidence", rule=reason, round=round_no)


def VerifierAgent(state, config: RunnableConfig):
    budget, settings = budget_from_config(config), get_settings()
    claims, evidence = state.get("claims") or {}, dict(state.get("evidence") or {})
    domain, scope, round_no = state["domain"], state.get("source_scope") or [], state.get("retry_round", 0)
    pending = [c for c in (state.get("pending_claim_ids") or []) if c in claims]
    if not pending:
        return {"pending_claim_ids": [], "trace": [step("verify", claims=0, mesh_calls=0)]}

    shown = {cid: list(dict.fromkeys((state.get("shown_evidence") or {}).get(cid, [])
                                      + claims[cid].candidate_evidence_ids)) for cid in pending}
    new_evidence, notices = {}, []

    # Independent retrieval: each claim is searched again by its own text (one batched embedding),
    # so the verifier also sees evidence the extractor did not pick, including contradicting passages.
    todo = [c for c in pending if c not in (state.get("independently_searched") or [])]
    if "documents" in scope and todo and not state.get("termination_reason"):
        try:
            results = make_retriever(budget, settings).search_many([claims[c].text for c in todo],
                                                                    top_k=INDEPENDENT_TOP_K)
            for cid, hits in zip(todo, results):
                for h in hits:
                    ev = doc_evidence(h, f"independent:{cid}")
                    evidence.setdefault(ev.evidence_id, ev)
                    new_evidence.setdefault(ev.evidence_id, evidence[ev.evidence_id])
                    shown[cid].append(ev.evidence_id)
        except BudgetExceeded as e:
            notices.append(f"Independent retrieval skipped ({e.reason})")
        except Exception as e:
            notices.append(f"Independent retrieval unavailable: {type(e).__name__}")
    previous = state.get("verifications") or {}
    fresh = state.get("fresh_evidence") or {}
    for cid in shown:
        prev = previous.get(cid)
        accepted = (prev.supporting_evidence_ids + prev.contradicting_evidence_ids) if prev else []
        shown[cid] = rank_for_claim(claims[cid].candidate_evidence_ids, shown[cid], evidence, MAX_EVIDENCE_PER_CLAIM,
                                    accepted=accepted, fresh=fresh.get(cid, []))

    verifications, calls, stop = {}, 0, state.get("termination_reason")
    for i in range(0, len(pending), VERIFY_BATCH_SIZE):
        batch = pending[i:i + VERIFY_BATCH_SIZE]
        remaining = budget.llm_calls_remaining()
        if stop or (remaining is not None and remaining <= synthesis_reserve(state)):
            reason = stop or "not verified: LLM calls reserved for synthesis"
            for cid in batch:
                verifications[cid] = _not_verified(cid, round_no, reason)
            continue
        ev_ids = list(dict.fromkeys(e for cid in batch for e in shown[cid]))
        claims_txt = "\n".join(f'Claim {cid}: "{claims[cid].text}"  (evidence to judge: {", ".join(shown[cid]) or "none"})'
                               for cid in batch)
        prompt = PROMPT.format(untrusted=UNTRUSTED_NOTE, claims=claims_txt,
                               evidence="\n\n".join(evidence_block(evidence[e]) for e in ev_ids) or "(none)")
        try:
            out: VerificationBatchOutput = structured("verifier", VerificationBatchOutput, budget).invoke(prompt)
            calls += 1
        except BudgetExceeded as e:
            stop = e.reason
            for cid in batch:
                verifications[cid] = _not_verified(cid, round_no, f"not verified: {e.reason}")
            continue
        except MeshError as e:
            logger.error(f"VerifierAgent | verification call failed: {type(e).__name__}")
            notices.append(f"Verification call failed: {type(e).__name__}")
            for cid in batch:
                verifications[cid] = _not_verified(cid, round_no, "not verified: verification call failed")
            continue
        for cid in batch:
            verifications[cid] = verify_claim(claims[cid], out.judgments, evidence, domain, set(shown[cid]), round_no)

    # Keep the audit trail across rounds: earlier rejected judgments are carried forward.
    for cid, v in verifications.items():
        if cid in previous and previous[cid].rejected:
            seen = {(r.evidence_id, r.reason) for r in v.rejected}
            carried = [r for r in previous[cid].rejected if (r.evidence_id, r.reason) not in seen]
            verifications[cid] = v.model_copy(update={"rejected": carried + v.rejected})

    counts = {}
    for v in verifications.values():
        counts[v.status] = counts.get(v.status, 0) + 1
    out = {"verifications": verifications, "shown_evidence": shown, "evidence": new_evidence,
           "independently_searched": list(dict.fromkeys((state.get("independently_searched") or []) + todo)),
           "pending_claim_ids": [], "notices": notices, "fresh_evidence": {},
           "trace": [step("verify", agent="Verifier", round=round_no, claims=len(pending), claim_ids=pending,
                          verifier_calls=calls, statuses=counts,
                          fresh_shown={c: [e for e in fresh.get(c, []) if e in shown[c]] for c in pending if fresh.get(c)},
                          rejected_judgments=sum(len(v.rejected) for v in verifications.values()))]}
    if stop and not state.get("termination_reason"):
        out["termination_reason"] = stop
    return out
