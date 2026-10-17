"""
Rule-based claim status aggregation. Model confidence is never an input.

  supported            >= required independent sources with accepted "supports" judgments,
                       and no accepted contradiction
  contradicted         accepted contradiction(s) and no accepted support
  unresolved           accepted support AND contradiction (conflicting evidence), or still
                       insufficient after the final retry round (see finalize_status)
  insufficient_evidence anything else (triggers targeted retry while budget allows)
"""

from schemas.domain import DomainConfig
from schemas.research import Claim, ClaimVerification, EvidenceJudgment, RejectedJudgment
from verification.citation_integrity import check_quote, check_relevance, independence_key


def required_sources(claim: Claim, policy: DomainConfig) -> int:
    return policy.min_independent_sources_high_risk if claim.risk == "high" else policy.min_independent_sources


def verify_claim(claim: Claim, judgments: list, evidence: dict, policy: DomainConfig,
                 shown_evidence_ids: set, round_no: int = 0) -> ClaimVerification:
    supports, contradicts, quotes, rejected = [], [], {}, []
    for j in judgments:
        j = j if isinstance(j, EvidenceJudgment) else EvidenceJudgment(**j)
        if j.claim_id != claim.claim_id or j.stance == "irrelevant":
            continue
        reason = None
        ev = evidence.get(j.evidence_id)
        if ev is None:
            reason = "unknown_evidence_id"
        elif j.evidence_id not in shown_evidence_ids:
            reason = "evidence_not_shown_for_this_claim"
        else:
            reason = check_quote(j.quote, ev.text) or check_relevance(claim.text, j.quote, j.stance, ev.text)
        if reason:
            loc = ev.locator if ev is not None else None
            rejected.append(RejectedJudgment(claim_id=claim.claim_id, evidence_id=j.evidence_id,
                                             stance=j.stance, reason=reason, quote=j.quote,
                                             source=loc.source if loc else None, page=loc.page if loc else None,
                                             url=loc.url if loc else None))
            continue
        target = supports if j.stance == "supports" else contradicts
        if j.evidence_id not in target:
            target.append(j.evidence_id)
            quotes[j.evidence_id] = j.quote

    independent = len({independence_key(evidence[e]) for e in supports})
    need = required_sources(claim, policy)
    if supports and contradicts:
        status, rule = "unresolved", "conflicting accepted evidence (supports and contradicts)"
    elif contradicts:
        status, rule = "contradicted", "accepted contradicting evidence, no accepted support"
    elif independent >= need:
        status, rule = "supported", f"{independent} independent supporting source(s) >= {need} required"
    elif supports:
        status, rule = "insufficient_evidence", f"{independent} independent source(s) < {need} required"
    else:
        status, rule = "insufficient_evidence", "no accepted supporting or contradicting evidence"

    return ClaimVerification(
        claim_id=claim.claim_id, status=status, supporting_evidence_ids=supports,
        contradicting_evidence_ids=contradicts, accepted_quotes=quotes, rejected=rejected,
        independent_sources=independent, required_sources=need, rule=rule,
        needs_human_review=policy.human_review_high_risk and claim.risk == "high", round=round_no)


def finalize_status(v: ClaimVerification, reason: str) -> ClaimVerification:
    """After the last retry round (or when the budget is exhausted) insufficient -> unresolved."""
    if v.status != "insufficient_evidence":
        return v
    return v.model_copy(update={"status": "unresolved", "rule": f"{v.rule}; {reason}"})


def is_conflict(v: ClaimVerification) -> bool:
    return bool(v.supporting_evidence_ids and v.contradicting_evidence_ids)


def explain_status(v: ClaimVerification) -> str:
    """Plain-language reason for a status (shown in the UI and the exported report)."""
    if v.status == "supported":
        return (f"Supported: {v.independent_sources} independent source(s) with an accepted supporting quote "
                f"(≥ {v.required_sources} required) and no accepted contradiction.")
    if v.status == "contradicted":
        return "Contradicted: accepted contradicting evidence and no accepted supporting evidence."
    if v.status == "unresolved" and is_conflict(v):
        return ("Unresolved — conflicting evidence: accepted quotes both support and contradict this claim. "
                "DeepDive does not pick a winner by source ranking; resolve it by human review.")
    if v.status == "unresolved":
        return f"Unresolved: not enough accepted evidence ({v.rule})."
    return f"Insufficient evidence so far ({v.rule})."
