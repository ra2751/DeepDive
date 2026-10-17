"""Citation check (deterministic): keep only sentences backed by supported claims, build and validate
page-level citations from the accepted evidence."""

from langchain_core.runnables import RunnableConfig

from agents.research_common import step
from schemas.research import ReportSentence
from verification.citation_integrity import build_citations, validate_citations, validate_sentences


def CitationCheckerAgent(state, config: RunnableConfig):
    claims, verifications = state.get("claims") or {}, state.get("verifications") or {}
    evidence = state.get("evidence") or {}
    kept, removed = validate_sentences(state.get("draft_sentences") or [], verifications, claims)

    # Deterministic fallback: every supported claim left without a valid sentence (because generated
    # sentences failed the integrity or grounding checks, or were never written) is reported with its own
    # verified claim text. The fallback sentences go through exactly the same checks as model sentences.
    fallback_ids = []
    covered = {c for s in kept for c in s.claim_ids}
    uncovered = [cid for cid, v in verifications.items() if v.status == "supported" and cid not in covered]
    if uncovered and (removed or kept):
        fb_kept, fb_removed = validate_sentences(
            [ReportSentence(text=claims[c].text, claim_ids=[c], origin="fallback") for c in uncovered],
            verifications, claims)
        kept += fb_kept
        removed += fb_removed
        fallback_ids = [s.claim_ids[0] for s in fb_kept]
    citations = build_citations(verifications, evidence)
    issues = validate_citations(citations, verifications, evidence)
    if issues:  # defensive: drop any sentence relying on a claim with an invalid citation
        bad = {i.split(":", 1)[0] for i in issues}
        removed += [{"text": s.text, "claim_ids": s.claim_ids, "reason": "citation_integrity_failure"}
                    for s in kept if set(s.claim_ids) & bad]
        kept = [s for s in kept if not set(s.claim_ids) & bad]
        citations = [c for c in citations if c.marker not in bad]
    notices = [f"{len(fallback_ids)} finding(s) shown as their verified claim text because no generated sentence "
               f"for them passed the citation and grounding checks (deterministic fallback): {', '.join(fallback_ids)}"] if fallback_ids else []
    return {"verified_sentences": kept, "citations": citations, "removed_sentences": removed,
            "citation_issues": issues, "notices": notices,
            "trace": [step("cite", agent="Citation checker", kept=len(kept), removed=len(removed), citations=len(citations),
                           integrity_issues=len(issues), fallback_sentences=len(fallback_ids),
                           fallback_claims=fallback_ids)]}
