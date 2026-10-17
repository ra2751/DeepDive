"""
Shared state for the self-verifying research graph.

Each node reads what it needs and returns only the keys it owns, so hand-offs
between agents are explicit:

  intake      -> domain, source_scope, status
  plan        -> plan
  research    -> evidence (merge)
  extract     -> claims (merge), pending_claim_ids
  verify      -> verifications (merge), shown_evidence (merge), pending_claim_ids
  retry       -> evidence (merge), shown_evidence (merge), pending_claim_ids, retry_round
  synthesize  -> verifications (final statuses), draft_sentences
  cite        -> verified_sentences, citations, removed_sentences, citation_issues
  report      -> report, status, usage
"""

import operator
from typing import Annotated, Optional, TypedDict


def merge(a: Optional[dict], b: Optional[dict]) -> dict:
    """Keyed upsert reducer: later values replace earlier ones for the same key."""
    return {**(a or {}), **(b or {})}


class ResearchState(TypedDict, total=False):
    query: str
    domain_id: str
    source_scope: list                       # subset of ["documents", "web"]
    claims_to_check: list                    # optional user-supplied claims (fact-check mode)
    prior_context: Optional[dict]            # follow-up: previous question, verified findings, accepted
                                             # evidence (same session only; see research_graph.prior_context_from)

    domain: object                           # schemas.domain.DomainConfig
    plan: object                             # schemas.research.ResearchPlan
    evidence: Annotated[dict, merge]         # evidence_id -> Evidence
    claims: Annotated[dict, merge]           # claim_id -> Claim
    verifications: Annotated[dict, merge]    # claim_id -> ClaimVerification
    shown_evidence: Annotated[dict, merge]   # claim_id -> evidence ids shown to the verifier
    independently_searched: list             # claim ids that already had independent retrieval
    pending_claim_ids: list                  # claims awaiting (re-)verification
    fresh_evidence: dict                     # claim_id -> evidence ids found by the latest retry round
    retry_round: int
    retry_found_new_evidence: bool

    draft_sentences: list
    verified_sentences: list
    citations: list
    removed_sentences: list
    citation_issues: list

    report: object                           # schemas.research.ResearchReport
    status: str
    termination_reason: Optional[str]
    trace: Annotated[list, operator.add]     # one entry per node hand-off
    notices: Annotated[list, operator.add]
    usage: dict
