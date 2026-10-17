"""
Typed contracts for the self-verifying research workflow.

Two kinds of models live here:
- Records the workflow stores and hands between nodes (Evidence, Claim, ...).
- *Output models the LLM must return (validated by the Mesh gateway).

Claim status is never produced by a model: it is computed by
verification.aggregation from validated evidence judgments.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from verification.source_quality import SourceQuality

ClaimStatus = Literal["supported", "contradicted", "insufficient_evidence", "unresolved"]
Stance = Literal["supports", "contradicts", "irrelevant"]
SourceKind = Literal["documents", "web"]


# ------------------------------------------------------------------ records
class SourceLocator(BaseModel):
    """Where an evidence passage came from (page-level provenance)."""
    source: str                                  # document file name, or URL for web results
    source_type: Literal["document", "web"]
    page: Optional[int] = None                   # 1-based page for documents
    chunk_id: Optional[str] = None               # Milvus primary key, when available
    url: Optional[str] = None
    retrieved_at: Optional[str] = None           # when DeepDive fetched it
    published_at: Optional[str] = None           # publication date, only if the provider reported one


class Evidence(BaseModel):
    evidence_id: str                             # deterministic: hash of source, page and text
    text: str                                    # the exact passage shown to the verifier
    locator: SourceLocator
    retrieved_via: str                           # the query that surfaced it
    sub_question_id: Optional[str] = None
    quality: Optional[SourceQuality] = None      # provisional ranking signal; never affects claim status
    origin: Literal["new", "reused"] = "new"     # "reused" = carried over from the previous question


class SubQuestion(BaseModel):
    id: str
    text: str
    sources: List[SourceKind] = ["documents", "web"]


class ResearchPlan(BaseModel):
    in_scope: bool
    scope_reason: str
    sub_questions: List[SubQuestion] = []
    resolved_question: str = ""                  # follow-up question rewritten to be self-contained


class Claim(BaseModel):
    claim_id: str
    text: str
    sub_question_id: Optional[str] = None
    claim_type: str = "fact"
    risk: Literal["low", "high"] = "low"
    candidate_evidence_ids: List[str] = []
    created_round: int = 0
    user_supplied: bool = False


class EvidenceJudgment(BaseModel):
    """One verifier judgment for a (claim, evidence) pair, as returned by the model."""
    claim_id: str
    evidence_id: str
    stance: Stance
    quote: str = Field(description="Exact sentence or phrase copied verbatim from the evidence text")
    reasoning: str = ""


class RejectedJudgment(BaseModel):
    claim_id: str
    evidence_id: str
    stance: Stance                               # what the verifier proposed
    reason: str
    quote: str = ""                              # the verifier's proposed quote, kept for audit
    source: Optional[str] = None                 # locator of the evidence, when it exists
    page: Optional[int] = None
    url: Optional[str] = None


class ClaimVerification(BaseModel):
    claim_id: str
    status: ClaimStatus
    supporting_evidence_ids: List[str] = []
    contradicting_evidence_ids: List[str] = []
    accepted_quotes: dict = {}                   # evidence_id -> verified quote
    rejected: List[RejectedJudgment] = []
    independent_sources: int = 0
    required_sources: int = 1
    rule: str
    needs_human_review: bool = False
    round: int = 0


class Citation(BaseModel):
    marker: str                                  # e.g. "C1"
    claim_id: str
    evidence_id: str
    source: str
    page: Optional[int] = None
    url: Optional[str] = None
    quote: str


class ReportSentence(BaseModel):
    text: str
    claim_ids: List[str]
    origin: Literal["model", "template", "fallback"] = "model"   # how the sentence was produced


class ResearchReport(BaseModel):
    question: str
    domain_id: str
    status: Literal["completed", "partial", "no_verified_findings", "out_of_scope"]
    verified_sentences: List[ReportSentence] = []
    citations: List[Citation] = []
    unverified: List[dict] = []                  # contradicted / unresolved claims with reasons
    human_review: List[str] = []                 # claim ids flagged for review
    removed_sentences: List[dict] = []           # dropped by the citation check, with reasons
    coverage: List[str] = []                     # requested items vs what was verified (deterministic)
    markdown: str = ""
    termination_reason: Optional[str] = None


# ------------------------------------------------------- LLM output models
class PlanOutput(BaseModel):
    in_scope: bool = Field(description="False only if the question is outside the research domain's scope")
    scope_reason: str
    sub_questions: List[str] = Field(description="Up to 4 focused research sub-questions")
    resolved_question: str = Field(default="", description="For follow-ups: the question rewritten to be "
                                   "self-contained, resolving references to the previous research")


class ExtractedClaim(BaseModel):
    text: str = Field(description="One atomic, checkable factual statement")
    sub_question_index: int = Field(description="1-based index of the sub-question it answers")
    claim_type: str
    evidence_ids: List[str] = Field(description="IDs of evidence passages the claim is based on")


class ClaimExtractionOutput(BaseModel):
    claims: List[ExtractedClaim]


class VerificationBatchOutput(BaseModel):
    judgments: List[EvidenceJudgment]


class SynthesisOutput(BaseModel):
    sentences: List[ReportSentence] = Field(
        description="Report sentences; each cites the claim IDs it is based on and adds no other facts")
