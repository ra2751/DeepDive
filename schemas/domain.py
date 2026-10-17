from typing import List, Literal

from pydantic import BaseModel


class DomainConfig(BaseModel):
    """Research-domain configuration (data, not code). See domains/configs/*.yaml."""
    id: str
    version: int
    name: str
    description: str
    scope_guidance: str
    out_of_scope_examples: List[str] = []
    blocked_terms: List[str] = []                # deterministic pre-check; normally empty
    preferred_sources: List[Literal["documents", "web"]] = ["documents", "web"]
    web_allowed_domains: List[str] = []
    web_blocked_domains: List[str] = []
    claim_types: List[str]
    high_risk_claim_types: List[str] = []
    min_independent_sources: int = 1
    min_independent_sources_high_risk: int = 1
    human_review_high_risk: bool = True
    planning_guidance: str = ""
    report_disclaimer: str = ""
