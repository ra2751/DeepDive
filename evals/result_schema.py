"""
Evaluation-result format (schema_version 1.0) for DeepDive live evaluations.

Results files are written by evals/run_live_eval.py to evals/results/<UTC timestamp>_<commit>.json
together with a Markdown summary. Numbers in the README must come from a committed results file.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class CaseUsage(BaseModel):
    mesh_llm_calls: int
    mesh_embedding_calls: int
    web_searches: int
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    cost_basis: Literal["reported", "estimated", "unknown"]
    cost_incomplete: bool = False
    elapsed_s: float


class CaseResult(BaseModel):
    case_id: str
    mode: Literal["fact_check", "research"]
    domain: str
    expected: dict
    observed: dict                      # claim statuses and/or report status
    passed: bool
    report_status: str
    verified_sentences: int
    unsupported_claim_leakage: int      # verified sentences citing a non-supported claim (target 0)
    citation_integrity_issues: int      # target 0
    termination_reason: Optional[str] = None
    usage: CaseUsage
    error: Optional[str] = None


class Metric(BaseModel):
    numerator: int
    denominator: int
    rate: Optional[float] = None


class EvalRun(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    kind: Literal["live", "offline_fake"]   # offline_fake runs are pipeline checks, never accuracy claims
    created_at: str
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None          # uncommitted changes present when the run started
    code_sha256: Optional[str] = None         # fingerprint of all project .py/.yaml/.jsonl files
    dataset: str
    dataset_sha256: str
    models: Dict[str, str]                   # role -> requested Mesh model id
    models_served: List[str] = []
    limits: dict
    cases: List[CaseResult]
    metrics: Dict[str, Metric]
    totals: CaseUsage
    notes: List[str] = []
