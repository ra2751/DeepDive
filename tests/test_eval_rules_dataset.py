"""Runs the labelled deterministic-rules dataset (evals/datasets/verification_rules_v1.jsonl).

Model judgments in this dataset are hand-written: results describe the rules, not model accuracy.
"""

import json
from pathlib import Path

import pytest

from agents.research_common import evidence_id
from domains.registry import get_domain
from schemas.research import Claim, Evidence, SourceLocator
from verification.aggregation import verify_claim

CASES = [json.loads(l) for l in (Path(__file__).resolve().parents[1] / "evals/datasets/verification_rules_v1.jsonl")
         .read_text().splitlines() if l.strip()]


@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_rule_case(case):
    evs = []
    for d in case["evidence"]:
        evs.append(Evidence(evidence_id=evidence_id(d["source"], d["page"], d["text"]), text=d["text"],
                            retrieved_via="eval", locator=SourceLocator(source=d["source"], source_type="document",
                                                                        page=d["page"])))
    store = {e.evidence_id: e for e in evs}
    judgments = [{"claim_id": "C1", "evidence_id": j.get("evidence_id") or evs[j["evidence_index"]].evidence_id,
                  "stance": j["stance"], "quote": j["quote"]} for j in case["judgments"]]
    v = verify_claim(Claim(claim_id="C1", text=case["claim"], risk=case["risk"]), judgments, store,
                     get_domain(case["domain"]), set(store))
    assert v.status == case["expected_status"], v.rule
    reasons = [r.reason for r in v.rejected]
    for prefix in case["expected_rejection_prefixes"]:
        assert any(r.startswith(prefix) for r in reasons), reasons
    if not case["expected_rejection_prefixes"]:
        assert reasons == []
    assert v.needs_human_review == case["expected_human_review"]


def test_dataset_is_complete():
    assert len(CASES) == 17 and len({c["case_id"] for c in CASES}) == 17
    assert {c["expected_status"] for c in CASES} == {"supported", "contradicted", "insufficient_evidence", "unresolved"}
