"""Grounded synthesis: recorded GR1 sentences as regression fixtures, plus positive and negative cases.

The grounding check is conservative: it rejects sentences that visibly assert more than their cited
claims; it does not prove entailment. Rejected sentences fall back to the verified claim text.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.citation_checker_agent import CitationCheckerAgent
from agents.report_agent import ReportAgent, coverage_lines, requested_items
from budget import BudgetLimits, BudgetManager
from domains.registry import get_domain
from schemas.research import Claim, ClaimVerification, Evidence, ReportSentence, SourceLocator
from verification.citation_integrity import check_quote, validate_sentences
from verification.grounding import grounding_issue
from verification.source_quality import evaluate
from verification.text import negation_conflict, normalize

FX = json.loads((Path(__file__).parent / "fixtures" / "gr1_synthesis.json").read_text())
CLAIMS = {c: Claim(claim_id=c, text=v["text"], claim_type=v["type"]) for c, v in FX["claims"].items()}


def gr1_state():
    quotes = {}
    for v in FX["verifications"].values():
        for e, q in v["accepted_quotes"].items():
            quotes.setdefault(e, []).append(q)
    evidence = {e: Evidence(evidence_id=e, text=" ".join(dict.fromkeys(qs)), retrieved_via="gr1",
                            locator=SourceLocator(source=FX["evidence"][e]["url"], source_type="web",
                                                  url=FX["evidence"][e]["url"], retrieved_at="2026-10-17T10:52:00+00:00"))
                for e, qs in quotes.items()}
    verifs = {c: ClaimVerification(claim_id=c, status=v["status"], supporting_evidence_ids=v["supporting_evidence_ids"],
                                   contradicting_evidence_ids=v["contradicting_evidence_ids"],
                                   accepted_quotes=v["accepted_quotes"], rule=v["rule"] or "recorded",
                                   independent_sources=v["independent_sources"])
              for c, v in FX["verifications"].items()}
    return {"query": FX["question"], "domain": get_domain("general"), "claims": CLAIMS, "verifications": verifs,
            "evidence": evidence, "draft_sentences": [ReportSentence(**s) for s in FX["recorded_sentences"]]}


def run_cite_and_report(state):
    cfg = {"configurable": {"budget": BudgetManager(BudgetLimits())}}
    state = {**state, **CitationCheckerAgent(state, cfg)}
    return state, ReportAgent(state, cfg)["report"]


# ------------------------------------------------------------------ 1. recorded GR1 defects reproduced

def test_gr1_recorded_sentences_all_rejected_with_reasons():
    claims = {c: SimpleNamespace(text=v.text) for c, v in CLAIMS.items()}
    verifs = {c: SimpleNamespace(status=v["status"]) for c, v in FX["verifications"].items()}
    kept, removed = validate_sentences(FX["recorded_sentences"], verifs, claims)
    assert kept == []
    assert [r["reason"] for r in removed] == ["unsupported_ranking_term:biggest",
                                              "unsupported_ranking_term:biggest,significant",
                                              "unsupported_causal_term:if,may"]


def test_gr1_after_fix_report_shows_only_verified_claim_text():
    state, report = run_cite_and_report(gr1_state())
    supported = [c for c, v in FX["verifications"].items() if v["status"] == "supported"]
    assert [s.origin for s in report.verified_sentences] == ["fallback"] * len(supported)
    assert sorted(c for s in report.verified_sentences for c in s.claim_ids) == sorted(supported)  # C2 now shown too
    for s in report.verified_sentences:
        assert s.text == CLAIMS[s.claim_ids[0]].text
    shown = report.markdown.split("## Verified findings")[1].split("## Evidence")[0]   # all findings sections
    for phrase in ("three biggest", "may present risks", "domestic supply chains", "significant quantities"):
        assert phrase not in shown
    assert not state["citation_issues"]
    assert all(c.url.startswith("https://") for c in report.citations)


def test_gr1_forecast_in_its_own_section_and_coverage_explicit():
    _, report = run_cite_and_report(gr1_state())
    md = report.markdown
    findings = md.split("## Verified findings\n")[1].split("\n## ")[0]
    forecasts = md.split("## Supported forecasts and projections\n")[1].split("\n## ")[0]
    assert "projected to grow 48%" in forecasts and "projected to grow 48%" not in findings
    assert "lithium" in findings
    assert report.coverage[0] == ("Growth opportunities: 3 requested; 0 verified claim(s) explicitly describe one "
                                  "(none) — fewer than requested")
    assert report.coverage[1] == ("Commercialization risks: 3 requested; 1 verified claim(s) explicitly describe one "
                                  "(C7) — fewer than requested")
    assert "does not verify rankings" in report.coverage[2]
    assert "## Unresolved claims (excluded from the findings)" in md and "**C5 — unresolved" in md


# ------------------------------------------------------------------ 2. grounding rules: negatives

C_FACT = "India imports 82% of lithium and 76% of silicon from China, risking production delays for precision weapons."
C_FUND = "The National Critical Mineral Mission earmarks INR 16,000 crore to secure 50 overseas mines."
C_FORECAST = "India's defence market is projected to grow 48% from $27 billion in 2025 to $40 billion by 2030."


@pytest.mark.parametrize("sentence, claims, reason", [
    ("The leading risk is that India imports 82% of lithium and 76% of silicon from China.", [C_FACT], "unsupported_ranking_term:leading"),
    ("The most important finding: India imports 82% of lithium and 76% of silicon from China.", [C_FACT], "unsupported_ranking_term:important,most"),
    ("All three risks were found: India imports 82% of lithium and 76% of silicon from China.", [C_FACT], "unsupported_counting_term:all,three"),
    ("Because India imports 82% of lithium from China, startups will lose contracts.", [C_FACT], "unsupported_causal_term:because"),
    ("The mission earmarks INR 16,000 crore for 50 overseas mines, which could cut supply risk.", [C_FUND], "unsupported_causal_term:could"),
    ("The mission's INR 16,000 crore for 50 overseas mines is an opportunity for startups.", [C_FUND], "unsupported_category_term:opportunity"),
    ("India's defence market grew 48% from $27 billion in 2025 to $40 billion by 2030.", [C_FORECAST], "forecast_stated_as_fact"),
    ("India does not import 82% of lithium and 76% of silicon from China.", [C_FACT], "negation_not_in_cited_claims"),
    ("Taken together, lithium imports and mine funding show India's supply strategy is failing badly.",
     [C_FACT, C_FUND], "unsupported_causal_term:together"),
    ("India imports 82% of lithium and 76% of silicon from China, harming export competitiveness and startup valuations.",
     [C_FACT], "unsupported_content:compe,expor,harm,start,valua"),
])
def test_grounding_rejects_assertions_beyond_claims(sentence, claims, reason):
    assert grounding_issue(sentence, claims) == reason


@pytest.mark.parametrize("sentence, claims", [
    ("India imports 82% of lithium and 76% of silicon from China, risking production delays for precision weapons.", [C_FACT]),
    ("According to the source, India imports 82% of lithium and 76% of silicon from China, risking production delays.", [C_FACT]),
    ("India's defence market is projected to grow 48% from $27 billion in 2025 to $40 billion by 2030.", [C_FORECAST]),
    ("The National Critical Mineral Mission earmarks INR 16,000 crore to secure 50 overseas mines.", [C_FUND]),
])
def test_grounding_accepts_faithful_restatements(sentence, claims):
    assert grounding_issue(sentence, claims) is None


def test_ranking_word_allowed_when_the_claim_itself_uses_it():
    claim = "India is the fifth largest military spender in the world."
    assert grounding_issue("India is the fifth largest military spender in the world.", [claim]) is None


def test_fallback_still_runs_every_existing_check():
    # a fallback sentence is the claim text and still passes through numbers and provenance checks
    claims = {"C1": SimpleNamespace(text=C_FACT)}
    kept, removed = validate_sentences([{"text": C_FACT, "claim_ids": ["C1"]}], {"C1": SimpleNamespace(status="unresolved")}, claims)
    assert kept == [] and removed[0]["reason"] == "cites_unsupported_or_unknown_claims:C1"


def test_requested_items_parser():
    assert requested_items(FX["question"]) == [(3, "growth opportunities"), (3, "commercialization risks")]
    assert requested_items("What are the top 5 suppliers of lithium in 2026?") == [(5, "suppliers")]
    assert requested_items("How many attention heads does the base model use?") == []


def test_coverage_lists_sub_questions_without_verified_findings():
    plan = SimpleNamespace(sub_questions=[SimpleNamespace(id="Q1", text="policies"), SimpleNamespace(id="Q2", text="funding risks")])
    claims = {"C1": SimpleNamespace(text="A policy exists.", sub_question_id="Q1")}
    verifs = {"C1": SimpleNamespace(status="supported")}
    assert coverage_lines("What policies exist?", claims, verifs, plan) == [
        "No verified evidence found for sub-question Q2: funding risks"]


# ------------------------------------------------------------------ 3. source quality: Indian government domains

@pytest.mark.parametrize("url, tier", [
    ("https://www.pib.gov.in/PressReleasePage.aspx?PRID=2227612", "primary"),
    ("https://mod.gov.in/en/press-releases", "primary"),
    ("https://ddpmod.nic.in/policy", "primary"),
    ("https://pib.gov.in.evil.example/x", "unverified"),       # lookalike: suffix not at the end
    ("https://fakegov.in/report", "unverified"),                # no dot before gov.in
    ("https://gov.in.example.com/x", "unverified"),
    ("https://mynic.in/x", "unverified"),
])
def test_indian_government_domains(url, tier):
    q = evaluate("web", url, url)
    assert q.tier == tier, (url, q)


# ------------------------------------------------------------------ 4. small verification fixes

def test_never_before_idiom_no_longer_a_negation():
    c8 = FX["c8_rejected"]
    assert c8["reason"] == "negation_mismatch"
    assert negation_conflict(FX["claims"]["C8"]["text"], c8["quote"]) is None
    assert negation_conflict("Prices never fell in 2025.", "Prices fell in 2025.") == "claim_negation_not_supported"
    assert negation_conflict("Prices fell in 2025.", "Prices never fell in 2025.") == "negation_mismatch"


def test_invisible_format_characters_normalized():
    evidence = "Startups raised $432 Mn in H1 2025,⁣ with drone and anti-drone tech accounting for 71% of that funding.⁣⁣"
    assert check_quote("with drone and anti-drone tech accounting for 71% of that funding.", evidence) is None
    assert check_quote("with drone and anti​-drone tech accounting for 71% of that funding.⁣", evidence) is None
    assert normalize("soft­hyphen zero​width") == "softhyphen zerowidth"
    # substantive differences still fail
    assert check_quote("with drone tech accounting for 81% of that funding.", evidence) == "quote_not_in_evidence"
