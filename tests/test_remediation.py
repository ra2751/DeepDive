"""Stage A remediation (D1–D5): regression tests, including the live L08 output as a fixture.

Scripted fixtures only; nothing here is a live accuracy measurement.
"""

import json
import logging
from pathlib import Path

import pytest

from budget import BudgetLimits, BudgetManager
from domains.registry import get_domain
from graph_builder.research_graph import run_research
from schemas.research import Claim, ClaimVerification, Evidence, SourceLocator
from tests.research_fixtures import AUDIT, CONTRACT, SCORECARD, TRIAL_A, eid, judgment
from tests.research_world import QUESTION, install_world, plan, script_procurement
from verification.aggregation import verify_claim
from verification.citation_integrity import (check_relevance, citation_markers, strip_citation_markers,
                                             validate_sentences)
from verification.text import numbers

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "stage_a_l08.json").read_text())
GENERAL = get_domain("general")


def _verifs(statuses):
    return {cid: ClaimVerification(claim_id=cid, status=s, rule="fixture") for cid, s in statuses.items()}


def _claims(texts):
    return {cid: Claim(claim_id=cid, text=t) for cid, t in texts.items()}


# ---------------------------------------------------------------- D1 markers
def test_live_l08_sentences_are_now_kept():
    """Stage A: all 11 sentences were removed (sentence_numbers_not_in_cited_claims:<marker digit>)."""
    assert all(r["reason"].startswith("sentence_numbers_not_in_cited_claims") for r in FIXTURE["removed_sentences"])
    kept, removed = validate_sentences(FIXTURE["removed_sentences"], _verifs(FIXTURE["statuses"]),
                                       _claims(FIXTURE["claims"]))
    assert len(kept) == 11 and removed == []


def test_marker_digits_are_not_factual_numbers_but_real_numbers_still_are():
    s = "The encoder of the Transformer is composed of 6 identical layers (C3)."
    assert numbers(s) == {"6", "3"} and numbers(strip_citation_markers(s)) == {"6"}
    assert strip_citation_markers(s).startswith("The encoder of the Transformer is composed of 6")


@pytest.mark.parametrize("text,ids", [
    ("A fact (C3).", ["C3"]), ("A fact [C3].", ["C3"]), ("A fact (C1, C2).", ["C1", "C2"]),
    ("A fact [C1; C2].", ["C1", "C2"]), ("A fact `[C3]`.", ["C3"]), ("A fact [C1][C2].", ["C1", "C2"]),
])
def test_supported_marker_formats_are_parsed_structurally(text, ids):
    assert citation_markers(text) == ids


def test_unsupported_numbers_still_rejected_even_with_valid_markers():
    v, c = _verifs({"C3": "supported"}), _claims({"C3": "The encoder is composed of 6 identical layers."})
    kept, removed = validate_sentences([{"text": "The encoder has 6 identical layers, introduced in 2017 (C3).",
                                         "claim_ids": ["C3"]}], v, c)
    assert kept == [] and removed[0]["reason"] == "sentence_numbers_not_in_cited_claims:2017"


def test_bare_unbracketed_ids_are_not_treated_as_markers():
    v, c = _verifs({"C3": "supported"}), _claims({"C3": "The encoder is composed of 6 identical layers."})
    kept, removed = validate_sentences([{"text": "As C3 shows, the encoder has 6 identical layers.",
                                         "claim_ids": ["C3"]}], v, c)
    assert kept == [] and removed[0]["reason"].endswith(":3")      # conservative: digit treated as a number


@pytest.mark.parametrize("text,claim_ids,reason", [
    ("The encoder has 6 identical layers (C99).", ["C3"], "cites_unsupported_or_unknown_claims:C99"),
    ("The encoder has 6 identical layers (C2).", ["C3"], "cites_unsupported_or_unknown_claims:C2"),   # contradicted
    ("The encoder has 6 identical layers (C1).", ["C3"], "inline_marker_not_in_cited_claims:C1"),
])
def test_invalid_inline_citations_fail(text, claim_ids, reason):
    v = _verifs({"C1": "supported", "C2": "contradicted", "C3": "supported"})
    c = _claims({"C1": "x", "C2": "y", "C3": "The encoder is composed of 6 identical layers."})
    kept, removed = validate_sentences([{"text": text, "claim_ids": claim_ids}], v, c)
    assert kept == [] and removed[0]["reason"] == reason


def test_graph_with_live_style_markers_reports_verified_findings(monkeypatch, settings, tmp_path):
    w = install_world(monkeypatch, settings)
    script_procurement(w)
    w.generator.structured_script[1] = {"sentences": [
        {"text": "The master supply agreement with Acme Metals runs until 31 December 2027 (C1).", "claim_ids": ["C1"]}]}
    out = run_research(QUESTION, BudgetManager(BudgetLimits()), domain_id="procurement", source_scope=["documents"])
    assert out["report"].status == "completed" and len(out["report"].verified_sentences) == 1
    assert out["report"].verified_sentences[0].origin == "model"


# ---------------------------------------------------------------- fallback
def test_deterministic_fallback_when_generated_sentences_fail(monkeypatch, settings):
    w = install_world(monkeypatch, settings)
    script_procurement(w)
    w.generator.structured_script[1] = {"sentences": [
        {"text": "The agreement runs until 31 December 2027 and renews in 2030.", "claim_ids": ["C1"]}]}
    out = run_research(QUESTION, BudgetManager(BudgetLimits()), domain_id="procurement", source_scope=["documents"])
    rep = out["report"]
    assert [(s.text, s.origin) for s in rep.verified_sentences] == [
        ("The master supply agreement with Acme Metals runs until 31 December 2027.", "fallback")]
    assert [(c.marker, c.source, c.page) for c in rep.citations] == [("C1", "contract.pdf", 1)]
    cite = next(t for t in out["trace"] if t["node"] == "cite")
    assert cite["fallback_sentences"] == 1 and cite["fallback_claims"] == ["C1"]
    assert "deterministic fallback" in rep.markdown and "renews in 2030" not in rep.markdown.split("## Removed")[0]
    assert all(s.claim_ids != ["C2"] for s in rep.verified_sentences)   # contradicted never falls back in


# ---------------------------------------------------------------- D2 negation
def _rel(claim, quote):
    return check_relevance(claim, quote, "supports")


@pytest.mark.parametrize("claim,quote,expected", [
    # direct contradictions mislabelled as support stay rejected
    ("Acme Metals holds ISO 27001 certification.", "Acme Metals does not hold ISO 27001 certification", "negation_mismatch"),
    ("The 2026 screening found sanctions listings for Acme Metals.", "The 2026 screening found no sanctions listings for Acme Metals", "negation_mismatch"),
    # negation about other words in the quote no longer rejects valid support
    ("The Transformer relies entirely on self-attention.", "The Transformer relies entirely on self-attention without using sequence-aligned recurrence", None),
    ("Drug X is approved in the United States.", "Drug X is approved in the United States but not in Canada", None),
    ("The decoder uses masked self-attention.", "Unlike recurrent decoders, which lack parallelism, the decoder uses masked self-attention", None),
    # negated claims need negated support
    ("The 2026 screening found no sanctions listings for Acme Metals.", "The 2026 screening found no sanctions listings for Acme Metals", None),
    ("The encoder lacks masked self-attention.", "Unlike the decoder, the encoder lacks masked self-attention sub-layers", None),
    ("Acme Metals is not a certified supplier.", "Acme Metals is a certified supplier", "claim_negation_not_supported"),
    # uncertain compound/comparative claims prefer insufficient evidence (Stage A C9 pattern)
    ("The decoder includes masked multi-head attention, which the encoder lacks.",
     "The decoder inserts a third sub-layer which performs masked multi-head attention", "claim_negation_not_supported"),
])
def test_scoped_negation(claim, quote, expected):
    assert _rel(claim, quote) == expected


# ---------------------------------------------------------------- D3 rejected quotes
def test_rejected_judgment_keeps_quote_stance_and_locator(caplog):
    ev = Evidence(evidence_id=eid(SCORECARD), text=SCORECARD["text"], retrieved_via="t",
                  locator=SourceLocator(source=SCORECARD["source"], source_type="document", page=SCORECARD["page"]))
    proposed = "Acme Metals delivered all purchase orders on time in 2025"
    v = verify_claim(Claim(claim_id="C4", text="Acme Metals delivered 98 percent of purchase orders on time during 2025."),
                     [judgment("C4", SCORECARD, "supports", proposed)], {ev.evidence_id: ev}, GENERAL, {ev.evidence_id})
    r = v.rejected[0]
    assert (r.claim_id, r.evidence_id, r.stance, r.reason, r.quote, r.source, r.page) == (
        "C4", eid(SCORECARD), "supports", "quote_not_in_evidence", proposed, "supplier_scorecard.pdf", 4)


def test_rejected_quotes_exported_but_not_logged(monkeypatch, settings, caplog):
    w = install_world(monkeypatch, settings)
    script_procurement(w)
    with caplog.at_level(logging.DEBUG):
        out = run_research(QUESTION, BudgetManager(BudgetLimits()), domain_id="procurement", source_scope=["documents"])
    md = out["report"].markdown
    assert ('rejected supports quote from supplier_scorecard.pdf, p. 4: "Acme Metals delivered all purchase orders '
            'on time in 2025" — quote_not_in_evidence') in md
    assert "delivered all purchase orders" not in caplog.text and SCORECARD["text"] not in caplog.text


# ---------------------------------------------------------------- D4 claim cap
def _many_claims(n_per_sub, subs):
    return {"claims": [{"text": f"Acme Metals fact number {s}-{i} about the agreement.", "sub_question_index": s,
                        "claim_type": "fact", "evidence_ids": [eid(CONTRACT)]}
                       for s in range(1, subs + 1) for i in range(n_per_sub)]}


def test_research_claims_capped_at_eight_with_subquestion_coverage(monkeypatch, settings):
    w = install_world(monkeypatch, settings)
    w.planner.structured_script = [plan("Acme Metals agreement term", "Acme Metals delivery performance",
                                        "Acme Metals certification")]
    w.generator.structured_script = [_many_claims(5, 3), {"sentences": []}]
    w.verifier.structured_script = [{"judgments": []}]
    b = BudgetManager(BudgetLimits())
    out = run_research(QUESTION, b, domain_id="general", source_scope=["documents"])
    claims = out["claims"]
    assert len(claims) == 8
    assert {c.sub_question_id for c in claims.values()} == {"Q1", "Q2", "Q3"}
    first_verify = next(t for t in out["trace"] if t["node"] == "verify")
    assert first_verify["verifier_calls"] == 2                     # ceil(8 / 5), was 3 with 15 claims
    assert next(t for t in out["trace"] if t["node"] == "extract")["proposed"] == 15


def test_claim_cap_is_configurable(monkeypatch, use_settings):
    s = use_settings(MAX_RESEARCH_CLAIMS=3)
    w = install_world(monkeypatch, s)
    w.planner.structured_script = [plan("Acme Metals agreement term", "Acme Metals delivery performance")]
    w.generator.structured_script = [_many_claims(4, 2), {"sentences": []}]
    w.verifier.structured_script = [{"judgments": []}]
    out = run_research(QUESTION, BudgetManager(BudgetLimits()), domain_id="general", source_scope=["documents"])
    assert len(out["claims"]) == 3


# ---------------------------------------------------------------- D5 fact-check template
def test_fact_check_uses_template_without_synthesis_call(monkeypatch, settings):
    w = install_world(monkeypatch, settings)
    w.docs = [CONTRACT, AUDIT]
    w.verifier.structured_script = [{"judgments": [
        judgment("C1", CONTRACT, "supports", "runs until 31 December 2027"),
        judgment("C2", AUDIT, "contradicts", "Acme Metals does not hold ISO 27001 certification")]}]
    out = run_research("Check these", BudgetManager(BudgetLimits()), domain_id="general", source_scope=["documents"],
                       claims_to_check=["The master supply agreement with Acme Metals runs until 31 December 2027.",
                                        "Acme Metals holds ISO 27001 certification."])
    rep = out["report"]
    assert w.generator.calls == 0 and w.planner.calls == 0            # no planning, extraction or synthesis calls
    assert [(s.claim_ids, s.origin) for s in rep.verified_sentences] == [(["C1"], "template")]
    assert out["verifications"]["C2"].status == "contradicted" and "C2" not in {c for s in rep.verified_sentences for c in s.claim_ids}
    assert [(c.marker, c.source, c.page) for c in rep.citations] == [("C1", "contract.pdf", 1)]


def test_fact_check_claims_are_never_truncated_silently(monkeypatch, use_settings):
    s = use_settings(MAX_FACT_CHECK_CLAIMS=2)
    w = install_world(monkeypatch, s)
    w.docs = [TRIAL_A]
    w.verifier.structured_script = [{"judgments": []}]
    out = run_research("Check", BudgetManager(BudgetLimits()), domain_id="general", source_scope=["documents"],
                       claims_to_check=["Claim one about Drug X.", "Claim two about Drug X.", "Claim three about Drug X."])
    assert len(out["claims"]) == 2
    assert any("Only the first 2 of 3 supplied claims were checked" in n and "Claim three about Drug X." in n
               for n in out["notices"])
    assert "## Notices" in out["report"].markdown and "Claim three about Drug X." in out["report"].markdown


# ---------------------------------------------------------------- budget
def test_research_run_with_eight_claims_stays_within_budget(monkeypatch, settings):
    w = install_world(monkeypatch, settings)
    w.planner.structured_script = [plan("Acme Metals agreement term", "Acme Metals delivery performance")]
    w.generator.structured_script = [_many_claims(6, 2), {"sentences": []}]
    w.verifier.structured_script = [{"judgments": []}]
    b = BudgetManager(BudgetLimits())
    run_research(QUESTION, b, domain_id="general", source_scope=["documents"])
    snap = b.snapshot()
    # plan 1 + extract 1 + verify 2 + up to 2 retry rounds x 2 batches; synthesis skipped (no supported claims)
    assert snap.mesh_llm_calls <= 8 and snap.retry_rounds <= 2 and snap.mesh_embedding_calls <= 4
