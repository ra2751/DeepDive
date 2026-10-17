"""B4 Q2 recall remediation: regression tests on the RECORDED B4-Q2 data, plus negative tests.

tests/fixtures/b4_q2.json holds the claims, judgments, rejection reasons and removed sentences exactly
as archived in evals/results/20261017_B4_ui_pilot_results.json, plus the full passage text of the three
evidence chunks involved (read once, read-only, from Milvus by chunk id; the archive stores only locators).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from budget import BudgetLimits, BudgetManager
from domains.registry import get_domain
from graph_builder.research_graph import run_research
from schemas.research import Claim, Evidence, SourceLocator
from verification.aggregation import verify_claim
from verification.citation_integrity import (check_quote, check_relevance, citation_markers,
                                             strip_citation_markers, validate_sentences)
from verification.text import content_stems, negation_conflict, numbers

FX = json.loads((Path(__file__).parent / "fixtures" / "b4_q2.json").read_text())
C1 = FX["claims"]["C1"]
C1_J = FX["judgments"]["C1"][0]
C1_PASSAGE = FX["evidence"][C1_J["evidence_id"]]["text"]
C7_J, C8_J = FX["judgments"]["C7"][0], FX["judgments"]["C8"][0]


def evidence(eid):
    e = FX["evidence"][eid]
    return Evidence(evidence_id=eid, text=e["text"], retrieved_via="b4-q2",
                    locator=SourceLocator(source=e["source"], source_type="document", page=e["page"],
                                          chunk_id=str(e["chunk_id"])))


# ---------------------------------------------------------------- 1-2. reproduce C1 and explain 0.29

def test_c1_recorded_rejection_reproduces_without_passage():
    assert C1 == "The base model of the Transformer architecture uses 8 attention heads."
    assert C1_J["quote"] == "In this work we employ h = 8 parallel attention layers, or heads."
    assert C1_J["reason"] == "quote_not_about_claim(overlap=0.29)"
    assert check_quote(C1_J["quote"], C1_PASSAGE) is None                      # verbatim in the recorded passage
    # the old call (claim, quote, stance) gives exactly the recorded reason
    assert check_relevance(C1, C1_J["quote"], "supports") == C1_J["reason"]


def test_c1_overlap_is_two_of_seven_stems():
    claim, quote = content_stems(C1), content_stems(C1_J["quote"])
    assert claim == {"archi", "atten", "base", "model", "trans", "use", "head"}
    assert claim & quote == {"atten", "head"}                                   # 2 / 7 = 0.29 < 0.34
    assert numbers(C1) == {"8"} and "8" in numbers(C1_J["quote"])              # the number fact matches


# ---------------------------------------------------------------- 3. narrow, number-anchored fix

def test_c1_now_accepted_with_its_passage_and_verified_supported():
    assert check_relevance(C1, C1_J["quote"], "supports", C1_PASSAGE) is None
    ev = {C1_J["evidence_id"]: evidence(C1_J["evidence_id"])}
    v = verify_claim(Claim(claim_id="C1", text=C1), [C1_J], ev, get_domain("general"), {C1_J["evidence_id"]})
    assert v.status == "supported" and v.supporting_evidence_ids == [C1_J["evidence_id"]] and not v.rejected


def test_threshold_unchanged():
    from verification import citation_integrity
    assert citation_integrity.MIN_CONTENT_OVERLAP == 0.34


@pytest.mark.parametrize("claim, reason", [
    # wrong number: still a hard rejection
    ("The base model of the Transformer architecture uses 16 attention heads.", "claim_numbers_missing_from_quote:16"),
    # wrong entity: the number phrase matches, but BERT is not in the passage
    ("The base model of the BERT architecture uses 8 attention heads.", "quote_not_about_claim"),
    # wrong quantity: "8 encoder layers" is not what the 8 counts in the quote
    ("The base model of the Transformer architecture has 8 encoder layers.", "quote_not_about_claim"),
    # no number: the anchored path never applies
    ("The base model of the Transformer architecture uses multi-head attention heads.", "quote_not_about_claim"),
    # negated claim: negation is checked before relevance and is not bypassed
    ("The base model of the Transformer architecture does not use 8 attention heads.",
     "claim_negation_not_supported"),
])
def test_negative_claims_against_c1_quote(claim, reason):
    assert check_relevance(claim, C1_J["quote"], "supports", C1_PASSAGE).startswith(reason)


def test_unrelated_evidence_rejected():
    # a real quote from a different passage (survey p18) cannot support C1
    assert check_relevance(C1, C7_J["quote"], "supports", FX["evidence"][C7_J["evidence_id"]]["text"]).startswith(
        "claim_numbers_missing_from_quote")
    # same number and noun, but a passage about something else entirely
    other = "In this work we employ h = 8 parallel attention layers, or heads, in our recommender for movies."
    assert check_relevance("The base model of the Transformer architecture uses 8 attention heads.",
                           "we employ h = 8 parallel attention layers, or heads", "supports", other) \
        .startswith("quote_not_about_claim")


def test_passage_must_cover_subject_not_just_the_quote():
    # if the passage is only the quote itself, the subject ("Transformer", "base model") is not covered
    assert check_relevance(C1, C1_J["quote"], "supports", C1_J["quote"]).startswith("quote_not_about_claim")


def test_anchored_path_only_for_supports():
    assert check_relevance(C1, C1_J["quote"], "contradicts", C1_PASSAGE).startswith("quote_not_about_claim")


def test_fabricated_quotation_rejected():
    fake = "In this work we employ h = 8 parallel attention heads in the base Transformer model."
    assert check_quote(fake, C1_PASSAGE) == "quote_not_in_evidence"
    ev = {C1_J["evidence_id"]: evidence(C1_J["evidence_id"])}
    v = verify_claim(Claim(claim_id="C1", text=C1), [{**C1_J, "quote": fake}], ev, get_domain("general"),
                     {C1_J["evidence_id"]})
    assert v.status == "insufficient_evidence" and v.rejected[0].reason == "quote_not_in_evidence"


def test_provenance_and_independence_preserved():
    ev = {C1_J["evidence_id"]: evidence(C1_J["evidence_id"])}
    v = verify_claim(Claim(claim_id="C1", text=C1), [C1_J], ev, get_domain("general"), set())
    assert v.rejected[0].reason == "evidence_not_shown_for_this_claim"
    v = verify_claim(Claim(claim_id="C1", text=C1), [{**C1_J, "evidence_id": "Eunknown"}], ev,
                     get_domain("general"), {"Eunknown"})
    assert v.rejected[0].reason == "unknown_evidence_id"
    # a high-risk pharma claim still needs 2 independent sources; one accepted passage is not enough
    v = verify_claim(Claim(claim_id="C1", text=C1, risk="high"), [C1_J], ev, get_domain("pharma"),
                     {C1_J["evidence_id"]})
    assert v.status == "insufficient_evidence" and v.independent_sources == 1 and v.required_sources == 2


# ---------------------------------------------------------------- 7. negation, recorded fixtures only

def test_c7_recorded_false_rejection_fixed():
    assert C7_J["reason"] == "negation_mismatch"
    assert negation_conflict(FX["claims"]["C7"], C7_J["quote"]) is None        # both say "no mechanism ..."
    assert check_relevance(FX["claims"]["C7"], C7_J["quote"], "supports") is None


def test_c8_still_rejected():
    # quote's "nor ... mechanism for heads to interact" heads a different negation than the claim's
    # "do not explicitly interact": left rejected (conservative), see the report.
    assert negation_conflict(FX["claims"]["C8"], C8_J["quote"]) == "negation_mismatch"


@pytest.mark.parametrize("claim", [
    "Multi-head attention has a mechanism to guarantee that different attention heads capture distinct features.",
    "There is a mechanism to guarantee that different attention heads indeed capture distinct features.",
])
def test_negation_still_blocks_positive_claim_from_negative_quote(claim):
    assert negation_conflict(claim, C7_J["quote"]) == "negation_mismatch"


def test_existing_negation_cases_unchanged():
    assert negation_conflict("Acme Metals holds ISO 27001 certification.",
                             "Acme Metals does not hold ISO 27001 certification.") == "negation_mismatch"
    assert negation_conflict("Acme Metals does not hold ISO 27001 certification.",
                             "Acme Metals holds ISO 27001 certification.") == "claim_negation_not_supported"


# ---------------------------------------------------------------- 6. prose claim references

def _ns(statuses):
    claims = {c: SimpleNamespace(text=t) for c, t in FX["all_claims"].items()}
    verifs = {c: SimpleNamespace(status=s) for c, s in statuses.items()}
    return verifs, claims


def test_recorded_prose_reference_sentences_now_kept():
    verifs, claims = _ns(FX["statuses"])
    removed = FX["removed_sentences"]
    assert [r["reason"].split(":")[0] for r in removed] == ["sentence_numbers_not_in_cited_claims"] * 4
    assert [citation_markers(r["text"]) for r in removed] == [["C2", "C3", "C6"], ["C3", "C2"], ["C5", "C2"], ["C7"]]
    kept, still_removed = validate_sentences([{"text": r["text"], "claim_ids": r["claim_ids"]} for r in removed],
                                             verifs, claims)
    # none is removed for prose references any more; the stricter grounding check (final hardening)
    # rejects the first sentence because it adds "base model of the Transformer architecture"
    assert len(kept) == 3
    assert [r["reason"] for r in still_removed] == ["unsupported_content:archi,base,emplo,trans"]


@pytest.mark.parametrize("text, ids, reason", [
    ("Heads attend jointly, as stated in claim IDs C2 and C9.", ["C2"], "cites_unsupported_or_unknown_claims:C9"),
    ("The base model uses eight heads, as noted in claim ID C1.", ["C2"], "cites_unsupported_or_unknown_claims:C1"),
    ("Heads attend jointly, as stated in claim IDs C2 and C3.", ["C2"], "inline_marker_not_in_cited_claims:C3"),
    ("Heads attend jointly, see C3.", ["C3"], "sentence_numbers_not_in_cited_claims:3"),      # bare ID: not a ref
    ("Heads attend jointly in 3 ways, per claim C3.", ["C3"], "sentence_numbers_not_in_cited_claims:3"),
])
def test_prose_reference_negatives(text, ids, reason):
    verifs, claims = _ns(FX["statuses"])
    kept, removed = validate_sentences([{"text": text, "claim_ids": ids}], verifs, claims)
    assert kept == [] and removed[0]["reason"] == reason


def test_prose_reference_forms():
    assert citation_markers("as claim C7 says") == ["C7"]
    assert citation_markers("per Claims C2; C3") == ["C2", "C3"]
    assert citation_markers("(C1, C2) and claim IDs C3, C4, and C5") == ["C1", "C2", "C3", "C4", "C5"]
    assert citation_markers("the claim c7 was") == []                           # IDs are upper-case C only
    assert numbers(strip_citation_markers("Six layers, per claim IDs C2 and C6.")) == set()


# ---------------------------------------------------------------- source selection

def test_source_options():
    from ui.research_view import SOURCE_OPTIONS, source_label
    assert SOURCE_OPTIONS == {"Documents only": ["documents"], "Web only": ["web"],
                              "Documents + Web": ["documents", "web"]}
    assert source_label(["documents"]) == "Documents only"


def test_documents_only_run_makes_no_web_search(monkeypatch, settings):
    from tests.research_world import QUESTION, install_world, script_procurement
    script_procurement(install_world(monkeypatch, settings))
    budget = BudgetManager(BudgetLimits())
    run_research(QUESTION, budget, domain_id="procurement", source_scope=["documents"])
    assert budget.snapshot().web_searches == 0


def test_app_defaults_to_web_only(monkeypatch, settings, tmp_path):
    from streamlit.testing.v1 import AppTest
    from tests.research_world import install_world
    monkeypatch.chdir(tmp_path)
    install_world(monkeypatch, settings)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    src = at.sidebar.radio[1]
    assert src.label == "Sources" and src.value == "Web only"
    assert list(src.options) == ["Documents only", "Web only", "Documents + Web"]
