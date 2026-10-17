"""Stage B hardening (B-D1 … B-D5) — offline regression tests with scripted fakes."""

import pytest
from langchain_core.messages import AIMessage

from agents.research_common import LLM_CALLS_RESERVED_FOR_SYNTHESIS, evidence_id, rank_for_claim, synthesis_reserve
from budget import BudgetLimits, BudgetManager, PriceTable, format_usage
from budget.manager import CACHE_UNCONFIRMED
from domains.registry import get_domain
from graph_builder.research_graph import run_research
from schemas.research import Claim, Evidence, SourceLocator
from tests.research_fixtures import SCORECARD, eid, judgment
from tests.research_world import install_world, plan
from verification.aggregation import verify_claim
from verification.citation_integrity import check_relevance
from verification.meta_claims import is_meta_claim

GENERAL = get_domain("general")
ACME_DOCS = [{"text": f"Acme Metals supplier note {i}: delivery logistics for steel coil orders in region {i}.",
              "source": f"note_{i}.pdf", "page": i} for i in range(1, 9)]


def _ev(d):
    return Evidence(evidence_id=eid(d), text=d["text"], retrieved_via="t",
                    locator=SourceLocator(source=d["source"], source_type="document", page=d["page"]))


# ------------------------------------------------------------------ B-D1
def test_rank_puts_accepted_then_fresh_then_candidates_and_dedupes():
    store = {e.evidence_id: e for e in map(_ev, ACME_DOCS)}
    ids = list(store)
    accepted, fresh, cands, old = [ids[0]], [ids[7], ids[6]], [ids[1], ids[2], ids[3]], [ids[4], ids[5], ids[1]]
    shown = rank_for_claim(cands, old + fresh + accepted, store, limit=6, accepted=accepted, fresh=fresh)
    assert shown == [ids[0], ids[7], ids[6], ids[1], ids[2], ids[3]]          # fresh survives the cap
    assert len(shown) == len(set(shown))


def test_retry_evidence_is_shown_to_the_verifier(monkeypatch, settings):
    w = install_world(monkeypatch, settings)
    w.docs = ACME_DOCS
    claim_text = "Acme Metals delivered steel coil orders on time in region 9."
    w.planner.structured_script = [plan("Acme Metals supplier note delivery logistics")]
    w.generator.structured_script = [{"claims": [{"text": claim_text, "sub_question_index": 1, "claim_type": "fact",
                                                  "evidence_ids": [eid(d) for d in ACME_DOCS[:3]]}]}]
    w.verifier.structured_script = [{"judgments": []}]
    out = run_research("Acme Metals delivery", BudgetManager(BudgetLimits()), domain_id="general",
                       source_scope=["documents"])
    retry = next(t for t in out["trace"] if t["node"] == "retry")
    fresh_ids = retry["new_evidence_ids"]["C1"]
    assert fresh_ids                                                           # retry found new passages
    second_verify = [t for t in out["trace"] if t["node"] == "verify"][1]
    assert second_verify["fresh_shown"]["C1"] == fresh_ids                     # ...and they were shown
    assert all(e in w.verifier.prompts[1] for e in fresh_ids)                  # ...in the verifier prompt
    assert w.verifier.prompts[0] != w.verifier.prompts[1]                      # no identical re-send


# ------------------------------------------------------------------ B-D2
def test_synthesis_reserve_only_in_research_mode():
    assert synthesis_reserve({"claims_to_check": ["x"]}) == 0
    assert synthesis_reserve({}) == LLM_CALLS_RESERVED_FOR_SYNTHESIS == 2


def test_fact_check_retry_is_reachable_with_three_call_share(monkeypatch, settings):
    w = install_world(monkeypatch, settings)
    w.docs = ACME_DOCS
    w.verifier.structured_script = [{"judgments": []}]
    b = BudgetManager(BudgetLimits(max_llm_calls=3, max_embedding_calls=4))   # Stage B fact-check share
    out = run_research("check", b, domain_id="general", source_scope=["documents"],
                       claims_to_check=["Acme Metals delivered steel coil orders on time in region 9."])
    nodes = [t["node"] for t in out["trace"]]
    assert "retry" in nodes and w.verifier.calls >= 2                          # Stage B: retry never ran
    assert b.snapshot().mesh_llm_calls <= 3 and w.generator.calls == 0


# ------------------------------------------------------------------ B-D3
def _msg(headers=None, rid="chatcmpl-abc"):
    return AIMessage(content="x", response_metadata={"model_name": "gemini-2.5-flash", "id": rid,
                                                     **({"headers": headers} if headers else {})},
                     usage_metadata={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500})


def test_cache_detection_documented_header_and_observed_prefix():
    from gateway.mesh import _usage_from_message

    u = _usage_from_message(_msg({"X-Cache": "HIT"}))
    assert (u.cached, u.cache_marker) == (True, "x-cache-header")
    u2 = _usage_from_message(_msg(rid="chatcmpl-cached-req_01ABC"))
    assert (u2.cached, u2.cache_marker) == (True, "id-prefix")
    assert _usage_from_message(_msg({"x-request-id": "1"})).cached is False


def test_cached_responses_count_as_requests_and_cost_rules():
    table = PriceTable({"google/gemini-2.5-flash": {"input_per_1m_usd": 0.30, "output_per_1m_usd": 2.50,
                                                    "source": "t", "retrieved": "2026-10-16"}})
    b = BudgetManager(BudgetLimits(max_llm_calls=2), pricing=table)
    b.record_mesh_attempt("structured", "google/gemini-2.5-flash", "ok", input_tokens=1000, output_tokens=500,
                          cached=True, cache_marker="x-cache-header")
    b.record_mesh_attempt("structured", "google/gemini-2.5-flash", "ok", input_tokens=1000, output_tokens=500,
                          cached=True, cache_marker="id-prefix")
    snap = b.snapshot()
    assert snap.mesh_chat_calls == 2 and snap.mesh_cached_responses == 2      # still counted against the budget
    assert b.events[0]["cost_usd"] == 0.0                                      # documented: hits are not billed
    assert b.events[1]["cost_usd"] == pytest.approx((1000 * 0.30 + 500 * 2.50) / 1e6)   # undocumented marker
    assert CACHE_UNCONFIRMED in snap.cost_incomplete_reasons
    assert snap.mesh_input_tokens == 1000                                      # cached original not new work
    assert "2 cached response(s)" in format_usage(snap, b.limits)
    with pytest.raises(Exception):
        b.check("llm")                                                         # cap reached by cached calls


def test_cache_bypass_is_opt_in_and_documented_header_only(use_settings):
    from gateway.mesh import _make_chat_client, _make_embeddings_client

    s = use_settings()
    assert _make_chat_client("m", s).default_headers in (None, {})
    assert _make_chat_client("m", s).include_response_headers is True
    s2 = use_settings(MESH_CACHE_MODE="no-store")
    assert _make_chat_client("m", s2).default_headers == {"X-Mesh-Cache": "no-store"}
    assert _make_embeddings_client(s2).default_headers == {"X-Mesh-Cache": "no-store"}


# ------------------------------------------------------------------ B-D4 (deferred; limitation pinned)
SURVEY_ABSTRACT = ("we introduce the various X-formers from three perspectives: architectural modification, "
                   "pre-training, and applications")
L06_CLAIM = "The survey categorizes X-formers from three perspectives: architectural modification, pre-training, and applications."


def test_threshold_accepts_genuine_classification_quote():
    assert check_relevance(L06_CLAIM, SURVEY_ABSTRACT, "supports") is None


def test_known_false_negatives_remain_rejected_until_a_safe_exception_exists():
    l06_quote = "we mainly focus on the general architecture variants and just briefly discuss the specific variants on pre-training and applications."
    assert check_relevance(L06_CLAIM, l06_quote, "supports").startswith("quote_not_about_claim")
    # L07: a truncated bibliographic entry that may name a DIFFERENT survey; accepting it would risk
    # evidence about another entity, so no exception is implemented.
    assert check_relevance("The Transformer survey was published in 2010.", "formers: A survey,” arXiv, 2020.",
                           "contradicts").startswith("quote_not_about_claim")


# ------------------------------------------------------------------ B-D5
@pytest.mark.parametrize("text,meta", [
    ("No standard payment terms commonly used by suppliers in the industry are mentioned in the evidence.", True),
    ("The impact of payment terms on the financial health of suppliers is not specified in the evidence.", True),
    ("The provided documents do not mention any supplier audits.", True),
    ("There is no information about late payments in the retrieved passages.", True),
    ("The supplier has no outstanding sanctions.", False),
    ("The 2026 screening found no sanctions listings for Acme Metals.", False),
    ("No adverse events were reported in the phase 3 trial.", False),
    ("Audit documents are not provided to suppliers under the contract.", False),
])
def test_meta_claim_detection(text, meta):
    assert is_meta_claim(text) is meta


def test_meta_claims_filtered_but_genuine_negatives_verified(monkeypatch, settings):
    w = install_world(monkeypatch, settings)
    w.docs = [{"text": "The 2026 screening found no sanctions listings for Acme Metals or its directors.",
               "source": "screening.pdf", "page": 2}]
    sanc = w.docs[0]
    w.planner.structured_script = [plan("Acme Metals sanctions screening")]
    w.generator.structured_script = [
        {"claims": [{"text": "Payment terms are not specified in the evidence.", "sub_question_index": 1,
                     "claim_type": "fact", "evidence_ids": []},
                    {"text": "The 2026 screening found no sanctions listings for Acme Metals.", "sub_question_index": 1,
                     "claim_type": "fact", "evidence_ids": [eid(sanc)]}]},
        {"sentences": [{"text": "The 2026 screening found no sanctions listings for Acme Metals.", "claim_ids": ["C1"]}]}]
    w.verifier.structured_script = [{"judgments": [judgment("C1", sanc, "supports",
                                                            "The 2026 screening found no sanctions listings for Acme Metals")]}]
    out = run_research("Acme sanctions?", BudgetManager(BudgetLimits()), domain_id="general", source_scope=["documents"])
    assert [c.text for c in out["claims"].values()] == ["The 2026 screening found no sanctions listings for Acme Metals."]
    assert out["verifications"]["C1"].status == "supported"
    assert next(t for t in out["trace"] if t["node"] == "extract")["meta_claims_filtered"] == 1
    assert any("not treated as findings" in n for n in out["notices"])


def test_all_meta_claims_leave_an_honest_empty_report(monkeypatch, settings):
    w = install_world(monkeypatch, settings)
    w.planner.structured_script = [plan("Acme Metals payment terms")]
    w.generator.structured_script = [{"claims": [
        {"text": "No standard payment terms are mentioned in the evidence.", "sub_question_index": 1,
         "claim_type": "fact", "evidence_ids": []}]}]
    out = run_research("Payment terms?", BudgetManager(BudgetLimits()), domain_id="procurement", source_scope=["documents"])
    assert out["claims"] == {} and out["report"].status == "no_verified_findings"
    assert w.verifier.calls == 0
