"""Increment 3: source quality, contradiction visibility, export integrity, follow-up reuse, isolation.

Scripted fixtures only: these tests prove behaviour, not live model accuracy.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from agents.research_common import evidence_id, rank_for_claim
from budget import BudgetLimits, BudgetManager
from domains.registry import get_domain
from graph_builder.research_graph import prior_context_from, run_research
from schemas.research import Claim, Evidence, SourceLocator
from tests.research_fixtures import AUDIT, CONTRACT, PROCUREMENT_DOCS, SCORECARD, eid, judgment
from tests.research_world import QUESTION, install_world, plan, script_procurement
from ui.research_view import source_label
from verification.aggregation import explain_status, verify_claim
from verification.citation_integrity import build_citations
from verification.source_quality import evaluate, load_rules

APP = str(Path(__file__).resolve().parents[1] / "app.py")
AUDIT2025 = {"text": "The 2025 audit found Acme Metals delivered only 71 percent of purchase orders on time, not 98 percent.",
             "source": "audit_2025.pdf", "page": 3}
GENERAL = get_domain("general")


# ------------------------------------------------------------ source quality
@pytest.mark.parametrize("url,tier", [
    ("https://www.fda.gov/drugs/label", "primary"),
    ("https://data.cms.gov/x", "primary"),                         # .gov suffix rule
    ("https://www.nature.com/articles/abc", "strong_secondary"),
    ("https://pubmed.ncbi.nlm.nih.gov/123", "strong_secondary"),   # specific host beats parent nih.gov
    ("https://www.ncbi.nlm.nih.gov/books/x", "primary"),           # parent nih.gov (no more specific rule)
    ("https://arxiv.org/abs/1706.03762", "general_secondary"),
    ("https://en.wikipedia.org/wiki/Transformer", "general_secondary"),
    ("https://medium.com/@someone/post", "unverified"),
    ("https://fda.gov.example.com/fake", "unverified"),            # lookalike host is not trusted
])
def test_web_rule_selection(url, tier):
    assert evaluate("web", url, url).tier == tier


def test_unknown_document_is_unverified_and_nothing_is_invented():
    q = evaluate("document", "AttentionAllYouNeed.pdf")
    assert q.tier == "unverified" and q.score == 0.35
    assert q.publisher is None and q.category is None and q.provenance is None
    assert "no recorded publisher" in q.reason


def test_document_registry_uses_only_recorded_metadata():
    rules = dict(load_rules())
    rules["document_registry"] = {
        "contract.pdf": {"category": "contract", "publisher": "Northwind legal"},
        "brochure.pdf": {"category": "glossy_flyer"}}
    q = evaluate("document", "contract.pdf", rules=rules)
    assert (q.tier, q.publisher, q.provenance) == ("primary", "Northwind legal", None)
    assert evaluate("document", "brochure.pdf", rules=rules).tier == "unverified"   # unknown category
    assert evaluate("document", "other.pdf", rules=rules).tier == "unverified"


def test_config_rejects_unknown_tiers(tmp_path):
    bad = tmp_path / "sq.yaml"
    bad.write_text("tiers: {primary: {score: 1.0, label: P}}\nweb_rules: [{tier: gold, reason: x, hosts: [a.com]}]\n")
    with pytest.raises(ValueError):
        load_rules(str(bad))


def _ev(doc=None, url=None, text=None):
    if url:
        e = Evidence(evidence_id=evidence_id(url, None, text), text=text, retrieved_via="t",
                     locator=SourceLocator(source=url, source_type="web", url=url), quality=evaluate("web", url, url))
    else:
        e = Evidence(evidence_id=eid(doc), text=doc["text"], retrieved_via="t",
                     locator=SourceLocator(source=doc["source"], source_type="document", page=doc["page"]),
                     quality=evaluate("document", doc["source"]))
    return e


def test_low_quality_source_does_not_veto_and_high_quality_does_not_prove():
    claim = Claim(claim_id="C1", text="Acme Metals delivered 98 percent of purchase orders on time during 2025.")
    low = _ev(SCORECARD)                                            # unverified document
    v = verify_claim(claim, [judgment("C1", SCORECARD, "supports", "delivered 98 percent of purchase orders on time during 2025")],
                     {low.evidence_id: low}, GENERAL, {low.evidence_id})
    assert low.quality.tier == "unverified" and v.status == "supported"   # quality is not an input to status
    gov_text = "Federal records show Acme Metals shipped steel coil to three agencies in 2024."
    gov = _ev(url="https://www.gsa.gov/acme", text=gov_text)
    v2 = verify_claim(claim, [{"claim_id": "C1", "evidence_id": gov.evidence_id, "stance": "supports",
                               "quote": "Acme Metals delivered 98 percent of purchase orders on time during 2025"}],
                      {gov.evidence_id: gov}, GENERAL, {gov.evidence_id})
    assert gov.quality.tier == "primary" and v2.status == "insufficient_evidence"
    assert v2.rejected[0].reason == "quote_not_in_evidence"          # reputation never bypasses quote integrity


def test_primary_contradiction_does_not_win_over_support():
    claim = Claim(claim_id="C1", text="Acme Metals delivered 98 percent of purchase orders on time during 2025.")
    sup = _ev(SCORECARD)
    gov_text = "The 2025 federal audit found Acme Metals delivered only 71 percent of purchase orders on time."
    gov = _ev(url="https://www.gsa.gov/audit", text=gov_text)
    store = {sup.evidence_id: sup, gov.evidence_id: gov}
    v = verify_claim(claim, [judgment("C1", SCORECARD, "supports", "delivered 98 percent of purchase orders on time during 2025"),
                             {"claim_id": "C1", "evidence_id": gov.evidence_id, "stance": "contradicts",
                              "quote": "Acme Metals delivered only 71 percent of purchase orders on time"}],
                     store, GENERAL, set(store))
    assert v.status == "unresolved"                                   # no winner chosen by ranking
    assert "does not pick a winner" in explain_status(v)


def test_ranking_orders_but_keeps_claim_candidates_first():
    a, b = _ev(SCORECARD), _ev(url="https://www.nature.com/x", text="Some peer-reviewed text about suppliers.")
    c = _ev(url="https://www.fda.gov/y", text="Some regulator text about suppliers.")
    store = {e.evidence_id: e for e in (a, b, c)}
    ranked = rank_for_claim([a.evidence_id], [b.evidence_id, c.evidence_id, a.evidence_id], store, limit=2)
    assert ranked == [a.evidence_id, c.evidence_id]                   # candidate first, then primary over strong


def test_citations_present_stronger_sources_first():
    claim = Claim(claim_id="C1", text="Acme Metals delivered 98 percent of purchase orders on time during 2025.")
    doc = _ev(SCORECARD)
    gov_text = "Acme Metals delivered 98 percent of purchase orders on time during 2025, per agency data."
    gov = _ev(url="https://www.gsa.gov/perf", text=gov_text)
    store = {doc.evidence_id: doc, gov.evidence_id: gov}
    v = verify_claim(claim, [judgment("C1", SCORECARD, "supports", "delivered 98 percent of purchase orders on time during 2025"),
                             {"claim_id": "C1", "evidence_id": gov.evidence_id, "stance": "supports",
                              "quote": "Acme Metals delivered 98 percent of purchase orders on time during 2025"}],
                     store, GENERAL, set(store))
    cites = build_citations({"C1": v}, store)
    assert [c.evidence_id for c in cites] == [gov.evidence_id, doc.evidence_id]


# --------------------------------------------------- contradiction outcomes
def script_conflict(world):
    world.docs = PROCUREMENT_DOCS + [AUDIT2025]
    script_procurement(world)
    world.verifier.structured_script = [
        {"judgments": [
            judgment("C1", CONTRACT, "supports", "runs until 31 December 2027"),
            judgment("C2", AUDIT, "contradicts", "Acme Metals does not hold ISO 27001 certification"),
            judgment("C4", SCORECARD, "supports", "delivered 98 percent of purchase orders on time during 2025"),
            judgment("C4", AUDIT2025, "contradicts", "Acme Metals delivered only 71 percent of purchase orders on time"),
        ]},
        {"judgments": []}]


@pytest.fixture
def world(monkeypatch, settings, tmp_path):
    monkeypatch.chdir(tmp_path)
    return install_world(monkeypatch, settings)


def test_contradicted_vs_unresolved_conflict_in_graph_and_export(world):
    script_conflict(world)
    out = run_research(QUESTION, BudgetManager(BudgetLimits()), domain_id="procurement", source_scope=["documents"])
    v = out["verifications"]
    assert v["C2"].status == "contradicted" and not v["C2"].supporting_evidence_ids
    assert v["C4"].status == "unresolved" and v["C4"].supporting_evidence_ids and v["C4"].contradicting_evidence_ids
    md = out["report"].markdown
    assert "**C4 — unresolved (conflicting evidence):**" in md and "does not pick a winner" in md
    assert "supporting: supplier_scorecard.pdf, p. 4" in md and "contradicting: audit_2025.pdf, p. 3" in md
    assert "**C2 — contradicted:**" in md and "(conflicting evidence)" not in md.split("**C2")[1].split("\n")[0]
    # export integrity
    assert "## Verified findings" in md and "[C1]" in md
    assert "| C1 | supported | contract.pdf, p. 1 | Unverified (0.35) | new |" in md
    assert "## Human review required" in md and "## Verification limitations" in md
    assert "provisional ranking signals" in md


def run_app(question=QUESTION, domain="procurement"):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox[0].set_value(domain)
    at.sidebar.radio[1].set_value(source_label(["documents"]))
    at.run()
    at.chat_input[0].set_value(question).run()
    assert not at.exception, at.exception
    return at


def test_evidence_tab_separates_supporting_and_contradicting_columns(world):
    script_conflict(world)
    at = run_app()
    ev = at.tabs[2]
    md = [m.value for m in ev.markdown]
    assert "#### ⚖️ Conflicting evidence (unresolved)" in md and "#### ❌ Contradicted claims" in md
    assert any("does not pick a winner" in w.value for w in ev.warning)
    cols = ev.columns
    assert len(cols) == 2
    left = [m.value for m in cols[0].markdown]
    right = [m.value for m in cols[1].markdown]
    assert left[0] == "**Supporting**" and "**supplier_scorecard.pdf · page 4**" in left
    assert right[0] == "**Contradicting**" and "**audit_2025.pdf · page 3**" in right
    assert "> “Acme Metals delivered only 71 percent of purchase orders on time”" in right
    caps = " ".join(c.value for c in ev.caption)
    assert "source quality: Unverified (0.35)" in caps and "🆕 newly retrieved" in caps and "evidence E" in caps
    claims = " ".join(c.value for c in at.tabs[1].caption)
    assert "Contradicted: accepted contradicting evidence and no accepted supporting evidence." in claims
    assert "Unresolved — conflicting evidence" in claims


def test_audit_shows_agents_handoffs_retry_evidence_and_final_statuses(world):
    script_procurement(world)
    at = run_app()
    audit = at.tabs[3]
    trace = audit.dataframe[0].value
    assert list(trace["agent"]) == ["Intake", "Planner", "Researcher", "Claim extractor", "Verifier",
                                    "Targeted retry (researcher)", "Verifier", "Targeted retry (researcher)",
                                    "Synthesizer", "Citation checker", "Reporter"]
    md = " ".join(m.value for m in audit.markdown)
    assert "**Researcher → Claim extractor:** 4 evidence passages (4 new, 0 reused)" in md
    assert "**Claim extractor → Verifier:** claims C1, C2, C3, C4" in md
    assert "Round 1: retried C3, C4 · new evidence: C3: E" in md and "Round 2: retried C3, C4 · new evidence: none found" in md
    assert "C1: ✅ Supported · C2: ❌ Contradicted · C3: ⚠️ Unresolved · C4: ⚠️ Unresolved" in md


# ----------------------------------------------------------- follow-up
def first_run(world):
    script_procurement(world)
    return run_research(QUESTION, BudgetManager(BudgetLimits()), domain_id="procurement", source_scope=["documents"])


def script_follow_up(world):
    world.planner.structured_script = [{"in_scope": True, "scope_reason": "follow-up",
                                        "resolved_question": "When does the Acme Metals master supply agreement expire?",
                                        "sub_questions": ["Acme Metals master supply agreement expiry date"]}]
    world.generator.structured_script = [
        {"claims": [{"text": "The master supply agreement with Acme Metals runs until 31 December 2027.",
                     "sub_question_index": 1, "claim_type": "contract_term", "evidence_ids": [eid(CONTRACT)]}]},
        {"sentences": [{"text": "The agreement runs until 31 December 2027.", "claim_ids": ["C1"]}]}]
    world.verifier.structured_script = [{"judgments": [judgment("C1", CONTRACT, "supports", "runs until 31 December 2027")]}]


def test_prior_context_carries_only_accepted_evidence(world):
    out1 = first_run(world)
    prior = prior_context_from(out1)
    assert set(prior["evidence"]) == {eid(CONTRACT), eid(AUDIT)}     # C1 support + C2 contradiction only
    assert prior["findings"] == ["The Acme Metals master supply agreement runs until 31 December 2027."]
    assert prior["domain_id"] == "procurement"


def test_follow_up_reuses_evidence_and_reverifies(world):
    prior = prior_context_from(first_run(world))
    calls_before = world.verifier.calls
    script_follow_up(world)
    b = BudgetManager(BudgetLimits())
    out = run_research("When does it expire?", b, domain_id="procurement", source_scope=["documents"],
                       prior_context=prior)
    assert out["plan"].resolved_question == "When does the Acme Metals master supply agreement expire?"
    research = next(t for t in out["trace"] if t["node"] == "research")
    assert research["reused_evidence"] == 2
    assert out["evidence"][eid(CONTRACT)].origin == "reused"
    assert world.verifier.calls > calls_before                        # new conclusions verified again
    assert out["verifications"]["C1"].status == "supported"
    assert out["report"].citations[0].evidence_id == eid(CONTRACT)
    md = out["report"].markdown
    assert "**Follow-up to:**" in md and "| reused |" in md
    snap = b.snapshot()                                               # same per-run limits
    assert b.limits.max_llm_calls == 25 and snap.mesh_llm_calls <= 25 and snap.retry_rounds <= 2


def test_follow_up_respects_source_scope_and_domain(world, monkeypatch):
    import agents.research_agent
    from retriever.web_search import ReplayWebSearch

    prior = prior_context_from(first_run(world))
    monkeypatch.setattr(agents.research_agent, "make_web", lambda b, s: ReplayWebSearch(b, recordings={}))
    world.planner.structured_script = [plan("anything")]
    world.generator.structured_script = [{"claims": []}]
    out = run_research("And on the web?", BudgetManager(BudgetLimits()), domain_id="procurement",
                       source_scope=["web"], prior_context=prior)
    assert not any(e.origin == "reused" for e in (out.get("evidence") or {}).values())  # documents not in scope
    out2 = run_research("Same question in pharma", BudgetManager(BudgetLimits()), domain_id="pharma",
                        source_scope=["documents"], prior_context=prior)
    assert out2.get("prior_context") is None and any("not reused" in n for n in out2["notices"])


def test_runs_do_not_share_state_without_prior_context(world):
    first_run(world)
    script_procurement(world)
    out = run_research(QUESTION, BudgetManager(BudgetLimits()), domain_id="procurement", source_scope=["documents"])
    assert not any(e.origin == "reused" for e in out["evidence"].values())
    assert next(t for t in out["trace"] if t["node"] == "intake")["follow_up"] is False


def test_follow_up_in_ui_and_session_isolation(world):
    script_procurement(world)
    a = run_app()
    assert len(a.toggle) == 1 and a.toggle[0].value is True           # follow-up offered after a first run
    script_follow_up(world)
    a.chat_input[0].set_value("When does it expire?").run()
    assert not a.exception
    report_caps = " ".join(c.value for c in a.tabs[0].caption)
    assert "↪ Follow-up to: Review Acme Metals as a supplier" in report_caps
    assert "♻️ reused from the previous question" in " ".join(c.value for c in a.tabs[2].caption)

    # a separate browser session starts empty: no history, no follow-up toggle, nothing reused
    script_procurement(world)
    b = AppTest.from_file(APP, default_timeout=60).run()
    assert len(b.toggle) == 0
    b.sidebar.selectbox[0].set_value("procurement")
    b.sidebar.radio[1].set_value(source_label(["documents"]))
    b.run()
    b.chat_input[0].set_value(QUESTION).run()
    audit = " ".join(m.value for m in b.tabs[3].markdown)
    assert "(4 new, 0 reused)" in audit
