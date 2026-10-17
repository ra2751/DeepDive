"""Streamlit integration of the verified-research graph (scripted fixtures, network blocked)."""

import logging
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from budget import BudgetLimits, BudgetManager
from tests.research_fixtures import CONTRACT
from tests.research_world import QUESTION, install_world, script_procurement
from ui.research_view import source_label

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def world(monkeypatch, settings, tmp_path):
    monkeypatch.chdir(tmp_path)
    return install_world(monkeypatch, settings)


def run_app(question=QUESTION, domain="procurement", sources=("documents",), claims=""):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.sidebar.radio[0].value == "Verified research (default)"
    at.sidebar.selectbox[0].set_value(domain)
    at.sidebar.radio[1].set_value(source_label(list(sources)))
    if claims:
        at.sidebar.text_area[0].set_value(claims)
    at.run()
    at.chat_input[0].set_value(question).run()
    assert not at.exception, at.exception
    return at


def texts(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def test_new_graph_is_default_and_shows_four_views(world):
    script_procurement(world)
    at = run_app()
    assert [t.label for t in at.tabs] == ["📄 Report", "🧾 Claims", "📚 Evidence", "🔍 Audit"]
    assert world.verifier.calls >= 1  # the claim-level verifier ran (not the legacy validator)


def test_report_separates_verified_from_excluded_claims(world):
    script_procurement(world)
    at = run_app()
    report = at.tabs[0]
    assert any("Verified findings: every finding below passed claim-level verification" in s.value for s in report.success)
    md = " ".join(m.value for m in report.markdown)
    assert "The Acme Metals master supply agreement runs until 31 December 2027. `[C1]`" in md
    assert "**C2** ❌ Contradicted" in md and "**C3** ⚠️ Unresolved" in md and "**C4** ⚠️ Unresolved" in md
    assert "holds ISO 27001 certification. `[C2]`" not in md          # contradicted claim never a finding
    assert any("Human review required for: C2" in w.value for w in report.warning)
    assert len(report.get("download_button")) == 1


def test_claims_tab_shows_statuses_rules_and_rejections(world):
    script_procurement(world)
    at = run_app()
    claims = at.tabs[1]
    body = " ".join([m.value for m in claims.markdown] + [c.value for c in claims.caption])
    for badge in ("✅ Supported (1)", "❌ Contradicted (1)", "⚠️ Unresolved (2)"):
        assert badge in body
    assert "human review required" in body
    assert "rejected evidence" in body and "quote_not_in_evidence" in body
    assert len(claims.dataframe) == 1


def test_evidence_tab_shows_page_citation_and_exact_quote(world):
    script_procurement(world)
    at = run_app()
    ev = at.tabs[2]
    md = [m.value for m in ev.markdown]
    assert "**[C1] contract.pdf · page 1**" in md
    assert "> “runs until 31 December 2027”" in md
    assert any("contradicts C2" in m and "audit_report.pdf · page 7" in m for m in md)


def test_audit_tab_shows_retries_usage_rejections_and_citation_check(world):
    script_procurement(world)
    at = run_app()
    audit = at.tabs[3]
    md = " ".join(m.value for m in audit.markdown)
    assert "Retry rounds used: **2/2**" in md
    assert "Mesh LLM calls" in md and "Cost unknown" in md       # fake models are unpriced -> unknown, not exact
    assert "Integrity issues: **0**" in md and "Sentences removed: **2**" in md
    assert len(audit.dataframe) >= 4                              # trace, Mesh calls, rejections, removed sentences


def test_zero_evidence_search_is_graceful(world):
    world.docs = []                                               # nothing in the index matches
    world.planner.structured_script = [{"in_scope": True, "scope_reason": "ok", "sub_questions": ["anything"]}]
    at = run_app(question="What does the index say about quantum annealing suppliers?")
    report = at.tabs[0]
    assert any("No verified findings" in w.value for w in report.warning)
    assert "No evidence was found in the selected sources" in texts(report)
    assert world.verifier.calls == 0 and world.generator.calls == 0  # no wasted model calls


def test_budget_termination_is_shown_and_not_verified(world, monkeypatch):
    import ui.research_view

    script_procurement(world)
    # patch the binding the view actually uses (it imports new_research_budget by name)
    monkeypatch.setattr(ui.research_view, "new_research_budget",
                        lambda s: BudgetManager(BudgetLimits(max_llm_calls=1)))
    at = run_app()
    report = at.tabs[0]
    assert not report.success                                     # never shown as verified
    assert any("No verified findings" in w.value or "Run stopped early" in w.value for w in report.warning)
    assert "termination: max_llm_calls" in " ".join(m.value for m in at.tabs[3].markdown)


def test_fact_check_mode_from_sidebar(world):
    from tests.research_fixtures import TRIAL_A, judgment

    world.docs = [TRIAL_A]
    world.verifier.structured_script = [{"judgments": [
        judgment("C1", TRIAL_A, "supports", "Drug X reduced HbA1c by 1.2 percent versus placebo at 26 weeks")]}]
    at = run_app(question="Is Drug X effective?", domain="pharma",
                 claims="Drug X reduced HbA1c by 1.2 percent at 26 weeks.")
    body = " ".join([m.value for m in at.tabs[1].markdown] + [c.value for c in at.tabs[1].caption])
    assert "⚠️ Unresolved (1)" in body and "1/2 required" in body   # pharma high-risk needs 2 sources
    assert world.planner.calls == 0                                  # fact-check mode skips planning


def test_report_markdown_download_content(world):
    from graph_builder.research_graph import run_research
    from ui.research_view import report_markdown

    script_procurement(world)
    b = BudgetManager(BudgetLimits())
    out = run_research(QUESTION, b, domain_id="procurement", source_scope=["documents"])
    md = report_markdown({"out": out, "usage": b.snapshot(), "limits": b.limits, "at": "t", "wall_s": 0.1})
    assert md.startswith("# Research report")
    assert "| C1 | supported | contract.pdf, p. 1 | Unverified (0.35) | new | \"runs until 31 December 2027\" |" in md
    assert "## Contradicted claims (excluded from the findings)" in md
    assert "Rejected evidence judgments" in md and "quote_not_in_evidence" in md
    assert CONTRACT["text"] not in md                                 # no full document passages in the download


def test_confidential_document_text_not_logged(world, caplog):
    secret = "CONFIDENTIAL-7731 the master supply agreement pricing schedule is attached"
    world.docs = [{"text": secret, "source": "secret.pdf", "page": 2}] + world.docs
    script_procurement(world)
    with caplog.at_level(logging.DEBUG):
        run_app()
    assert "CONFIDENTIAL-7731" not in caplog.text
