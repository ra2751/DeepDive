"""General-purpose, Web-only research through the full nine-node graph (offline).

The two representative questions run with scripted Mesh fakes and replayed web results.
All web passages below are SYNTHETIC test fixtures on reserved .example hosts: they exercise
the workflow logic and are not real market data, and the tests are not measurements of live
research quality.
"""

import pytest

from agents.intake_agent import IntakeAgent
from agents.research_common import evidence_id
from budget import BudgetLimits, BudgetManager
from graph_builder.research_graph import prior_context_from, run_research
from retriever.web_search import ReplayWebSearch
from tests.research_fixtures import CONTRACT, eid, judgment
from tests.research_world import install_world, plan
from verification.temporal import is_forecast, time_basis

EV_Q = ("What are the three biggest growth opportunities and three biggest commercialization risks in the EV "
        "battery market in 2026, based on evidence available as of October 2026?")
INDIA_Q = ("What are the three biggest growth opportunities and three biggest commercialization risks for India's "
           "defence-tech startups through 2030, based on developments and policies available as of October 2026?")


def web(url, content, published=None, retrieved="2026-10-17T09:00:00+00:00"):
    return {"url": url, "content": content, "published_date": published, "retrieved_at": retrieved}


def wid(r):
    return evidence_id(r["url"], None, r["content"])


def section(md, title):
    """Lines of one '## title' section of the report markdown."""
    body = md.split(f"## {title}\n", 1)[1]
    return body.split("\n## ", 1)[0].splitlines()


def wj(cid, r, stance, quote):
    return {"claim_id": cid, "evidence_id": wid(r), "stance": stance, "quote": quote, "reasoning": "test"}


# ------------------------------------------------------------------ EV battery fixture (synthetic)
EV_SUBS = ["EV battery market growth opportunities 2026", "EV battery commercialization risks 2026",
           "EV battery market forecasts to 2030"]
EV1 = web("https://energy-agency.example/ev-outlook-2026",
          "Battery demand for electric vehicles grew 25 percent in 2025 compared with 2024. Sodium-ion cells "
          "entered commercial production in 2025 for low-cost vehicles.", published="2026-05-14")
EV2 = web("https://battery-news.example/risks",
          "Lithium price volatility remains a key commercialization risk for cell makers. Several gigafactory "
          "projects were reviewed by their investors.")
EV3 = web("https://market-research.example/forecast",
          "The firm projects that global battery pack prices will fall below 80 dollars per kWh by 2030.",
          published="2026-08-02")
EV4 = web("https://trade-journal.example/demand",
          "Battery demand for electric vehicles grew 15 percent in 2025, according to the journal's own tally.")
EV_NO_URL = {"url": "", "content": "Battery makers announced record profits in 2026."}
EV5 = web("https://industry-tracker.example/gigafactories",
          "Several gigafactory projects were delayed in 2025 due to weak demand, the tracker reported.")
EV_CLAIMS = {
    "C1": "Battery demand for electric vehicles grew 25 percent in 2025.",
    "C2": "Sodium-ion cells entered commercial production in 2025.",
    "C3": "Lithium price volatility is a key commercialization risk for battery cell makers.",
    "C4": "According to one market research firm, global battery pack prices will fall below 80 dollars per kWh by 2030.",
    "C5": "Several gigafactory projects were delayed in 2025 due to weak demand.",
}


def script_ev(world, monkeypatch):
    import agents.research_agent
    import agents.retry_agent
    recordings = {EV_SUBS[0]: [EV1, EV_NO_URL], EV_SUBS[1]: [EV2, EV4], EV_SUBS[2]: [EV3], EV_CLAIMS["C5"]: [EV5]}
    for mod in (agents.research_agent, agents.retry_agent):
        monkeypatch.setattr(mod, "make_web", lambda b, s: ReplayWebSearch(b, recordings=recordings))
    world.planner.structured_script = [plan(*EV_SUBS)]
    world.generator.structured_script = [
        {"claims": [
            {"text": EV_CLAIMS["C1"], "sub_question_index": 1, "claim_type": "quantity", "evidence_ids": [wid(EV1), wid(EV4)]},
            {"text": EV_CLAIMS["C2"], "sub_question_index": 1, "claim_type": "fact", "evidence_ids": [wid(EV1)]},
            {"text": EV_CLAIMS["C3"], "sub_question_index": 2, "claim_type": "fact", "evidence_ids": [wid(EV2)]},
            {"text": EV_CLAIMS["C4"], "sub_question_index": 3, "claim_type": "forecast", "evidence_ids": [wid(EV3)]},
            {"text": EV_CLAIMS["C5"], "sub_question_index": 2, "claim_type": "fact", "evidence_ids": [wid(EV2)]},
        ]},
        {"sentences": [
            {"text": "Sodium-ion cells entered commercial production in 2025.", "claim_ids": ["C2"]},
            {"text": "Lithium price volatility is a key commercialization risk for battery cell makers.", "claim_ids": ["C3"]},
            {"text": "One market research firm projects battery pack prices below 80 dollars per kWh by 2030.", "claim_ids": ["C4"]},
            # invented current market figure: not in any verified claim
            {"text": "The EV battery market was worth 150 billion dollars in 2026.", "claim_ids": ["C3"]},
            {"text": "Several gigafactory projects were delayed in 2025 due to weak demand.", "claim_ids": ["C5"]},
        ]},
    ]
    world.verifier.structured_script = [
        {"judgments": [
            wj("C1", EV1, "supports", "Battery demand for electric vehicles grew 25 percent in 2025"),
            wj("C1", EV4, "contradicts", "Battery demand for electric vehicles grew 15 percent in 2025"),
            wj("C2", EV1, "supports", "Sodium-ion cells entered commercial production in 2025 for low-cost vehicles"),
            wj("C3", EV2, "supports", "Lithium price volatility remains a key commercialization risk for cell makers"),
            wj("C4", EV3, "supports", "The firm projects that global battery pack prices will fall below 80 dollars per kWh by 2030"),
            wj("C5", EV2, "supports", "Many gigafactory projects were cancelled in 2025 due to weak demand"),   # fabricated
        ]},
        {"judgments": [wj("C5", EV5, "supports", "Several gigafactory projects were delayed in 2025 due to weak demand")]},
    ]


@pytest.fixture
def world(monkeypatch, settings):
    return install_world(monkeypatch, settings)


@pytest.fixture
def ev_run(world, monkeypatch):
    script_ev(world, monkeypatch)
    budget = BudgetManager(BudgetLimits())
    return run_research(EV_Q, budget, domain_id="general", source_scope=["web"]), budget, world


def test_both_questions_accepted_by_general_intake():
    for q in (EV_Q, INDIA_Q):
        out = IntakeAgent({"query": q, "domain_id": "general", "source_scope": ["web"]}, {})
        assert out["status"] == "running" and out["source_scope"] == ["web"]


def test_ev_web_only_uses_no_documents_milvus_or_embeddings(ev_run):
    out, budget, world = ev_run
    assert out["evidence"] and all(e.locator.source_type == "web" for e in out["evidence"].values())
    assert world.milvus is None and world.embeddings.calls == 0
    snap = budget.snapshot()
    assert snap.mesh_embedding_calls == 0 and snap.web_searches == 4          # 3 sub-questions + 1 retry
    assert snap.mesh_llm_calls == 5                                         # plan, extract, 2 verify, synthesize


def test_ev_every_accepted_web_citation_has_a_real_url(ev_run):
    out, _, _ = ev_run
    cits = out["report"].citations
    assert cits and all(c.url and c.url.startswith("https://") and c.page is None for c in cits)
    assert not out.get("citation_issues")
    assert not any(e.locator.url in ("", "Web Search") for e in out["evidence"].values())   # no-URL result dropped


def test_ev_publication_and_retrieval_dates_are_distinct(ev_run):
    out, _, _ = ev_run
    ev1 = out["evidence"][wid(EV1)].locator
    assert ev1.published_at == "2026-05-14" and ev1.retrieved_at.startswith("2026-10-17")
    assert out["evidence"][wid(EV2)].locator.published_at is None
    md = out["report"].markdown
    assert "published 2026-05-14 · retrieved 2026-10-17" in md
    assert "published not reported · retrieved 2026-10-17" in md


def test_ev_forecast_distinguished_from_historical(ev_run):
    out, _, _ = ev_run
    claims = out["claims"]
    assert time_basis(claims["C4"]) == "forecast" and time_basis(claims["C2"]) == "historical/current"
    md = out["report"].markdown
    assert any("80 dollars per kWh" in x for x in section(md, "Supported forecasts and projections"))
    assert not any("80 dollars per kWh" in x for x in section(md, "Verified findings"))
    assert any(x.startswith("- Sodium-ion") for x in section(md, "Verified findings"))


def test_ev_invented_market_figure_removed(ev_run):
    out, _, _ = ev_run
    removed = out["report"].removed_sentences
    assert sorted(r["reason"] for r in removed) == ["forecast_stated_as_fact",            # conservative: replaced
                                                    "sentence_numbers_not_in_cited_claims:150,2026"]  # by claim text
    assert "150 billion" not in "\n".join(s.text for s in out["report"].verified_sentences)


def test_ev_contradiction_and_fabricated_quote_visible_and_retry_recovers(ev_run):
    out, _, _ = ev_run
    v = out["verifications"]
    assert v["C1"].status == "unresolved" and v["C1"].contradicting_evidence_ids == [wid(EV4)]
    md = out["report"].markdown
    assert "C1 — unresolved (conflicting evidence)" in md and "15 percent" in md
    # C5: fabricated quote rejected in round 0, then supported by targeted web retry evidence
    assert any(r.reason == "quote_not_in_evidence" for r in v["C5"].rejected)
    assert v["C5"].status == "supported" and v["C5"].supporting_evidence_ids == [wid(EV5)]
    assert [t["round"] for t in out["trace"] if t["node"] == "retry"][:1] == [1]
    assert {v[c].status for c in ("C2", "C3", "C4")} == {"supported"}
    assert out["report"].status == "completed"


def test_ev_mesh_budget_enforced(world, monkeypatch):
    script_ev(world, monkeypatch)
    budget = BudgetManager(BudgetLimits(max_llm_calls=2))
    out = run_research(EV_Q, budget, domain_id="general", source_scope=["web"])
    assert budget.snapshot().mesh_llm_calls <= 2
    assert out["report"].status in ("partial", "no_verified_findings")
    assert all(v.status != "supported" for v in out["verifications"].values())


# ------------------------------------------------------------------ India defence-tech fixture (synthetic)
IN_SUBS = ["India defence-tech startup growth opportunities through 2030",
           "India defence-tech startup commercialization risks", "India defence procurement policy 2026"]
IN1 = web("https://defence-ministry.example/innovation-challenge",
          "The innovation challenge programme had awarded contracts to 400 startups by March 2026, according "
          "to the ministry.", published="2026-03-10")
IN2 = web("https://policy-review.example/startup-risks",
          "Long procurement cycles remain the main commercialization risk for defence startups. Startups "
          "reported trial-to-order times of three to five years.")
IN3 = web("https://defence-ministry.example/exports",
          "The ministry expects defence exports to reach 50,000 crore rupees by 2029.", published="2026-04-01")


def script_india(world, monkeypatch, recordings=None):
    import agents.research_agent
    import agents.retry_agent
    recordings = recordings or {IN_SUBS[0]: [IN1, IN3], IN_SUBS[1]: [IN2], IN_SUBS[2]: []}
    for mod in (agents.research_agent, agents.retry_agent):
        monkeypatch.setattr(mod, "make_web", lambda b, s: ReplayWebSearch(b, recordings=recordings))
    world.planner.structured_script = [plan(*IN_SUBS)]
    world.generator.structured_script = [
        {"claims": [
            {"text": "The innovation challenge programme had awarded contracts to 400 startups by March 2026.",
             "sub_question_index": 1, "claim_type": "quantity", "evidence_ids": [wid(IN1)]},
            {"text": "Long procurement cycles are the main commercialization risk for defence startups.",
             "sub_question_index": 2, "claim_type": "fact", "evidence_ids": [wid(IN2)]},
            {"text": "The ministry expects defence exports to reach 50,000 crore rupees by 2029.",
             "sub_question_index": 1, "claim_type": "forecast", "evidence_ids": [wid(IN3)]},
        ]},
        {"sentences": [
            {"text": "The innovation challenge programme had awarded contracts to 400 startups by March 2026.", "claim_ids": ["C1"]},
            {"text": "Long procurement cycles are the main commercialization risk for defence startups.", "claim_ids": ["C2"]},
            {"text": "The ministry expects defence exports to reach 50,000 crore rupees by 2029.", "claim_ids": ["C3"]},
        ]},
    ]
    world.verifier.structured_script = [{"judgments": [
        wj("C1", IN1, "supports", "The innovation challenge programme had awarded contracts to 400 startups by March 2026"),
        wj("C2", IN2, "supports", "Long procurement cycles remain the main commercialization risk for defence startups"),
        wj("C3", IN3, "supports", "The ministry expects defence exports to reach 50,000 crore rupees by 2029"),
    ]}]


def test_india_web_only_research_with_tight_web_cap(world, monkeypatch):
    script_india(world, monkeypatch)
    budget = BudgetManager(BudgetLimits(max_web_searches=2))
    out = run_research(INDIA_Q, budget, domain_id="general", source_scope=["web"])
    assert budget.snapshot().web_searches == 2 and world.milvus is None
    assert {v.status for v in out["verifications"].values()} == {"supported"}
    assert out["report"].status == "completed" and len(out["report"].verified_sentences) == 3
    notices = " ".join(out["notices"])
    assert "remaining sub-questions were not searched on the web" in notices and "documents only" not in notices
    md = out["report"].markdown
    assert any("defence exports" in x for x in section(md, "Supported forecasts and projections"))
    assert all(c.url.startswith("https://") for c in out["report"].citations)


def test_india_follow_up_obeys_source_scope(world, monkeypatch):
    script_india(world, monkeypatch)
    first = run_research(INDIA_Q, BudgetManager(BudgetLimits()), domain_id="general", source_scope=["web"])
    prior = prior_context_from(first)
    assert prior["evidence"] and all(e.locator.source_type == "web" for e in prior["evidence"].values())

    # follow-up restricted to documents: no web evidence is reused, and no web search is made
    world.planner.structured_script = [plan("What is the Acme Metals contract term?")]
    world.generator.structured_script = [
        {"claims": [{"text": "The master supply agreement with Acme Metals runs until 31 December 2027.",
                     "sub_question_index": 1, "claim_type": "fact", "evidence_ids": [eid(CONTRACT)]}]},
        {"sentences": [{"text": "The Acme Metals agreement runs until 31 December 2027.", "claim_ids": ["C1"]}]}]
    world.verifier.structured_script = [{"judgments": [judgment("C1", CONTRACT, "supports", "runs until 31 December 2027")]}]
    budget = BudgetManager(BudgetLimits())
    out = run_research("And the supplier contract?", budget, domain_id="general", source_scope=["documents"],
                       prior_context=prior)
    assert all(e.locator.source_type == "document" for e in out["evidence"].values())
    assert next(t for t in out["trace"] if t["node"] == "research")["reused_evidence"] == 0
    assert budget.snapshot().web_searches == 0


def test_forecast_detector():
    assert is_forecast("Prices will fall below 80 dollars per kWh by 2030.", as_of_year=2026)
    assert is_forecast("Exports are expected to double.", as_of_year=2026)
    assert is_forecast("Capacity of 3 TWh in 2031.", as_of_year=2026)                 # future year
    assert is_forecast("Anything", claim_type="forecast")
    assert not is_forecast("Battery demand grew 25 percent in 2025.", as_of_year=2026)
    assert not is_forecast("400 startups had contracts by March 2026.", as_of_year=2026)


def test_replay_and_tavily_drop_results_without_url():
    from retriever.web_search import TavilyWebSearch

    class FakeTavily:
        def search(self, **kw):
            return {"results": [{"url": None, "content": "x"}, {"url": "Web Search", "content": "y"},
                                {"url": "https://ok.example/a", "content": "z", "published_date": "2026-09-01"}]}

    b = BudgetManager(BudgetLimits())
    from config.settings import get_settings
    res = TavilyWebSearch(b, get_settings(), client=FakeTavily()).search("q")
    assert [(r.url, r.published_date) for r in res] == [("https://ok.example/a", "2026-09-01")]
    rep = ReplayWebSearch(b, recordings={"q": [{"url": "", "content": "a"}, {"url": "https://ok.example/b", "content": "b"}]})
    assert [r.url for r in rep.search("q")] == ["https://ok.example/b"]
