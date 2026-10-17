"""B4 harness, exercised offline with scripted fakes (no live calls)."""

import json

import pytest

from evals.ui_pilot import B4_CAPS, PER_QUESTION, QUESTIONS, PilotLedger, main, question_budget, run_pilot
from tests.research_fixtures import CONTRACT, eid, judgment
from tests.research_world import install_world, plan


@pytest.fixture
def world(monkeypatch, settings, tmp_path):
    import agents.research_agent
    from retriever.web_search import ReplayWebSearch

    monkeypatch.chdir(tmp_path)
    w = install_world(monkeypatch, settings)
    url = "https://arxiv.org/abs/1706.03762"
    web_text = "The master supply agreement between Northwind Foods and Acme Metals runs until 31 December 2027, per the filing."
    w.web_calls = []

    def make_web(b, s):
        rep = ReplayWebSearch(b, recordings={})
        original = rep.search

        def search(query, max_results=None, **kw):
            rep.recordings[query] = [{"url": url, "content": web_text}]
            out = original(query, max_results=max_results)    # raises BudgetExceeded before any search
            w.web_calls.append(query)                          # count only searches that actually ran
            return out

        rep.search = search
        return rep

    monkeypatch.setattr(agents.research_agent, "make_web", make_web)
    w.planner.structured_script = [plan("Acme Metals agreement term", "Acme Metals contract expiry", "Acme Metals renewal"),
                                   {"in_scope": True, "scope_reason": "follow-up",
                                    "resolved_question": "When does the Acme Metals agreement expire?",
                                    "sub_questions": ["Acme Metals agreement expiry"]}]
    claim = {"text": "The master supply agreement with Acme Metals runs until 31 December 2027.",
             "sub_question_index": 1, "claim_type": "fact", "evidence_ids": [eid(CONTRACT)]}
    w.generator.structured_script = [{"claims": [claim]},
                                     {"sentences": [{"text": "The agreement runs until 31 December 2027 (C1).", "claim_ids": ["C1"]}]},
                                     {"claims": [claim]},
                                     {"sentences": [{"text": "It runs until 31 December 2027 (C1).", "claim_ids": ["C1"]}]}]
    w.verifier.structured_script = [{"judgments": [judgment("C1", CONTRACT, "supports", "runs until 31 December 2027")]}]
    return w


def test_dry_run_makes_no_calls(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["ui_pilot"])
    main()
    out = capsys.readouterr().out
    assert "Dry run only (no calls made)" in out and "13 Mesh per question" in out and "cap 26 Mesh / 2 web" in out


def test_per_question_budget_is_13_and_web_capped(tmp_path, settings):
    led = PilotLedger(tmp_path / "l.json")
    b1 = question_budget(led, 2, settings)
    assert (b1.limits.max_llm_calls, b1.limits.max_embedding_calls, b1.limits.max_web_searches) == (9, 4, 2)
    led.data.update(mesh=20, web=2)
    b2 = question_budget(led, 2, settings)
    assert b2.limits.max_llm_calls + b2.limits.max_embedding_calls <= 6 and b2.limits.max_web_searches == 0


def test_pilot_runs_question_and_follow_up_within_caps(world, tmp_path):
    ledger = tmp_path / "stage_b_B4_ledger.json"
    results, led = run_pilot(ledger, timeout=60)
    assert [r["id"] for r in results] == ["B4-Q1", "B4-Q2"] and all(not r["exception"] for r in results)
    q1, q2 = results
    for r in results:
        u, lim = r["usage"], r["limits"]
        assert u["mesh_chat_calls"] + u["mesh_vision_calls"] <= PER_QUESTION["llm"] == lim["max_llm_calls"]
        assert u["mesh_embedding_calls"] <= PER_QUESTION["embedding"]
    assert q1["usage"]["web_searches"] <= 2 and q2["usage"]["web_searches"] == 0 and q2["limits"]["max_web_searches"] == 0
    assert led["mesh"] <= B4_CAPS["mesh"] and led["web"] <= B4_CAPS["web"]
    assert any("Verified findings" in s for s in q1["tabs"]["📄 Report"]["success"])
    assert any("Follow-up to" in c for c in q2["tabs"]["📄 Report"]["caption"])
    assert any("♻️ reused" in c for c in q2["tabs"]["📚 Evidence"]["caption"])
    assert (tmp_path / "stage_b_B4_results.json").exists()


def test_pilot_refuses_rerun_and_respects_remaining_cap(world, tmp_path):
    ledger = tmp_path / "stage_b_B4_ledger.json"
    ledger.write_text(json.dumps({"mesh": 20, "web": 0, "questions": {"B4-Q1": {}}}))
    results, led = run_pilot(ledger, timeout=60)
    assert results[0]["skipped"] == "already executed (ledger)"
    assert "B4 cap 26 would be exceeded" in results[1]["skipped"]          # 20 used + 13 > 26
    assert led["mesh"] == 20


def test_web_attempts_stop_at_cap(world, tmp_path):
    results, _ = run_pilot(tmp_path / "stage_b_B4_ledger.json", questions=QUESTIONS[:1], timeout=60)
    assert results[0]["usage"]["web_searches"] == 2 and len(world.web_calls) == 2   # 3 sub-questions, cap 2
    notices = " ".join(results[0]["tabs"]["📄 Report"]["caption"])
    assert "Web search limit reached (max_web_searches)" in notices


def test_follow_up_reverifies_and_uses_only_approved_paths(world, tmp_path):
    results, _ = run_pilot(tmp_path / "stage_b_B4_ledger.json", timeout=60)
    q1, q2 = results
    q2_calls = [e for e in q2["events"] if e.get("provider") == "mesh" and e["operation"] == "structured"]
    assert any(e["model_requested"] == "test/verifier" for e in q2_calls)       # follow-up re-verified
    providers = {e.get("provider") for r in results for e in r["events"]}
    assert providers <= {"mesh", "local", "replay"}                            # gateway + replay web only
    ops = {e.get("operation") for r in results for e in r["events"] if e.get("provider") == "local"}
    assert ops <= {"milvus_search"}                                            # Milvus: search only (read-only)
    assert world.milvus is not None and not hasattr(world.milvus, "insert")



def test_pilot_records_export_and_details(world, tmp_path):
    results, _ = run_pilot(tmp_path / "stage_b_B4_ledger.json", timeout=60)
    q1, q2 = results
    assert q1["export_markdown"].startswith("# Research report") and q1["download_buttons"] == 1
    assert "## Audit summary" in q1["export_markdown"] and "## Verification limitations" in q1["export_markdown"]
    assert q2["follow_up_to"] and "**Follow-up to:**" in q2["export_markdown"]
    assert q2["details"]["verifications"]["C1"]["status"] == "supported"
    assert any(v["origin"] == "reused" for v in q2["details"]["evidence_locators"].values())


def test_gr1_pilot_budget_is_web_only_with_hard_caps(tmp_path, settings):
    from evals.ui_pilot import GR1_CAPS, GR1_PER_QUESTION, GR1_QUESTIONS
    led = PilotLedger(tmp_path / "gr1.json", GR1_CAPS)
    b = question_budget(led, GR1_QUESTIONS[0]["web_cap"], settings, GR1_PER_QUESTION)
    assert (b.limits.max_llm_calls, b.limits.max_embedding_calls, b.limits.max_web_searches) == (9, 0, 8)
    assert GR1_QUESTIONS[0]["sources"] == ["web"] and GR1_CAPS == {"mesh": 9, "web": 8}


def test_gr1_pilot_dry_run_makes_no_calls(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["ui_pilot", "--pilot", "gr1"])
    main()
    out = capsys.readouterr().out
    assert "GR1 plan: 1 questions · 9 Mesh per question (9 LLM + 0 embedding)" in out and "cap 9 Mesh / 8 web" in out
    assert "Dry run only" in out
