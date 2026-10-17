"""The live-eval runner and result format, exercised offline with scripted fakes (kind=offline_fake)."""

import json

from evals.result_schema import EvalRun
from evals.run_live_eval import main, run
from tests.research_fixtures import TRIAL_A, TRIAL_B, judgment
from tests.research_world import install_world


def test_runner_writes_schema_valid_results(monkeypatch, settings, tmp_path):
    w = install_world(monkeypatch, settings)
    w.docs = [TRIAL_A, TRIAL_B]
    w.verifier.structured_script = [{"judgments": [
        judgment("C1", TRIAL_A, "supports", "Drug X reduced HbA1c by 1.2 percent versus placebo at 26 weeks")]}]
    w.planner.structured_script = [{"in_scope": False, "scope_reason": "personal dosing", "sub_questions": []}]
    ds = tmp_path / "ds.jsonl"
    ds.write_text("\n".join(json.dumps(c) for c in [
        {"case_id": "T1", "mode": "fact_check", "domain": "general", "claims": ["Drug X reduced HbA1c by 1.2 percent at 26 weeks."],
         "expected": {"C1": ["supported"]}, "source_scope": ["documents"]},
        {"case_id": "T2", "mode": "research", "domain": "pharma", "question": "What dose should I take?",
         "expected": {"report_status": ["out_of_scope"]}, "source_scope": ["documents"]},
    ]))
    run_obj, path = run(ds, max_mesh=23, max_web=0, out_dir=tmp_path / "out", kind="offline_fake")  # 7 + 16
    parsed = EvalRun.model_validate_json(path.read_text())
    assert parsed.kind == "offline_fake" and len(parsed.cases) == 2
    assert [c.passed for c in parsed.cases] == [True, True]
    assert parsed.metrics["unsupported_claim_leakage"].numerator == 0
    assert parsed.metrics["citation_integrity_issues"].numerator == 0
    assert parsed.totals.mesh_llm_calls + parsed.totals.mesh_embedding_calls <= 23
    assert (tmp_path / "out" / f"{parsed.run_id}.md").exists()


def test_runner_respects_run_level_cap(monkeypatch, settings, tmp_path):
    w = install_world(monkeypatch, settings)
    w.planner.structured_script = [{"in_scope": False, "scope_reason": "test", "sub_questions": []}]  # 1 call per case
    ds = tmp_path / "ds.jsonl"
    ds.write_text("\n".join(json.dumps({"case_id": f"T{i}", "mode": "research", "domain": "general",
                                        "question": "q", "expected": {}, "source_scope": ["documents"]})
                            for i in range(5)))
    # cap 16 = one worst-case research share: case 1 runs (uses 1), then 1 + 16 > 16 so the rest are skipped
    run_obj, _ = run(ds, max_mesh=16, max_web=0, out_dir=tmp_path, kind="offline_fake")
    assert len(run_obj.cases) == 1 and len(run_obj.notes) == 4
    assert run_obj.totals.mesh_llm_calls + run_obj.totals.mesh_embedding_calls <= 16
    # with room for every worst-case share, all cases run and the total still stays under the cap
    run_obj, _ = run(ds, max_mesh=80, max_web=0, out_dir=tmp_path, kind="offline_fake")
    assert len(run_obj.cases) == 5 and run_obj.totals.mesh_llm_calls == 5


def test_cli_is_dry_run_without_confirmation(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_live_eval"])
    main()
    out = capsys.readouterr().out
    assert "Dry run only (no calls made)" in out and "10 cases" in out


def test_persistent_ledger_caps_across_runs_and_refuses_reruns(monkeypatch, settings, tmp_path):
    w = install_world(monkeypatch, settings)
    w.planner.structured_script = [{"in_scope": False, "scope_reason": "test", "sub_questions": []}]
    ds = tmp_path / "ds.jsonl"
    ds.write_text("\n".join(json.dumps({"case_id": f"T{i}", "mode": "research", "domain": "general",
                                        "question": "q", "expected": {}, "source_scope": ["documents"]})
                            for i in range(3)))
    ledger = tmp_path / "ledger.json"
    r1, _ = run(ds, max_mesh=17, max_web=0, out_dir=tmp_path, kind="offline_fake", case_ids=["T0"], ledger=ledger)
    assert [c.case_id for c in r1.cases] == ["T0"] and json.loads(ledger.read_text())["mesh"] == 1
    # second session: ledger says 1 used; T1 fits (1 + 16 <= 17), then 2 + 16 > 17 so T2 is skipped
    r2, _ = run(ds, max_mesh=17, max_web=0, out_dir=tmp_path, kind="offline_fake", ledger=ledger)
    assert [c.case_id for c in r2.cases] == ["T1"]
    assert any("already executed" in n for n in r2.notes) and any("cap 17" in n for n in r2.notes)
    led = json.loads(ledger.read_text())
    assert led["mesh"] <= 17 and led["cases"] == ["T0", "T1"]


def test_strict_shares_bound_worst_case():
    from evals.run_live_eval import PER_CASE_MESH
    assert PER_CASE_MESH["fact_check"] * 2 + PER_CASE_MESH["research"] == 30   # Stage A worst case
