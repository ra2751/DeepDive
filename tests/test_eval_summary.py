"""The evaluation summary is generated from recorded result files only and never modifies them."""

import hashlib
from pathlib import Path

from evals.summarize_results import RESULTS, RUNS, load, render, summarize


def _hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for _, rid in RUNS for p in RESULTS.glob(f"{rid}*")}


def test_summary_is_derived_from_recorded_files_without_modifying_them():
    before = _hashes()
    obs = load()
    s = summarize(obs)
    md = render(obs, s)
    assert _hashes() == before
    assert s["questions"] == 10 and s["observations"] == 11             # L08-r1 is not a separate question
    assert [o["case"]["case_id"] for o in s["questions_map"]["L08"]] == ["L08", "L08-r1"]
    assert s["leakage"] == 0 and s["issues"] == 0
    assert "Small sample" in md and "L08-r1 is a re-run of L08" in md


def test_committed_summary_matches_regenerated_summary():
    obs = load()
    assert (RESULTS / "SUMMARY.md").read_text() == render(obs, summarize(obs))
