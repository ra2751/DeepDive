"""The G2 probe keeps token-usage metadata, enforces its caps and never prints sample text."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def probe(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("mesh_probe", ROOT / "scripts" / "mesh_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path)
    monkeypatch.setattr(mod, "LEDGER", tmp_path / "g2_ledger.json")
    return mod


def test_structured_report_retains_token_usage_and_respects_cap(probe, fake_mesh, tmp_path):
    from tests.fakes import ScriptedChat

    answer = {"claim_id": "C1", "stance": "supports", "supporting_quote": probe.G2_EVIDENCE, "reasoning": "r"}
    for m in ("m/one", "m/two", "m/three", "m/four"):
        fake_mesh.all[m] = ScriptedChat(m, structured_script=[answer])
    probe.g2_structured(["m/one", "m/two", "m/three", "m/four"])

    report = json.loads(next(tmp_path.glob("g2_structured_*.json")).read_text())
    done = [r for r in report["results"] if r.get("schema_valid")]
    assert len(done) == 3
    assert all(r["attempts"][0]["input_tokens"] == 10 and r["attempts"][0]["output_tokens"] == 5 for r in done)
    assert report["results"][3]["skipped"].startswith("chat cap reached")
    assert report["ledger"]["chat"] == 3
    assert report["budget"]["mesh_chat_calls"] == 3


def test_ledger_refuses_to_exceed_authorized_totals(probe):
    probe._spend("milvus_query")
    probe._spend("milvus_query")
    with pytest.raises(SystemExit):
        probe._spend("milvus_query")


def test_sensitivity_scan_flags_obvious_patterns(probe):
    assert probe._sensitivity_scan("A Survey of Transformers. Transformers have achieved great success.")["clean"]
    hits = probe._sensitivity_scan("CONFIDENTIAL. Contact jane.doe@example.com, api_key=abc")["hits"]
    assert {"email", "secret_like", "confidentiality_marker"} <= set(hits)


def test_probe_never_prints_sample_text():
    src = (ROOT / "scripts" / "mesh_probe.py").read_text()
    assert 'print(sample["text"])' not in src and "SAMPLE TEXT" not in src
