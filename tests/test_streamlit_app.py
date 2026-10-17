"""T22: the Streamlit app still loads and shows honest status labels (fakes, offline)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.graph_helpers import patch_retriever, route, verdict

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_loads(fake_mesh, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("DeepDive AI" in t.value for t in at.title)


def test_failed_check_is_shown_as_not_validated(fake_mesh, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    patch_retriever(monkeypatch)
    fake_mesh.supervisor.script = [route("rag_agent")]
    fake_mesh.generator.script = ["Draft that fails."]
    fake_mesh.verifier.structured_script = [verdict("no")]

    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Legacy answer-level (fallback)").run()   # legacy fallback still works
    assert any("Fallback mode" in i.value for i in at.info)
    at.chat_input[0].set_value("Explain multi-head attention in the attached papers").run()

    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "Not validated" in warnings and "max_retry_rounds" in warnings
    assert not at.success  # never shown as passed
