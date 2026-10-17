"""
LIVE Streamlit smoke test: the real app.py (Streamlit AppTest harness) answering a
research question from the EXISTING Milvus index. No PDF upload. Uses the G3
ledger's remaining share (3 Mesh LLM calls + 1 embedding, 0 Tavily).
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.live

APP = str(Path(__file__).resolve().parents[2] / "app.py")
QUESTION = ("According to the attached Transformer survey paper, how do the authors categorize "
            "the different Transformer variants (X-formers)?")


def test_streamlit_rag_smoke(live_settings, g3, once, monkeypatch, tmp_path):
    import budget as budget_module
    from streamlit.testing.v1 import AppTest

    b = g3.budget("STREAMLIT", llm=3, embedding=1, web=0)
    import ui.legacy_view
    import ui.research_view

    # patch every binding that creates the per-question budget (modules import it by name)
    for mod in (budget_module, ui.research_view, ui.legacy_view):
        monkeypatch.setattr(mod, "new_research_budget", lambda settings: b)
    monkeypatch.chdir(tmp_path)  # the app creates data/raw_pdfs relative to cwd; keep the repo clean

    at = AppTest.from_file(APP, default_timeout=240).run()
    at.chat_input[0].set_value(QUESTION).run()

    banners = {"success": [e.value for e in at.success], "warning": [e.value for e in at.warning],
               "info": [e.value for e in at.info], "error": [e.value for e in at.error]}
    markdown = [m.value for m in at.markdown if len(m.value) > 80]
    captions = [c.value for c in at.caption]
    rows_after = None
    try:
        from pymilvus import MilvusClient
        s = live_settings
        rows_after = MilvusClient(uri=s.milvus_endpoint, token=s.milvus_api_key).get_collection_stats(
            s.milvus_collection).get("row_count")
    except Exception as e:  # report, never fail the smoke test on the stats read
        rows_after = f"stats read failed: {type(e).__name__}"
    g3.commit(once, b, question=QUESTION, exception=[str(x) for x in at.exception], banners=banners,
              answer_excerpt=(markdown[-1] if markdown else "")[:1200], captions=captions, rows_after=rows_after)

    assert not at.exception
    assert any(banners.values())   # an answer-status label is always shown
    assert not (banners["success"] and b.snapshot().mesh_llm_calls == 0)
