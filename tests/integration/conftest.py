"""
G3 live-test harness: a persistent ledger caps the TOTAL requests across all live
tests (approved: 10 Mesh inference requests, 2 Tavily searches; Milvus reads only)
and refuses to re-run a test that already spent budget.
"""

import json
import os
from pathlib import Path

import pytest

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from budget import BudgetLimits, BudgetManager, PriceTable  # noqa: E402
from config.settings import get_settings, reset_settings_cache, set_settings_override  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "data" / "probe"
LEDGER = PROBE / "g3_ledger.json"
RESULTS = PROBE / "g3_results.json"
CAPS = {"llm": 8, "embedding": 2, "web": 2}   # 8 chat + 2 embedding = 10 Mesh requests


def _load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


class G3Ledger:
    def used(self):
        return _load(LEDGER, {"llm": 0, "embedding": 0, "web": 0})

    def remaining(self, kind):
        return max(0, CAPS[kind] - self.used()[kind])

    def already_ran(self, test_id):
        return test_id in _load(RESULTS, {})

    def budget(self, name, llm=0, embedding=0, web=0):
        s = get_settings()
        limits = BudgetLimits.from_settings(s).model_copy(update={
            "max_llm_calls": min(llm, self.remaining("llm")),
            "max_embedding_calls": min(embedding, self.remaining("embedding")),
            "max_web_searches": min(web, self.remaining("web")),
        })
        return BudgetManager(limits, pricing=PriceTable.load(s.mesh_pricing_file), name=name)

    def commit(self, test_id, budget, **extra):
        snap = budget.snapshot()
        used = self.used()
        used["llm"] += snap.mesh_llm_calls
        used["embedding"] += snap.mesh_embedding_calls
        used["web"] += snap.web_searches
        PROBE.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(json.dumps(used))
        results = _load(RESULTS, {})
        results[test_id] = {"usage": snap.model_dump(), "limits": budget.limits.model_dump(),
                            "events": budget.events, **extra}
        RESULTS.write_text(json.dumps(results, indent=2, default=str))


@pytest.fixture(scope="session")
def live_settings():
    set_settings_override(None)
    reset_settings_cache()
    s = get_settings()
    missing = [n for n, v in (("MESH_API_KEY", s.mesh_api_key),) if not v]
    if missing:
        pytest.skip("missing " + ", ".join(missing))
    return s


@pytest.fixture(scope="session")
def g3():
    return G3Ledger()


@pytest.fixture
def once(request, g3):
    """Skip a live test that already ran (and spent budget) in an earlier session."""
    test_id = request.node.name
    if g3.already_ran(test_id) and not os.environ.get("G3_ALLOW_RERUN"):
        pytest.skip(f"{test_id} already ran; see data/probe/g3_results.json")
    return test_id
