"""
B4 live Streamlit pilot harness (prepared; requires explicit approval to run live).

    python -m evals.ui_pilot                    # dry run: prints the plan, makes NO calls
    python -m evals.ui_pilot --confirm-live     # runs the real app.py via Streamlit AppTest

Two questions through the real Streamlit app: (1) a research question over documents + web,
(2) a conversational follow-up (documents only) reusing the previous evidence. Each question
gets its own BudgetManager capped at 13 Mesh requests (9 LLM + 4 embedding); web searches are
capped at 2 for question 1 and 0 for the follow-up. A persistent ledger caps B4 at 26 Mesh
requests and 2 web searches in total and refuses to re-run a question that already spent budget.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from budget import BudgetLimits, BudgetManager, PriceTable  # noqa: E402
from config.settings import get_settings  # noqa: E402
from ui.research_view import source_label  # noqa: E402

APP = str(ROOT / "app.py")
PER_QUESTION = {"llm": 9, "embedding": 4}
B4_CAPS = {"mesh": 26, "web": 2}
DEFAULT_LEDGER = ROOT / "data" / "probe" / "stage_b_B4_ledger.json"
QUESTIONS = [
    {"id": "B4-Q1", "domain": "general", "sources": ["documents", "web"], "follow_up": False,
     "question": "According to the attached papers, how does multi-head attention work in the Transformer?",
     "web_cap": 2},
    {"id": "B4-Q2", "domain": "general", "sources": ["documents"], "follow_up": True,
     "question": "How many attention heads does the base model use?", "web_cap": 0},
]

# General-research Web-only pilot (proposed; runs only with --pilot gr1 --confirm-live after approval).
# No documents in scope and 0 embeddings allowed, so Milvus is never contacted.
GR1_PER_QUESTION = {"llm": 9, "embedding": 0}
GR1_CAPS = {"mesh": 9, "web": 8}
GR1_LEDGER = ROOT / "data" / "probe" / "general_research_gr1_ledger.json"
GR1_QUESTIONS = [
    {"id": "GR1-India", "domain": "general", "sources": ["web"], "follow_up": False, "web_cap": 8,
     "question": "What are the three biggest growth opportunities and three biggest commercialization risks for "
                 "India's defence-tech startups through 2030, based on developments and policies available as of "
                 "October 2026?"},
]
PILOTS = {"b4": (QUESTIONS, B4_CAPS, PER_QUESTION, DEFAULT_LEDGER),
          "gr1": (GR1_QUESTIONS, GR1_CAPS, GR1_PER_QUESTION, GR1_LEDGER)}


class PilotLedger:
    def __init__(self, path, caps=None):
        self.path = Path(path)
        self.caps = caps or B4_CAPS
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"mesh": 0, "web": 0, "questions": {}}

    def remaining(self, kind):
        return max(0, self.caps[kind] - self.data[kind])

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=str))


def question_budget(ledger: PilotLedger, web_cap: int, settings, per_question=None) -> BudgetManager:
    per_question = per_question or PER_QUESTION
    mesh_left = ledger.remaining("mesh")
    llm = min(per_question["llm"], max(0, mesh_left - per_question["embedding"]))
    emb = min(per_question["embedding"], mesh_left - llm)
    limits = BudgetLimits.from_settings(settings).model_copy(update={
        "max_llm_calls": llm, "max_embedding_calls": max(0, emb),
        "max_web_searches": min(web_cap, ledger.remaining("web"))})
    return BudgetManager(limits, pricing=PriceTable.load(settings.mesh_pricing_file), name="ui-pilot")


def _texts(container, kinds=("success", "warning", "info", "error", "markdown", "caption")):
    out = {}
    for k in kinds:
        out[k] = [e.value for e in getattr(container, k)]
    return out


def run_pilot(ledger_path=DEFAULT_LEDGER, questions=QUESTIONS, app=APP, timeout=600, caps=None, per_question=None):
    from streamlit.testing.v1 import AppTest

    import ui.research_view as view

    settings = get_settings()
    per_question = per_question or PER_QUESTION
    ledger = PilotLedger(ledger_path, caps)
    budgets = []

    def factory(_settings):
        return budgets[-1]

    original = view.new_research_budget
    view.new_research_budget = factory
    results = []
    try:
        at = AppTest.from_file(app, default_timeout=timeout).run()
        for q in questions:
            if q["id"] in ledger.data["questions"]:
                results.append({"id": q["id"], "skipped": "already executed (ledger)"})
                continue
            if ledger.remaining("mesh") < per_question["llm"] + per_question["embedding"]:
                results.append({"id": q["id"], "skipped": f"{'B4' if ledger.caps is B4_CAPS else 'pilot'} cap {ledger.caps['mesh']} would be exceeded"})
                continue
            budgets.append(question_budget(ledger, q["web_cap"], settings, per_question))
            at.sidebar.selectbox[0].set_value(q["domain"])
            at.sidebar.radio[1].set_value(source_label(q["sources"]))
            at.run()
            if at.toggle:
                at.toggle[0].set_value(q["follow_up"])
            at.chat_input[0].set_value(q["question"]).run()
            b = budgets[-1]
            snap = b.snapshot()
            ledger.data["mesh"] += snap.mesh_llm_calls + snap.mesh_embedding_calls
            ledger.data["web"] += snap.web_searches
            tabs = {t.label: _texts(t) for t in at.tabs}
            entry = {"id": q["id"], "question": q["question"], "sources": q["sources"], "follow_up": q["follow_up"],
                     "exception": [str(x) for x in at.exception], "limits": b.limits.model_dump(),
                     "usage": snap.model_dump(), "events": b.events, "tabs": tabs,
                     "cache_mode": settings.mesh_cache_mode,
                     "download_buttons": len(at.get("download_button"))}
            history = at.session_state["research_history"] if "research_history" in at.session_state else []
            if history:
                from evals.run_live_eval import case_details
                from ui.research_view import report_markdown

                entry["export_markdown"] = report_markdown(history[-1])     # exactly what the download serves
                entry["details"] = case_details(history[-1]["out"], b, None)
                entry["follow_up_to"] = history[-1].get("follow_up_to")
            ledger.data["questions"][q["id"]] = {k: entry[k] for k in ("usage", "limits", "exception")}
            ledger.save()
            results.append(entry)
    finally:
        view.new_research_budget = original
    out = Path(ledger_path).with_name(Path(ledger_path).stem.replace("_ledger", "") + "_results.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    return results, ledger.data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm-live", action="store_true")
    ap.add_argument("--pilot", choices=sorted(PILOTS), default="b4")
    ap.add_argument("--ledger", default=None)
    args = ap.parse_args()
    questions, caps, per_question, default_ledger = PILOTS[args.pilot]
    ledger = args.ledger or str(default_ledger)
    print(f"{args.pilot.upper()} plan: {len(questions)} questions · {per_question['llm'] + per_question['embedding']} "
          f"Mesh per question ({per_question['llm']} LLM + {per_question['embedding']} embedding) "
          f"· cap {caps['mesh']} Mesh / {caps['web']} web · ledger {ledger}")
    for q in questions:
        print(f"  {q['id']}: {q['question']} · sources {q['sources']} · follow-up {q['follow_up']} · web cap {q['web_cap']}")
    if not args.confirm_live:
        print("Dry run only (no calls made). Re-run with --confirm-live after approval.")
        return
    results, led = run_pilot(ledger, questions=questions, caps=caps, per_question=per_question)
    print(json.dumps({"ledger": {k: led[k] for k in ("mesh", "web")},
                      "questions": [{"id": r["id"], "skipped": r.get("skipped")} for r in results]}, indent=2))


if __name__ == "__main__":
    main()
