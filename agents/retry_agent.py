"""Targeted retry: new evidence only for claims with insufficient evidence (no model call).

Queries are the claim texts themselves (documents: one batched embedding; web: one search per
claim while the web budget allows). Bounded by MAX_RETRY_ROUNDS via the BudgetManager.
"""

from langchain_core.runnables import RunnableConfig

from agents.research_agent import make_retriever, make_web
from agents.research_common import doc_evidence, step, web_domain_allowed, web_evidence
from budget import BudgetExceeded, budget_from_config
from config.settings import get_settings

MAX_RETRY_CLAIMS = 5
RETRY_DOC_TOP_K = 5


def RetryAgent(state, config: RunnableConfig):
    budget, settings = budget_from_config(config), get_settings()
    claims, evidence = state.get("claims") or {}, state.get("evidence") or {}
    verifications, shown_all = state.get("verifications") or {}, state.get("shown_evidence") or {}
    domain, scope = state["domain"], state.get("source_scope") or []
    try:
        round_no = budget.next_retry_round()
    except BudgetExceeded as e:
        return {"retry_found_new_evidence": False, "trace": [step("retry", result="no rounds left", reason=e.reason)]}

    targets = [c for c, v in verifications.items() if v.status == "insufficient_evidence"][:MAX_RETRY_CLAIMS]
    new_ev, shown, notices = {}, {}, []

    def add(cid, ev):
        if ev.evidence_id in shown_all.get(cid, []) or ev.evidence_id in shown.get(cid, []):
            return
        new_ev.setdefault(ev.evidence_id, evidence.get(ev.evidence_id, ev))
        shown.setdefault(cid, list(shown_all.get(cid, []))).append(ev.evidence_id)

    if "documents" in scope and targets:
        try:
            results = make_retriever(budget, settings).search_many([claims[c].text for c in targets],
                                                                    top_k=RETRY_DOC_TOP_K)
            for cid, hits in zip(targets, results):
                for h in hits:
                    add(cid, doc_evidence(h, f"retry{round_no}:{cid}"))
        except BudgetExceeded as e:
            notices.append(f"Retry document search stopped ({e.reason})")
        except Exception as e:
            notices.append(f"Retry document search unavailable: {type(e).__name__}")

    if "web" in scope and targets:
        try:
            web = make_web(budget, settings)
            for cid in targets:
                for r in web.search(claims[cid].text, max_results=3):
                    if web_domain_allowed(r.url, domain):
                        add(cid, web_evidence(r, f"retry{round_no}:{cid}"))
        except BudgetExceeded as e:
            notices.append(f"Retry web search stopped ({e.reason})")
        except Exception as e:
            notices.append(f"Retry web search unavailable: {type(e).__name__}")

    pending = [c for c in targets if c in shown]
    fresh = {c: [e for e in shown.get(c, []) if e in new_ev] for c in pending}
    return {"retry_round": round_no, "evidence": new_ev, "shown_evidence": shown, "pending_claim_ids": pending,
            "fresh_evidence": fresh,
            "retry_found_new_evidence": bool(pending), "notices": notices,
            "trace": [step("retry", agent="Targeted retry (researcher)", round=round_no, targeted_claims=targets, new_evidence=len(new_ev),
                           claims_with_new_evidence=pending,
                           new_evidence_ids={c: [e for e in shown.get(c, []) if e in new_ev] for c in pending})]}
