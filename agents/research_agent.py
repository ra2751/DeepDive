"""Research: collect evidence for each sub-question from documents (Milvus) and the web (Tavily).

Document search embeds all sub-questions in ONE batched Mesh request.
"""

from langchain_core.runnables import RunnableConfig

from agents.research_common import doc_evidence, step, web_domain_allowed, web_evidence
from budget import BudgetExceeded, budget_from_config
from config.settings import get_settings
from monitoring.logger import setup_logger
from retriever.retriever_pipeline import RetrieverPipeline
from retriever.web_search import get_web_search_provider

logger = setup_logger()
DOC_TOP_K = 3
WEB_RESULTS = 3


def make_retriever(budget, settings):
    return RetrieverPipeline(settings.milvus_collection, budget=budget, settings=settings)


def make_web(budget, settings):
    return get_web_search_provider(budget, settings)


def ResearchAgent(state, config: RunnableConfig):
    budget, settings = budget_from_config(config), get_settings()
    plan, domain, scope = state["plan"], state["domain"], state.get("source_scope") or []
    subs = plan.sub_questions
    evidence, notices, stop = {}, [], None

    # Follow-up: reuse previously ACCEPTED evidence from this session, only for the selected sources.
    allowed_types = {"documents": "document", "web": "web"}
    allowed = {allowed_types[s] for s in scope if s in allowed_types}
    prior = state.get("prior_context") or {}
    for eid, ev in (prior.get("evidence") or {}).items():
        if ev.locator.source_type in allowed:
            evidence[eid] = ev.model_copy(update={"origin": "reused"})
    reused = len(evidence)

    if "documents" in scope and subs:
        try:
            results = make_retriever(budget, settings).search_many([s.text for s in subs], top_k=DOC_TOP_K)
            for sq, hits in zip(subs, results):
                for h in hits:
                    ev = doc_evidence(h, sq.text, sq.id)
                    evidence.setdefault(ev.evidence_id, ev)
        except BudgetExceeded as e:
            stop = e.reason
        except Exception as e:  # Milvus unavailable, incompatible index, Mesh error
            logger.error(f"ResearchAgent | document search failed: {type(e).__name__}")
            notices.append(f"Document search unavailable: {type(e).__name__}")

    if "web" in scope and subs and not stop:
        web = None
        for sq in subs:
            try:
                web = web or make_web(budget, settings)
                kwargs = {}
                if domain.web_allowed_domains:
                    kwargs["include_domains"] = domain.web_allowed_domains
                if domain.web_blocked_domains:
                    kwargs["exclude_domains"] = domain.web_blocked_domains
                for r in web.search(sq.text, max_results=WEB_RESULTS, **kwargs):
                    if web_domain_allowed(r.url, domain):
                        ev = web_evidence(r, sq.text, sq.id)
                        evidence.setdefault(ev.evidence_id, ev)
            except BudgetExceeded as e:
                notices.append(f"Web search limit reached ({e.reason}); remaining sub-questions were not searched "
                               "on the web" + (" (documents only)" if "documents" in scope else ""))
                break
            except Exception as e:
                notices.append(f"Web search unavailable: {type(e).__name__}")
                break

    if not evidence and not stop and not prior:
        notices.append("No evidence was found in the selected sources ("
                       + (", ".join(scope) or "none selected") + "); nothing could be verified.")
    out = {"evidence": evidence, "notices": notices,
           "trace": [step("research", agent="Researcher", sub_questions=len(subs), evidence=len(evidence),
                          reused_evidence=reused, new_evidence=len(evidence) - reused,
                          documents=sum(1 for e in evidence.values() if e.locator.source_type == "document"),
                          web=sum(1 for e in evidence.values() if e.locator.source_type == "web"))]}
    if stop:
        out["termination_reason"] = stop
    return out
