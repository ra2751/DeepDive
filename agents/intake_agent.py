"""Intake: load the domain configuration and apply deterministic scope checks (no model call).

Replaces the old hard-coded AI-only keyword guardrail: scope now comes from the
selected domain's configuration.
"""

from langchain_core.runnables import RunnableConfig

from agents.research_common import step
from domains.registry import DEFAULT_DOMAIN, get_domain

MAX_QUERY_CHARS = 2000


def IntakeAgent(state, config: RunnableConfig):
    query = (state.get("query") or "").strip()
    domain_id = state.get("domain_id") or DEFAULT_DOMAIN
    try:
        domain = get_domain(domain_id)
    except KeyError as e:
        return {"status": "out_of_scope", "termination_reason": "unknown_domain",
                "notices": [str(e)], "trace": [step("intake", result="unknown domain", domain_id=domain_id)]}
    reason = None
    if not query:
        reason = "empty question"
    elif len(query) > MAX_QUERY_CHARS:
        reason = f"question longer than {MAX_QUERY_CHARS} characters"
    else:
        hit = [t for t in domain.blocked_terms if t.lower() in query.lower()]
        if hit:
            reason = f"blocked by domain configuration: {hit}"
    scope = [s for s in (state.get("source_scope") or domain.preferred_sources) if s in domain.preferred_sources]
    prior, prior_note = state.get("prior_context"), []
    if prior and prior.get("domain_id") != domain.id:
        prior_note = [f"Previous research was in domain '{prior.get('domain_id')}'; it was not reused for "
                      f"'{domain.id}'."]
        prior = None
    if reason:
        return {"domain": domain, "source_scope": scope, "status": "out_of_scope", "notices": [reason],
                "trace": [step("intake", result="rejected", reason=reason, domain=domain.id)]}
    return {"domain": domain, "source_scope": scope, "status": "running", "retry_round": 0,
            "prior_context": prior, "notices": prior_note,
            "trace": [step("intake", agent="Intake", result="accepted", domain=domain.id,
                           domain_version=domain.version, source_scope=scope, follow_up=bool(prior))]}
