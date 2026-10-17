"""
Verified-research view (default workflow).

Runs graph_builder.research_graph and shows Report, Claims, Evidence and Audit tabs.
A finding is shown as verified only if its claim is `supported` by the deterministic
claim-level rules AND it passed the citation-integrity check.
"""

import json
import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from langgraph.errors import GraphRecursionError

from budget import BudgetConfigError, format_cost, format_usage, new_research_budget
from config.settings import ConfigError, get_settings
from graph_builder.research_graph import build_research_graph, prior_context_from, run_research
from agents.report_agent import FORECAST_NOTE, forecast_sentence
from verification.aggregation import explain_status, is_conflict
from verification.temporal import time_basis

# Source scopes the user can choose. Web is never added implicitly to a Documents-only run.
SOURCE_OPTIONS = {"Documents only": ["documents"], "Web only": ["web"], "Documents + Web": ["documents", "web"]}


def source_label(scope) -> str:
    return next(label for label, s in SOURCE_OPTIONS.items() if s == list(scope))


REPORT_STATUS = {
    "completed": ("success", "Verified findings: every finding below passed claim-level verification "
                             "and the citation-integrity check."),
    "partial": ("warning", "Run stopped early ({reason}). The findings shown passed verification, but the "
                           "research is incomplete; see Claims and Audit."),
    "no_verified_findings": ("warning", "No verified findings: no claim passed claim-level verification. "
                                        "Nothing below should be read as a verified conclusion."),
    "out_of_scope": ("info", "Not researched: the question is outside the selected domain's scope."),
}
CLAIM_BADGE = {"supported": "✅ Supported", "contradicted": "❌ Contradicted",
               "unresolved": "⚠️ Unresolved", "insufficient_evidence": "⏳ Insufficient evidence"}
STATUS_ORDER = ["supported", "contradicted", "unresolved", "insufficient_evidence"]


@st.cache_resource
def get_research_graph():
    return build_research_graph()


# ------------------------------------------------------------------ running
def execute(query, domain_id, scope, claims_to_check=None, prior_context=None):
    settings = get_settings()
    budget = new_research_budget(settings)   # follow-ups get a fresh budget with the same limits
    started = time.monotonic()
    out = run_research(query, budget, domain_id=domain_id, source_scope=scope,
                       claims_to_check=claims_to_check or None, graph=get_research_graph(),
                       prior_context=prior_context)
    return {"query": query, "domain_id": domain_id, "scope": list(scope), "fact_check": bool(claims_to_check),
            "follow_up_to": prior_context.get("question") if prior_context else None,
            "out": out, "events": list(budget.events), "limits": budget.limits,
            "usage": budget.snapshot(), "wall_s": round(time.monotonic() - started, 2),
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}


def report_markdown(result) -> str:
    """Downloadable report: the verified report plus an audit summary (no document passages)."""
    out, usage, limits = result["out"], result["usage"], result["limits"]
    lines = [out["report"].markdown, "", "## Audit summary",
             f"- Generated: {result['at']} · wall time {result['wall_s']} s",
             f"- Retry rounds: {usage.retry_rounds}/{limits.max_retry_rounds}",
             f"- Usage: {format_usage(usage, limits)}",
             f"- Citation-integrity issues: {len(out.get('citation_issues') or [])}",
             "- Workflow trace:"]
    lines += [f"  - {t['node']}: " + ", ".join(f"{k}={v}" for k, v in t.items() if k != "node")
              for t in out.get("trace") or []]
    rejected = [(cid, r) for cid, v in (out.get("verifications") or {}).items() for r in v.rejected]
    if rejected:
        lines += ["- Rejected evidence judgments:"]
        lines += [f"  - {cid} / {r.evidence_id} ({r.stance}, {r.source or r.url or 'unknown'}"
                  + (f" p. {r.page}" if r.page is not None else "") + f"): {r.reason} — proposed quote \"{r.quote}\""
                  for cid, r in rejected]
    return "\n".join(lines)


# --------------------------------------------------------------- rendering
def _loc(ev):
    loc = ev.locator
    return f"{loc.source} · page {loc.page}" if loc.page is not None else (loc.url or loc.source)


ORIGIN = {"new": "🆕 newly retrieved", "reused": "♻️ reused from the previous question"}
NODE_ROLE = {"intake": "Intake", "plan": "Planner", "research": "Researcher", "extract": "Claim extractor",
             "verify": "Verifier", "retry": "Targeted retry (researcher)", "synthesize": "Synthesizer",
             "cite": "Citation checker", "report": "Reporter"}


def _quality_caption(ev):
    q = ev.quality
    tier = f"source quality: {q.label} ({q.score:.2f}) — {q.reason}" if q else "source quality: n/a"
    meta = "".join(f" · {k}: {v}" for k, v in (("publisher", q.publisher), ("category", q.category),
                                                 ("provenance", q.provenance)) if q and v)
    return (f"{tier}{meta} · {ORIGIN.get(ev.origin, ev.origin)} · evidence {ev.evidence_id}"
            + (f" · chunk {ev.locator.chunk_id}" if ev.locator.chunk_id else "")
            + ((f" · published {ev.locator.published_at[:10]}" if ev.locator.published_at
                else " · publication date not reported") if ev.locator.source_type == "web" else "")
            + (f" · retrieved {ev.locator.retrieved_at}" if ev.locator.retrieved_at else ""))


def _evidence_card(ev, quote, label=None):
    st.markdown(f"**{label + ' ' if label else ''}{_loc(ev)}**")
    st.markdown(f"> “{quote}”")
    st.caption(_quality_caption(ev))


def render_status(report):
    kind, text = REPORT_STATUS.get(report.status, ("warning", report.status))
    text = text.format(reason=report.termination_reason or "budget or time limit")
    getattr(st, kind)(text)


def render_report_tab(result, key):
    out = result["out"]
    report = out["report"]
    render_status(report)
    if result.get("follow_up_to"):
        plan = out.get("plan")
        st.caption(f"↪ Follow-up to: {result['follow_up_to']}"
                   + (f" · interpreted as: {plan.resolved_question}" if plan and plan.resolved_question else ""))
    origin_note = {"model": "", "template": " _(verified claim text)_",
                   "fallback": " _(verified claim text — deterministic fallback)_"}
    claims = out.get("claims") or {}
    forecasts = [s for s in report.verified_sentences if forecast_sentence(s, claims)]
    facts = [s for s in report.verified_sentences if s not in forecasts]
    st.markdown("#### Verified findings")
    for s in facts:
        st.markdown(f"- {s.text} " + " ".join(f"`[{c}]`" for c in s.claim_ids) + origin_note.get(s.origin, ""))
    if not facts:
        st.markdown("_No verified findings._" if not forecasts else "_None besides the forecasts below._")
    if forecasts:
        st.markdown("#### Supported forecasts and projections")
        st.caption(FORECAST_NOTE.strip(" _()").capitalize())
        for s in forecasts:
            st.markdown(f"- {s.text} " + " ".join(f"`[{c}]`" for c in s.claim_ids) + origin_note.get(s.origin, ""))
    if report.coverage:
        st.markdown("#### Requested coverage")
        for line in report.coverage:
            st.markdown(f"- {line}")
    excluded = [u for u in report.unverified]
    if excluded:
        st.markdown(f"#### Excluded from the findings ({len(excluded)})")
        for u in excluded:
            conflict = " (conflicting evidence)" if u.get("conflict") else ""
            st.markdown(f"- **{u['claim_id']}** {CLAIM_BADGE.get(u['status'], u['status'])}{conflict}: {u['claim']}")
    if report.human_review:
        st.warning("🧑‍⚖️ Human review required for: " + ", ".join(report.human_review))
    for n in dict.fromkeys(out.get("notices") or []):
        st.caption(f"⚠️ {n}")
    st.download_button("⬇️ Download report (Markdown)", data=report_markdown(result),
                       file_name=f"deepdive-report-{key}.md", mime="text/markdown", key=f"dl-{key}")


def render_claims_tab(result):
    out = result["out"]
    claims, verifs = out.get("claims") or {}, out.get("verifications") or {}
    if not claims:
        st.info("No claims were extracted (see Audit for the reason).")
        return
    rows = []
    for cid, c in claims.items():
        v = verifs.get(cid)
        rows.append({"claim": cid, "status": v.status if v else "not verified", "text": c.text,
                     "type": c.claim_type, "time basis": time_basis(c), "risk": c.risk,
                     "sources": f"{v.independent_sources}/{v.required_sources}" if v else "",
                     "human review": "yes" if v and v.needs_human_review else "",
                     "rejected judgments": len(v.rejected) if v else 0})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for status in STATUS_ORDER:
        group = [cid for cid, v in verifs.items() if v.status == status]
        if not group:
            continue
        st.markdown(f"#### {CLAIM_BADGE[status]} ({len(group)})")
        for cid in group:
            v, c = verifs[cid], claims[cid]
            flag = " · 🧑‍⚖️ human review required" if v.needs_human_review else ""
            st.markdown(f"**{cid}** — {c.text}")
            st.caption(explain_status(v))
            st.caption(f"{CLAIM_BADGE[status]} · rule: {v.rule} · independent sources "
                       f"{v.independent_sources}/{v.required_sources} required{flag}")
            for r in v.rejected:
                st.caption(f"↳ rejected evidence {r.evidence_id} ({r.stance}): {r.reason}")


def render_evidence_tab(result):
    out = result["out"]
    evidence, report = out.get("evidence") or {}, out["report"]
    verifs, claims = out.get("verifications") or {}, out.get("claims") or {}
    st.caption("Source-quality tiers are provisional ranking signals from config/source_quality.yaml: "
               "Primary (1.00) official regulator, standards body or government domain · Strong secondary (0.85) "
               "peer-reviewed journal or scholarly publisher · General secondary (0.65) established news agency, "
               "reference work, preprint server or official project documentation · Unverified (0.35) anything "
               "not in the curated lists, including documents without registered metadata. They order evidence "
               "but never decide whether a claim is supported.")
    st.markdown("#### Supporting evidence for verified findings")
    if not report.citations:
        st.info("No supporting evidence was accepted for any claim.")
    for c in report.citations:
        ev = evidence.get(c.evidence_id)
        if ev:
            _evidence_card(ev, c.quote, label=f"[{c.marker}]")
            with st.expander("Full passage"):
                st.text(ev.text)

    conflicts = [cid for cid, v in verifs.items() if is_conflict(v)]
    if conflicts:
        st.markdown("#### ⚖️ Conflicting evidence (unresolved)")
        for cid in conflicts:
            v = verifs[cid]
            st.markdown(f"**{cid}** — {claims[cid].text}")
            st.warning(explain_status(v))
            left, right = st.columns(2)
            with left:
                st.markdown("**Supporting**")
                for eid in v.supporting_evidence_ids:
                    _evidence_card(evidence[eid], v.accepted_quotes.get(eid, ""))
            with right:
                st.markdown("**Contradicting**")
                for eid in v.contradicting_evidence_ids:
                    _evidence_card(evidence[eid], v.accepted_quotes.get(eid, ""))

    contradicted = [cid for cid, v in verifs.items() if v.status == "contradicted"]
    if contradicted:
        st.markdown("#### ❌ Contradicted claims")
        for cid in contradicted:
            v = verifs[cid]
            st.markdown(f"**{cid}** — {claims[cid].text}")
            st.caption(explain_status(v))
            for eid in v.contradicting_evidence_ids:
                _evidence_card(evidence[eid], v.accepted_quotes.get(eid, ""), label=f"contradicts {cid}:")


def render_audit_tab(result):
    out, usage, limits = result["out"], result["usage"], result["limits"]
    trace = out.get("trace") or []
    st.markdown("#### Workflow trace (agent hand-offs)")
    st.dataframe(pd.DataFrame([{"step": i + 1, "agent": t.get("agent") or NODE_ROLE.get(t["node"], t["node"]),
                                "node": t["node"],
                                "details": json.dumps({k: v for k, v in t.items() if k not in ("node", "agent")},
                                                      default=str)} for i, t in enumerate(trace)]),
                 hide_index=True, width="stretch")
    research = next((t for t in trace if t["node"] == "research"), None)
    first_verify = next((t for t in trace if t["node"] == "verify"), None)
    if research:
        st.markdown(f"- **Researcher → Claim extractor:** {research.get('evidence', 0)} evidence passages "
                    f"({research.get('new_evidence', research.get('evidence', 0))} new, "
                    f"{research.get('reused_evidence', 0)} reused)")
    if first_verify:
        st.markdown(f"- **Claim extractor → Verifier:** claims {', '.join(first_verify.get('claim_ids') or []) or 'none'}")
    retries = [t for t in trace if t["node"] == "retry"]
    st.markdown("#### Targeted retries")
    if not retries:
        st.caption("No retry was needed.")
    for r in retries:
        added = r.get("new_evidence_ids") or {}
        st.markdown(f"- Round {r.get('round')}: retried {', '.join(r.get('targeted_claims') or []) or 'none'}"
                    " · new evidence: " + ("; ".join(f"{c}: {', '.join(ids)}" for c, ids in added.items() if ids)
                                           or "none found"))
    final = next((t for t in trace if t["node"] == "report"), {})
    if final.get("final_statuses"):
        st.markdown("#### Final claim statuses")
        st.markdown(" · ".join(f"{cid}: {CLAIM_BADGE.get(s, s)}" for cid, s in final["final_statuses"].items()))
    st.markdown("#### Retries, usage and cost")
    st.markdown(f"- Retry rounds used: **{usage.retry_rounds}/{limits.max_retry_rounds}**")
    st.markdown(f"- {format_usage(usage, limits)}")
    st.markdown(f"- Cost: {format_cost(usage)}"
                + (" — " + "; ".join(usage.cost_incomplete_reasons) if usage.cost_incomplete_reasons else ""))
    st.markdown(f"- Wall time: {result['wall_s']} s · termination: {out.get('termination_reason') or 'none'}")
    calls = [e for e in result["events"] if e.get("provider") == "mesh"]
    if calls:
        st.dataframe(pd.DataFrame([{k: e.get(k) for k in ("operation", "model_requested", "model_served", "status",
                                                           "cached", "cache_marker", "latency_s", "input_tokens",
                                                           "output_tokens", "reasoning_tokens", "cost_usd", "cost_basis")}
                                   for e in calls]), hide_index=True, width="stretch")
    st.markdown("#### Rejected evidence judgments")
    evidence = out.get("evidence") or {}
    rej = [{"claim": cid, "evidence": r.evidence_id,
            "source": (f"{r.source} · page {r.page}" if r.page is not None else (r.url or r.source)) or "unknown",
            "stance": r.stance, "proposed quote": r.quote, "reason": r.reason}
           for cid, v in (out.get("verifications") or {}).items() for r in v.rejected]
    if rej:
        st.dataframe(pd.DataFrame(rej), hide_index=True, width="stretch")
    else:
        st.caption("None.")
    st.markdown("#### Citation check")
    issues = out.get("citation_issues") or []
    st.markdown(f"- Integrity issues: **{len(issues)}**")
    for i in issues:
        st.error(i)
    removed = out.get("removed_sentences") or []
    st.markdown(f"- Sentences removed: **{len(removed)}**")
    cite_step = next((t for t in out.get("trace") or [] if t["node"] == "cite"), {})
    if cite_step.get("fallback_sentences"):
        st.markdown(f"- Deterministic fallback: **{cite_step['fallback_sentences']}** finding(s) shown as verified "
                    f"claim text ({', '.join(cite_step.get('fallback_claims') or [])})")
    if removed:
        st.dataframe(pd.DataFrame(removed), hide_index=True, width="stretch")
    for n in dict.fromkeys(out.get("notices") or []):
        st.caption(f"⚠️ {n}")


def render_result(result, key):
    st.markdown(f"**Question:** {result['query']}  \n"
                f"<small>domain: {result['domain_id']} · sources: {', '.join(result['scope']) or 'none'}"
                f"{' · fact-check mode' if result['fact_check'] else ''} · {result['at']}</small>",
                unsafe_allow_html=True)
    tabs = st.tabs(["📄 Report", "🧾 Claims", "📚 Evidence", "🔍 Audit"])
    with tabs[0]:
        render_report_tab(result, key)
    with tabs[1]:
        render_claims_tab(result)
    with tabs[2]:
        render_evidence_tab(result)
    with tabs[3]:
        render_audit_tab(result)


def render_research(domain_id, scope, claims_to_check):
    history = st.session_state.setdefault("research_history", [])
    for i, r in enumerate(history[:-1]):
        with st.expander(f"Earlier question {i + 1}: {r['query']} — {r['out']['report'].status}"):
            render_status(r["out"]["report"])
            st.markdown(r["out"]["report"].markdown)
    if history:
        render_result(history[-1], key=len(history))
    if not scope:
        st.warning("Select at least one source in the sidebar.")
    follow_up = False
    if history:
        follow_up = st.toggle("Follow-up: reuse the previous research in this session", value=True,
                              help="Reuses the previous question, its verified findings and its accepted evidence "
                                   "(this browser session only). Everything is verified again.")
    if query := st.chat_input("Ask a research question…"):
        if not scope:
            return
        prior = prior_context_from(history[-1]["out"]) if (follow_up and history) else None
        try:
            with st.spinner("Planning, researching and verifying claims…"):
                result = execute(query, domain_id, scope, claims_to_check, prior_context=prior)
        except (ConfigError, BudgetConfigError) as e:
            st.error(f"Configuration error: {e}")
            return
        except GraphRecursionError:
            st.error("The run was stopped by the graph safety limit.")
            return
        history.append(result)
        rep = result["out"]["report"]
        logs = st.session_state.setdefault("system_logs", [])
        logs.append(f"Research run: status={rep.status} · claims={len(result['out'].get('claims') or {})} · "
                    f"verified sentences={len(rep.verified_sentences)}")
        logs.append(f"Usage: {format_usage(result['usage'], result['limits'])}")
        st.rerun()
