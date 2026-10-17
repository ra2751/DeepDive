"""Report: assemble the final research report (markdown + structured), with an honest status."""

from langchain_core.runnables import RunnableConfig

from agents.research_common import step
from budget import budget_from_config, format_usage
from schemas.research import ResearchReport
from verification.aggregation import explain_status, is_conflict
import re

from verification.temporal import is_forecast
from verification.text import content_stems, stem

LIMITATIONS = [
    "Claim statuses come from deterministic rules applied to verifier-model judgments: a quote must be verbatim "
    "in the cited passage, contain the claim's numbers, not negate it, and cover its key terms. Deeper semantic "
    "support still depends on the verifier model.",
    "Source-quality tiers are provisional ranking signals from configurable rules, not probabilities, and never "
    "prove that a source supports a claim.",
    "Findings are limited to the sources searched in this run; absence of evidence is reported as unresolved, "
    "not as false.",
    "Figures come only from the cited sources, as they stood when retrieved; nothing is updated to today. "
    "Findings marked forecast are projections reported by a source, not observed outcomes. Web publication "
    "dates are shown only when the search provider reports them.",
]
FORECAST_NOTE = " _(forecast: a projection reported by the cited source, not an observed outcome)_"


def forecast_sentence(sentence, claims) -> bool:
    return any(c in claims and is_forecast(claims[c].text, claims[c].claim_type) for c in sentence.claim_ids)


_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10}
_REQUEST = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})\s+(?:(?:biggest|largest|main|key|top|"
                      r"most important|major|primary|greatest|leading)\s+)?([a-z][a-z\-]*(?:\s+[a-z][a-z\-]*)?)"
                      r"(?=\s+(?:and|for|in|of|to|facing|through|by|from|across)\b|\s*[,.?;]|\s*$)")


def requested_items(question: str) -> list:
    """Counted categories the question asks for, e.g. [(3, "growth opportunities"), (3, "commercialization risks")]."""
    out = []
    for n, phrase in _REQUEST.findall((question or "").lower()):
        count = _COUNT_WORDS.get(n) or int(n)
        if phrase.split()[-1].endswith("s") and 1 < count <= 20:
            out.append((count, phrase))
    return out


def coverage_lines(question, claims, verifications, plan) -> list:
    """Deterministic statement of what was asked versus what was verified. Category membership is only
    counted when a verified claim itself uses the category word (e.g. "risk"); DeepDive never ranks
    findings or adds findings to reach a requested number."""
    supported = [c for c, v in verifications.items() if v.status == "supported"]
    lines = []
    for count, phrase in requested_items(question):
        head = stem(phrase.split()[-1])
        explicit = [c for c in supported if head in content_stems(claims[c].text)]
        lines.append(f"{phrase.capitalize()}: {count} requested; {len(explicit)} verified claim(s) explicitly describe "
                     f"one ({', '.join(explicit) or 'none'})" + (" — fewer than requested" if len(explicit) < count else ""))
    if lines:
        lines.append("Other verified findings are listed as stated by their sources; DeepDive does not verify rankings "
                     "such as \"biggest\", and does not add findings to reach a requested number.")
    for sq in (plan.sub_questions if plan else []):
        if not any(claims[c].sub_question_id == sq.id for c in supported):
            lines.append(f"No verified evidence found for sub-question {sq.id}: {sq.text}")
    return lines


def dates(ev) -> str:
    loc = ev.locator
    if loc.source_type != "web":
        return "document"
    published = loc.published_at[:10] if loc.published_at else "not reported"
    retrieved = loc.retrieved_at[:10] if loc.retrieved_at else "unknown"
    return f"published {published} · retrieved {retrieved}"


def _loc(ev):
    loc = ev.locator
    return f"{loc.source}, p. {loc.page}" if loc.page is not None else (loc.url or loc.source)


def _q(ev):
    return f"{ev.quality.label} ({ev.quality.score:.2f})" if ev.quality else "n/a"


def _side(eids, v, evidence):
    return [{"evidence_id": e, "source": _loc(evidence[e]), "quality": _q(evidence[e]),
             "origin": evidence[e].origin, "quote": v.accepted_quotes.get(e, "")} for e in eids if e in evidence]


def ReportAgent(state, config: RunnableConfig):
    budget = budget_from_config(config)
    domain = state.get("domain")
    claims, verifications = state.get("claims") or {}, state.get("verifications") or {}
    evidence = state.get("evidence") or {}
    sentences, citations = state.get("verified_sentences") or [], state.get("citations") or []
    reason, plan = state.get("termination_reason"), state.get("plan")
    prior = state.get("prior_context")

    unverified = []
    for cid, v in verifications.items():
        if v.status == "supported":
            continue
        entry = {"claim_id": cid, "claim": claims[cid].text, "status": v.status, "rule": v.rule,
                 "conflict": is_conflict(v), "explanation": explain_status(v),
                 "human_review": v.needs_human_review,
                 "rejected_judgments": [r.model_dump() for r in v.rejected]}
        if v.supporting_evidence_ids:
            entry["supporting"] = _side(v.supporting_evidence_ids, v, evidence)
        if v.contradicting_evidence_ids:
            entry["contradicting"] = _side(v.contradicting_evidence_ids, v, evidence)
        unverified.append(entry)
    review = [cid for cid, v in verifications.items() if v.needs_human_review]

    if state.get("status") == "out_of_scope":
        status = "out_of_scope"
    elif not sentences:
        status = "no_verified_findings"
    elif reason:
        status = "partial"
    else:
        status = "completed"

    md = ["# Research report", f"**Question:** {state.get('query', '')}"]
    if prior:
        md.append(f"**Follow-up to:** {prior.get('question', '')}")
    if plan and plan.resolved_question:
        md.append(f"**Interpreted as:** {plan.resolved_question}")
    md += [f"**Domain:** {domain.name if domain else state.get('domain_id')} · **Status:** {status}"
           + (f" · stopped: {reason}" if reason else ""), ""]
    if status == "out_of_scope":
        md += ["Not researched: " + "; ".join(state.get("notices") or ["out of scope"])]
    origin_note = {"model": "", "template": " _(verified claim text)_", "fallback": " _(verified claim text — fallback)_"}

    def line(s):
        return f"- {s.text} " + "".join(f"[{c}]" for c in s.claim_ids) + origin_note.get(s.origin, "")

    facts = [s for s in sentences if not forecast_sentence(s, claims)]
    forecasts = [s for s in sentences if forecast_sentence(s, claims)]
    md += ["## Verified findings"] + ([line(s) for s in facts] or
                                      ["No claim could be verified." if not forecasts else "None besides the forecasts below."])
    if forecasts:
        md += ["", "## Supported forecasts and projections",
               "_Projections reported by the cited sources, not observed outcomes._"] + [line(s) for s in forecasts]
    coverage = [] if status == "out_of_scope" else coverage_lines(state.get("query", ""), claims, verifications, plan)
    if coverage:
        md += ["", "## Requested coverage"] + [f"- {x}" for x in coverage]
    if citations:
        md += ["", "## Evidence", "| Claim | Status | Source | Source quality | Origin | Quote | Dates |",
               "|---|---|---|---|---|---|---|"]
        for c in citations:
            ev = evidence.get(c.evidence_id)
            md.append(f"| {c.marker} | supported | {c.source}{', p. ' + str(c.page) if c.page is not None else ''} | "
                      f"{_q(ev) if ev else 'n/a'} | {ev.origin if ev else ''} | \"{c.quote}\" | {dates(ev) if ev else ''} |")
    groups = [("Contradicted claims (excluded from the findings)", [u for u in unverified if u["status"] == "contradicted"]),
              ("Unresolved claims (excluded from the findings)", [u for u in unverified if u["status"] != "contradicted"])]
    for title, group in groups:
        if not group:
            continue
        md += ["", f"## {title}"]
        for u in group:
            md.append(f"- **{u['claim_id']} — {u['status']}{' (conflicting evidence)' if u['conflict'] else ''}:** "
                      f"{u['claim']}")
            md.append(f"  - {u['explanation']}")
            for side in ("supporting", "contradicting"):
                for e in u.get(side, []):
                    md.append(f"  - {side}: {e['source']} · {e['quality']} · \"{e['quote']}\" ({e['evidence_id']})")
            for r in u.get("rejected_judgments", []):
                where = f"{r.get('source')}, p. {r.get('page')}" if r.get("page") is not None else (r.get("url") or r.get("source") or "unknown source")
                md.append(f"  - rejected {r['stance']} quote from {where}: \"{r.get('quote', '')}\" — {r['reason']}")
    if review:
        md += ["", "## Human review required", "- " + ", ".join(review) + " (high-risk claim types)"]
    removed = state.get("removed_sentences") or []
    if removed:
        md += ["", "## Removed by citation check"] + [f"- \"{r['text']}\" ({r['reason']})" for r in removed]
    notices = list(dict.fromkeys(state.get("notices") or []))
    if notices:
        md += ["", "## Notices"] + [f"- {n}" for n in notices]
    md += ["", "## Verification limitations"] + [f"- {x}" for x in LIMITATIONS]
    usage = budget.snapshot()
    md += ["", "---", f"*{format_usage(usage, budget.limits)}*"]
    if domain and domain.report_disclaimer:
        md.append(f"*{domain.report_disclaimer}*")

    report = ResearchReport(question=state.get("query", ""), domain_id=domain.id if domain else state.get("domain_id", ""),
                            status=status, verified_sentences=sentences, citations=citations, unverified=unverified,
                            human_review=review, removed_sentences=removed, coverage=coverage, markdown="\n".join(md),
                            termination_reason=reason)
    return {"report": report, "status": status, "usage": usage.model_dump(),
            "trace": [step("report", agent="Reporter", status=status, verified_sentences=len(sentences),
                           unverified=len(unverified), human_review=len(review), termination_reason=reason,
                           final_statuses={cid: v.status for cid, v in verifications.items()})]}
