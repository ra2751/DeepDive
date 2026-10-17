"""
Live evaluation runner (prepared; requires explicit approval to run).

    python -m evals.run_live_eval                       # dry run: prints the plan, makes NO calls
    python -m evals.run_live_eval --confirm-live        # live run under a hard request cap

Every case runs through the real research graph with its own BudgetManager; a run-level
cap (default 90 Mesh requests, 2 web searches) is enforced before each case. Results are
written in the evals.result_schema format.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from budget import BudgetLimits, BudgetManager, PriceTable  # noqa: E402
from config.settings import get_settings  # noqa: E402
from evals.result_schema import CaseResult, CaseUsage, EvalRun, Metric  # noqa: E402
from graph_builder.research_graph import build_research_graph, run_research  # noqa: E402
from verification.citation_integrity import validate_citations  # noqa: E402

DEFAULT_DATASET = ROOT / "evals" / "datasets" / "live_eval_v1.jsonl"
# Strict per-case shares (LLM calls, embedding calls). A case can never exceed llm + emb Mesh requests,
# so the run-level cap holds even in the worst case (retries and failed attempts included).
PER_CASE_SHARE = {"fact_check": (3, 4), "research": (12, 4)}
PER_CASE_MESH = {m: llm + emb for m, (llm, emb) in PER_CASE_SHARE.items()}


def _ledger_load(path):
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else {"mesh": 0, "web": 0, "cases": []}


def _ledger_save(path, led):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(led, indent=2))


def code_fingerprint():
    h = hashlib.sha256()
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_file() and rel.parts[0] not in {".venv", ".git", "data", "docs", "logs"} and "__pycache__" not in rel.parts \
                and p.suffix in {".py", ".yaml", ".yml", ".jsonl", ".txt", ".ini"}:
            h.update(str(rel).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def load_cases(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def case_budget(mode, settings, remaining_mesh, remaining_web):
    llm, emb = PER_CASE_SHARE[mode]
    if llm + emb > remaining_mesh:          # callers skip such cases; never hand out more than remains
        llm, emb = 0, 0
    limits = BudgetLimits.from_settings(settings).model_copy(update={
        "max_llm_calls": min(llm, settings.max_llm_calls or llm),
        "max_embedding_calls": min(emb, settings.max_embedding_calls or emb),
        "max_web_searches": remaining_web})
    return BudgetManager(limits, pricing=PriceTable.load(settings.mesh_pricing_file), name="eval")


def score_case(case, out):
    report, verifs = out["report"], out.get("verifications") or {}
    exp = case["expected"]
    observed, ok = {"report_status": report.status}, True
    for cid, accepted in exp.items():
        if cid.startswith("C"):
            got = verifs[cid].status if cid in verifs else "missing"
            observed[cid] = got
            ok &= got in accepted
    if "report_status" in exp:
        ok &= report.status in exp["report_status"]
    if exp.get("min_supported_claims"):
        n = sum(1 for v in verifs.values() if v.status == "supported")
        observed["supported_claims"] = n
        ok &= n >= exp["min_supported_claims"]
    if exp.get("citations_include"):
        srcs = {c.source for c in report.citations}
        observed["citation_sources"] = sorted(srcs)
        ok &= exp["citations_include"] in srcs
    if exp.get("not_out_of_scope"):
        ok &= report.status != "out_of_scope"
    return observed, bool(ok)


def run(dataset=DEFAULT_DATASET, max_mesh=90, max_web=2, out_dir=ROOT / "evals" / "results",
        kind="live", graph=None, case_ids=None, ledger=None):
    settings = get_settings()
    cases = [c for c in load_cases(dataset) if not case_ids or c["case_id"] in case_ids]
    graph = graph or build_research_graph()
    led = _ledger_load(ledger)
    used_mesh, used_web = led["mesh"], led["web"]          # persistent global ledger (survives restarts)
    results, served, notes, details = [], set(), [], {}
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                    text=True, check=True).stdout.strip())
    except Exception:
        commit, dirty = None, None
    for case in cases:
        need = PER_CASE_MESH[case["mode"]]
        if case["case_id"] in led["cases"]:
            notes.append(f"{case['case_id']} skipped: already executed (ledger); not re-run")
            continue
        if used_mesh + need > max_mesh:
            notes.append(f"{case['case_id']} skipped: run-level Mesh cap {max_mesh} would be exceeded")
            continue
        b = case_budget(case["mode"], settings, max_mesh - used_mesh, max_web - used_web)
        led["cases"].append(case["case_id"])
        err = None
        try:
            out = run_research(case.get("question") or " / ".join(case.get("claims", [])), b,
                               domain_id=case["domain"], source_scope=case.get("source_scope", ["documents"]),
                               claims_to_check=case.get("claims"), graph=graph)
        except Exception as e:  # recorded, never hidden
            out, err = None, f"{type(e).__name__}: {e}"
        snap = b.snapshot()
        used_mesh += snap.mesh_llm_calls + snap.mesh_embedding_calls
        used_web += snap.web_searches
        led["mesh"], led["web"] = used_mesh, used_web
        _ledger_save(ledger, led)
        details[case["case_id"]] = case_details(out, b, err)
        served |= {e["model_served"] for e in b.events if e.get("provider") == "mesh" and e.get("model_served")}
        usage = CaseUsage(mesh_llm_calls=snap.mesh_llm_calls, mesh_embedding_calls=snap.mesh_embedding_calls,
                          web_searches=snap.web_searches, input_tokens=snap.mesh_input_tokens,
                          output_tokens=snap.mesh_output_tokens, cost_usd=snap.mesh_cost_usd,
                          cost_basis=snap.cost_basis, cost_incomplete=snap.cost_estimate_incomplete,
                          elapsed_s=snap.elapsed_s)
        if out is None:
            results.append(CaseResult(case_id=case["case_id"], mode=case["mode"], domain=case["domain"],
                                      expected=case["expected"], observed={}, passed=False, report_status="error",
                                      verified_sentences=0, unsupported_claim_leakage=0,
                                      citation_integrity_issues=0, usage=usage, error=err))
            continue
        report, verifs = out["report"], out.get("verifications") or {}
        leakage = sum(1 for s in report.verified_sentences for c in s.claim_ids
                      if c not in verifs or verifs[c].status != "supported")
        issues = len(validate_citations(report.citations, verifs, out.get("evidence") or {}))
        observed, ok = score_case(case, out)
        results.append(CaseResult(case_id=case["case_id"], mode=case["mode"], domain=case["domain"],
                                  expected=case["expected"], observed=observed, passed=ok,
                                  report_status=report.status, verified_sentences=len(report.verified_sentences),
                                  unsupported_claim_leakage=leakage, citation_integrity_issues=issues,
                                  termination_reason=out.get("termination_reason"), usage=usage))

    def rate(n, d):
        return Metric(numerator=n, denominator=d, rate=round(n / d, 4) if d else None)

    claim_rows = [(r.observed.get(k), v) for r in results for k, v in r.expected.items() if k.startswith("C")]
    totals = CaseUsage(
        mesh_llm_calls=sum(r.usage.mesh_llm_calls for r in results),
        mesh_embedding_calls=sum(r.usage.mesh_embedding_calls for r in results),
        web_searches=sum(r.usage.web_searches for r in results),
        input_tokens=None if any(r.usage.input_tokens is None for r in results) else sum(r.usage.input_tokens for r in results),
        output_tokens=None if any(r.usage.output_tokens is None for r in results) else sum(r.usage.output_tokens for r in results),
        cost_usd=None if any(r.usage.cost_usd is None for r in results) else round(sum(r.usage.cost_usd for r in results), 6),
        cost_basis="unknown" if any(r.usage.cost_basis == "unknown" for r in results) else (
            "estimated" if any(r.usage.cost_basis == "estimated" for r in results) else "reported"),
        cost_incomplete=any(r.usage.cost_incomplete for r in results),
        elapsed_s=round(sum(r.usage.elapsed_s for r in results), 2))
    now = datetime.now(timezone.utc)
    run_obj = EvalRun(
        run_id=f"{now.strftime('%Y%m%dT%H%M%SZ')}_{commit or 'nocommit'}", kind=kind, created_at=now.isoformat(),
        git_commit=commit, git_dirty=dirty, code_sha256=code_fingerprint(), dataset=str(Path(dataset).relative_to(ROOT)) if str(dataset).startswith(str(ROOT)) else str(dataset),
        dataset_sha256=hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        models={"supervisor": settings.mesh_model_supervisor, "generator": settings.mesh_model_generator,
                "verifier": settings.mesh_model_verifier, "embeddings": settings.mesh_embedding_model},
        models_served=sorted(served), limits={"max_mesh_requests": max_mesh, "max_web_searches": max_web},
        cases=results, notes=notes,
        metrics={"case_pass": rate(sum(r.passed for r in results), len(results)),
                 "claim_status_agreement": rate(sum(1 for got, acc in claim_rows if got in acc), len(claim_rows)),
                 "unsupported_claim_leakage": rate(sum(r.unsupported_claim_leakage for r in results),
                                                   sum(r.verified_sentences for r in results)),
                 "citation_integrity_issues": rate(sum(r.citation_integrity_issues for r in results), len(results))},
        totals=totals)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_obj.run_id}.json"
    path.write_text(run_obj.model_dump_json(indent=2))
    md = [f"# DeepDive evaluation {run_obj.run_id} ({kind})", "",
          f"Dataset `{run_obj.dataset}` (sha256 {run_obj.dataset_sha256[:12]}…) · commit {commit}", "",
          "| Case | Mode | Passed | Report status | Observed | Mesh calls | Cost |", "|---|---|---|---|---|---|---|"]
    for r in results:
        cost = "unknown" if r.usage.cost_usd is None else f"≈${r.usage.cost_usd:.5f} ({r.usage.cost_basis}{', incomplete' if r.usage.cost_incomplete else ''})"
        md.append(f"| {r.case_id} | {r.mode} | {'yes' if r.passed else 'NO'} | {r.report_status} | "
                  f"{json.dumps(r.observed)} | {r.usage.mesh_llm_calls}+{r.usage.mesh_embedding_calls} | {cost} |")
    md += ["", "| Metric | Value |", "|---|---|"]
    md += [f"| {k} | {m.numerator}/{m.denominator}" + (f" ({m.rate})" if m.rate is not None else "") + " |"
           for k, m in run_obj.metrics.items()]
    md += notes and ["", "Notes:"] + [f"- {n}" for n in notes] or []
    (out_dir / f"{run_obj.run_id}.md").write_text("\n".join(md))
    (out_dir / f"{run_obj.run_id}_details.json").write_text(json.dumps(details, indent=2, default=str))
    return run_obj, path


def case_details(out, budget, err):
    """Per-case audit detail: claims, verifications, rejections, citations, trace and per-call usage.
    Contains accepted quotes only (no full evidence passages)."""
    if out is None:
        return {"error": err, "events": budget.events}
    verifs, claims, evidence = out.get("verifications") or {}, out.get("claims") or {}, out.get("evidence") or {}
    return {
        "claims": {cid: {"text": c.text, "type": c.claim_type, "risk": c.risk,
                         "candidate_evidence": c.candidate_evidence_ids} for cid, c in claims.items()},
        "verifications": {cid: v.model_dump() for cid, v in verifs.items()},
        "evidence_locators": {e: {"source": ev.locator.source, "page": ev.locator.page, "chunk_id": ev.locator.chunk_id,
                                  "origin": ev.origin, "quality": ev.quality.tier if ev.quality else None,
                                  "retrieved_via": ev.retrieved_via} for e, ev in evidence.items()},
        "shown_evidence": out.get("shown_evidence"),
        "citations": [c.model_dump() for c in out["report"].citations],
        "verified_sentences": [s.model_dump() for s in out["report"].verified_sentences],
        "removed_sentences": out.get("removed_sentences"), "citation_issues": out.get("citation_issues"),
        "trace": out.get("trace"), "notices": out.get("notices"), "events": budget.events,
        "plan": out["plan"].model_dump() if out.get("plan") else None, "report_markdown": out["report"].markdown}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--max-mesh", type=int, default=90)
    ap.add_argument("--max-web", type=int, default=2)
    ap.add_argument("--confirm-live", action="store_true", help="required to make live Mesh/Milvus calls")
    ap.add_argument("--cases", default="", help="comma-separated case IDs (default: all)")
    ap.add_argument("--ledger", default=None, help="persistent global ledger file (caps across runs)")
    args = ap.parse_args()
    ids = [c.strip() for c in args.cases.split(",") if c.strip()] or None
    cases = [c for c in load_cases(args.dataset) if not ids or c["case_id"] in ids]
    worst = sum(PER_CASE_MESH[c["mode"]] for c in cases)
    print(f"{len(cases)} cases · worst-case {worst} Mesh requests · cap {args.max_mesh} Mesh / {args.max_web} web")
    if not args.confirm_live:
        print("Dry run only (no calls made). Re-run with --confirm-live after approval.")
        return
    run_obj, path = run(args.dataset, args.max_mesh, args.max_web, case_ids=ids, ledger=args.ledger)
    print(f"Wrote {path}")
    for k, m in run_obj.metrics.items():
        print(f"  {k}: {m.numerator}/{m.denominator}")


if __name__ == "__main__":
    main()
