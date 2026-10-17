"""
Controlled Mesh API capability probe (approval gates G1 and G2).

  G1  python scripts/mesh_probe.py --catalog
      GET /v1/models only (metadata, no inference).

  G2  (authorized totals: 3 chat, 2 embedding, 1 read-only Milvus query; no retries;
       enforced across runs by data/probe/g2_ledger.json)
      python scripts/mesh_probe.py --g2-structured openai/gpt-4o-mini,google/gemini-2.5-flash,anthropic/claude-haiku-4.5
      python scripts/mesh_probe.py --g2-milvus-sample      # 1 query; prints text for review; sends nothing to Mesh
      python scripts/mesh_probe.py --g2-embed --with-compat  # 1 synthetic + 1 sample embedding

The API key is never printed. Results are written to data/probe/ (git-ignored).
"""

import argparse
import json
import math
import os
import sys
from typing import Literal
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Keep LangSmith tracing disabled for probes, whatever .env contains.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from pydantic import BaseModel  # noqa: E402

from budget import BudgetExceeded, BudgetLimits, BudgetManager  # noqa: E402
from config.settings import get_settings  # noqa: E402

OUT_DIR = Path("data/probe")


def _write(name, payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {path}")


def catalog():
    from gateway import list_models

    models = list_models()
    rows = []
    for m in models:
        if not isinstance(m, dict):
            continue
        rows.append({
            "id": m.get("id"),
            "input_modalities": m.get("input_modalities") or m.get("architecture", {}).get("input_modalities"),
            "supports_structured_output": m.get("supports_structured_output"),
            "pricing_fields": sorted((m.get("pricing") or {}).keys()) if isinstance(m.get("pricing"), dict) else None,
        })
    print(f"{len(rows)} models listed by Mesh")
    for r in rows:
        print(f"  {r['id']:<55} structured={r['supports_structured_output']} inputs={r['input_modalities']}")
    _write("catalog", {"models": rows, "raw_count": len(models)})


# ---------------------------------------------------------------- G2 probe
# Authorized totals for G2 (enforced across runs by a persistent ledger).
# G2 approval: 3 chat, 2 embedding, 1 Milvus query. G2-completion approval (2026-10-16):
# +1 read-only Milvus query; the remaining 2nd embedding covers the compatibility check.
G2_CAPS = {"chat": 3, "embedding": 2, "milvus_query": 2}
LEDGER = OUT_DIR / "g2_ledger.json"

G2_EVIDENCE = "The Eiffel Tower was completed in 1889 for the Exposition Universelle in Paris."
G2_CLAIM = "The Eiffel Tower was completed in 1889."
G2_PROMPT = (
    "You are verifying a research claim against evidence.\n"
    f"Evidence E1: \"{G2_EVIDENCE}\"\n"
    f"Claim C1: \"{G2_CLAIM}\"\n"
    "Return claim_id 'C1', a stance (supports, contradicts or insufficient_evidence), "
    "a supporting_quote copied verbatim from E1, and one sentence of reasoning."
)


class ClaimCheck(BaseModel):
    claim_id: str
    stance: Literal["supports", "contradicts", "insufficient_evidence"]
    supporting_quote: str
    reasoning: str


def _ledger():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"chat": 0, "embedding": 0, "milvus_query": 0}


def _spend(kind, n=1):
    led = _ledger()
    if led[kind] + n > G2_CAPS[kind]:
        raise SystemExit(f"G2 cap reached for {kind}: {led[kind]}/{G2_CAPS[kind]} already used; refusing.")
    led[kind] += n
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led))


def _no_retry_settings():
    s = get_settings()
    return s.model_copy(update={"mesh_max_attempts": 1})  # gateway loop: 1 attempt; clients: max_retries=0


def g2_structured(models):
    """Structured-output probe. The report is written after EVERY model, so token usage
    and cost metadata survive even if a later step fails."""
    from budget import PriceTable
    from gateway import structured
    from gateway.errors import MeshError

    settings = _no_retry_settings()
    prices = PriceTable.load(settings.mesh_pricing_file)
    remaining = G2_CAPS["chat"] - _ledger()["chat"]
    budget = BudgetManager(BudgetLimits(max_llm_calls=max(remaining, 0), max_embedding_calls=0,
                                        max_web_searches=0, max_tool_calls=0, max_retry_rounds=0,
                                        max_run_seconds=300, max_mesh_cost_usd=None),
                           pricing=prices, name="g2-structured")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = OUT_DIR / f"g2_structured_{stamp}.json"
    report = {"test": "structured", "prompt_is_synthetic": True, "results": []}

    def save():
        report["ledger"] = _ledger()
        report["budget"] = budget.snapshot().model_dump()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, default=str))

    for model in models:
        s = settings.model_copy(update={"mesh_model_verifier": model})
        before = len(budget.events)
        entry = {"model_requested": model}
        try:
            out = structured("verifier", ClaimCheck, budget, s).invoke(G2_PROMPT)
            entry.update(schema_valid=True, parsed=out.model_dump(),
                         quote_verbatim=out.supporting_quote.strip().strip('"') in G2_EVIDENCE,
                         stance_correct=out.stance == "supports")
        except BudgetExceeded as e:
            entry.update(schema_valid=None, skipped=f"chat cap reached ({e.reason})")
        except MeshError as e:
            entry.update(schema_valid=False, error=f"{type(e).__name__}: {e}")
        attempts = budget.events[before:]
        if attempts:
            _spend("chat", len(attempts))
        entry["attempts"] = [
            {k: a.get(k) for k in ("status", "model_served", "latency_s", "input_tokens", "output_tokens",
                                   "cached_input_tokens", "reasoning_tokens", "cost_usd", "cost_basis",
                                   "priced_as", "cost_incomplete_reasons", "request_id", "error")}
            for a in attempts
        ]
        entry["method"] = (["json_schema", "function_calling"][len(attempts) - 1]
                           if entry.get("schema_valid") and 1 <= len(attempts) <= 2 else None)
        report["results"].append(entry)
        save()
        print(json.dumps(entry, default=str))
    save()
    print(f"\nWrote {report_path}")


_SENSITIVE_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "phone": r"\+?\d[\d\s().-]{8,}\d",
    "card_or_account_number": r"\b(?:\d[ -]?){13,19}\b",
    "national_id_like": r"\b\d{3}-\d{2}-\d{4}\b",
    "secret_like": r"(?i)(api[_-]?key|secret|password|token)\s*[:=]",
    "confidentiality_marker": r"(?i)\b(confidential|proprietary|internal use only|do not distribute|privileged)\b",
}


def _sensitivity_scan(text):
    import re

    hits = {name: len(re.findall(pat, text)) for name, pat in _SENSITIVE_PATTERNS.items()}
    return {"hits": {k: v for k, v in hits.items() if v}, "clean": not any(hits.values())}


def g2_milvus_sample(min_chars=200):
    """ONE read-only query; picks a suitable chunk; nothing is sent to Mesh here."""
    from pymilvus import MilvusClient

    settings = get_settings()
    settings.require("milvus_endpoint", "milvus_api_key")
    _spend("milvus_query")
    try:
        client = MilvusClient(uri=settings.milvus_endpoint, token=settings.milvus_api_key)
        exists = client.has_collection(settings.milvus_collection)  # metadata check, not a data query
        print(json.dumps({"connected": True, "collection": settings.milvus_collection, "exists": exists}))
        if not exists:
            _write("g2_milvus_sample_meta", {"result": "inconclusive: collection not found"})
            return
        rows = client.query(collection_name=settings.milvus_collection, filter="id >= 0",
                            output_fields=["text", "source", "page", "embedding"], limit=20)  # the one read-only query
    except Exception as e:  # report without hostnames or credentials
        msg = {"connected": False, "error_type": type(e).__name__, "code": getattr(e, "code", None)}
        print(json.dumps(msg))
        _write("g2_milvus_sample_meta", {"result": "inconclusive", **msg})
        return
    suitable = [r for r in rows if isinstance(r.get("text"), str) and len(r["text"].strip()) >= min_chars
                and isinstance(r.get("embedding"), list) and len(r["embedding"]) > 0]
    info = {"rows_returned": len(rows), "suitable_rows": len(suitable)}
    if not suitable:
        info["result"] = "inconclusive: no suitable sample"
        print(json.dumps(info))
        _write("g2_milvus_sample_meta", info)
        return
    r = suitable[0]
    sample = {"id": r.get("id"), "source": r.get("source"), "page": r.get("page"), "text": r["text"],
              "stored_dim": len(r["embedding"]), "embedding": r["embedding"]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "g2_milvus_sample.json").write_text(json.dumps(sample))
    info.update(id=sample["id"], source=sample["source"], page=sample["page"],
                text_chars=len(sample["text"]), stored_dim=sample["stored_dim"])
    info["sensitivity_scan"] = _sensitivity_scan(sample["text"])
    print(json.dumps(info))
    _write("g2_milvus_sample_meta", info)


def g2_embed(include_compat, compat_only=False):
    from budget import PriceTable
    from gateway import embeddings
    from gateway.errors import MeshError

    settings = _no_retry_settings()
    n = 1 if compat_only else (2 if include_compat else 1)
    led = _ledger()
    if led["embedding"] + n > G2_CAPS["embedding"]:
        raise SystemExit("G2 embedding cap would be exceeded; refusing.")
    budget = BudgetManager(BudgetLimits(max_llm_calls=0, max_embedding_calls=n, max_web_searches=0,
                                        max_tool_calls=0, max_retry_rounds=0, max_run_seconds=120,
                                        max_mesh_cost_usd=None),
                           pricing=PriceTable.load(settings.mesh_pricing_file), name="g2-embed")
    emb = embeddings(budget, settings)
    report = {"test": "embedding", "model": settings.mesh_embedding_model}
    if not compat_only:
        try:
            _spend("embedding")
            vec = emb.embed_query("DeepDive embedding dimension probe (synthetic text).")
            report["dimension"] = {"dim": len(vec), "passes": len(vec) == 1536}
        except MeshError as e:  # dimension mismatch also lands here
            report["dimension"] = {"error": f"{type(e).__name__}: {e}", "passes": False}
        print(json.dumps(report["dimension"]))
    report["usage_metadata"] = "missing: LangChain OpenAIEmbeddings does not expose Mesh token usage"
    if include_compat or compat_only:
        sample = json.loads((OUT_DIR / "g2_milvus_sample.json").read_text())
        try:
            _spend("embedding")
            fresh = emb.embed_query(sample["text"])
            stored = sample["embedding"]
            dot = sum(a * b for a, b in zip(stored, fresh))
            cos = dot / (math.sqrt(sum(a * a for a in stored)) * math.sqrt(sum(b * b for b in fresh)))
            report["milvus_compat"] = {"sample_id": sample["id"], "source": sample["source"],
                                       "page": sample["page"], "stored_dim": sample["stored_dim"],
                                       "fresh_dim": len(fresh), "cosine": round(cos, 6),
                                       "threshold": 0.99, "result": "pass" if cos >= 0.99 else "fail"}
        except MeshError as e:
            report["milvus_compat"] = {"result": "inconclusive", "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(report["milvus_compat"]))
    report["ledger"] = _ledger()
    report["budget"] = budget.snapshot().model_dump()
    _write("g2_embed", report)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", action="store_true")
    ap.add_argument("--g2-structured", metavar="MODELS", help="comma-separated Mesh model IDs, in order")
    ap.add_argument("--g2-milvus-sample", action="store_true")
    ap.add_argument("--g2-embed", action="store_true")
    ap.add_argument("--with-compat", action="store_true", help="with --g2-embed: also embed the saved Milvus sample")
    ap.add_argument("--compat-only", action="store_true", help="with --g2-embed: only embed the saved Milvus sample")
    args = ap.parse_args()
    if args.catalog:
        catalog()
    elif args.g2_structured:
        g2_structured([m.strip() for m in args.g2_structured.split(",") if m.strip()])
    elif args.g2_milvus_sample:
        g2_milvus_sample()
    elif args.g2_embed:
        g2_embed(args.with_compat, args.compat_only)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
