"""
Source Quality Evaluator: deterministic, configurable source tiers (config/source_quality.yaml).

The score is a provisional ranking signal only. It orders evidence for the verifier and for
presentation; claim status is decided solely by verification.aggregation and is never affected
by source quality. Unknown sources are labelled "unverified" and missing metadata is never invented.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel

CONFIG = Path(__file__).resolve().parents[1] / "config" / "source_quality.yaml"


class SourceQuality(BaseModel):
    tier: str
    label: str
    score: float
    reason: str
    publisher: Optional[str] = None
    category: Optional[str] = None
    provenance: Optional[str] = None


@lru_cache(maxsize=4)
def load_rules(path: str = str(CONFIG)) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    tiers = data["tiers"]
    for rule in data.get("web_rules", []):
        if rule["tier"] not in tiers:
            raise ValueError(f"unknown tier {rule['tier']!r} in web_rules")
    for cat, tier in (data.get("document_category_tiers") or {}).items():
        if tier not in tiers:
            raise ValueError(f"unknown tier {tier!r} for document category {cat!r}")
    return data


def _q(rules, tier, reason, **meta) -> SourceQuality:
    t = rules["tiers"][tier]
    return SourceQuality(tier=tier, label=t["label"], score=float(t["score"]), reason=reason, **meta)


def host_of(url: str) -> str:
    host = urlparse(url if "://" in url else "https://" + url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def evaluate(source_type: str, source: str, url: Optional[str] = None, rules: Optional[dict] = None) -> SourceQuality:
    rules = rules or load_rules()
    if source_type == "web":
        host = host_of(url or source)
        web_rules = rules.get("web_rules", [])
        # Precedence (most specific first): exact host > longest listed parent domain > broad suffix.
        # e.g. pubmed.ncbi.nlm.nih.gov (listed) beats nih.gov (parent) and ".gov" (suffix).
        for rule in web_rules:
            if host in set(rule.get("hosts") or []):
                return _q(rules, rule["tier"], f"{rule['reason']} ({host})")
        parents = [(len(h), rule, h) for rule in web_rules for h in rule.get("hosts") or [] if host.endswith("." + h)]
        if parents:
            _, rule, h = max(parents, key=lambda x: x[0])
            return _q(rules, rule["tier"], f"{rule['reason']} ({host}, under {h})")
        for rule in web_rules:
            if any(host.endswith(sfx) for sfx in rule.get("host_suffixes") or []):
                return _q(rules, rule["tier"], f"{rule['reason']} ({host})")
        return _q(rules, "unverified", f"web source {host or 'unknown'} is not in a curated quality list")
    entry = (rules.get("document_registry") or {}).get(source)
    if entry:
        category = entry.get("category")
        tier = (rules.get("document_category_tiers") or {}).get(category)
        if tier:
            return _q(rules, tier, f"registered document, category '{category}'",
                      publisher=entry.get("publisher"), category=category, provenance=entry.get("provenance"))
        return _q(rules, "unverified", f"registered document with unrecognised category {category!r}",
                  publisher=entry.get("publisher"), category=category, provenance=entry.get("provenance"))
    return _q(rules, "unverified", "document has no recorded publisher or provenance metadata")
