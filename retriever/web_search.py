"""
Web search providers.

Tavily is a replaceable, NON-MODEL web-search provider: it is
used only to fetch search results, never for AI inference (no Tavily-generated
answers). Every attempt is checked against and recorded in the run's
BudgetManager, with a per-request timeout and bounded retries.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

import requests
from pydantic import BaseModel

from budget.manager import BudgetExceeded, BudgetManager
from config.settings import Settings, get_settings

logger = logging.getLogger("ai_research_assistant")

_sleep = time.sleep  # patchable for tests


class WebSearchError(RuntimeError):
    """Web search failed after the allowed attempts, or failed permanently."""


class WebSearchQuotaError(WebSearchError):
    """The provider's usage quota is exhausted."""


class WebSearchResult(BaseModel):
    url: str
    title: Optional[str] = None
    content: str
    score: Optional[float] = None
    retrieved_at: str
    provider: str
    query_hash: str
    published_date: Optional[str] = None   # only when the provider reports one; never inferred


def has_source_url(url: Optional[str]) -> bool:
    """A web result is usable as evidence only if it carries an actual http(s) source URL."""
    return bool(url) and url.lower().startswith(("http://", "https://")) and len(url) > len("https://")


class WebSearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: Optional[int] = None) -> list: ...


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify(exc: BaseException) -> str:
    name = type(exc).__name__
    if name in ("UsageLimitExceededError", "TavilyKeylessLimitError"):
        return "quota"
    if name in ("InvalidAPIKeyError", "MissingAPIKeyError", "BadRequestError", "ForbiddenError"):
        return "fatal"
    if name == "TimeoutError" or isinstance(exc, (TimeoutError, ConnectionError, requests.Timeout, requests.ConnectionError)):
        return "retry"
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return "retry" if status == 429 or status >= 500 else "fatal"
    return "fatal"


class TavilyWebSearch:
    name = "tavily"

    def __init__(self, budget: BudgetManager, settings: Optional[Settings] = None, client=None):
        self.settings = settings or get_settings()
        self.budget = budget
        self._client = client

    def _get_client(self):
        if self._client is None:
            self.settings.require("tavily_api_key")
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self.settings.tavily_api_key)
        return self._client

    def search(self, query: str, max_results: Optional[int] = None,
               include_domains: Optional[list] = None, exclude_domains: Optional[list] = None) -> list:
        s = self.settings
        client = self._get_client()
        n = min(max_results or s.web_search_max_results, 5)
        qh = query_hash(query)
        credits = 1.0 if s.web_search_depth == "basic" else 2.0
        attempts = max(1, s.web_search_max_attempts)
        for attempt in range(1, attempts + 1):
            self.budget.check("web_search")
            started = time.monotonic()
            try:
                kwargs = dict(query=query, search_depth=s.web_search_depth, max_results=n,
                              timeout=s.web_search_timeout_s)
                if include_domains:
                    kwargs["include_domains"] = include_domains
                if exclude_domains:
                    kwargs["exclude_domains"] = exclude_domains
                response = client.search(**kwargs)
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                decision = _classify(exc)
                self.budget.record_search_attempt(
                    self.name, "error", latency_s=time.monotonic() - started,
                    query_hash=qh, error=type(exc).__name__,
                )
                logger.warning("WebSearch | tavily | query=%s | attempt %d/%d failed: %s (%s)",
                               qh, attempt, attempts, type(exc).__name__, decision)
                if decision == "quota":
                    raise WebSearchQuotaError("Tavily usage quota exhausted") from exc
                if decision == "fatal" or attempt == attempts:
                    raise WebSearchError(f"Tavily search failed: {type(exc).__name__}") from exc
                _sleep(min(0.5 * (2 ** (attempt - 1)), 4.0))
                continue
            items = (response or {}).get("results", []) if isinstance(response, dict) else []
            results = [
                WebSearchResult(
                    url=item["url"],
                    title=item.get("title"),
                    content=item.get("content") or "",
                    score=item.get("score"),
                    retrieved_at=_now(),
                    provider=self.name,
                    query_hash=qh,
                    published_date=item.get("published_date"),
                )
                for item in items if has_source_url(item.get("url"))   # no URL -> not citable -> dropped
            ]
            self.budget.record_search_attempt(
                self.name, "ok", result_count=len(results), latency_s=time.monotonic() - started,
                credits_estimated=credits, query_hash=qh,
            )
            logger.info("WebSearch | tavily | query=%s | %d results", qh, len(results))
            return results
        return []


class ReplayWebSearch:
    """Deterministic provider that replays recorded results (offline tests and evals)."""

    name = "replay"

    def __init__(self, budget: BudgetManager, recordings: Optional[dict] = None, path: Optional[str] = None):
        self.budget = budget
        if recordings is None and path:
            recordings = json.loads(Path(path).read_text())
        self.recordings = recordings or {}

    def search(self, query: str, max_results: Optional[int] = None, **_ignored) -> list:
        self.budget.check("web_search")
        qh = query_hash(query)
        items = self.recordings.get(query, self.recordings.get(qh, []))
        results = [
            WebSearchResult(url=i["url"], title=i.get("title"), content=i.get("content", ""),
                            score=i.get("score"), retrieved_at=i.get("retrieved_at", _now()),
                            provider=self.name, query_hash=qh, published_date=i.get("published_date"))
            for i in [i for i in items if has_source_url(i.get("url"))][: (max_results or 5)]
        ]
        self.budget.record_search_attempt(self.name, "ok", result_count=len(results), query_hash=qh)
        return results


def get_web_search_provider(budget: BudgetManager, settings: Optional[Settings] = None):
    s = settings or get_settings()
    if s.web_search_provider == "replay":
        return ReplayWebSearch(budget, path=s.web_search_replay_file)
    return TavilyWebSearch(budget, s)
