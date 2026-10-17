"""T13: Tavily wrapper — timeouts, bounded retries, quotas and usage tracking."""

import logging

import pytest
import requests
import tavily.errors as te

from budget import BudgetExceeded, BudgetLimits, BudgetManager
from retriever.web_search import (ReplayWebSearch, TavilyWebSearch, WebSearchError, WebSearchQuotaError,
                                  query_hash)
from tests.fakes import FakeTavilyClient

OK = {"results": [{"url": "https://example.org/a", "title": "A", "content": "alpha", "score": 0.9}]}


def http_error(code):
    resp = requests.Response()
    resp.status_code = code
    return requests.HTTPError(f"{code}", response=resp)


def provider(settings, script, **limits):
    b = BudgetManager(BudgetLimits(**limits))
    client = FakeTavilyClient(script)
    return TavilyWebSearch(b, settings, client=client), b, client


def test_success_records_usage_and_passes_timeout(settings):
    p, b, client = provider(settings, [OK])
    results = p.search("what is attention")
    assert results[0].url == "https://example.org/a" and results[0].provider == "tavily"
    assert client.kwargs[0]["timeout"] == 15.0 and client.kwargs[0]["search_depth"] == "basic"
    assert client.kwargs[0]["max_results"] <= 5
    snap = b.snapshot()
    assert snap.web_searches == 1 and snap.tool_calls == 1 and snap.web_search_credits_estimated == 1.0


def test_timeout_then_success_is_two_attempts(settings):
    p, b, client = provider(settings, [te.TimeoutError(15), OK])
    assert len(p.search("q")) == 1
    assert client.calls == 2 and b.snapshot().web_searches == 2 and b.snapshot().web_search_failures == 1


def test_429_retried_but_attempts_bounded(settings):
    p, b, client = provider(settings, [http_error(429)])
    with pytest.raises(WebSearchError):
        p.search("q")
    assert client.calls == 2  # WEB_SEARCH_MAX_ATTEMPTS


def test_auth_error_not_retried(settings):
    p, b, client = provider(settings, [te.InvalidAPIKeyError("bad key")])
    with pytest.raises(WebSearchError):
        p.search("q")
    assert client.calls == 1


def test_quota_error_not_retried(settings):
    p, b, client = provider(settings, [te.UsageLimitExceededError("quota")])
    with pytest.raises(WebSearchQuotaError):
        p.search("q")
    assert client.calls == 1


def test_ninth_search_is_blocked(settings):
    p, b, client = provider(settings, [OK])
    for _ in range(8):
        p.search("q")
    with pytest.raises(BudgetExceeded, match="max_web_searches"):
        p.search("q")
    assert client.calls == 8


def test_raw_query_is_not_logged(settings, caplog):
    p, b, _ = provider(settings, [te.TimeoutError(15), OK])
    with caplog.at_level(logging.INFO, logger="ai_research_assistant"):
        p.search("confidential supplier merger plans")
    assert "confidential supplier" not in caplog.text
    assert query_hash("confidential supplier merger plans") in caplog.text


def test_replay_provider_is_deterministic_and_metered():
    b = BudgetManager(BudgetLimits(max_web_searches=1))
    r = ReplayWebSearch(b, recordings={"q": [{"url": "https://x", "content": "c"}]})
    assert [x.url for x in r.search("q")] == ["https://x"]
    with pytest.raises(BudgetExceeded):
        r.search("q")
