"""T06–T09: limits, attempt counting and honest cost labels."""

import pytest

from budget import BudgetExceeded, BudgetLimits, BudgetManager, PriceTable, format_cost, format_usage


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _ok(b, op="chat", **kw):
    b.record_mesh_attempt(op, "m", "ok", input_tokens=kw.get("tin", 100), output_tokens=kw.get("tout", 10),
                          reported_cost_usd=kw.get("cost"), model_served=kw.get("served"))


@pytest.mark.parametrize("kind,limit_field,limit,record", [
    ("llm", "max_llm_calls", 25, lambda b: _ok(b, "chat")),
    ("embedding", "max_embedding_calls", 10, lambda b: _ok(b, "embedding")),
    ("web_search", "max_web_searches", 8, lambda b: b.record_search_attempt("tavily", "ok")),
    ("tool", "max_tool_calls", 30, lambda b: b.record_tool("milvus_search")),
])
def test_each_limit_blocks_exactly_the_next_attempt(kind, limit_field, limit, record):
    b = BudgetManager(BudgetLimits())
    for _ in range(limit):
        b.check(kind)
        record(b)
    with pytest.raises(BudgetExceeded) as exc:
        b.check(kind)
    assert exc.value.reason == limit_field


def test_web_searches_also_count_as_tool_calls():
    b = BudgetManager(BudgetLimits(max_web_searches=None, max_tool_calls=3))
    for _ in range(3):
        b.check("web_search")
        b.record_search_attempt("tavily", "ok")
    with pytest.raises(BudgetExceeded, match="max_tool_calls"):
        b.check("web_search")


def test_retry_rounds_limit():
    b = BudgetManager(BudgetLimits())
    assert b.next_retry_round() == 1 and b.next_retry_round() == 2
    assert b.retry_rounds_remaining() == 0
    with pytest.raises(BudgetExceeded, match="max_retry_rounds"):
        b.next_retry_round()


def test_elapsed_time_limit_blocks_every_kind():
    clock = Clock()
    b = BudgetManager(BudgetLimits(), clock=clock)
    clock.t = 299.9
    b.check("llm")
    clock.t = 300.0
    for kind in ("llm", "embedding", "web_search", "tool"):
        with pytest.raises(BudgetExceeded, match="max_run_seconds"):
            b.check(kind)
    assert b.exhausted() == "max_run_seconds"


def test_failed_attempts_count_toward_limits():
    b = BudgetManager(BudgetLimits(max_llm_calls=2))
    b.record_mesh_attempt("chat", "m", "error", error="RateLimitError")
    b.record_mesh_attempt("chat", "m", "error", error="RateLimitError")
    with pytest.raises(BudgetExceeded, match="max_llm_calls"):
        b.check("llm")
    assert b.snapshot().mesh_failed_attempts == 2


def test_unknown_cost_basis_does_not_enforce_cost_cap():
    b = BudgetManager(BudgetLimits(max_llm_calls=None, max_mesh_cost_usd=0.0001))
    for _ in range(50):
        b.check("llm")
        _ok(b)
    snap = b.snapshot()
    assert snap.cost_basis == "unknown" and snap.mesh_cost_usd is None
    assert snap.cost_cap_enforced is False
    assert format_cost(snap) == "Cost unknown; limited by request caps"


def test_estimated_cost_enforced_with_safety_factor():
    prices = PriceTable({"m": {"input_per_1m_usd": 1000.0, "output_per_1m_usd": 0.0,
                               "source": "test", "retrieved": "2026-10-16"}})
    b = BudgetManager(BudgetLimits(max_llm_calls=None, max_mesh_cost_usd=0.50), pricing=prices)
    _ok(b, tin=350)  # 0.35 USD estimated; x1.25 = 0.4375 < 0.50
    b.check("llm")
    _ok(b, tin=60)   # 0.41 USD estimated; x1.25 = 0.5125 >= 0.50 -> blocked
    with pytest.raises(BudgetExceeded, match="max_mesh_cost_usd"):
        b.check("llm")
    snap = b.snapshot()
    assert snap.cost_basis == "estimated" and snap.cost_cap_enforced
    assert format_cost(snap).startswith("≈ $0.4100") and "estimated" in format_cost(snap)


def test_reported_cost_is_labelled_reported_and_weakest_basis_wins():
    b = BudgetManager(BudgetLimits())
    _ok(b, cost=0.01)
    assert b.snapshot().cost_basis == "reported"
    assert "reported by Mesh" in format_cost(b.snapshot())
    _ok(b)  # no cost, no price -> unknown
    assert b.snapshot().cost_basis == "unknown"


def test_format_never_shows_a_bare_amount():
    for snap in (BudgetManager().snapshot(),):
        text = format_cost(snap)
        assert any(w in text for w in ("reported", "estimated", "unknown"))
    line = format_usage(BudgetManager().snapshot(), BudgetLimits())
    assert "Mesh LLM calls 0/25" in line and "embeddings 0/10" in line and "web searches 0/8" in line
    assert "tools 0/30" in line and "retry rounds 0/2" in line and "Cost unknown" in line


def test_ingestion_budget_is_separate_and_embedding_only():
    from budget import new_ingestion_budget
    from config.settings import load_settings

    b = new_ingestion_budget(load_settings(environ={}))
    assert b.limits.max_embedding_calls == 50 and b.limits.max_llm_calls == 0
    with pytest.raises(BudgetExceeded):
        b.check("llm")
