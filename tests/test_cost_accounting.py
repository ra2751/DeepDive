"""Budget Manager cost fixes: requested-vs-served price lookup, incomplete estimates,
reported vs estimated vs unknown, and unchanged request counting."""

import pytest
from langchain_core.messages import AIMessage

from budget import BudgetLimits, BudgetManager, PriceTable, format_cost
from budget.pricing import CACHED_UNPRICED, REASONING_UNPRICED

P = {"source": "test", "retrieved": "2026-10-16"}
TABLE = PriceTable({
    "openai/gpt-4o-mini": {"input_per_1m_usd": 0.15, "output_per_1m_usd": 0.60, **P},
    "google/gemini-2.5-flash": {"input_per_1m_usd": 0.30, "output_per_1m_usd": 2.50, **P},
})


def mgr(table=TABLE):
    return BudgetManager(BudgetLimits(max_llm_calls=None), pricing=table)


@pytest.mark.parametrize("requested,served,expected", [
    ("openai/gpt-4o-mini", "gpt-4o-mini-2024-07-18", "openai/gpt-4o-mini"),        # dated served name
    ("google/gemini-2.5-flash", "gemini-2.5-flash", "google/gemini-2.5-flash"),    # unprefixed served name
    ("openai/some-alias", "gpt-4o-mini-2024-07-18", "openai/gpt-4o-mini"),         # requested unpriced: served resolves
    ("google/alias", "gemini-2.5-flash", "google/gemini-2.5-flash"),               # provider prefix from request
    ("x/unknown", "unknown-model", None),                                          # nothing matches
])
def test_price_resolution(requested, served, expected):
    assert TABLE.resolve(requested, served) == expected


def test_primary_models_now_get_estimated_cost_with_observed_served_names():
    """Regression for the G2 defect: served names previously produced 'unknown'."""
    real = PriceTable.load("config/mesh_pricing.yaml")
    for requested, served in [("openai/gpt-4o-mini", "gpt-4o-mini-2024-07-18"),
                              ("google/gemini-2.5-flash", "gemini-2.5-flash"),
                              ("anthropic/claude-haiku-4.5", "anthropic/claude-haiku-4.5")]:
        b = mgr(real)
        b.record_mesh_attempt("structured", requested, "ok", model_served=served, input_tokens=250, output_tokens=40)
        snap = b.snapshot()
        assert snap.cost_basis == "estimated", requested
        assert b.events[-1]["priced_as"] == requested
    # Active since G3: the gateway now reads embedding token usage from Mesh responses
    assert real.estimate("openai/text-embedding-3-small", 1_000_000, 0) == pytest.approx(0.02)


def test_arithmetic_matches_the_approved_formula():
    b = mgr()
    b.record_mesh_attempt("chat", "openai/gpt-4o-mini", "ok", input_tokens=1000, output_tokens=200)
    assert b.snapshot().mesh_estimated_cost_usd == pytest.approx((1000 * 0.15 + 200 * 0.60) / 1e6)


def test_reasoning_tokens_without_rate_mark_estimate_incomplete():
    b = mgr()
    b.record_mesh_attempt("structured", "google/gemini-2.5-flash", "ok", model_served="gemini-2.5-flash",
                          input_tokens=100, output_tokens=50, reasoning_tokens=30)
    snap = b.snapshot()
    assert snap.cost_basis == "estimated" and snap.cost_estimate_incomplete
    assert snap.cost_incomplete_reasons == [REASONING_UNPRICED]
    assert snap.mesh_reasoning_tokens == 30
    assert snap.mesh_cost_usd == pytest.approx((100 * 0.30 + 50 * 2.50) / 1e6)
    assert "INCOMPLETE" in format_cost(snap)


def test_reasoning_rate_makes_estimate_complete():
    table = PriceTable({"google/gemini-2.5-flash": {"input_per_1m_usd": 0.30, "output_per_1m_usd": 2.50,
                                                    "reasoning_output_per_1m_usd": 3.50, **P}})
    b = mgr(table)
    b.record_mesh_attempt("chat", "google/gemini-2.5-flash", "ok", input_tokens=100, output_tokens=50,
                          reasoning_tokens=30)
    snap = b.snapshot()
    assert not snap.cost_estimate_incomplete
    assert snap.mesh_cost_usd == pytest.approx((100 * 0.30 + 20 * 2.50 + 30 * 3.50) / 1e6)


def test_cached_tokens_with_and_without_rate():
    b = mgr()
    b.record_mesh_attempt("chat", "openai/gpt-4o-mini", "ok", input_tokens=1000, output_tokens=0,
                          cached_input_tokens=400)
    assert b.snapshot().cost_incomplete_reasons == [CACHED_UNPRICED]
    table = PriceTable({"openai/gpt-4o-mini": {"input_per_1m_usd": 0.15, "output_per_1m_usd": 0.60,
                                               "cached_input_per_1m_usd": 0.075, **P}})
    b2 = mgr(table)
    b2.record_mesh_attempt("chat", "openai/gpt-4o-mini", "ok", input_tokens=1000, output_tokens=0,
                           cached_input_tokens=400)
    snap = b2.snapshot()
    assert not snap.cost_estimate_incomplete
    assert snap.mesh_cost_usd == pytest.approx((600 * 0.15 + 400 * 0.075) / 1e6)


def test_approximated_tokens_are_never_priced():
    table = PriceTable({"openai/text-embedding-3-small": {"input_per_1m_usd": 0.02, **P}})
    b = mgr(table)
    b.record_mesh_attempt("embedding", "openai/text-embedding-3-small", "ok", input_tokens=120,
                          output_tokens=0, tokens_estimated=True)
    snap = b.snapshot()
    assert snap.cost_basis == "unknown" and snap.mesh_cost_usd is None
    assert snap.mesh_input_tokens is None  # totals are not presented as exact
    assert format_cost(snap) == "Cost unknown; limited by request caps"


def test_reported_and_estimated_costs_are_kept_apart():
    b = mgr()
    b.record_mesh_attempt("chat", "openai/gpt-4o-mini", "ok", input_tokens=100, output_tokens=10,
                          reported_cost_usd=0.01)
    b.record_mesh_attempt("chat", "openai/gpt-4o-mini", "ok", input_tokens=1000, output_tokens=200)
    snap = b.snapshot()
    assert snap.mesh_reported_cost_usd == pytest.approx(0.01)
    assert snap.mesh_estimated_cost_usd == pytest.approx(0.00027)
    assert snap.cost_basis == "estimated"  # weakest basis wins
    assert b.events[0]["cost_basis"] == "reported" and b.events[1]["cost_basis"] == "estimated"


def test_unpriced_model_makes_run_cost_unknown():
    b = mgr()
    b.record_mesh_attempt("chat", "openai/gpt-4o-mini", "ok", input_tokens=100, output_tokens=10)
    b.record_mesh_attempt("chat", "x/unpriced", "ok", input_tokens=100, output_tokens=10)
    assert b.snapshot().cost_basis == "unknown" and not b.snapshot().cost_cap_enforced


def test_request_counts_unchanged_and_failures_not_costed():
    b = mgr()
    b.record_mesh_attempt("chat", "openai/gpt-4o-mini", "error", error="RateLimitError")
    b.record_mesh_attempt("chat", "openai/gpt-4o-mini", "ok", input_tokens=100, output_tokens=10)
    snap = b.snapshot()
    assert snap.mesh_chat_calls == 2 and snap.mesh_failed_attempts == 1
    assert len(b.events) == 2 and b.events[0]["cost_usd"] is None
    assert snap.mesh_estimated_cost_usd == pytest.approx((100 * 0.15 + 10 * 0.60) / 1e6)


def test_gateway_passes_cached_and_reasoning_details(fake_mesh, monkeypatch):
    from gateway import chat_model

    msg = AIMessage(content="ok", response_metadata={"model_name": "gemini-2.5-flash"},
                    usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
                                    "input_token_details": {"cache_read": 20},
                                    "output_token_details": {"reasoning": 30}})
    fake_mesh.generator.script = [msg]
    b = mgr(PriceTable({"test/generator": {"input_per_1m_usd": 0.30, "output_per_1m_usd": 2.50, **P}}))
    chat_model("generator", b).invoke("hi")
    snap = b.snapshot()
    assert snap.mesh_cached_input_tokens == 20 and snap.mesh_reasoning_tokens == 30
    assert set(snap.cost_incomplete_reasons) == {CACHED_UNPRICED, REASONING_UNPRICED}
