"""
Research Budget Manager.

A run-scoped object that is checked before, and updated after, every Mesh API
attempt and every web search attempt. Retries count as attempts. Costs are
labelled "reported", "estimated" or "unknown" and an estimate is never
presented as an exact figure.
"""

import threading
import time
from typing import Callable, Literal, Optional

from pydantic import BaseModel

from budget.pricing import PriceTable

CheckKind = Literal["llm", "embedding", "web_search", "tool"]
CostBasis = Literal["reported", "estimated", "unknown"]
_BASIS_RANK = {"unknown": 0, "estimated": 1, "reported": 2}


CACHE_UNCONFIRMED = ("cached response identified by an undocumented marker; token-based estimate kept because "
                     "billing is unconfirmed")


class BudgetExceeded(Exception):
    """Raised before a call is sent when a limit would be exceeded."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class BudgetConfigError(RuntimeError):
    """Raised when a graph node runs without a run-scoped budget."""


class BudgetLimits(BaseModel):
    max_llm_calls: Optional[int] = 25
    max_embedding_calls: Optional[int] = 10
    max_web_searches: Optional[int] = 8
    max_tool_calls: Optional[int] = 30
    max_retry_rounds: int = 2
    max_run_seconds: Optional[float] = 300.0
    max_mesh_cost_usd: Optional[float] = 0.50
    cost_safety_factor: float = 1.25

    @classmethod
    def from_settings(cls, settings) -> "BudgetLimits":
        return cls(
            max_llm_calls=settings.max_llm_calls,
            max_embedding_calls=settings.max_embedding_calls,
            max_web_searches=settings.max_web_searches,
            max_tool_calls=settings.max_tool_calls,
            max_retry_rounds=settings.max_retry_rounds,
            max_run_seconds=settings.max_run_seconds,
            max_mesh_cost_usd=settings.max_mesh_cost_usd,
            cost_safety_factor=settings.cost_estimate_safety_factor,
        )

    @classmethod
    def ingestion_from_settings(cls, settings) -> "BudgetLimits":
        # Ingestion budgets are separate from research-run budgets: embeddings only.
        return cls(
            max_llm_calls=0,
            max_embedding_calls=settings.max_ingest_embedding_calls,
            max_web_searches=0,
            max_tool_calls=None,
            max_retry_rounds=0,
            max_run_seconds=None,
            max_mesh_cost_usd=None,
            cost_safety_factor=settings.cost_estimate_safety_factor,
        )


class UsageLedger(BaseModel):
    mesh_chat_calls: int = 0
    mesh_vision_calls: int = 0
    mesh_embedding_calls: int = 0
    mesh_failed_attempts: int = 0
    mesh_input_tokens: Optional[int] = 0
    mesh_output_tokens: Optional[int] = 0
    web_searches: int = 0
    web_search_failures: int = 0
    tool_calls: int = 0
    retry_rounds: int = 0
    elapsed_s: float = 0.0
    mesh_cached_responses: int = 0               # gateway response-cache hits (still count as requests)
    mesh_cached_input_tokens: int = 0
    mesh_reasoning_tokens: int = 0
    mesh_cost_usd: Optional[float] = None          # reported + estimated, shown only if basis is not unknown
    mesh_reported_cost_usd: float = 0.0            # authoritative costs returned by Mesh
    mesh_estimated_cost_usd: float = 0.0           # local estimates from token usage x price table
    cost_basis: CostBasis = "unknown"
    cost_estimate_incomplete: bool = False
    cost_incomplete_reasons: list = []
    cost_cap_enforced: bool = False
    web_search_credits_estimated: Optional[float] = None

    @property
    def mesh_llm_calls(self) -> int:
        return self.mesh_chat_calls + self.mesh_vision_calls


class BudgetManager:
    def __init__(
        self,
        limits: Optional[BudgetLimits] = None,
        pricing: Optional[PriceTable] = None,
        clock: Callable[[], float] = time.monotonic,
        name: str = "research",
    ):
        self.limits = limits or BudgetLimits()
        self.pricing = pricing or PriceTable({})
        self.name = name
        self._clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self._ledger = UsageLedger()
        self._cost_basis: Optional[CostBasis] = None  # None until a priced call happens
        self._cost_usd = 0.0
        self._reported_usd = 0.0
        self._estimated_usd = 0.0
        self._incomplete_reasons: list[str] = []
        self.events: list[dict] = []

    # ------------------------------------------------------------------ checks
    def elapsed(self) -> float:
        return self._clock() - self._started

    def cost_cap_enforceable(self) -> bool:
        return (
            self.limits.max_mesh_cost_usd is not None
            and self._cost_basis in ("reported", "estimated")
        )

    def check(self, kind: CheckKind) -> None:
        """Raise BudgetExceeded if one more attempt of `kind` is not allowed."""
        with self._lock:
            reason = self._blocking_reason(kind)
        if reason:
            raise BudgetExceeded(reason)

    def _blocking_reason(self, kind: CheckKind) -> Optional[str]:
        lim, led = self.limits, self._ledger
        if lim.max_run_seconds is not None and self.elapsed() >= lim.max_run_seconds:
            return "max_run_seconds"
        if kind == "llm":
            if lim.max_llm_calls is not None and led.mesh_llm_calls >= lim.max_llm_calls:
                return "max_llm_calls"
        elif kind == "embedding":
            if lim.max_embedding_calls is not None and led.mesh_embedding_calls >= lim.max_embedding_calls:
                return "max_embedding_calls"
        elif kind == "web_search":
            if lim.max_web_searches is not None and led.web_searches >= lim.max_web_searches:
                return "max_web_searches"
            if lim.max_tool_calls is not None and led.tool_calls >= lim.max_tool_calls:
                return "max_tool_calls"
        elif kind == "tool":
            if lim.max_tool_calls is not None and led.tool_calls >= lim.max_tool_calls:
                return "max_tool_calls"
        if kind in ("llm", "embedding") and self.cost_cap_enforceable():
            projected = self._cost_usd
            if self._cost_basis == "estimated":
                projected *= lim.cost_safety_factor
            if projected >= lim.max_mesh_cost_usd:
                return "max_mesh_cost_usd"
        return None

    def exhausted(self) -> Optional[str]:
        """Reason why no further model call can be made, or None."""
        with self._lock:
            return self._blocking_reason("llm")

    def llm_calls_remaining(self) -> Optional[int]:
        """Model calls still allowed (None = unlimited); 0 also when time/cost limits block."""
        if self.exhausted():
            return 0
        if self.limits.max_llm_calls is None:
            return None
        return max(0, self.limits.max_llm_calls - self._ledger.mesh_llm_calls)

    def retry_rounds_remaining(self) -> int:
        return max(0, self.limits.max_retry_rounds - self._ledger.retry_rounds)

    def next_retry_round(self) -> int:
        with self._lock:
            if self._ledger.retry_rounds >= self.limits.max_retry_rounds:
                raise BudgetExceeded("max_retry_rounds")
            self._ledger.retry_rounds += 1
            return self._ledger.retry_rounds

    # --------------------------------------------------------------- recording
    def record_mesh_attempt(
        self,
        operation: Literal["chat", "structured", "vision", "embedding"],
        model_requested: str,
        status: Literal["ok", "error"],
        model_served: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        tokens_estimated: bool = False,
        reported_cost_usd: Optional[float] = None,
        cached_input_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        cached: bool = False,
        cache_marker: Optional[str] = None,
        latency_s: Optional[float] = None,
        request_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            led = self._ledger
            if operation == "embedding":
                led.mesh_embedding_calls += 1
            elif operation == "vision":
                led.mesh_vision_calls += 1
            else:
                led.mesh_chat_calls += 1
            if status != "ok":
                led.mesh_failed_attempts += 1

            basis: Optional[CostBasis] = None
            cost: Optional[float] = None
            priced_as: Optional[str] = None
            reasons: list[str] = []
            if status == "ok" and cached:
                led.mesh_cached_responses += 1
            if status == "ok" and cached and cache_marker == "x-cache-header":
                # Mesh documents gateway cache hits as not billed; usage reflects the ORIGINAL response,
                # so tokens are not added as new work.
                basis, cost = "estimated", 0.0
            elif status == "ok":
                if not tokens_estimated:
                    led.mesh_cached_input_tokens += cached_input_tokens or 0
                    led.mesh_reasoning_tokens += reasoning_tokens or 0
                if led.mesh_input_tokens is not None:
                    if input_tokens is None or tokens_estimated:
                        led.mesh_input_tokens = None   # totals are only shown when every call reported usage
                        led.mesh_output_tokens = None
                    else:
                        led.mesh_input_tokens += input_tokens
                        led.mesh_output_tokens = (led.mesh_output_tokens or 0) + (output_tokens or 0)
                if reported_cost_usd is not None:
                    basis, cost = "reported", reported_cost_usd
                    self._reported_usd += cost
                elif tokens_estimated:
                    # Token counts were approximated locally: no fabricated precision.
                    basis = "unknown"
                else:
                    quote = self.pricing.quote(model_requested, model_served, input_tokens, output_tokens,
                                               cached_input_tokens, reasoning_tokens)
                    if quote is not None:
                        basis, cost, priced_as = "estimated", quote.cost_usd, quote.priced_as
                        reasons = list(quote.incomplete_reasons)
                        if cached:   # detected only by an undocumented marker: billing unconfirmed
                            reasons.append(CACHE_UNCONFIRMED)
                        self._estimated_usd += cost
                        for r in reasons:
                            if r not in self._incomplete_reasons:
                                self._incomplete_reasons.append(r)
                    else:
                        basis = "unknown"
            # Failed attempts without a response are not assumed to be billed and
            # do not change the cost basis.
            if basis is not None:
                if cost is not None:
                    self._cost_usd += cost
                if self._cost_basis is None or _BASIS_RANK[basis] < _BASIS_RANK[self._cost_basis]:
                    self._cost_basis = basis
            self._event(
                provider="mesh", operation=operation, model_requested=model_requested,
                model_served=model_served, status=status, input_tokens=input_tokens,
                output_tokens=output_tokens, tokens_estimated=tokens_estimated,
                cached_input_tokens=cached_input_tokens, reasoning_tokens=reasoning_tokens,
                cost_usd=cost, cost_basis=basis, priced_as=priced_as, cached=cached, cache_marker=cache_marker,
                cost_incomplete_reasons=reasons or None, latency_s=latency_s,
                request_id=request_id, error=error,
            )

    def record_search_attempt(
        self,
        provider: str,
        status: Literal["ok", "error"],
        result_count: int = 0,
        latency_s: Optional[float] = None,
        credits_estimated: Optional[float] = None,
        query_hash: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            led = self._ledger
            led.web_searches += 1
            led.tool_calls += 1
            if status != "ok":
                led.web_search_failures += 1
            if credits_estimated is not None:
                led.web_search_credits_estimated = (led.web_search_credits_estimated or 0.0) + credits_estimated
            self._event(provider=provider, operation="search", status=status,
                        result_count=result_count, latency_s=latency_s,
                        credits_estimated=credits_estimated, query_hash=query_hash, error=error)

    def record_tool(self, name: str, status: Literal["ok", "error"] = "ok") -> None:
        with self._lock:
            self._ledger.tool_calls += 1
            self._event(provider="local", operation=name, status=status)

    def _event(self, **fields) -> None:
        fields["budget"] = self.name
        fields["t_s"] = round(self.elapsed(), 3)
        self.events.append(fields)

    # ---------------------------------------------------------------- snapshot
    def snapshot(self) -> UsageLedger:
        with self._lock:
            led = self._ledger.model_copy()
        led.elapsed_s = round(self.elapsed(), 3)
        led.cost_basis = self._cost_basis or "unknown"
        led.mesh_cost_usd = round(self._cost_usd, 8) if self._cost_basis in ("reported", "estimated") else None
        led.mesh_reported_cost_usd = round(self._reported_usd, 8)
        led.mesh_estimated_cost_usd = round(self._estimated_usd, 8)
        led.cost_estimate_incomplete = bool(self._incomplete_reasons)
        led.cost_incomplete_reasons = list(self._incomplete_reasons)
        led.cost_cap_enforced = self.cost_cap_enforceable()
        return led


def format_cost(ledger: UsageLedger) -> str:
    """Human-readable cost that always carries its basis label."""
    if ledger.cost_basis == "reported" and ledger.mesh_cost_usd is not None:
        return f"${ledger.mesh_cost_usd:.4f} (reported by Mesh)"
    if ledger.cost_basis == "estimated" and ledger.mesh_cost_usd is not None:
        if ledger.cost_estimate_incomplete:
            return (f"≈ ${ledger.mesh_cost_usd:.4f} (estimated, INCOMPLETE: some token types have no "
                    f"configured rate)")
        return f"≈ ${ledger.mesh_cost_usd:.4f} (estimated from token usage and the dated price table)"
    return "Cost unknown; limited by request caps"


def format_usage(ledger: UsageLedger, limits: BudgetLimits) -> str:
    def lim(v):
        return "∞" if v is None else str(v)

    parts = [
        f"Mesh LLM calls {ledger.mesh_llm_calls}/{lim(limits.max_llm_calls)}",
        f"embeddings {ledger.mesh_embedding_calls}/{lim(limits.max_embedding_calls)}",
        f"web searches {ledger.web_searches}/{lim(limits.max_web_searches)}",
        f"tools {ledger.tool_calls}/{lim(limits.max_tool_calls)}",
        f"retry rounds {ledger.retry_rounds}/{limits.max_retry_rounds}",
        f"{ledger.elapsed_s:.1f} s/{lim(limits.max_run_seconds)} s",
        *([f"{ledger.mesh_cached_responses} cached response(s)"] if ledger.mesh_cached_responses else []),
        format_cost(ledger),
    ]
    return " · ".join(parts)
