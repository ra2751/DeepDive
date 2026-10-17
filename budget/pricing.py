"""
Mesh price table used to *estimate* cost from token usage.

Every entry must record where the price came from and when. With no entry for a
model, cost is "unknown" and is never guessed.

Token accounting follows the OpenAI-compatible usage convention: input tokens
include cached input tokens, and output tokens include reasoning tokens. When a
response reports cached or reasoning tokens but no specific rate is configured,
those tokens are priced at the ordinary input/output rate and the estimate is
marked INCOMPLETE.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")

CACHED_UNPRICED = "cached-input tokens priced at the full input rate (no cached rate configured)"
REASONING_UNPRICED = "reasoning tokens priced at the output rate (no reasoning rate configured)"


class ModelPrice(BaseModel):
    input_per_1m_usd: float
    output_per_1m_usd: float = 0.0
    cached_input_per_1m_usd: Optional[float] = None
    reasoning_output_per_1m_usd: Optional[float] = None
    source: str
    retrieved: str


@dataclass
class CostQuote:
    cost_usd: float
    priced_as: str
    incomplete_reasons: list = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.incomplete_reasons


class PriceTable:
    def __init__(self, models: dict):
        self.models = {k: (v if isinstance(v, ModelPrice) else ModelPrice(**v)) for k, v in models.items()}

    @classmethod
    def load(cls, path: str = "config/mesh_pricing.yaml") -> "PriceTable":
        p = Path(path)
        if not p.exists():
            return cls({})
        data = yaml.safe_load(p.read_text()) or {}
        return cls(data.get("models") or {})  # "inactive_models" are deliberately not loaded

    def resolve(self, model_requested: Optional[str], model_served: Optional[str] = None) -> Optional[str]:
        """Find the price entry for a call.

        Mesh bills the requested route; the served name may be a dated snapshot
        ("gpt-4o-mini-2024-07-18") or lack the provider prefix ("gemini-2.5-flash").
        """
        candidates = []
        if model_requested:
            candidates.append(model_requested)
        if model_served:
            provider = model_requested.split("/", 1)[0] if model_requested and "/" in model_requested else None
            base = _DATE_SUFFIX.sub("", model_served)
            candidates += [model_served, base]
            if provider and "/" not in model_served:
                candidates += [f"{provider}/{model_served}", f"{provider}/{base}"]
        for c in candidates:
            if c in self.models:
                return c
        return None

    def quote(self, model_requested: Optional[str], model_served: Optional[str], input_tokens: Optional[int],
              output_tokens: Optional[int], cached_input_tokens: Optional[int] = None,
              reasoning_tokens: Optional[int] = None) -> Optional[CostQuote]:
        if input_tokens is None:
            return None
        key = self.resolve(model_requested, model_served)
        if key is None:
            return None
        price = self.models[key]
        out = output_tokens or 0
        cached = min(cached_input_tokens or 0, input_tokens)
        reasoning = min(reasoning_tokens or 0, out)
        reasons = []

        if cached and price.cached_input_per_1m_usd is not None:
            input_cost = (input_tokens - cached) * price.input_per_1m_usd + cached * price.cached_input_per_1m_usd
        else:
            input_cost = input_tokens * price.input_per_1m_usd
            if cached:
                reasons.append(CACHED_UNPRICED)

        if reasoning and price.reasoning_output_per_1m_usd is not None:
            output_cost = (out - reasoning) * price.output_per_1m_usd + reasoning * price.reasoning_output_per_1m_usd
        else:
            output_cost = out * price.output_per_1m_usd
            if reasoning:
                reasons.append(REASONING_UNPRICED)

        return CostQuote(cost_usd=(input_cost + output_cost) / 1_000_000, priced_as=key, incomplete_reasons=reasons)

    def estimate(self, model: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
        q = self.quote(model, None, input_tokens, output_tokens)
        return q.cost_usd if q else None
