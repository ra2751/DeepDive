"""Time basis of a claim: forecast/projection vs. historical or current fact (deterministic).

A claim is a forecast if the extractor typed it so, if it uses projection language, or if it
names a year after the run year. Forecasts are shown as projections reported by the cited
source, never as observed outcomes.
"""

import re
from datetime import datetime, timezone
from typing import Optional

from verification.text import normalize

_PROJECTION = (r"forecasts?|forecasted|projected (?:to|at)|projects (?:that|to)|"
               r"projections? (?:for|of|show|suggest)|expected to|is expected|are expected|"
               r"anticipated|outlook|estimated to (?:reach|grow|rise|exceed)|"
               r"(?:will|could|may|might|would) (?:reach|grow|rise|fall|exceed|double|triple|account|be|become|"
               r"surpass|increase|decline|expand|hit)|cagr")
_WORDING = re.compile(rf"\b(?:{_PROJECTION})\b")
_CUES = re.compile(rf"\b(?:{_PROJECTION}|by (?:the )?end of 20\d\d|through 20\d\d|by 20\d\d)\b")


def forecast_wording(text: str) -> bool:
    """True if the text uses projection language (verbs and nouns of projection; a target year alone
    such as "by 2030" does not count)."""
    return bool(_WORDING.search(normalize(text)))


def is_forecast(text: str, claim_type: Optional[str] = None, as_of_year: Optional[int] = None) -> bool:
    if claim_type == "forecast":
        return True
    t = normalize(text)
    if _CUES.search(t):
        return True
    year = as_of_year or datetime.now(timezone.utc).year
    return any(int(y) > year for y in re.findall(r"\b(20\d\d)\b", t))


def time_basis(claim, as_of_year: Optional[int] = None) -> str:
    return "forecast" if is_forecast(claim.text, claim.claim_type, as_of_year) else "historical/current"
