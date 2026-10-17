"""
Grounding check for model-written report sentences (deterministic, conservative).

A report sentence may only restate its cited, verified claims. This check REJECTS sentences that
visibly add something the claims do not say; it cannot PROVE that an accepted sentence is entailed
by its claims (that would need semantic judgment). A rejected sentence is never repaired: the
citation checker replaces it with the verified atomic claim text (deterministic fallback).

Rules (each term must already appear in the cited claims, otherwise the sentence is rejected):
1. Ranking and importance words ("biggest", "leading", "most important", ...).
2. Counting and completeness words ("three", "all", "only", ...): a sentence cannot claim that the
   requested number of items was found.
3. Causal, conditional and implication language ("could lead to", "because", "may present", ...).
4. Evaluative category words ("risk", "opportunity", "challenge", ...): calling a fact a risk or an
   opportunity is itself an assertion.
5. Negation that the claims do not contain.
6. A forecast claim restated without forecast wording (estimate turned into fact); a future year alone
   does not count as forecast wording when the claim used projection language.
7. More than MAX_NOVEL_TERMS content words that none of the cited claims contain (a flag for
   paraphrase drift; word overlap alone is never treated as proof of support).
"""

import re

from verification.temporal import forecast_wording, is_forecast
from verification.text import content_stems, negated, normalize, stem

MAX_NOVEL_TERMS = 2

RANKING = {"biggest", "largest", "greatest", "leading", "top", "best", "worst", "most", "least", "foremost",
           "main", "major", "key", "primary", "principal", "crucial", "critical", "important", "significant",
           "significantly", "dominant", "notable", "notably", "prominent", "chief", "strongest", "weakest"}
COUNTING = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "all", "every",
            "only", "sole", "entire", "complete", "comprehensive", "several", "various", "numerous", "many",
            "both", "first", "second", "third"}
CAUSAL = {"because", "therefore", "thus", "hence", "consequently", "cause", "causes", "caused", "causing",
          "lead", "leads", "leading", "led", "result", "results", "resulted", "due", "drive", "drives",
          "driven", "driving", "enable", "enables", "enabled", "implies", "imply", "implication",
          "implications", "suggest", "suggests", "indicate", "indicates", "means", "may", "might", "could",
          "would", "likely", "unlikely", "potential", "potentially", "if", "unless", "so", "overall",
          "together", "combined", "since", "despite", "although", "boost", "boosts", "fuel", "fuels",
          "impact", "impacts", "affect", "affects", "contribute", "contributes", "help", "helps"}
CATEGORY = {"risk", "risks", "risky", "opportunity", "opportunities", "challenge", "challenges", "threat",
            "threats", "barrier", "barriers", "benefit", "benefits", "advantage", "advantages",
            "disadvantage", "weakness", "weaknesses", "strength", "strengths", "concern", "concerns",
            "headwind", "headwinds", "tailwind", "tailwinds", "growth", "promising", "attractive"}
# Attribution and connective words that restate rather than assert.
ALLOWED_NOVEL = {stem(w) for w in ("also", "additionally", "including", "includes", "further", "furthermore",
                                    "moreover", "stated", "states", "noted", "notes", "shown", "according",
                                    "reported", "reports", "described", "mentioned", "cited", "illustrated",
                                    "supported")}


def _words(text: str) -> list:
    return re.findall(r"[a-z][a-z'\-]*", normalize(text))


def grounding_issue(sentence: str, claim_texts: list):
    """Return a rejection reason if the sentence asserts more than its claims, else None."""
    claims_joined = " ".join(claim_texts)
    claim_words = set(_words(claims_joined))
    words = _words(sentence)
    for label, vocab in (("ranking", RANKING), ("counting", COUNTING), ("causal", CAUSAL), ("category", CATEGORY)):
        added = sorted({w for w in words if w in vocab and w not in claim_words})
        if added:
            return f"unsupported_{label}_term:{','.join(added)}"
    if negated(sentence) and not negated(claims_joined):
        return "negation_not_in_cited_claims"
    if (any(forecast_wording(c) for c in claim_texts) and not forecast_wording(sentence)) or \
            (any(is_forecast(c) for c in claim_texts) and not is_forecast(sentence)):
        return "forecast_stated_as_fact"
    novel = sorted(content_stems(sentence) - content_stems(claims_joined) - ALLOWED_NOVEL)
    if len(novel) > MAX_NOVEL_TERMS:
        return "unsupported_content:" + ",".join(novel)
    return None
