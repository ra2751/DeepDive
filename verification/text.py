"""Text normalization helpers shared by the deterministic verification checks."""

import re
import unicodedata

_QUOTES = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"',
           "„": '"', "′": "'", "″": '"'}
_DASHES = {"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"}
_STOP = set("""a an and are as at be been but by can could did do does for from had has have how in into is it
its may might more most must not of on or our over such than that the their them then there these they this
those to under was were what when where which while who will with would also about after before between both
each other per via""".split())


def normalize(text: str) -> str:
    """Strict normalization: unicode, quote/dash forms, line-break hyphenation, whitespace, case."""
    t = unicodedata.normalize("NFKC", text or "")
    # invisible formatting characters (zero-width space/joiners, invisible separators, soft hyphen, BOM)
    # carry no words; removing them never changes the substantive text
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")
    for src, dst in {**_QUOTES, **_DASHES}.items():
        t = t.replace(src, dst)
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)   # re-join words hyphenated across lines
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def strip_outer(text: str) -> str:
    return normalize(text).strip(" \"'`.,;:")


def numbers(text: str) -> set:
    """Numeric tokens (integers, decimals, percentages, years), normalized."""
    found = re.findall(r"\d+(?:[.,]\d+)*", normalize(text))
    return {f.replace(",", "") for f in found}


def stem(word: str) -> str:
    """Tiny deterministic stemmer: strip one common suffix, then keep 5 characters."""
    for suf in ("ing", "ed", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)]
            break
    return word[:5]


def content_stems(text: str) -> set:
    words = re.findall(r"[a-z][a-z\-]{3,}", normalize(text))
    return {stem(w) for w in words if w not in _STOP}


# Explicit negators and privatives. A negator applies to the next few content words (its scope).
_NEGATORS = {"not", "no", "never", "without", "none", "neither", "nor", "cannot", "lacks", "lack", "lacking",
             "absent", "fails", "failed"}
NEGATION_WINDOW = 3


def _is_negator(token: str) -> bool:
    return token in _NEGATORS or token.endswith("n't")


def negation_spans(text: str) -> list:
    """One (head, stems) pair per negator: the content stems in its scope, head = the first of them.

    Forward scope is the next NEGATION_WINDOW content words up to the clause boundary; a
    clause-final negator ("..., which the encoder lacks.") scopes the preceding words instead.
    """
    tokens = re.findall(r"[a-z][a-z'\-]*|[,;.:!?()]", normalize(text))
    spans = []

    def collect(seq):
        found = []
        for tok in seq:
            if _is_negator(tok) or tok in ",;.:!?()":   # scope ends at the clause boundary
                break
            if len(tok) >= 4 and tok not in _STOP:
                found.append(stem(tok))
                if len(found) == NEGATION_WINDOW:
                    break
        return found

    for i, tok in enumerate(tokens):
        if tok == "never" and tokens[i + 1:i + 2] == ["before"]:
            continue   # idiom ("like never before") intensifies; it negates nothing in the claim
        if _is_negator(tok):
            found = collect(tokens[i + 1:]) or collect(reversed(tokens[:i]))
            if found:
                spans.append((found[0], set(found)))
    return spans


def negation_scope(text: str) -> set:
    """Content stems that fall inside the scope of any negator."""
    return set().union(*(stems for _, stems in negation_spans(text)))


def negated(text: str) -> bool:
    """True if the text contains any explicit negator (kept for callers that need a coarse signal)."""
    return any(_is_negator(t) for t in re.findall(r"[a-z][a-z'\-]*", normalize(text)))


def negation_conflict(claim: str, quote: str):
    """Return a reason if negation changes the claim's proposition between claim and quote, else None.

    1. The quote negates content the claim asserts            -> "negation_mismatch"
       (e.g. claim "holds ISO 27001 certification" vs quote "does not hold ISO 27001 certification").
    2. The claim negates content the quote never negates      -> "claim_negation_not_supported"
       (a negated proposition needs explicitly negated support; when unsure, prefer insufficient).
    Negations in the quote about words that are not in the claim (e.g. "unlike RNNs, which cannot
    be parallelised") are ignored. A quote negation whose head the claim also negates ("there is
    no mechanism to guarantee that different heads ..." for a claim "has no mechanism to guarantee
    ... different heads") negates the same proposition, so its other scoped words are not a mismatch.
    """
    claim_stems = content_stems(claim)
    c_spans = negation_spans(claim)
    c_scope = set().union(*(stems for _, stems in c_spans))
    c_heads = {head for head, _ in c_spans}
    q_spans = negation_spans(quote)
    q_scope = set().union(*(stems for _, stems in q_spans))
    q_unaligned = set().union(*(stems for head, stems in q_spans if head not in c_heads))
    if (q_unaligned - c_scope) & claim_stems:
        return "negation_mismatch"
    if c_scope and not (c_scope & q_scope):
        return "claim_negation_not_supported"
    return None
