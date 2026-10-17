"""
Deterministic evidence and citation integrity checks.

A model's judgment that a passage supports a claim is accepted only if:
1. the evidence ID exists and was actually shown to the verifier for that claim;
2. the quoted text appears verbatim (after strict normalization) in that evidence;
3. the quote is about the claim: every number in the claim appears in the quote
   (for "supports"), the quote does not negate what the claim asserts (for "supports"),
   and enough of the claim's content words appear in it. A numeric claim whose quote falls
   short of the overlap threshold can still pass the narrow number-anchored check below.
A quote merely existing in a document, or merely sharing the claim's words and numbers,
is therefore not sufficient. Remaining semantic judgment rests with the verifier model and
is measured in the live evaluation.
"""

from typing import Optional

from schemas.research import Citation, ClaimVerification, Evidence, ReportSentence
import re
import unicodedata

from verification.grounding import grounding_issue
from verification.text import _STOP, content_stems, negation_conflict, normalize, numbers, stem, strip_outer

MIN_QUOTE_CHARS = 12
MIN_QUOTE_WORDS = 3
MIN_CONTENT_OVERLAP = 0.34
NUMBER_ANCHOR_WINDOW = 5      # tokens either side of a number that describe what it counts
MIN_NUMBER_ANCHOR_TERMS = 2


def check_quote(quote: str, evidence_text: str) -> Optional[str]:
    """Return a rejection reason, or None if the quote is verbatim evidence text."""
    q = strip_outer(quote)
    if not q:
        return "empty_quote"
    if len(q) < MIN_QUOTE_CHARS or len(q.split()) < MIN_QUOTE_WORDS:
        return "quote_too_short"
    if q not in normalize(evidence_text):
        return "quote_not_in_evidence"
    return None


def check_relevance(claim_text: str, quote: str, stance: str, evidence_text: Optional[str] = None) -> Optional[str]:
    """Return a rejection reason if the (verbatim) quote does not bear on the claim."""
    if stance == "supports":
        missing = numbers(claim_text) - numbers(quote)
        if missing:
            return "claim_numbers_missing_from_quote:" + ",".join(sorted(missing))
        reason = negation_conflict(claim_text, quote)
        if reason:
            # negation changes the claim's proposition: not genuine support
            return reason
    claim_stems = content_stems(claim_text)
    if claim_stems:
        overlap = len(claim_stems & content_stems(quote)) / len(claim_stems)
        if overlap < MIN_CONTENT_OVERLAP:
            if stance == "supports" and evidence_text and number_anchored(claim_text, quote, evidence_text):
                return None
            return f"quote_not_about_claim(overlap={overlap:.2f})"
    return None


def _number_contexts(text: str) -> dict:
    """For each number in the text, the content stems within NUMBER_ANCHOR_WINDOW tokens of it."""
    tokens = re.findall(r"\d+(?:[.,]\d+)*|[a-z][a-z\-]*", normalize(text))
    out = {}
    for i, tok in enumerate(tokens):
        if tok[0].isdigit():
            near = tokens[max(0, i - NUMBER_ANCHOR_WINDOW):i] + tokens[i + 1:i + 1 + NUMBER_ANCHOR_WINDOW]
            ctx = {stem(w) for w in near if not w[0].isdigit() and len(w) >= 4 and w not in _STOP}
            out.setdefault(tok.replace(",", ""), set()).update(ctx)
    return out


def _named_entities(text: str) -> set:
    """Stems of capitalised words after the first word (e.g. "Transformer", "BERT")."""
    words = re.findall(r"[A-Za-z][A-Za-z\-]+", unicodedata.normalize("NFKC", text or ""))
    return {stem(w.lower()) for w in words[1:] if w[0].isupper() and len(w) >= 3}


def number_anchored(claim_text: str, quote: str, evidence_text: str) -> bool:
    """Narrow relevance path for a numeric fact stated tersely ("we employ h = 8 parallel attention
    layers, or heads" for "The base model of the Transformer architecture uses 8 attention heads").

    Accepted only if ALL hold (numbers already matched and negation already checked by the caller):
    1. every claim number is described in the quote by >= MIN_NUMBER_ANCHOR_TERMS of the same
       content words that describe it in the claim (same quantity, same unit);
    2. the full evidence passage the quote comes from covers >= MIN_CONTENT_OVERLAP of the
       claim's content words (the subject is the passage's subject);
    3. every named entity in the claim appears in that passage (no "BERT" from a Transformer passage).
    """
    claim_ctx, quote_ctx = _number_contexts(claim_text), _number_contexts(quote)
    if not claim_ctx:
        return False
    for num, ctx in claim_ctx.items():
        if len(ctx & quote_ctx.get(num, set())) < MIN_NUMBER_ANCHOR_TERMS:
            return False
    claim_stems, passage_stems = content_stems(claim_text), content_stems(evidence_text)
    if len(claim_stems & passage_stems) / len(claim_stems) < MIN_CONTENT_OVERLAP:
        return False
    return _named_entities(claim_text) <= passage_stems


def independence_key(ev: Evidence) -> str:
    """Distinct documents, or distinct web sites, count as independent sources."""
    loc = ev.locator
    if loc.source_type == "web":
        from urllib.parse import urlparse

        host = urlparse(loc.url or loc.source).netloc.lower()
        return "web:" + (host[4:] if host.startswith("www.") else host)
    return "doc:" + loc.source.lower()


def build_citations(verifications: dict, evidence: dict) -> list:
    """Citations are derived only from supported claims and their accepted evidence."""
    out = []
    for cid, v in verifications.items():
        if v.status != "supported":
            continue
        # presentation order: stronger sources first (quality never decides support)
        ordered = sorted(v.supporting_evidence_ids,
                         key=lambda e: -(evidence[e].quality.score if evidence[e].quality else 0.0))
        for eid in ordered:
            ev = evidence[eid]
            out.append(Citation(marker=cid, claim_id=cid, evidence_id=eid, source=ev.locator.source,
                                page=ev.locator.page, url=ev.locator.url, quote=v.accepted_quotes.get(eid, "")))
    return out


def validate_citations(citations: list, verifications: dict, evidence: dict) -> list:
    """Return integrity issues; an empty list means every citation is valid."""
    issues = []
    for c in citations:
        v: Optional[ClaimVerification] = verifications.get(c.claim_id)
        ev: Optional[Evidence] = evidence.get(c.evidence_id)
        if v is None:
            issues.append(f"{c.marker}: unknown claim {c.claim_id}")
            continue
        if v.status != "supported":
            issues.append(f"{c.marker}: claim {c.claim_id} is {v.status}, not supported")
        if ev is None:
            issues.append(f"{c.marker}: unknown evidence {c.evidence_id}")
            continue
        if c.evidence_id not in v.supporting_evidence_ids:
            issues.append(f"{c.marker}: evidence {c.evidence_id} does not support claim {c.claim_id}")
        if ev.locator.source_type == "web" and not (c.url or "").lower().startswith(("http://", "https://")):
            issues.append(f"{c.marker}: web citation {c.evidence_id} has no source URL")
        if c.source != ev.locator.source or c.page != ev.locator.page:
            issues.append(f"{c.marker}: cited source {c.source} p.{c.page} does not match evidence "
                          f"{ev.locator.source} p.{ev.locator.page}")
        if check_quote(c.quote, ev.text):
            issues.append(f"{c.marker}: cited quote is not verbatim in {c.evidence_id}")
    return issues


# Claim-reference markers DeepDive accepts inside sentence text: (C3), [C3], (C1, C2), [C1; C2], `[C3]`,
# and prose references "claim C7", "claim ID C7", "claim IDs C2, C3, and C6". A bare "C3" with no
# bracket or "claim" keyword is not a reference: its digits stay subject to the number check.
_ID = r"C\d+"
_BRACKET_MARKER = rf"`?[\(\[]\s*({_ID}(?:\s*[,;]\s*{_ID})*)\s*[\)\]]`?"
_PROSE_MARKER = rf"\b(?i:claims?(?:\s+ids?)?)\s+({_ID}(?:\s*(?:,\s*and|,|;|and)\s*{_ID})*)\b"
CITATION_MARKER = re.compile(rf"{_BRACKET_MARKER}|{_PROSE_MARKER}")


def citation_markers(text: str) -> list:
    """Claim IDs referenced by structural or prose markers in the text, in order."""
    return [i for m in CITATION_MARKER.finditer(text) for i in re.findall(_ID, m.group(1) or m.group(2))]


def strip_citation_markers(text: str) -> str:
    """Text for FACTUAL checks only: valid marker syntax removed; the sentence itself is not changed."""
    return CITATION_MARKER.sub(" ", text)


def validate_sentences(sentences: list, verifications: dict, claims: dict) -> tuple:
    """Keep only report sentences fully backed by supported claims. Returns (kept, removed)."""
    kept, removed = [], []
    for s in sentences:
        sent = s if isinstance(s, ReportSentence) else ReportSentence(**s)
        ids = [i for i in sent.claim_ids if i]
        markers = citation_markers(sent.text)
        reason = None
        if not ids:
            reason = "no_claim_cited"
        else:
            bad = [i for i in dict.fromkeys(ids + markers)
                   if i not in verifications or verifications[i].status != "supported"]
            stray = [i for i in markers if i not in ids]
            if bad:
                reason = "cites_unsupported_or_unknown_claims:" + ",".join(bad)
            elif stray:
                reason = "inline_marker_not_in_cited_claims:" + ",".join(stray)
            else:
                allowed = set().union(*(numbers(claims[i].text) for i in ids))
                extra = numbers(strip_citation_markers(sent.text)) - allowed
                if extra:
                    reason = "sentence_numbers_not_in_cited_claims:" + ",".join(sorted(extra))
                else:
                    # wording may not assert more than the cited claims (rankings, causes, risks, ...)
                    reason = grounding_issue(strip_citation_markers(sent.text), [claims[i].text for i in ids])
        if reason:
            removed.append({"text": sent.text, "claim_ids": ids, "reason": reason})
        else:
            kept.append(sent)
    return kept, removed
