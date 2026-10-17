"""
Deterministic filter for evidence-absence META-claims ("X is not mentioned in the evidence").

These describe the retrieved material, not the world, so they are never research findings.
A claim is filtered only when it combines an absence verb with an explicit reference to the
evidence/documents/sources; genuine negative facts ("the supplier has no outstanding sanctions")
do not refer to the evidence and are kept.
"""

import re

from verification.text import normalize

_VERB = (r"(?:mentioned|specified|discussed|provided|described|stated|found|addressed|covered|available|given|"
         r"present|included|detailed|reported|identified|listed|explained|outlined)")
_REF = (r"(?:the\s+)?(?:provided\s+|retrieved\s+|available\s+|given\s+)?"
        r"(?:evidence|documents?|sources?|passages?|texts?|context|excerpts?|materials?|search results)")
_PATTERNS = [
    # "... is not specified in the evidence", "... are not discussed in the provided documents"
    re.compile(rf"\b(?:is|are|was|were)\s+not\s+{_VERB}\s+(?:in|by|within|across)\s+{_REF}\b"),
    # "No standard payment terms ... are mentioned in the evidence"
    re.compile(rf"^no\b.*\b(?:is|are|was|were)\s+{_VERB}\s+(?:in|by|within|across)\s+{_REF}\b"),
    # "The evidence does not mention / specify ..."
    re.compile(rf"\b{_REF}\s+(?:does|do|did)\s*(?:not|n't)\s+(?:mention|specify|discuss|provide|describe|state|"
               rf"address|cover|include|contain|detail|indicate|say)\b"),
    # "There is no information/mention ... in the evidence"
    re.compile(rf"\bno\s+(?:information|details|data|mention|indication)\b.*\b(?:in|within)\s+{_REF}\b"),
]


def is_meta_claim(text: str) -> bool:
    t = normalize(text)
    return any(p.search(t) for p in _PATTERNS)
