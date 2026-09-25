"""Tiny shared edit-distance helper.

Used by both the STT corrector (``src.transcribe``) and fuzzy entity
lookup (``src.wiki_index``) so the two agree on the exact same metric.
"""
from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    """Pure-Python Levenshtein distance — fine for short tokens."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    current[j - 1] + 1,            # insertion
                    prev[j] + 1,                   # deletion
                    prev[j - 1] + (ca != cb),      # substitution
                )
            )
        prev = current
    return prev[-1]
