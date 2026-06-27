"""
Critic — judges whether a candidate chain reproduces the analyst's revealed payload.

Problem 3 from the spec: exact match is too strict on text/bytecode. The critic supports
three scorers, cheapest first:
  exact        — byte-for-byte equality (the default; right for binary/network captures)
  token        — fraction of matching bytes (tolerant to a few wrong bytes)
  levenshtein  — 1 - edit_distance/max_len (tolerant to insert/delete, e.g. recompressed text)

A chain is accepted only if it passes the threshold on EVERY demo. At LLM scale this becomes
a learned "is-this-revealed?" judge; the interface (score a candidate against the goal) is
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


def _token_score(a: bytes, b: bytes) -> float:
    if not b:
        return 1.0 if not a else 0.0
    if len(a) != len(b):
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(b)


def _levenshtein_score(a: bytes, b: bytes) -> float:
    if not a and not b:
        return 1.0
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return 1.0 - prev[m] / max(n, m, 1)


@dataclass
class Critic:
    mode: str = "exact"          # "exact" | "token" | "levenshtein"
    threshold: float = 0.999

    def score(self, output: bytes, revealed: bytes) -> float:
        if self.mode == "exact":
            return 1.0 if output == revealed else 0.0
        if self.mode == "token":
            return _token_score(output, revealed)
        if self.mode == "levenshtein":
            return _levenshtein_score(output, revealed)
        raise ValueError(f"unknown critic mode: {self.mode}")

    def accepts(self, output: bytes, revealed: bytes) -> bool:
        return output is not None and self.score(output, revealed) >= self.threshold
