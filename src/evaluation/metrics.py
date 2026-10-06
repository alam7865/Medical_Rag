"""Ranking metrics with binary relevance.

`relevance` is a list of 0/1 flags for the ranked results (position 0 = rank 1).
`n_relevant` is how many relevant documents exist in the whole corpus for this
query (needed for the ideal DCG).

Recall@K follows this project's definition: did ANY correct document appear in
the top K (a.k.a. hit rate / success@K). This is the usual definition for QA
retrieval where one answer is sufficient, and it is not inflated by duplicate
copies of the answer in the corpus.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be a positive integer")


def recall_at_k(relevance: Sequence[int], k: int) -> float:
    _check_k(k)
    return 1.0 if any(relevance[:k]) else 0.0


def precision_at_k(relevance: Sequence[int], k: int) -> float:
    """Relevant results in the top K divided by K (missing positions count as non-relevant)."""
    _check_k(k)
    return sum(1 for r in relevance[:k] if r) / k


def reciprocal_rank(relevance: Sequence[int]) -> float:
    for i, r in enumerate(relevance):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def first_relevant_rank(relevance: Sequence[int]) -> int | None:
    for i, r in enumerate(relevance):
        if r:
            return i + 1
    return None


def dcg_at_k(relevance: Sequence[int], k: int) -> float:
    return sum((1.0 if r else 0.0) / math.log2(i + 2) for i, r in enumerate(relevance[:k]))


def ndcg_at_k(relevance: Sequence[int], k: int, n_relevant: int | None = None) -> float:
    """NDCG@K. If `n_relevant` is None it is taken as the number of relevant results retrieved."""
    _check_k(k)
    if n_relevant is None:
        n_relevant = sum(1 for r in relevance if r)
    ideal_hits = min(n_relevant, k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg_at_k(relevance, k) / idcg


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
