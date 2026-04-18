"""Standard IR metrics for evaluating retrieval quality.

Inputs are lists of "document identifiers" — the concrete implementation
uses `source_url` strings so golden queries can declare expected URLs
without knowing anything about chunk-level ids. Metrics here don't care
about the identifier type; they just compare retrieved vs relevant sets.

References:
    - Manning, Raghavan, Schütze, *Introduction to Information Retrieval*.
    - Järvelin & Kekäläinen (2002), "Cumulated Gain-Based Evaluation".
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant docs that appear in the top-k retrieved.

    recall_at_k([a, b, c], {a, d}, k=2) -> 0.5  (a is there; d isn't)
    recall_at_k([a, b], {}, k=2) -> 1.0  (vacuously — no relevant to miss)
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0
    top = set(retrieved[:k])
    return len(top & relevant_set) / len(relevant_set)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved that are relevant.

    Complements recall: precision answers "of what I returned, how much
    was right?" P@1 is the strictest — "did the FIRST result answer the
    query?" — and where reranking pays its keep most.

    precision_at_k([a, b, c], {a}, k=3) -> 1/3
    precision_at_k([a, b], {}, k=2) -> 0.0  (no relevant items defined)
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for doc in top if doc in relevant_set) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant doc in `retrieved`.

    Returns 0.0 if none of the retrieved items is relevant (used verbatim
    as the per-query contribution to MRR).
    """
    relevant_set = set(relevant)
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k, with binary relevance.

    DCG rewards putting relevant items at the top: each relevant hit at
    rank i contributes 1 / log2(i + 1). NDCG normalises against the ideal
    ordering (all relevant items packed at the top), so the output is
    always in [0, 1].

    Binary relevance is used rather than graded because the golden set
    declares per-query relevance as "these URLs should appear", not with
    a continuous score. Graded NDCG is future work.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0

    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, doc in enumerate(retrieved[:k], start=1)
        if doc in relevant_set
    )
    # Ideal DCG: every relevant item in the first min(|relevant|, k) ranks.
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean, safe on empty sequences."""
    xs = list(values)
    return sum(xs) / len(xs) if xs else 0.0
