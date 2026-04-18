# 007 — Reciprocal Rank Fusion server-side via Qdrant Query API

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** @heitor-am

## Context

For hybrid retrieval, dense scores and sparse (BM25) scores are not on the same scale and don't combine cleanly with arithmetic. Reciprocal Rank Fusion (RRF) sidesteps this by working only with *ranks*:

```
score(d) = Σ_{q ∈ queries} 1 / (k + rank_q(d))   — typically k=60
```

Two ways to compute it:

1. **Client-side fusion** — issue two queries to Qdrant (dense, sparse), merge result lists in Python, sort.
2. **Server-side fusion** — Qdrant Query API takes a `prefetch` list of sub-queries plus a `query=Fusion.RRF` directive, does the fusion in one round-trip.

Client-side fusion was the initial plan from the PRD (using `rank-bm25` in-process). Once BM25 moved into Qdrant as a named sparse vector (ADR-004), keeping fusion client-side meant: two round-trips per query, plus our own RRF implementation maintained next to Qdrant's.

## Decision

Use Qdrant's Query API with `Fusion.RRF` for the `hybrid` strategy:

```python
client.query_points(
    collection_name=collection,
    prefetch=[
        Prefetch(query=dense_vec, using="dense", limit=N),
        Prefetch(query=SparseVector(...), using="bm25", limit=N),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=top_k,
)
```

One round-trip; one source of truth for the fusion math. The `dense_only` strategy uses the same Query API but with a single prefetch and no fusion directive.

## Consequences

**Positive:**
- One network round-trip for hybrid (was two with client-side fusion).
- We delete our RRF implementation. Less code to maintain, less risk of subtle off-by-one in the rank arithmetic.
- The same code path serves all strategies — `build_pipeline(strategy)` swaps the prefetch shape, not the executor.

**Negative:**
- Coupled to a Qdrant feature. If we move to a vector store that doesn't support server-side fusion, we'd reintroduce client-side fusion.
- The Qdrant Python client's `Fusion.RRF` enum has changed name across minor versions — pinned in `pyproject.toml`.

**Trade-offs accepted:**
- The fusion `k` (60) is the Qdrant default — we don't tune it. Tuning would require running the eval harness across a `k` grid; the current eval shows hybrid at parity with dense, so the lever isn't load-bearing.

## Alternatives considered

- **Client-side RRF** — was the initial implementation. Refactored away in PR #13 once named vectors landed.
- **Convex combination of normalized scores** (`α·dense + (1-α)·sparse`) — needs score normalization (which depends on the score distribution per corpus) and an `α` to tune. RRF avoids both.
- **Learned sparse + ColBERT-style late interaction** — bigger lift; not justified at this corpus size.

## References

- `app/retrieval/stages/hybrid.py`
- `app/shared/qdrant/repository.py` (`query_points` wrapper)
- Qdrant Query API: https://qdrant.tech/documentation/concepts/hybrid-queries/
- Cormack et al., *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* (SIGIR 2009)
