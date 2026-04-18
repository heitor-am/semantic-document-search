# 004 — Qdrant: named vectors + sparse with `Modifier.IDF`

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** @heitor-am

## Context

The collection has to hold both a dense vector (bge-m3, 1024-d cosine) and a BM25 sparse vector per chunk. Two natural shapes:

1. Two separate collections — one dense, one sparse — joined by `chunk_id` at query time
2. One collection with two named vectors per point

Querying separately doubles the round-trips and forces client-side fusion. Qdrant's Query API can fuse server-side via Reciprocal Rank Fusion *if* both vectors live on the same point.

For the sparse vector, Qdrant offers `Modifier.IDF` on the vector config — the server applies IDF at query time. If we apply it client-side instead, every collection re-index needs us to recompute the IDF table.

## Decision

One collection (`documents_bge-m3_v2`) with two named vectors:

```python
vectors_config = {
    "dense": VectorParams(size=1024, distance=Distance.COSINE),
}
sparse_vectors_config = {
    "bm25": SparseVectorParams(modifier=Modifier.IDF),
}
```

Each `VectorPoint` carries:
```python
{
    "id": chunk_id,
    "vectors": {
        "dense": [1024 floats],
        "bm25": SparseValue(indices=[...], values=[...]),  # TF-only, pre-IDF
    },
    "payload": {...},
}
```

Sparse vectors with no tokens (e.g., chunks of pure punctuation) are *omitted* from the point — Qdrant rejects empty sparse vectors.

## Consequences

**Positive:**
- One round-trip for hybrid search via Query API + RRF (see ADR-007).
- IDF lives where the corpus statistics live (the server). Re-ingestion automatically updates the term frequencies the IDF computes against.
- Indexing one point with both vectors is atomic — no risk of dense-without-sparse drift.

**Negative:**
- Coupled lifecycle: dropping the sparse vector means a collection migration. Mitigated by ADR-012 (collection versioning).
- `qdrant-client`'s typed sparse vector helpers feel less mature than the dense ones.

**Trade-offs accepted:**
- The sparse encoder still runs in the app process (FastEmbed, see ADR-006). Qdrant's IDF only operates on the values *we* upload — it doesn't tokenize for us.

## Alternatives considered

- **Two separate collections** — rejected: client-side fusion + double round-trips for every query.
- **`Modifier.NONE` + client-side IDF** — rejected: we'd recompute the IDF table on every re-ingest, and the math has to stay byte-identical to whatever the client expects.

## References

- `app/shared/qdrant/collections.py`
- `app/shared/qdrant/repository.py`
- Qdrant docs — Hybrid search: https://qdrant.tech/documentation/concepts/hybrid-queries/
