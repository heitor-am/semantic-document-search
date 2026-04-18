# 012 — Qdrant collection versioning (`documents_{model}_v{n}`)

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** @heitor-am

## Context

Two failure modes make a "single, evergreen collection name" risky:

1. **Embedding model swap.** `bge-m3` (1024-d) → `text-embedding-3-small` (1536-d) means the collection's vector dimension changes. Mixing dims in one collection is impossible by construction in Qdrant; mixing the same dim from different *models* is worse — silently bad search quality, no error.
2. **Schema breaking changes.** Adding a new sparse vector (BM25), changing the named-vector layout, or flipping `Modifier.IDF` aren't transparent — old points encoded under the previous schema still match queries against the new one, but with subtly wrong scores.

We need a way to swap schemas / models without contaminating the live collection, and without "delete-and-recreate-and-pray" downtime.

## Decision

Collection name encodes the embedding model + a schema version:

```
documents_bge-m3_v2
         │       │
         │       └── increment when the schema (vectors, modifiers, payload contract) breaks
         └────────── identifies the embedding model
```

The version is wired through:

- `QDRANT_COLLECTION_VERSION` env var (`v1`, `v2`, ...)
- `OPENROUTER_EMBEDDING_MODEL` env var, sanitised into the name
- `app/shared/qdrant/collections.py` derives the name from both

To roll a new version:

1. Bump the env var (`v2` → `v3`) in `.env` and `fly.toml`.
2. Boot the app — `ensure_collection` creates the new collection on first ingest.
3. Re-ingest the corpus (the `make seed` script is idempotent and points-aware).
4. Once verified, delete the old collection.

No live downtime — the old version keeps serving until the new one is verified.

## Consequences

**Positive:**
- Embedding-model migrations are *possible*, not "well, we'd have to take down search."
- Mistakes are visible: a stale `QDRANT_COLLECTION_VERSION` would point at an empty collection on first deploy, surfacing instantly via the readiness probe / health check.
- The collection name self-documents what's inside.

**Negative:**
- Migrations cost a re-ingest. ~5 minutes for the 50-article seed corpus; would be hours for a real corpus. Acceptable at this scale; large corpora would warrant a parallel-ingest script.
- The version isn't auto-derived from the schema. Someone has to remember to bump it when changing `vectors_config`. Mitigated by a comment at the top of `collections.py` calling this out, plus the BM25-introduction migration (Etapa 9) actually used the bump.

**Trade-offs accepted:**
- We don't attempt online dual-write or shadow-read during migration. The corpus is small enough to re-ingest within the deploy window; the complexity of dual-write isn't paid back.

## Alternatives considered

- **Single evergreen collection** — works until the first schema change, then becomes a poison pill.
- **Aliases** (Qdrant has them) — would let us atomically switch traffic between collection versions. Worth revisiting if we add online-reindex; not needed for a re-ingest-and-cut workflow.

## References

- `app/shared/qdrant/collections.py`
- `.env.example` (`QDRANT_COLLECTION_VERSION`)
- `fly.toml` (`[env]` block)
