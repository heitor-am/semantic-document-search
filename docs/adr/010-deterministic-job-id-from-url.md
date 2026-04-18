# 010 — Deterministic `job_id` from `sha256(url)[:16]`

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** @heitor-am

## Context

`POST /ingest` is exposed externally; clients (humans, scripts, the seed corpus loader) will retry, double-click, lose connections. If `job_id` is a fresh UUID per call, we get:

- Duplicate Qdrant points (one per re-submission)
- Duplicate jobs in the SQLite table
- Confused timelines ("which one is canonical?")

Two approaches to idempotency:

1. **Idempotency-Key header** (RFC 7240-style) — client provides the key, server dedupes for some TTL. Robust but pushes responsibility to the caller.
2. **Server-derived deterministic ID** from the request payload.

For ingest, the *natural* identity of a job is the URL being ingested. Same URL → same job. That makes the URL itself the key.

## Decision

`job_id = sha256(source_url)[:16]`. Computed in `app/ingestion/service.py`:

```python
def deterministic_job_id(source_url: str) -> str:
    return hashlib.sha256(source_url.encode()).hexdigest()[:16]
```

`POST /ingest` first looks up the existing job by this ID:

- **Job exists in a non-terminal state** (`pending`, `fetching`, `parsing`, ...) — return `200 OK` with the in-flight job; don't spawn a duplicate pipeline.
- **Job exists in `completed`** — return `200 OK` with the existing record; don't re-ingest.
- **Job exists in `failed`** — return `200 OK` with the failed record; the operator decides whether to `retry`. (Auto-retry is a footgun.)
- **No job** — create one and start it.

For chunk IDs, the same idea via `uuid5(NAMESPACE_URL, f"{source_url}#{chunk_index}")` — re-ingesting upserts the same Qdrant points instead of duplicating them.

## Consequences

**Positive:**
- The endpoint is genuinely idempotent. Doubled `POST /ingest` is a no-op, observable in the API response.
- Re-ingesting the same URL after a code change *upserts* the chunks (same `chunk_id`s) — we don't accumulate stale points.
- The notebook demo can re-run safely without flooding Qdrant.
- 16 hex chars = 64 bits of entropy. Collision probability is negligible at this scale.

**Negative:**
- Truncating SHA-256 is information loss. Two distinct URLs *could* in principle hash-collide in the first 16 chars. Birthday math: ~5×10⁹ URLs before ~50% collision odds. Not a problem at our scale; would need to revisit at >10⁸ URLs.
- The URL is the identity. URL canonicalization matters — `https://dev.to/foo` and `https://dev.to/foo/` would be different jobs. Acceptable: dev.to URLs come from one source (the API) and are already canonical.

**Trade-offs accepted:**
- We chose URL-as-key over `Idempotency-Key` header. For this project (one user, internal tooling) it's the right ergonomic. A public multi-tenant API would want both — URL for natural dedup, header for caller-driven retry semantics.

## Alternatives considered

- **Fresh UUID per call + dedup at Qdrant level** — pushes the dedup question to the indexer; doesn't solve the duplicate-job-in-SQLite problem.
- **`Idempotency-Key` header** — adds caller burden; overkill for the workload.
- **Full SHA-256 (64 chars)** — eye-bleed in URLs and logs; the truncation gives us all the dedup we need.

## References

- `app/ingestion/service.py` (`deterministic_job_id`)
- `app/ingestion/chunker.py` (`chunk_id` derivation)
- `app/ingestion/router.py` (lookup-then-spawn flow on `POST /ingest`)
