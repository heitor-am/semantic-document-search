# 009 — SQLite + transition log for job persistence

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** @heitor-am

## Context

Ingestion jobs need to survive process restarts so the API can answer `GET /ingest/jobs/{id}` after a redeploy or a Fly auto-stop. We also want the *transition history* — not just "current state", but "what happened, when". That powers the UI in the demo notebook and the timeline in the response payload.

Storage candidates:

- **In-memory dict** — fails on restart.
- **SQLite** — file-based, transactional, zero ops, async via `aiosqlite`.
- **Postgres** — proper RDBMS, but overkill for one operator; needs a separate Fly resource.
- **Redis** — fast, but everything we'd persist is already structured (job + transitions); Redis would be a glorified KV cache here.

The workload doesn't justify multi-worker or distributed coordination — there's exactly one uvicorn process serving one operator.

## Decision

SQLite via SQLAlchemy 2.0 (async) + Alembic migrations. Two tables:

```
ingest_jobs        (job_id PK, source_url, state, error, created_at, updated_at)
job_transitions    (id PK, job_id FK, from_state, to_state, at)
```

The DB file lives at `/data/jobs.db` on Fly.io, mounted from a persistent volume (`sds_data`). Local dev uses `./jobs.db`.

The transition log is append-only — every successful FSM trigger writes one row before the new `state` is persisted on the parent. Failed transitions don't get a row (the FSM raises before the commit).

## Consequences

**Positive:**
- Zero new infrastructure. One process, one file, one Fly volume.
- Same DB stack as the Q1 portfolio repo — no context switch for the reader.
- Transition history is queryable, not derived from logs. `GET /ingest/jobs/{id}` serves it in O(1) per job.
- Alembic migrations make schema changes safe to roll out (CI runs them on every deploy).

**Negative:**
- One-writer constraint. Multi-worker scaling would require a real DB.
- Fly volumes are AZ-pinned; the machine has to come back up in the same region. Acceptable — `primary_region = "gru"` is set in `fly.toml`.

**Trade-offs accepted:**
- Multi-tenancy and authentication are explicitly out of scope (PRD § 2.4). When that changes, SQLite gets replaced — the `ingest_jobs` schema is small enough that a Postgres migration is a script, not a project.

## Alternatives considered

- **In-memory only** — fails on every restart. Disqualifying.
- **Postgres on Fly** — adds a managed-DB resource, monthly cost, backup story. Buys nothing at one-operator scale.
- **Redis** — cache-shaped, not queue-shaped. The transition history is naturally relational.
- **A queue (SQS / Celery / Arq)** — solves task distribution we don't need.

## References

- `app/ingestion/models.py`
- `app/ingestion/repository.py`
- `alembic/versions/`
- `fly.toml` (`[[mounts]]` block)
