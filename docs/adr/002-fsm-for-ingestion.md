# 002 — Finite State Machine for ingestion

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** @heitor-am

## Context

Ingestion is a multi-stage pipeline (`fetch → parse → chunk → embed → index`) where:

- Each stage can fail in a recoverable way (rate limit, network blip) or unrecoverable way (404, malformed body).
- Jobs need to survive process restarts so we can show timelines in `GET /ingest/jobs/{id}`.
- An operator should be able to inspect *where* a job is at any point, not just "running" / "done".
- Invalid transitions (e.g. `pending → indexed` skipping the middle) must be impossible by construction, not just by code review.

A free-form Python class with `self.state = "..."` strings would let any caller set any state at any time — exactly the bug surface we want to eliminate.

## Decision

Use the [`transitions`](https://github.com/pytransitions/transitions) library. Declarative states + transitions:

```python
class JobState(StrEnum):
    PENDING = "pending"; FETCHING = "fetching"; PARSING = "parsing"
    CHUNKING = "chunking"; EMBEDDING = "embedding"; INDEXING = "indexing"
    COMPLETED = "completed"; FAILED = "failed"

TRANSITIONS = [
    {"trigger": "start_fetch", "source": "pending", "dest": "fetching"},
    ...
    {"trigger": "fail", "source": [<all non-terminal>], "dest": "failed"},
    {"trigger": "retry", "source": "failed", "dest": "pending"},
]
```

Invalid transitions raise `MachineError` automatically. `transitions` also ships a Graphviz exporter — `make diagram-states` regenerates `docs/diagrams/ingestion-fsm.png` from the same declaration.

## Consequences

**Positive:**
- Invalid transitions impossible. The test suite asserts every legal edge and `pytest.raises(MachineError)` for representative illegal edges.
- The state diagram (PNG) is the source of truth, regenerated from code — never drifts.
- Jobs persist with a transition log (`job_transitions` table), so the API can render the full timeline.
- `failed` is a *retomable* terminal — `trigger: retry` cycles back to `pending` without inventing a new state.

**Negative:**
- One more dependency.
- Graph rendering needs the system `graphviz` binary in the dev container, not just the Python package.

**Trade-offs accepted:**
- We don't need a distributed FSM (no Temporal, no AWS Step Functions). The workload is single-machine; `transitions` is enough.

## Alternatives considered

- **Hand-rolled `state` string + manual checks** — rejected: invalid transitions are caught by humans, not the type system.
- **Temporal / Cadence** — overkill for the workload (one operator, ~50 jobs/day in the demo).
- **Celery with task chains** — solves orchestration but not "show me where this job is now"; we'd still need a state model on top.

## References

- `app/ingestion/state.py`
- `tests/ingestion/test_state_machine.py`
- transitions library: https://github.com/pytransitions/transitions
