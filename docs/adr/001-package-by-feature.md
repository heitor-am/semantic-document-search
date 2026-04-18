# 001 — Package-by-feature, not by-layer

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** @heitor-am

## Context

This project has three distinct pipelines (ingestion, retrieval, evaluation) with different lifecycles, dependencies, and rates of change. The companion repo `virtual-library-api` (Q1) uses package-by-layer (`routers/`, `services/`, `repositories/`) because it's a single-domain CRUD where the layers are the natural boundaries.

The same shape would scatter each pipeline across four directories here, so a change to "how ingestion works" would touch four siblings of unrelated concerns.

## Decision

Top-level packages mirror the bounded context, not the layer:

```
app/
├── ingestion/      # FSM, parser, chunker, indexer, models, repository, router
├── retrieval/      # pipeline, stages/, service, router
├── evaluation/     # runner, metrics, schemas
└── shared/         # genuinely cross-cutting: ai/, qdrant/, db/, api/, core/
```

Inside each feature package, internal layering is allowed (`router.py`, `service.py`, `repository.py`) but **never re-exported across feature boundaries** — `retrieval` doesn't import from `ingestion.service`, only from `shared/`.

## Consequences

**Positive:**
- One pipeline, one folder. The radius of a change is obvious from the path.
- Cross-cutting code stays small. `app/shared/` only grows when something is *actually* shared by ≥2 features.
- New contributors onboard per pipeline, not per layer.

**Negative:**
- Some duplication between pipelines (each has its own `schemas.py`, its own `router.py`). Acceptable — the schemas have different shapes and lifecycles.
- The boundary requires discipline: it's tempting to reach into `ingestion.repository` from `retrieval` to "save a query." Linted via the import-graph review during PR.

**Trade-offs accepted:**
- The Q1 / Q3 portfolio repos diverge in structure. That's the point — picking the same shape for both would be cargo-culting one of them.

## Alternatives considered

- **Package-by-layer (Q1 style)** — rejected: scatters a single pipeline change across `routers/ingestion.py`, `services/ingestion.py`, etc.
- **Vertical slices with co-located tests** — partially adopted (`tests/ingestion/`, `tests/retrieval/`) but kept a parallel `tests/` tree because pytest discovery + coverage tooling are simpler that way.

## References

- Q1 companion (package-by-layer): https://github.com/heitor-am/virtual-library-api
- Vaughn Vernon, *Implementing Domain-Driven Design*, ch. 9 ("Modules")
