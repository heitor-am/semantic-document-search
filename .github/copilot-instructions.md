# Copilot Instructions — Semantic Document Search

Context for automated PR review. Focus on what matters for this RAG pipeline.

## Stack

- **Python 3.12**, async throughout (FastAPI + SQLAlchemy 2.0 async + httpx)
- **Pydantic v2** for HTTP contracts, separate from SQLAlchemy models
- **Qdrant** as vector store (Cloud in prod, local Docker in dev)
- **OpenRouter** as unified LLM gateway (`openai` SDK + custom `base_url`)
- **transitions** (Python lib) for ingestion state machine
- **rank-bm25** for sparse retrieval
- **langchain-text-splitters** only for the `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` pair — not LangChain the framework
- **uv** / **Ruff** / **mypy strict** / **pytest** with coverage

## Architecture — package-by-feature

```
app/
├── ingestion/     # docs INTO the store — FSM-driven pipeline
├── retrieval/     # docs OUT of the store — functional Pipeline with stages
├── evaluation/    # golden set + metrics + runner
├── shared/        # genuinely cross-feature (ai, qdrant, db, core, api, schemas)
├── config.py
└── main.py
```

**Not** layered (no `routers/`, `services/`, `repositories/` at top level). Each feature has its own `service.py`, `router.py`, `schemas.py`, etc. See ADR-008.

## Conventions

- **Conventional Commits** for every commit (`feat(scope): ...`, `fix(scope): ...`, `chore:`, etc). Flag commits that don't match.
- **RFC 7807 Problem Details** for every error response with `application/problem+json`.
- **Dependency injection via `Annotated[X, Depends(...)]`** (see `app/shared/api/deps.py`).
- **Ingestion uses FSM** (`transitions`): states `pending → fetching → parsing → chunking → embedding → indexing → completed`, with `failed` retomável via `retry`. Invalid transitions must be impossible by construction.
- **Retrieval uses functional Pipeline**: stages compose, optional stages (rerank, query_rewriter) fall back gracefully on error; they never 5xx.
- **Tests live in `tests/<feature>/`** mirroring `app/<feature>/`. Use `dependency_overrides` for integration tests.
- **Markdown:** use bullet lists or blank lines between consecutive `**Label:**` — CommonMark collapses single newlines.

## What to flag

- Routes doing pipeline work that belongs in `service.py`
- Cross-feature imports that should go through `shared/` (or are signs something isn't actually shared)
- State machine mutations that bypass the `transitions` triggers (e.g. `job.state = ...` direct assignment)
- Pipeline stages that don't respect `optional=True` semantics (failing hard when they should fall back)
- Qdrant calls that bypass the `VectorRepository` protocol (direct `qdrant-client` use outside `shared/qdrant/`)
- Hardcoded collection names (must use `documents_{model}_{version}` pattern)
- Missing idempotency on ingestion (same URL ingested twice should upsert, not duplicate)
- Sync I/O where async is expected
- Hardcoded model names/URLs that should be env-driven

## What to ignore

- Don't suggest Black, Flake8, isort, or Poetry — we use Ruff and uv.
- Don't ask for docstrings on every function; the project prefers self-documenting code with types.
- Don't flag intentional `# type: ignore` or `# noqa` with inline justification.
- Don't suggest renaming features to layer-style conventions.
- Don't complain about Portuguese content in seed data or queries; this is pt-BR-adjacent.

## Priorities

1. **Correctness** — logic errors, wrong HTTP codes, FSM invariants violated
2. **Security** — secrets, injection, unsafe deserialization
3. **Architectural drift** — package-by-feature violations, skipping DI, missing RFC 7807
4. **Test gaps** — untested branches, especially error paths and invalid state transitions
5. **Style nits** — only if they violate ruff config
