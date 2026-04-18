# PRD — Semantic Document Search

- **Project:** Semantic Document Search
- **Status:** In development — scaffold (`v0.0.1`) live; features landing incrementally
- **Live (scaffold):** https://semantic-document-search.fly.dev

## 1. Context

A production-grade RAG search system over technical articles. The explicit focus is **retrieval quality** — the project goes beyond "index and cosine" to demonstrate informed decisions on parsing, chunking, metadata modeling, hybrid search, reranking, and quantitative evaluation.

It's the third of three sibling portfolio projects and reuses the OpenRouter / tooling / shared-layer pattern established in the [Virtual Library API](https://github.com/heitor-am/virtual-library-api). Where it **diverges** is deliberate — see §4 on architecture.

## 2. Scope

### 2.1 Core requirements

- A corpus of documents (articles / blog posts)
- Generate embeddings with an embedding model
- Persist embeddings in a vector store
- A search function that returns documents ranked by semantic similarity
- Example queries demonstrating the end-to-end flow

### 2.2 RAG-quality differentiators (the core value)

Each decision below is documented as an ADR.

- **Parse to Markdown before chunking** — preserves semantic structure; headers become natural section boundaries
- **Hierarchical chunking** — primary split on `##`/`###` (parent chunks ~1024 tokens), recursive split for oversized parents (child chunks ~256 tokens with 12.5% overlap)
- **Empirically justified chunk size and overlap** — documented with trade-offs
- **Qdrant** as vector store — native metadata filtering, hybrid search, cloud-hosted free tier
- **Rich metadata schema** — source URL, title, section heading path, author, date, tags, `chunk_index`, `parent_chunk_id`, `embedding_model` (for migration safety)
- **Hybrid search** — BM25 + dense fused via Reciprocal Rank Fusion (k=60)
- **Cross-encoder reranking** after retrieval (marked optional in the pipeline — falls back gracefully on failure)
- **Parent-child retrieval** — small chunks for matching precision, parent chunks for context
- **Query rewriting** (optional) — LLM expands the query into variants for recall
- **Evaluation framework** — a golden set of queries with expected documents, recall@k / MRR / NDCG computed by a CLI

### 2.3 Architectural differentiators (raise the bar vs Q1)

- **Package-by-feature** structure: `app/ingestion/`, `app/retrieval/`, `app/evaluation/`, `app/shared/`. Each pipeline self-contained; genuinely cross-cutting code lives in `shared/`. The Q1 project is package-by-layer because it fits a single-domain CRUD; Q3's three-pipeline shape fits package-by-feature better. ADR-008 explains the divergence.
- **Finite State Machine for ingestion** (via the [`transitions`](https://github.com/pytransitions/transitions) library): `pending → fetching → parsing → chunking → embedding → indexing → completed`, with `failed` as a retomable terminal state. Invalid transitions impossible by construction. Graphviz export auto-generated for docs.
- **Functional Pipeline for retrieval**: composable stages (`QueryRewriter`, `DenseSearch`, `SparseSearch`, `RRFFusion`, `Reranker`, `ParentResolver`) with per-stage graceful degradation. Optional stages (rerank, query rewriter) fall back to partial results on error instead of 5xx-ing.
- **Idempotency on ingest** — source URL hashed into a deterministic `job_id`; re-submitting the same URL upserts, never duplicates.
- **Collection versioning** — Qdrant collection name `documents_{model}_v{n}`; swapping the embedding model means a new collection plus a migration, not contaminated dims.
- **Repository pattern for Qdrant** — `VectorRepository` protocol wraps `qdrant-client`; tests mock, and the underlying store is swappable (Milvus / Weaviate).
- **Job persistence in SQLite** — `ingest_jobs` table with a transitions log; jobs survive container restarts. Explicit upgrade path documented in ADR (Redis / Postgres for multi-worker scaling) but not implemented — that would be over-engineering at this workload.
- **Boot-time config validation** — `Settings.model_post_init` checks that Qdrant is reachable; fails fast instead of mid-request.
- **Per-stage structured logging** — `structlog.bind_contextvars(stage=..., job_id=...)` around each transition; job timelines visible in JSON logs and via `/ingest/jobs/{id}`.

### 2.4 Out of scope

- Real-time indexing (batch pipeline is enough for the demo)
- Multi-tenancy
- Authentication
- Multi-worker scaling (SQLite + in-process background tasks — Redis is over-engineering here)
- Advanced observability (OpenTelemetry, Prometheus)

## 3. Stack

| Layer | Choice | Notes |
|---|---|---|
| Web framework | FastAPI | Async native, consistent with Q1 |
| Vector store | Qdrant Cloud (free tier) | 1 GB cluster, metadata filtering, hybrid search |
| LLM gateway | OpenRouter (`openai` SDK + custom `base_url`) | Chat + embeddings, single key, swappable model |
| Chat model | `openai/gpt-4o-mini` | Cheap, adequate for query rewriting |
| Embedding model | `baai/bge-m3` (1024 dims) | Multilingual SOTA open-source |
| Reranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder, same vendor as embeddings |
| Parsing | `markdownify` for HTML → Markdown; `docling` reserved for PDFs | dev.to already serves markdown |
| Chunking | `langchain-text-splitters` (MarkdownHeader + Recursive) | Just the splitters, not LangChain the framework |
| BM25 | `rank-bm25` | Lightweight, fits in-process |
| State machine | `transitions` | Declarative FSM for ingestion; built-in graphviz |
| Job persistence | SQLite + SQLAlchemy 2.0 (async) | Same DB stack as Q1 |
| Evaluation | Custom: recall@k + MRR + NDCG against a YAML golden set | |
| Infrastructure | Docker (multi-stage), Fly.io, GitHub Actions, Dev Container | Same as Q1 |
| Quality | Ruff, mypy strict, pytest, Schemathesis, bandit, pip-audit, pre-commit | Same as Q1 |
| Package manager | uv | Same as Q1 |
| Notebook demo | Jupyter | End-to-end pipeline walkthrough |

## 4. Architecture — package-by-feature

```
app/
├── ingestion/                # docs INTO the store (FSM-driven)
│   ├── state.py              # JobState + transitions Machine
│   ├── models.py             # SQLAlchemy IngestJob + JobTransition
│   ├── repository.py
│   ├── parser.py             # dev.to fetch + normalize markdown
│   ├── chunker.py            # hierarchical MarkdownHeader + Recursive
│   ├── indexer.py            # Qdrant upsert + BM25 index rebuild
│   ├── service.py            # orchestrates the FSM
│   ├── router.py             # POST /ingest, GET /ingest/jobs/{id}
│   └── schemas.py
├── retrieval/                # docs OUT of the store (functional pipeline)
│   ├── pipeline.py           # Pipeline + Stage abstractions
│   ├── context.py            # Context flowing through stages
│   ├── stages/
│   │   ├── query_rewriter.py # optional LLM expansion
│   │   ├── dense.py          # Qdrant dense search
│   │   ├── sparse.py         # BM25 search
│   │   ├── fusion.py         # Reciprocal Rank Fusion
│   │   ├── reranker.py       # optional cross-encoder rerank
│   │   └── parent_child.py   # resolve parents + dedupe
│   ├── service.py
│   ├── router.py             # GET /search
│   └── schemas.py
├── evaluation/               # golden set + metrics CLI
│   ├── queries.yaml
│   ├── metrics.py            # recall@k, MRR, NDCG
│   ├── runner.py             # python -m app.evaluation.runner
│   └── schemas.py
├── shared/                   # genuinely cross-feature
│   ├── ai/{client,embeddings}.py
│   ├── qdrant/{client,collections,repository}.py
│   ├── db/database.py
│   ├── core/{exceptions,logging}.py
│   ├── api/{deps,routers/health}.py
│   └── schemas/problem.py    # RFC 7807
├── config.py                 # pydantic-settings + boot validation
└── main.py
```

**Why package-by-feature here?** A new contributor who wants to understand "how does ingestion work?" opens one folder. Deleting or replacing a feature is a single `rm -rf`. Cross-feature reuse is explicit via `shared/`. Layer-style (Q1) works fine for single-domain CRUD; here there are three distinct pipelines and their code barely overlaps.

## 5. Ingestion pipeline — FSM

```
┌─────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐
│ PENDING │──▶│ FETCHING │──▶│ PARSING │──▶│ CHUNKING │──▶│EMBEDDING │──▶│ INDEXING │──▶│ COMPLETED │
└─────────┘   └─────┬────┘   └────┬────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   └───────────┘
                    │             │              │              │              │
                    └─────────────┴──────────────┴──────────────┴──────────────┘
                                                  ▼
                                             ┌────────┐
                                             │ FAILED │ ─(retry)→ PENDING
                                             └────────┘
```

- Each stage has one happy-path transition and one failure edge
- `FAILED` is retomable — `retry` moves back to `PENDING` preserving the transitions log
- `COMPLETED` is fully terminal
- Transitions record `(state, timestamp, duration_ms)` in both the in-memory `.history` list and the SQLite `job_transitions` table
- The FSM is exportable to PNG via `make diagram-states` using `transitions.extensions.GraphMachine`

## 6. Retrieval pipeline — functional

```
query → (QueryRewriter?) → DenseSearch + SparseSearch → RRFFusion → (Reranker?) → ParentResolver → results
```

Each stage is a class with `async def run(ctx: Context) -> Context` and an `optional: bool`. If an optional stage raises, the pipeline logs and continues with the previous context — the call never returns 5xx because a non-core stage broke. Stages marked as required failing surface as `503 application/problem+json`.

## 7. API

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Submits an ingestion job — returns `202 Accepted` + `job_id` |
| `GET` | `/ingest/jobs/{id}` | Job state + transition timeline |
| `GET` | `/ingest/jobs` | List jobs (optional `?state=` filter) |
| `GET` | `/search` | `?q=&strategy=&top_k=&min_score=&tags=&source_type=&rewrite_query=` |
| `GET` | `/documents/{chunk_id}` | Fetch a stored chunk (primarily for debugging and demos) |
| `GET` | `/collections/stats` | Corpus statistics (total chunks, per-model counts) |
| `GET` | `/health` | Readiness check (DB + Qdrant reachability, version, commit SHA) |

`strategy` for search: `dense`, `bm25`, `hybrid`, `hybrid_rerank` (default).

All errors use RFC 7807 `application/problem+json` with an application `code` (`JOB_NOT_FOUND`, `INVALID_TRANSITION`, `LLM_UNAVAILABLE`, `VECTOR_STORE_UNAVAILABLE`, `VALIDATION_ERROR`).

## 8. Metadata schema on each chunk

```json
{
  "source_url": "https://dev.to/author/post",
  "source_type": "blog_post",
  "title": "Building RAG with Qdrant",
  "author": "jane_doe",
  "published_at": "2025-11-20T14:30:00Z",
  "tags": ["python", "rag", "qdrant"],
  "section_path": ["Introduction", "Why Qdrant"],
  "chunk_index": 3,
  "chunk_total": 12,
  "parent_doc_id": "post_abc123",
  "parent_chunk_id": "post_abc123_section_2",
  "char_count": 2048,
  "token_count": 487,
  "indexed_at": "2026-04-18T10:00:00Z",
  "embedding_model": "baai/bge-m3"
}
```

`section_path` enables breadcrumb-style filtering (e.g. "only introductions"). `parent_chunk_id` powers parent-child retrieval. `embedding_model` means we can detect and re-index stale vectors if the default model changes.

## 9. Evaluation

A hand-curated golden set (`app/evaluation/queries.yaml`) with ~20 queries, each with:

- `query` — the user-style question
- `expected_doc_ids` — the chunks that should appear in top-K
- `expected_tags` — a looser match criterion
- `min_recall_at_5` — the pass threshold

`python -m app.evaluation.runner` runs every query against every strategy (`dense_only`, `bm25_only`, `hybrid_rrf`, `hybrid_rrf_rerank`) and prints a comparative table:

```
Strategy              Recall@5   MRR      NDCG@10   Avg Latency
dense_only            0.65       0.52     0.68      142ms
bm25_only             0.55       0.48     0.61      38ms
hybrid_rrf            0.80       0.71     0.79      186ms
hybrid_rrf_rerank     0.90       0.84     0.88      421ms  ← default
```

The table is snapshotted into the README — rare portfolio signal (most candidates don't measure their RAG).

## 10. Corpus

**Default:** technical posts from [dev.to](https://dev.to) via the public API. The `body_markdown` field is already clean Markdown, the metadata (tags, author, date) is rich, and the content is coherent enough (Python / AI / backend tags) to make evaluation meaningful.

Seeded with 50-100 posts on `python`, `rag`, `ai`, `backend` tags.

## 11. Release plan

- **v0.0.1** — Scaffold: package-by-feature skeleton, shared layer copied from Q1, `/health` live on Fly.io, CI green
- **v0.1.0** — Ingestion MVP: FSM + parser + chunker + Qdrant integration + `POST /ingest`
- **v0.2.0** — Retrieval: hybrid + RRF + rerank + parent-child + `GET /search`
- **v1.0.0** — Production: eval table, ADRs, Mermaid diagrams (architecture + FSM auto-gen + retrieval pipeline), polished README, deployed on Fly.io with Qdrant Cloud

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Qdrant Cloud free tier insufficient | 1 GB easily fits 10k+ chunks at our sizes; fallback to local Qdrant via `docker-compose` documented |
| Reranker model download at deploy time | Pre-download in the Docker build layer |
| Golden set too small to be meaningful | Explicit acknowledgment in ADR-010; upgrade path mentioned (crowdsourced or LLM-as-judge) |
| Hybrid search drift between Qdrant versions | Pinned Qdrant client version; collection config versioned |
| Over-engineering | Explicit no-go list: no Redis, no Celery, no K8s, no OpenTelemetry — workload is single-user demo |

## 13. Portfolio companions

- **[Virtual Library API](https://github.com/heitor-am/virtual-library-api)** — FastAPI + SQLite + OpenRouter (released `v1.0.0`)
- **Python Tutor Chatbot** — Chainlit + LangChain + OpenRouter (planned)
- **Semantic Document Search** (this repo) — FastAPI + Qdrant + OpenRouter with package-by-feature, FSM, and functional pipeline

All three share the same `app/shared/ai/client.py` (OpenRouter wrapper) verbatim. Engineering conventions (Conventional Commits, RFC 7807, Ruff + uv + mypy, Scalar docs, Fly.io deploy) are identical. This project differs on purpose where the domain justifies it — documented in the ADRs.
