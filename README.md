# Semantic Document Search

[![CI](https://github.com/heitor-am/semantic-document-search/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/heitor-am/semantic-document-search/actions/workflows/ci.yml)
[![Deploy](https://github.com/heitor-am/semantic-document-search/actions/workflows/deploy.yml/badge.svg?branch=main)](https://github.com/heitor-am/semantic-document-search/actions/workflows/deploy.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Coverage 84%](https://img.shields.io/badge/coverage-84%25-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Production-grade RAG search over a curated dev.to corpus. Three retrieval strategies (`dense_only`, `hybrid` with RRF, `hybrid_rerank` with cross-encoder) deployed behind a FastAPI service backed by Qdrant Cloud and OpenRouter.

The architectural choices that earn this repo its name — package-by-feature, FSM-driven ingestion, functional retrieval pipeline — are documented in detail in [`docs/adr/`](docs/adr/). The eval table below isn't a vanity benchmark: it's the gate that tells me when a change to the retrieval pipeline is worth shipping.

> **Live:** <https://semantic-document-search.fly.dev> · **Docs (Scalar):** [`/docs`](https://semantic-document-search.fly.dev/docs)

## Quickstart

```bash
# 1. clone + install (uv)
git clone git@github.com:heitor-am/semantic-document-search.git
cd semantic-document-search
uv sync --all-extras

# 2. configure (Qdrant + OpenRouter keys)
cp .env.example .env
$EDITOR .env

# 3. apply DB schema and start the service
make migrate
make dev                       # uvicorn on http://127.0.0.1:8000

# 4. (in another terminal) load the 50-article seed corpus into Qdrant
make seed

# 5. try a query
curl "http://127.0.0.1:8000/search?q=replace+Redis+with+Postgres&strategy=hybrid_rerank&top_k=3"
```

Or skip steps 1–4 and hit production:

```bash
curl "https://semantic-document-search.fly.dev/search?q=replace+Redis+with+Postgres&strategy=hybrid_rerank&top_k=3"
```

Fly auto-stops the machine after idle, so the first call after a quiet period pays a few seconds of cold-start. The FastEmbed BM25 model is baked into the Docker image (ADR-014) so the cold-start doesn't include the ~50MB model download.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Web framework | FastAPI · Pydantic v2 · SQLAlchemy 2.0 (async) | Same stack as [Q1](https://github.com/heitor-am/virtual-library-api) — consistency across the portfolio |
| Vector store | Qdrant Cloud (sa-east-1) | Native hybrid search, `Modifier.IDF`, server-side RRF (ADR-004 / ADR-007) |
| Embeddings | OpenRouter `baai/bge-m3` (1024-d, cosine) | Multilingual SOTA open-weights |
| Sparse / BM25 | FastEmbed `Qdrant/bm25` (Snowball + stopwords + hashed indices) | Canonical tokenizer for Qdrant's IDF (ADR-006) |
| Reranker | OpenRouter Cohere `rerank-3.5` | +9.4pp on P@1 vs both baselines, no GPU dependency (ADR-008) |
| Job persistence | SQLite + Alembic on a Fly volume | Survives restarts; no extra infrastructure (ADR-009) |
| FSM | [`transitions`](https://github.com/pytransitions/transitions) | Declarative, illegal transitions impossible (ADR-002) |
| Quality gates | Ruff · mypy strict · pytest (84% cov) · Schemathesis · bandit · pip-audit | Same gates as Q1 — every PR runs them |
| Infra | Docker (multi-stage) · Fly.io · GitHub Actions · Dev Container | Auto-deploy on push to `main` |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest` | Submit a URL — returns `202` + `job_id` (idempotent on URL, ADR-010) |
| `GET`  | `/ingest/jobs/{id}` | Job state + transition timeline |
| `GET`  | `/ingest/jobs` | List jobs (optional `?state=` filter) |
| `GET`  | `/search` | `?q=...&strategy=dense_only|hybrid|hybrid_rerank&top_k=10` |
| `GET`  | `/health` | Liveness — version, commit, uptime, DB ping |
| `GET`  | `/docs` | Scalar-rendered OpenAPI |

Examples:

```bash
# 1. Ingest a fresh URL (idempotent — same URL, same job_id)
curl -X POST https://semantic-document-search.fly.dev/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_url": "https://dev.to/some/post"}'

# 2. Watch the FSM progress
curl https://semantic-document-search.fly.dev/ingest/jobs/<job_id>

# 3. Search with the production strategy (hybrid + cross-encoder rerank)
curl "https://semantic-document-search.fly.dev/search?q=production+RAG+with+FastAPI&strategy=hybrid_rerank&top_k=5"
```

## Evaluation

32 golden queries (mix of keyword-heavy, paraphrase, and adversarial) replayed against each retrieval strategy. Numbers below are **production**, against the 50-article dev.to corpus loaded into Qdrant Cloud:

| Strategy | P@1 | R@1 | R@3 | R@5 | R@10 | MRR | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| `dense_only` | 0.750 | 0.646 | 0.839 | 0.901 | 0.911 | 0.827 | 0.826 | 0.831 |
| `hybrid` (RRF) | 0.750 | 0.661 | 0.885 | 0.896 | 0.911 | 0.833 | 0.835 | 0.841 |
| `hybrid_rerank` | **0.844** | **0.740** | 0.865 | 0.896 | 0.911 | **0.878** | **0.863** | **0.869** |

Reranker contributes **+9.4pp on P@1** and **+5.1pp on MRR** over both baselines — sharper relevance at the top of the list, where users actually look. R@10 = 0.911 is the indexed-corpus ceiling for this query set; further gains require upstream changes (better chunking, query expansion, more diverse seeds).

The eval framework is the regression tripwire: every query has a `min_recall_at_5` floor in [`app/evaluation/queries.yaml`](app/evaluation/queries.yaml); the runner exits non-zero if any strategy drops below the declared floor.

- Full per-query breakdown: [`docs/eval-results-prod.txt`](docs/eval-results-prod.txt)
- Re-run against prod: `make eval APP_URL=https://semantic-document-search.fly.dev`
- Re-run locally: `make eval` (needs `make dev` running)

## Demo notebook

[`notebooks/pipeline-demo.ipynb`](notebooks/pipeline-demo.ipynb) — end-to-end walkthrough rendered with cell outputs:
ingestion FSM transitions, the three retrieval strategies side-by-side, and the evaluation summary. GitHub renders it inline.

## Architecture

- [`docs/diagrams/architecture.md`](docs/diagrams/architecture.md) — package-by-feature layout
- [`docs/diagrams/ingestion-fsm.md`](docs/diagrams/ingestion-fsm.md) — state machine (Mermaid + auto-generated PNG)
- [`docs/diagrams/retrieval-pipeline.md`](docs/diagrams/retrieval-pipeline.md) — `Stage` sequence by strategy
- [`docs/adr/`](docs/adr/) — 14 Architecture Decision Records covering every load-bearing choice

## Testing

```bash
make check       # lint + typecheck + tests (~3s)
make test        # tests only with coverage (gate ≥ 80%)
make smoke URL=https://dev.to/...   # end-to-end against real Qdrant + OpenRouter
```

Coverage is **84%** across 262 tests. The gate enforces ≥ 80% in CI.

## Project layout

```
app/
├── ingestion/      # POST /ingest — FSM-driven, persisted in SQLite
├── retrieval/      # GET /search — composable Stage pipeline
├── evaluation/     # CLI runner against the YAML golden set
├── shared/         # OpenRouter client, Qdrant repo, DB session, config
└── main.py
docs/
├── adr/            # 14 ADRs
├── diagrams/       # 3 Mermaid diagrams + auto-gen FSM PNG
├── eval-results-prod.txt
└── PRD.md
notebooks/
└── pipeline-demo.ipynb
scripts/
├── seed_corpus.py  # bulk-ingest seed_urls.txt (supports --app-url for prod)
├── smoke_ingestion.py
└── reencode_sparse.py
```

## Portfolio context

Companion to:

- **[Virtual Library API](https://github.com/heitor-am/virtual-library-api)** — Q1, FastAPI + SQLite + OpenRouter, package-by-layer
- **Python Tutor Chatbot** — Q2, Chainlit + LangChain + OpenRouter (planned)

All three share the OpenRouter wrapper verbatim and the same Conventional Commits / RFC 7807 / Ruff + uv + mypy / Fly.io conventions. This repo is the one where the architecture diverges on purpose — package-by-feature, FSM, functional pipeline — because the three-pipeline shape calls for it. The ADRs document every divergence.

## License

MIT — see [LICENSE](LICENSE).
