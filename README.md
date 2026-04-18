# Semantic Document Search

Production-grade RAG search: package-by-feature, FSM-driven ingestion, functional retrieval pipeline.

**Scaffold stage (`v0.0.1`) — features coming in the next etapas.**

## Stack (planned)

- **Web:** FastAPI · Pydantic v2 · SQLAlchemy 2.0 · SQLite (job persistence)
- **Vector store:** Qdrant Cloud
- **AI:** OpenRouter — chat `openai/gpt-4o-mini`, embeddings `baai/bge-m3`
- **Pipelines:** `transitions` (FSM for ingestion) · custom functional pipeline (retrieval)
- **Retrieval:** hybrid (BM25 + dense) + RRF + cross-encoder reranking + parent-child
- **Evaluation:** golden set + recall@k · MRR · NDCG
- **Quality:** Ruff · mypy (strict) · pytest · Schemathesis · bandit · pip-audit
- **Infra:** Docker · Fly.io · GitHub Actions · Dev Container

## Live deployment

- **Base URL:** https://semantic-document-search.fly.dev
- **OpenAPI / Scalar docs:** https://semantic-document-search.fly.dev/docs
- **Health:** https://semantic-document-search.fly.dev/health

Try it:

```bash
curl "https://semantic-document-search.fly.dev/search?q=replace+Redis+with+Postgres&strategy=hybrid_rerank&top_k=3"
```

Fly.io auto-stops the machine after idle, so the first call after a quiet period pays a few seconds of cold-start (the FastEmbed BM25 model is baked into the image to avoid the ~50MB download cost).

## Evaluation

32 golden queries (mix of keyword-heavy, paraphrase, and adversarial) replayed against each retrieval strategy. Numbers below are from production, against the same 50-article dev.to corpus loaded into Qdrant Cloud:

| Strategy | P@1 | R@1 | R@3 | R@5 | R@10 | MRR | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| `dense_only` | 0.750 | 0.646 | 0.839 | 0.901 | 0.911 | 0.827 | 0.826 | 0.831 |
| `hybrid` (RRF) | 0.750 | 0.661 | 0.885 | 0.896 | 0.911 | 0.833 | 0.835 | 0.841 |
| `hybrid_rerank` | **0.844** | **0.740** | 0.865 | 0.896 | 0.911 | **0.878** | **0.863** | **0.869** |

Reranker contributes **+9.4pp on P@1** and **+5.1pp on MRR** over both baselines — consistent with the cross-encoder literature: sharper relevance at the top of the list, where users actually look. R@10 = 0.911 is the indexed-corpus ceiling for this query set; further gains require upstream changes (better chunking, query expansion, more diverse seeds).

Full per-query breakdown: [`docs/eval-results-prod.txt`](docs/eval-results-prod.txt). Re-run with `make eval APP_URL=https://semantic-document-search.fly.dev` (or pass `--app-url` to `app.evaluation.runner`).

## Architecture

- [`docs/diagrams/architecture.md`](docs/diagrams/architecture.md) — package-by-feature layout, where the seams are
- [`docs/diagrams/ingestion-fsm.md`](docs/diagrams/ingestion-fsm.md) — ingestion state machine (Mermaid + auto-generated PNG)
- [`docs/diagrams/retrieval-pipeline.md`](docs/diagrams/retrieval-pipeline.md) — retrieval `Stage` sequence by strategy
- [`docs/adr/`](docs/adr/) — 14 Architecture Decision Records covering every load-bearing choice (FSM, hybrid search, RRF, parent-child, collection versioning, ...)

## Demo notebook

[`notebooks/pipeline-demo.ipynb`](notebooks/pipeline-demo.ipynb) — end-to-end walkthrough rendered with cell outputs:
ingestion FSM transitions, the three retrieval strategies side-by-side, and the evaluation summary.
GitHub renders it inline. To re-execute locally: `make migrate` (one-time SQLite schema), `make dev`
(uvicorn in another terminal), and — to reproduce the retrieval examples — `make seed` to load the
50-article corpus into Qdrant. Then open the notebook in Jupyter.

## License

MIT — see [LICENSE](LICENSE).
