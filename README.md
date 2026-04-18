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

## Demo

[`notebooks/pipeline-demo.ipynb`](notebooks/pipeline-demo.ipynb) — end-to-end walkthrough rendered with cell outputs:
ingestion FSM transitions, the three retrieval strategies side-by-side, and the evaluation summary.
GitHub renders it inline; locally run `make dev` first, then open in Jupyter to re-execute.

## License

MIT — see [LICENSE](LICENSE).
