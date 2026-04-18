# Architecture Decision Records

Each file documents one architectural choice: the context that forced the decision, the option taken, the consequences, and the alternatives that were rejected (and why). Format follows the [`template.md`](template.md).

| # | Decision |
|---|---|
| [001](001-package-by-feature.md) | Package-by-feature, not by-layer |
| [002](002-fsm-for-ingestion.md) | Finite State Machine (`transitions`) for ingestion |
| [003](003-functional-pipeline-for-retrieval.md) | Functional pipeline with composable `Stage`s |
| [004](004-qdrant-named-vectors-and-idf-modifier.md) | Qdrant: named vectors + sparse with `Modifier.IDF` |
| [005](005-hierarchical-chunking.md) | Hierarchical chunking (parent / child) |
| [006](006-fastembed-bm25-instead-of-handrolled.md) | FastEmbed `Qdrant/bm25` instead of a hand-rolled tokenizer |
| [007](007-rrf-server-side-via-query-api.md) | Reciprocal Rank Fusion server-side via Qdrant Query API |
| [008](008-cohere-rerank-via-openrouter.md) | Cross-encoder rerank via Cohere `rerank-3.5` over OpenRouter |
| [009](009-sqlite-job-persistence.md) | SQLite + transition log for job persistence |
| [010](010-deterministic-job-id-from-url.md) | Deterministic `job_id = sha256(url)[:16]` |
| [011](011-repository-pattern-for-qdrant.md) | Repository pattern for Qdrant (testability + swap seam) |
| [012](012-collection-versioning.md) | Collection versioning (`documents_{model}_v{n}`) |
| [013](013-per-stage-graceful-degradation.md) | Per-stage graceful degradation in retrieval |
| [014](014-fastembed-prewarm-in-docker.md) | Pre-warm FastEmbed model at Docker build time |

## When to add a new ADR

When a decision **closes off other options** and a future maintainer would benefit from knowing *why* — not "what does the code do" (the code answers that), but "why this and not that other reasonable thing." Bug fixes, refactors, and simple feature additions don't need ADRs. Architectural seams and irreversible choices do.

Copy [`template.md`](template.md), bump the number, write the context as if explaining to someone who'd otherwise repeat the rejected alternative.
