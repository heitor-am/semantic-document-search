# 003 — Functional pipeline for retrieval

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** @heitor-am

## Context

Retrieval is fundamentally a *composition* problem: dense → sparse → fusion → rerank → parent resolve. Different requests need different compositions:

- Fast path: dense only
- Production path: hybrid (dense + sparse + RRF)
- High-precision path: hybrid + cross-encoder rerank

A class hierarchy (`DenseRetriever extends BaseRetriever`) would force us to express composition through inheritance — fine for two strategies, painful for permutations.

We also need *graceful degradation*: if the reranker times out, the response should still come back with the unranked candidates plus a `warnings` field — not a 5xx.

## Decision

Each retrieval step is a `Stage` — a small, async, single-responsibility object that takes a `Context` and returns a `Context`:

```python
class Stage(Protocol):
    name: str
    optional: bool  # if True, errors → warning + skip; else propagate
    async def run(self, ctx: Context) -> Context: ...
```

A `Pipeline` is just a sequence of stages. `build_pipeline(strategy)` returns the right composition:

| Strategy | Stages |
|---|---|
| `dense_only` | `HybridSearchStage(sparse_enabled=False)` → `ParentChildStage` |
| `hybrid` | `HybridSearchStage` → `ParentChildStage` |
| `hybrid_rerank` | `HybridSearchStage` → `RerankerStage` → `ParentChildStage` |

Each stage carries an `optional` flag. When a stage with `optional=True` raises, the pipeline appends a `warnings[]` entry and skips it instead of failing the request.

## Consequences

**Positive:**
- New strategy = new sequence; no inheritance gymnastics. Reordering, swapping, A/B testing become trivial.
- Graceful degradation is a property of the pipeline, not duplicated in each stage.
- Stages are independently testable — mock the `Context`, run one stage, assert.
- The `Strategy` enum maps 1:1 to a published `/search?strategy=...` value, so the API surface and the internal composition stay aligned.

**Negative:**
- Slightly more boilerplate than a single `retrieve(query)` function — but the boilerplate *is* the composition contract.

**Trade-offs accepted:**
- We don't have a DAG executor (LangGraph, Haystack pipelines, etc.). All current strategies are linear; introducing branches would justify a real graph. Not yet.

## Alternatives considered

- **Class hierarchy** — rejected: composition is the wrong fit for inheritance.
- **LangChain / Haystack pipelines** — rejected: brings a large dependency for a 5-stage flow we control end-to-end.
- **Plain function calls inside the service** — rejected: graceful degradation would have to be re-implemented at each call site.

## References

- `app/retrieval/pipeline.py`
- `app/retrieval/stages/`
- `app/retrieval/service.py` (`build_pipeline`)
