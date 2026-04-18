# 008 — Cross-encoder rerank via Cohere `rerank-3.5` over OpenRouter

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** @heitor-am

## Context

Bi-encoder retrieval (dense or sparse) is fast but coarse — embeddings represent the query and the document independently. Cross-encoders score (query, document) pairs *jointly*, which is more accurate but costs an extra forward pass per candidate. The standard pattern is:

1. Retrieve a wide top-N (say, 50) cheaply with a bi-encoder.
2. Rerank those N with a cross-encoder.
3. Return the top-K (say, 10) after rerank.

Two model-host options:

- **Self-host a cross-encoder** (`BAAI/bge-reranker-v2-m3`) via FastEmbed or `sentence-transformers`. Adds ~500MB of model weights, GPU-friendly but CPU-painful, and an extra cold-start cost.
- **Hosted reranker via API**. Cohere `rerank-3.5` (and now via OpenRouter) is purpose-built; latency in the ~200ms range for 50 candidates.

The PRD originally specified `BAAI/bge-reranker-v2-m3` self-hosted. Once OpenRouter exposed Cohere reranking under the same key as our chat / embedding models, the operational savings tipped the choice.

## Decision

Use Cohere `rerank-3.5` via OpenRouter for the `hybrid_rerank` strategy. The reranker stage is **optional** in the pipeline (ADR-003 / ADR-013): if rerank fails or times out, the response carries a `warnings[]` entry and the unranked candidates come back.

```python
class RerankerStage(Stage):
    name = "reranker"
    optional = True   # cross-encoder failure → warning + skip, not 5xx
```

Rerank is performed against the **parent chunk text**, not the matched child — cross-encoders give meaningful weight to surrounding context, and the parent has more signal than the 256-token child.

## Consequences

**Positive:**
- One vendor (OpenRouter) for chat + embeddings + rerank. One key, one base URL, one rate-limit envelope.
- No GPU dependency; no second model in the Docker image.
- Eval verified the value: rerank delivers **+9.4pp on P@1** over both `dense_only` and `hybrid` baselines (32-query golden set, see `docs/eval-results-prod.txt`). That's the cross-encoder pulling its weight at the top of the list, which is where users actually look.
- Optional stage = graceful degradation if the OpenRouter rerank model is briefly unavailable.

**Negative:**
- API dependency in the hot retrieval path. Mitigated by `optional=True` and a per-request timeout.
- Cost per query is higher than a self-hosted model would be in steady state. Acceptable at demo scale; the $/query is dominated by embedding cost on `/ingest`, not query-time rerank.

**Trade-offs accepted:**
- We don't ship a fallback cross-encoder. If OpenRouter goes down for an extended period, `hybrid_rerank` degrades to `hybrid` quality (with a warning). That's the right trade for this demo's risk profile.

## Alternatives considered

- **`BAAI/bge-reranker-v2-m3` self-hosted** — was the original PRD pick. Rejected after Cohere via OpenRouter became available: same vendor as our other AI calls, no extra image weight.
- **No reranker** — leaves +9.4pp P@1 on the table. The eval makes the case directly.
- **`mxbai-rerank-large-v1`** — comparable quality, not yet on OpenRouter at the time of writing.

## References

- `app/retrieval/stages/reranker.py`
- `app/shared/ai/reranker.py`
- `docs/eval-results-prod.txt` (the +9.4pp number)
- Cohere rerank-3.5: https://docs.cohere.com/docs/rerank
