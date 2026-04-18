# Retrieval pipeline — sequence by strategy

`build_pipeline(strategy)` composes a sequence of `Stage`s. The same `Context` flows through every stage; each one mutates `ctx.results` and may add to `ctx.warnings`. Optional stages skip on error (ADR-013) — required stages propagate.

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Router as GET /search
    participant Svc as build_pipeline(strategy)
    participant Pipe as Pipeline executor
    participant Hyb as HybridSearchStage<br/>(required)
    participant Rer as RerankerStage<br/>(optional)
    participant PC as ParentChildStage<br/>(required)
    participant Embed as OpenRouter<br/>bge-m3
    participant Qdr as Qdrant Cloud
    participant Cohere as OpenRouter<br/>rerank-3.5

    Client->>Router: q, strategy, top_k
    Router->>Svc: Strategy.HYBRID_RERANK
    Svc-->>Pipe: [Hybrid, Reranker, ParentChild]
    Pipe->>Hyb: ctx
    Hyb->>Embed: embed(query)
    Embed-->>Hyb: 1024-d vector
    Hyb->>Qdr: query_points<br/>prefetch(dense + bm25 sparse)<br/>fusion=RRF
    Qdr-->>Hyb: top-N candidates (chunks)
    Hyb-->>Pipe: ctx.results = N candidates

    alt strategy == hybrid_rerank
        Pipe->>Rer: ctx
        Rer->>Qdr: scroll(parent_chunk_ids → parent text)
        Qdr-->>Rer: parent texts
        Rer->>Cohere: rerank(query, [parent_text...])
        alt rerank ok
            Cohere-->>Rer: relevance scores
            Rer-->>Pipe: ctx.results reordered
        else rerank fails
            Rer-->>Pipe: ctx.warnings += "reranker: ..."<br/>ctx.results unchanged
        end
    end

    Pipe->>PC: ctx
    Note over PC: dedup by parent_chunk_id<br/>keep best score per parent
    PC-->>Pipe: ctx.results sliced to top_k

    Pipe-->>Router: SearchResponse(results, warnings)
    Router-->>Client: 200 OK
```

## Strategies — what's in / out

| Strategy | Stages (in order) |
|---|---|
| `dense_only` | `HybridSearchStage(sparse_enabled=False)` → `ParentChildStage` |
| `hybrid` | `HybridSearchStage` → `ParentChildStage` |
| `hybrid_rerank` | `HybridSearchStage` → `RerankerStage` → `ParentChildStage` |

## Why this shape

- **One round-trip for fusion (step 6).** Qdrant's Query API does dense + sparse + RRF server-side — we don't merge result lists in Python. ADR-007 explains the trade.
- **Reranker is optional (steps 9–14).** When OpenRouter rerank hiccups, the request still returns the unranked hybrid list with a warning. ADR-013 covers the degradation contract.
- **Rerank scores parent text, not children (step 9).** Cross-encoders are sensitive to surrounding context; the 256-token child often doesn't carry enough. The parent (≈1024 tokens) does. Hierarchical chunking (ADR-005) is what makes this clean.
- **Parent-child dedup runs last (step 16).** It works on whatever stage produced the most recent ranking — so for `hybrid_rerank`, dedup respects rerank scores rather than fusion scores.
