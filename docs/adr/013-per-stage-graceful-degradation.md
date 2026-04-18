# 013 — Per-stage graceful degradation in the retrieval pipeline

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** @heitor-am

## Context

Retrieval depends on three external services:

- **Qdrant Cloud** — required; if it's down, there's nothing to return.
- **OpenRouter embeddings** (`bge-m3`) — required to embed the query before searching.
- **OpenRouter rerank** (`rerank-3.5`) — *enhances* a result list that already exists.

If a stage strictly required for the result fails (Qdrant, embeddings), a 5xx is correct — we have no answer. But if an *optional* stage like rerank fails, the choice is:

- Return 5xx → the user gets nothing, even though we had a perfectly serviceable hybrid result list before the rerank attempt.
- Return the unranked list → the user gets the next-best answer, with a signal that something degraded.

The second option matters more than it seems. Hosted reranker APIs have occasional hiccups (rate limits, model deploys, network); we don't want every blip to look like a service outage to the caller.

## Decision

Each `Stage` declares whether it is `optional`:

```python
class Stage(Protocol):
    name: str
    optional: bool
    async def run(self, ctx: Context) -> Context: ...
```

The `Pipeline` executor wraps each stage:

```python
try:
    ctx = await stage.run(ctx)
except Exception as exc:
    if stage.optional:
        ctx.warnings.append(f"{stage.name}: {exc}")
        continue              # skip this stage, go to the next
    raise                     # required stage → propagate, becomes a 5xx
```

The `SearchResponse` schema carries a `warnings: list[str]` field, populated from `ctx.warnings`. Clients (and the eval harness) can see exactly which optional stages were skipped. Required stages (Qdrant query, embeddings) propagate as before — those map cleanly to RFC 7807 problem responses via `app_error_handler`.

Current optionality:

| Stage | optional | Why |
|---|---|---|
| `HybridSearchStage` | False | No results without it |
| `RerankerStage` | True | Pre-rerank list is still useful |
| `ParentChildStage` | False | Cheap; in-memory; no external call |

## Consequences

**Positive:**
- Hosted reranker hiccups don't cascade into 5xxs. The eval harness has caught two real OpenRouter rate-limit windows; both came back as warnings, not failures.
- The `warnings` field is part of the API contract — clients can render "results may be less precise" instead of erroring out.
- Adding a future optional stage (query rewriter, query expansion) inherits this behavior for free.

**Negative:**
- The contract says "we tried" — it doesn't say "this is the best we could do." A monitoring story (alert on warning rate) would close that loop. Out of scope for the demo, called out as future work.
- Two response shapes (with vs without rerank scores) for the same `strategy` value. Documented in the OpenAPI; eval framework handles both.

**Trade-offs accepted:**
- We don't *retry* optional stages within a request. Tenacity wraps the embedding call (because it's required), but rerank gets one shot per request — failure is treated as "skip and move on." Retrying adds latency to a hot path that's already opted into degradation.

## Alternatives considered

- **Rerank failure → 5xx** — over-couples the API surface to a non-essential dependency.
- **Configurable optionality at request time** — feature creep for the demo. Not requested by any client; would dilute the eval signal.
- **Circuit breaker around the reranker** — reasonable at higher scale; overkill for a single-operator workload.

## References

- `app/retrieval/pipeline.py` (the executor)
- `app/retrieval/stages/reranker.py` (`optional = True`)
- `app/retrieval/schemas.py` (the `warnings` field on `SearchResponse`)
