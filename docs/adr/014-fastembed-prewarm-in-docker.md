# 014 — Pre-warm FastEmbed BM25 model at Docker build time

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** @heitor-am

## Context

Once the BM25 sparse encoder moved to FastEmbed `Qdrant/bm25` (ADR-006), the first request after a cold start blocks for **~10 seconds** while the ~50MB model downloads from Hugging Face into `~/.cache/fastembed`.

Fly.io is configured with `auto_stop_machines = "stop"` and `min_machines_running = 0` — the machine sleeps after idle, and the next request wakes it. So the first user-visible request of a quiet period would pay the ~10s penalty *plus* normal cold-start. Two failure modes follow:

- The Fly health check has a `grace_period = 10s`. Right at the boundary, the model download could time out the first probe and trigger a restart loop.
- Even when health passes, the user-perceived latency on the first request is unacceptable — and the BM25 download isn't visible in any logs the user understands.

## Decision

Download the model into a known cache directory **at Docker build time**, in the multi-stage image:

```dockerfile
# Builder stage — after `uv sync`, the venv has fastembed installed.
ENV FASTEMBED_CACHE_DIR=/app/.fastembed-cache
RUN /app/.venv/bin/python -c "from fastembed import SparseTextEmbedding; \
    SparseTextEmbedding(model_name='Qdrant/bm25', cache_dir='/app/.fastembed-cache')"

# Runtime stage inherits /app/.fastembed-cache via COPY --from=builder /app /app
ENV FASTEMBED_CACHE_DIR=/app/.fastembed-cache
```

`sparse_encoder.py` reads `FASTEMBED_CACHE_DIR` and passes it to `SparseTextEmbedding(cache_dir=...)`. Unset → falls back to `~/.cache/fastembed`, which is what local dev uses.

Two unit tests assert both branches (`test_passes_cache_dir_when_env_set`, `test_omits_cache_dir_when_env_unset`) — the env var coupling is production-critical and fails silently otherwise.

## Consequences

**Positive:**
- Cold-start cost on Fly drops from ~10s to ~1-2s (the FastAPI process startup itself).
- Health-check grace period is no longer a tightrope.
- Repeatable, version-pinned model — the model file in the image matches the `fastembed` version in `uv.lock`, no drift between image build and runtime.

**Negative:**
- Docker image grows by ~50MB. Acceptable; the image was ~250MB before, now ~300MB — well under any size constraint we care about.
- Builder needs network at build time (Hugging Face). CI already has it; if we ever moved to an offline build, we'd pre-stage the file via `COPY`.

**Trade-offs accepted:**
- The image is now coupled to one specific BM25 model. Swapping models requires a Dockerfile change. That's the right granularity — the model is part of the deployment artifact, not a runtime config knob.

## Alternatives considered

- **Lazy download on first request** — the original implementation. Rejected for the cold-start reasons above.
- **Mount the model on a Fly volume and download once at boot** — adds machinery (volume, boot script) for a problem the build can solve for free.
- **Bake into the image via `COPY` from a separate "model artifacts" build step** — works, but the Python one-liner is dead simple and produces the same result.

## References

- `Dockerfile` (the `RUN /app/.venv/bin/python -c "..."` line)
- `fly.toml`
- `app/shared/qdrant/sparse_encoder.py` (`FASTEMBED_CACHE_DIR` handling)
- `tests/shared/test_qdrant_sparse_encoder.py` (`TestModelCacheDir`)
