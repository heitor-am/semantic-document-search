"""Reciprocal Rank Fusion stage.

Merges the ranked candidate lists produced by upstream stages (dense +
sparse by default) into a single fused list.

RRF score for a document d across N ranked lists:

    RRF(d) = Σ_L  1 / (k + rank_L(d))

where `rank_L(d)` is 1-based. Documents absent from list L contribute 0.
k=60 is the canonical value from the original RRF paper (Cormack et al.);
it de-emphasises the exact rank in each list so fusion is robust to
wildly different score distributions (Qdrant cosine vs BM25 tf-idf).

The fused list populates both `ctx.state["fused"]` (for inspection/eval)
and `ctx.results` (so a pipeline that stops here without reranker still
returns something usable — the reranker stage in Etapa 10 overrides
`results` when it runs).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.retrieval.context import Candidate, Context
from app.retrieval.pipeline import Stage


class RRFFusionStage(Stage):
    name = "fusion_rrf"
    optional = False

    def __init__(
        self,
        *,
        sources: Sequence[str] = ("dense", "sparse"),
        k: int = 60,
        output_multiplier: int = 2,
    ) -> None:
        self._sources = tuple(sources)
        self._k = k
        # Keep more than top_k so the reranker (Etapa 10) has slack.
        self._output_multiplier = output_multiplier

    async def run(self, ctx: Context) -> Context:
        fused_scores: dict[str, float] = {}
        payloads: dict[str, Mapping[str, object]] = {}

        for source in self._sources:
            candidates = ctx.state.get(source) or []
            for rank_zero, cand in enumerate(candidates):
                contribution = 1.0 / (self._k + rank_zero + 1)
                fused_scores[cand.chunk_id] = fused_scores.get(cand.chunk_id, 0.0) + contribution
                # Stages downstream read payload (content, section_path,
                # parent_chunk_id, ...); first-seen wins so dense's richer
                # Qdrant payload takes precedence over BM25's.
                payloads.setdefault(cand.chunk_id, cand.payload)

        output_limit = max(ctx.top_k, ctx.top_k * self._output_multiplier)
        ranked_ids = sorted(
            fused_scores.keys(),
            key=lambda cid: fused_scores[cid],
            reverse=True,
        )[:output_limit]

        fused = [
            Candidate(
                chunk_id=cid,
                score=fused_scores[cid],
                payload=dict(payloads[cid]),
            )
            for cid in ranked_ids
        ]
        ctx.state["fused"] = fused
        ctx.results = fused[: ctx.top_k]
        return ctx
