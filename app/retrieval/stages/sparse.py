"""Sparse (BM25) search stage.

Uses the `BM25Index` instance injected at construction time (the lifespan
builds one shared index and passes it in). If the index is empty (fresh
app, nothing ingested yet, or rebuild pending), this stage produces an
empty list instead of failing — fusion merges whatever it gets.
"""

from __future__ import annotations

from app.retrieval.bm25_index import BM25Index
from app.retrieval.context import Context
from app.retrieval.pipeline import Stage


class SparseSearchStage(Stage):
    name = "sparse"
    # Empty BM25 output is legitimate (nothing ingested yet), so this stage
    # itself never fails. It's marked required so an actual bug inside
    # BM25Index.search would surface instead of being silently swallowed.
    optional = False

    def __init__(self, bm25_index: BM25Index, *, fetch_multiplier: int = 3) -> None:
        self._bm25_index = bm25_index
        self._fetch_multiplier = fetch_multiplier

    async def run(self, ctx: Context) -> Context:
        ctx.state["sparse"] = self._bm25_index.search(
            ctx.query,
            k=ctx.top_k * self._fetch_multiplier,
        )
        return ctx
