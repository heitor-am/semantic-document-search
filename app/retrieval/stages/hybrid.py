"""Hybrid search stage.

One round-trip to Qdrant's Query API: dense prefetch + sparse prefetch,
fused with RRF server-side. Replaces the old three-stage chain
(DenseSearch + SparseSearch + RRFFusion) — Qdrant does the fusion, and
the BM25 index is the sparse vector sitting on the same points in the
same collection, so there's no in-app corpus to keep in sync.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from app.retrieval.context import Candidate, Context
from app.retrieval.pipeline import Stage, StageError
from app.retrieval.sparse_encoder import encode_bm25_sparse
from app.shared.ai.embeddings import embed_texts
from app.shared.qdrant.repository import SparseValue, VectorRepository


class HybridSearchStage(Stage):
    name = "hybrid"
    optional = False

    def __init__(
        self,
        *,
        vector_repo: VectorRepository,
        openai_client: AsyncOpenAI,
        embedding_model: str,
        prefetch_multiplier: int = 4,
    ) -> None:
        self._vector_repo = vector_repo
        self._openai_client = openai_client
        self._embedding_model = embedding_model
        self._prefetch_multiplier = prefetch_multiplier

    async def run(self, ctx: Context) -> Context:
        dense_vectors = await embed_texts(
            [ctx.query],
            client=self._openai_client,
            model=self._embedding_model,
        )
        if not dense_vectors:
            raise StageError(self.name, cause=ValueError("empty query embedding"))

        sparse_indices, sparse_values = encode_bm25_sparse(ctx.query)
        sparse = (
            SparseValue(indices=sparse_indices, values=sparse_values) if sparse_indices else None
        )

        hits = await self._vector_repo.search_hybrid(
            dense_vector=dense_vectors[0],
            sparse=sparse,
            k=ctx.top_k,
            prefetch_limit=ctx.top_k * self._prefetch_multiplier,
            # Only children carry matching embeddings; parents are context
            # containers fetched later by id in the parent-child stage.
            filters={"is_parent": False},
        )
        candidates = [
            Candidate(chunk_id=hit.id, score=hit.score, payload=dict(hit.payload)) for hit in hits
        ]
        ctx.state["hybrid"] = candidates
        ctx.results = candidates
        return ctx
