"""Dense (vector) search stage.

Embeds the query via OpenRouter and asks Qdrant for the top-N closest child
chunks. N is `top_k * fetch_multiplier` so the fusion stage has material to
merge with sparse hits without starving the final top-k.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from app.retrieval.context import Candidate, Context
from app.retrieval.pipeline import Stage, StageError
from app.shared.ai.embeddings import embed_texts
from app.shared.qdrant.repository import VectorRepository


class DenseSearchStage(Stage):
    name = "dense"
    optional = False

    def __init__(
        self,
        *,
        vector_repo: VectorRepository,
        openai_client: AsyncOpenAI,
        embedding_model: str,
        fetch_multiplier: int = 3,
    ) -> None:
        self._vector_repo = vector_repo
        self._openai_client = openai_client
        self._embedding_model = embedding_model
        self._fetch_multiplier = fetch_multiplier

    async def run(self, ctx: Context) -> Context:
        vectors = await embed_texts(
            [ctx.query],
            client=self._openai_client,
            model=self._embedding_model,
        )
        if not vectors:
            raise StageError(self.name, cause=ValueError("empty query embedding"))

        hits = await self._vector_repo.search(
            vectors[0],
            k=ctx.top_k * self._fetch_multiplier,
            # Only children carry the matching embedding; parents are looked
            # up by id in the parent-child stage for context expansion.
            filters={"is_parent": False},
        )
        ctx.state["dense"] = [
            Candidate(chunk_id=hit.id, score=hit.score, payload=dict(hit.payload)) for hit in hits
        ]
        return ctx
