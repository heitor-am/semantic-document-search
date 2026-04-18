from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.retrieval.context import Context
from app.retrieval.pipeline import StageError
from app.retrieval.stages.hybrid import HybridSearchStage
from app.shared.qdrant.repository import SparseValue, VectorHit


def make_openai_client(vector: list[float] | None = None) -> MagicMock:
    client = MagicMock()
    vec = vector if vector is not None else [0.1] * 4

    async def create(**kwargs: Any) -> MagicMock:
        return MagicMock(data=[MagicMock(embedding=vec)])

    client.embeddings.create = AsyncMock(side_effect=create)
    return client


def make_vector_repo(hits: list[VectorHit] | None = None) -> MagicMock:
    repo = MagicMock()
    repo.search_hybrid = AsyncMock(return_value=hits or [])
    return repo


class TestHybridSearchStage:
    async def test_writes_candidates_to_state_and_results(self) -> None:
        hits = [
            VectorHit(id="c1", score=0.9, payload={"content": "a"}),
            VectorHit(id="c2", score=0.7, payload={"content": "b"}),
        ]
        stage = HybridSearchStage(
            vector_repo=make_vector_repo(hits),
            openai_client=make_openai_client(),
            embedding_model="baai/bge-m3",
        )

        ctx = await stage.run(Context(query="python async"))

        assert [c.chunk_id for c in ctx.state["hybrid"]] == ["c1", "c2"]
        assert ctx.results == ctx.state["hybrid"]

    async def test_calls_search_hybrid_with_dense_and_sparse(self) -> None:
        repo = make_vector_repo()
        stage = HybridSearchStage(
            vector_repo=repo,
            openai_client=make_openai_client([0.5] * 4),
            embedding_model="baai/bge-m3",
        )
        await stage.run(Context(query="python async", top_k=5))

        _, kwargs = repo.search_hybrid.await_args
        assert kwargs["dense_vector"] == [0.5, 0.5, 0.5, 0.5]
        sparse = kwargs["sparse"]
        assert isinstance(sparse, SparseValue)
        assert len(sparse.indices) == 2  # "python", "async"
        assert kwargs["k"] == 5
        assert kwargs["filters"] == {"is_parent": False}

    async def test_empty_query_tokens_collapses_to_dense_only(self) -> None:
        # Query "!!! ???" has no tokens; stage should still run and pass
        # sparse=None (or empty) so the repo falls back to dense-only.
        repo = make_vector_repo()
        stage = HybridSearchStage(
            vector_repo=repo,
            openai_client=make_openai_client(),
            embedding_model="baai/bge-m3",
        )
        await stage.run(Context(query="!!! ???"))

        _, kwargs = repo.search_hybrid.await_args
        assert kwargs["sparse"] is None

    async def test_sparse_disabled_skips_sparse_even_with_tokens(self) -> None:
        # Even a perfectly tokenisable query must produce sparse=None when
        # the stage is configured with sparse_enabled=False (Strategy.DENSE_ONLY).
        repo = make_vector_repo()
        stage = HybridSearchStage(
            vector_repo=repo,
            openai_client=make_openai_client(),
            embedding_model="baai/bge-m3",
            sparse_enabled=False,
        )
        await stage.run(Context(query="python async programming"))

        _, kwargs = repo.search_hybrid.await_args
        assert kwargs["sparse"] is None

    async def test_prefetch_limit_uses_multiplier(self) -> None:
        repo = make_vector_repo()
        stage = HybridSearchStage(
            vector_repo=repo,
            openai_client=make_openai_client(),
            embedding_model="baai/bge-m3",
            prefetch_multiplier=3,
        )
        await stage.run(Context(query="python", top_k=7))

        _, kwargs = repo.search_hybrid.await_args
        assert kwargs["prefetch_limit"] == 21  # 7 * 3

    async def test_uses_configured_embedding_model(self) -> None:
        openai = make_openai_client()
        stage = HybridSearchStage(
            vector_repo=make_vector_repo(),
            openai_client=openai,
            embedding_model="custom/model-x",
        )
        await stage.run(Context(query="hi"))
        _, kwargs = openai.embeddings.create.await_args
        assert kwargs["model"] == "custom/model-x"

    async def test_empty_embedding_response_raises_stage_error(self) -> None:
        openai = MagicMock()
        openai.embeddings.create = AsyncMock(return_value=MagicMock(data=[]))
        stage = HybridSearchStage(
            vector_repo=make_vector_repo(),
            openai_client=openai,
            embedding_model="baai/bge-m3",
        )
        with pytest.raises(StageError) as exc_info:
            await stage.run(Context(query="hi"))
        assert exc_info.value.stage_name == "hybrid"

    async def test_no_hits_produces_empty_state_and_results(self) -> None:
        stage = HybridSearchStage(
            vector_repo=make_vector_repo([]),
            openai_client=make_openai_client(),
            embedding_model="baai/bge-m3",
        )
        ctx = await stage.run(Context(query="obscure query"))
        assert ctx.state["hybrid"] == []
        assert ctx.results == []
