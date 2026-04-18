from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.retrieval.context import Context
from app.retrieval.pipeline import StageError
from app.retrieval.stages.dense import DenseSearchStage
from app.shared.qdrant.repository import VectorHit


def make_openai_client(vector: list[float] | None = None) -> MagicMock:
    client = MagicMock()
    vec = vector if vector is not None else [0.1] * 4

    async def create(**kwargs: Any) -> MagicMock:
        return MagicMock(data=[MagicMock(embedding=vec)])

    client.embeddings.create = AsyncMock(side_effect=create)
    return client


def make_vector_repo(hits: list[VectorHit] | None = None) -> MagicMock:
    repo = MagicMock()
    repo.search = AsyncMock(return_value=hits or [])
    return repo


class TestDenseSearchStage:
    async def test_writes_candidates_to_state_dense(self) -> None:
        hits = [
            VectorHit(id="c1", score=0.9, payload={"content": "a"}),
            VectorHit(id="c2", score=0.7, payload={"content": "b"}),
        ]
        stage = DenseSearchStage(
            vector_repo=make_vector_repo(hits),
            openai_client=make_openai_client(),
            embedding_model="baai/bge-m3",
        )
        ctx = Context(query="python")

        result = await stage.run(ctx)

        assert [c.chunk_id for c in result.state["dense"]] == ["c1", "c2"]
        assert result.state["dense"][0].score == 0.9
        assert result.state["dense"][0].payload["content"] == "a"

    async def test_uses_configured_embedding_model(self) -> None:
        openai = make_openai_client()
        stage = DenseSearchStage(
            vector_repo=make_vector_repo(),
            openai_client=openai,
            embedding_model="custom/model-x",
        )
        await stage.run(Context(query="hi"))
        _, kwargs = openai.embeddings.create.await_args
        assert kwargs["model"] == "custom/model-x"

    async def test_fetches_top_k_times_multiplier(self) -> None:
        repo = make_vector_repo()
        stage = DenseSearchStage(
            vector_repo=repo,
            openai_client=make_openai_client(),
            embedding_model="baai/bge-m3",
            fetch_multiplier=5,
        )
        await stage.run(Context(query="hi", top_k=4))
        _, kwargs = repo.search.await_args
        assert kwargs["k"] == 20  # 4 * 5

    async def test_filters_to_children_only(self) -> None:
        repo = make_vector_repo()
        stage = DenseSearchStage(
            vector_repo=repo,
            openai_client=make_openai_client(),
            embedding_model="baai/bge-m3",
        )
        await stage.run(Context(query="hi"))
        _, kwargs = repo.search.await_args
        assert kwargs["filters"] == {"is_parent": False}

    async def test_empty_embedding_response_raises_stage_error(self) -> None:
        openai = MagicMock()
        openai.embeddings.create = AsyncMock(return_value=MagicMock(data=[]))
        stage = DenseSearchStage(
            vector_repo=make_vector_repo(),
            openai_client=openai,
            embedding_model="baai/bge-m3",
        )
        with pytest.raises(StageError) as exc_info:
            await stage.run(Context(query="hi"))
        assert exc_info.value.stage_name == "dense"

    async def test_no_hits_produces_empty_state_entry(self) -> None:
        stage = DenseSearchStage(
            vector_repo=make_vector_repo([]),
            openai_client=make_openai_client(),
            embedding_model="baai/bge-m3",
        )
        ctx = await stage.run(Context(query="hi"))
        assert ctx.state["dense"] == []
