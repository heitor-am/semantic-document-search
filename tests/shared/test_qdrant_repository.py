from unittest.mock import AsyncMock, MagicMock

from qdrant_client.http import models

from app.shared.qdrant.repository import (
    QdrantRepository,
    VectorHit,
    VectorPoint,
    _build_filter,
)


def make_repo() -> tuple[QdrantRepository, AsyncMock]:
    client = AsyncMock()
    return QdrantRepository(client, collection="docs_v1"), client


class TestUpsert:
    async def test_sends_point_structs_to_client(self) -> None:
        repo, client = make_repo()
        points = [
            VectorPoint(id="id-1", vector=[0.1, 0.2], payload={"content": "a"}),
            VectorPoint(id="id-2", vector=[0.3, 0.4], payload={"content": "b"}),
        ]

        await repo.upsert(points)

        client.upsert.assert_awaited_once()
        _, kwargs = client.upsert.await_args
        assert kwargs["collection_name"] == "docs_v1"
        sent = kwargs["points"]
        assert len(sent) == 2
        assert sent[0].id == "id-1"
        assert sent[0].vector == [0.1, 0.2]
        assert sent[0].payload == {"content": "a"}

    async def test_empty_points_is_a_noop(self) -> None:
        repo, client = make_repo()
        await repo.upsert([])
        client.upsert.assert_not_awaited()


class TestSearch:
    async def test_returns_vector_hits_from_qdrant_response(self) -> None:
        repo, client = make_repo()
        # Qdrant's query_points returns an object with .points = [ScoredPoint...]
        scored_points = [
            MagicMock(id="hit-1", score=0.9, payload={"title": "A"}),
            MagicMock(id="hit-2", score=0.7, payload={"title": "B"}),
        ]
        client.query_points = AsyncMock(return_value=MagicMock(points=scored_points))

        hits = await repo.search([0.1] * 4, k=5)

        assert hits == [
            VectorHit(id="hit-1", score=0.9, payload={"title": "A"}),
            VectorHit(id="hit-2", score=0.7, payload={"title": "B"}),
        ]
        _, kwargs = client.query_points.await_args
        assert kwargs["collection_name"] == "docs_v1"
        assert kwargs["limit"] == 5
        assert kwargs["query_filter"] is None

    async def test_builds_filter_from_mapping(self) -> None:
        repo, client = make_repo()
        client.query_points = AsyncMock(return_value=MagicMock(points=[]))

        await repo.search([0.0] * 4, filters={"is_parent": False, "tags": ["rag", "ai"]})

        _, kwargs = client.query_points.await_args
        qfilter = kwargs["query_filter"]
        assert isinstance(qfilter, models.Filter)
        assert qfilter.must is not None
        assert len(qfilter.must) == 2

    async def test_handles_payload_none_as_empty_dict(self) -> None:
        repo, client = make_repo()
        scored = [MagicMock(id="hit-1", score=0.5, payload=None)]
        client.query_points = AsyncMock(return_value=MagicMock(points=scored))

        hits = await repo.search([0.1] * 4)

        assert hits[0].payload == {}


class TestScroll:
    async def test_returns_hits_and_offset_from_client(self) -> None:
        repo, client = make_repo()
        scroll_points = [
            MagicMock(id="c1", payload={"content": "a"}),
            MagicMock(id="c2", payload={"content": "b"}),
        ]
        client.scroll = AsyncMock(return_value=(scroll_points, "next-offset"))

        hits, offset = await repo.scroll(filters={"is_parent": False}, limit=2)

        assert [h.id for h in hits] == ["c1", "c2"]
        assert hits[0].score == 0.0  # scroll doesn't produce scores
        assert hits[0].payload == {"content": "a"}
        assert offset == "next-offset"

        _, kwargs = client.scroll.await_args
        assert kwargs["collection_name"] == "docs_v1"
        assert kwargs["limit"] == 2
        assert kwargs["with_vectors"] is False
        assert kwargs["query_filter" if "query_filter" in kwargs else "scroll_filter"] is not None

    async def test_none_offset_signals_end(self) -> None:
        repo, client = make_repo()
        client.scroll = AsyncMock(return_value=([], None))

        hits, offset = await repo.scroll()

        assert hits == []
        assert offset is None


class TestDeleteBySource:
    async def test_deletes_points_matching_source_url(self) -> None:
        repo, client = make_repo()

        await repo.delete_by_source("https://dev.to/a/post")

        client.delete.assert_awaited_once()
        _, kwargs = client.delete.await_args
        assert kwargs["collection_name"] == "docs_v1"
        selector = kwargs["points_selector"]
        assert isinstance(selector, models.FilterSelector)
        conditions = selector.filter.must
        assert conditions is not None
        assert len(conditions) == 1


class TestBuildFilter:
    def test_scalar_becomes_match_value(self) -> None:
        f = _build_filter({"is_parent": False})
        assert f.must is not None
        assert isinstance(f.must[0].match, models.MatchValue)

    def test_list_becomes_match_any(self) -> None:
        f = _build_filter({"tags": ["a", "b"]})
        assert f.must is not None
        assert isinstance(f.must[0].match, models.MatchAny)

    def test_multiple_keys_all_required(self) -> None:
        f = _build_filter({"is_parent": False, "source_type": "dev.to"})
        assert f.must is not None
        assert len(f.must) == 2
