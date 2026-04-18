from unittest.mock import AsyncMock, MagicMock

from qdrant_client.http import models

from app.shared.qdrant.collections import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from app.shared.qdrant.repository import (
    QdrantRepository,
    SparseValue,
    VectorPoint,
    _build_filter,
)


def make_repo() -> tuple[QdrantRepository, AsyncMock]:
    client = AsyncMock()
    return QdrantRepository(client, collection="docs_v2"), client


class TestUpsert:
    async def test_sends_point_structs_with_named_dense_vector(self) -> None:
        repo, client = make_repo()
        points = [
            VectorPoint(
                id="id-1",
                vectors={DENSE_VECTOR_NAME: [0.1, 0.2]},
                payload={"content": "a"},
            ),
        ]

        await repo.upsert(points)

        client.upsert.assert_awaited_once()
        _, kwargs = client.upsert.await_args
        assert kwargs["collection_name"] == "docs_v2"
        sent = kwargs["points"]
        assert len(sent) == 1
        assert sent[0].id == "id-1"
        assert sent[0].vector == {DENSE_VECTOR_NAME: [0.1, 0.2]}
        assert sent[0].payload == {"content": "a"}

    async def test_upsert_includes_sparse_vector_when_present(self) -> None:
        repo, client = make_repo()
        points = [
            VectorPoint(
                id="id-1",
                vectors={
                    DENSE_VECTOR_NAME: [0.1, 0.2],
                    SPARSE_VECTOR_NAME: SparseValue(indices=[7, 12], values=[1.0, 2.0]),
                },
                payload={},
            ),
        ]
        await repo.upsert(points)

        _, kwargs = client.upsert.await_args
        sent = kwargs["points"][0]
        assert isinstance(sent.vector[SPARSE_VECTOR_NAME], models.SparseVector)
        assert sent.vector[SPARSE_VECTOR_NAME].indices == [7, 12]
        assert sent.vector[SPARSE_VECTOR_NAME].values == [1.0, 2.0]

    async def test_empty_points_is_a_noop(self) -> None:
        repo, client = make_repo()
        await repo.upsert([])
        client.upsert.assert_not_awaited()


class TestSearchHybrid:
    async def test_builds_prefetch_and_fusion_query(self) -> None:
        repo, client = make_repo()
        client.query_points = AsyncMock(
            return_value=MagicMock(
                points=[
                    MagicMock(id="c1", score=0.9, payload={"title": "A"}),
                    MagicMock(id="c2", score=0.7, payload={"title": "B"}),
                ]
            )
        )

        hits = await repo.search_hybrid(
            dense_vector=[0.1, 0.2, 0.3, 0.4],
            sparse=SparseValue(indices=[1, 2], values=[1.0, 1.0]),
            k=5,
        )

        assert [h.id for h in hits] == ["c1", "c2"]
        _, kwargs = client.query_points.await_args
        assert kwargs["collection_name"] == "docs_v2"
        assert kwargs["limit"] == 5
        # FusionQuery(RRF) is in the main query
        assert isinstance(kwargs["query"], models.FusionQuery)
        assert kwargs["query"].fusion == models.Fusion.RRF
        # Two prefetches: dense + sparse
        prefetches = kwargs["prefetch"]
        assert len(prefetches) == 2
        assert prefetches[0].using == DENSE_VECTOR_NAME
        assert prefetches[1].using == SPARSE_VECTOR_NAME
        assert isinstance(prefetches[1].query, models.SparseVector)

    async def test_skips_sparse_prefetch_when_sparse_is_none(self) -> None:
        repo, client = make_repo()
        client.query_points = AsyncMock(return_value=MagicMock(points=[]))

        await repo.search_hybrid(dense_vector=[0.1] * 4, sparse=None, k=3)

        _, kwargs = client.query_points.await_args
        assert len(kwargs["prefetch"]) == 1
        assert kwargs["prefetch"][0].using == DENSE_VECTOR_NAME

    async def test_skips_sparse_prefetch_when_sparse_is_empty(self) -> None:
        # A query that tokenises to nothing (pure punctuation) yields an
        # empty SparseValue; it should collapse to dense-only, not send
        # an empty SparseVector (which Qdrant rejects).
        repo, client = make_repo()
        client.query_points = AsyncMock(return_value=MagicMock(points=[]))

        await repo.search_hybrid(
            dense_vector=[0.1] * 4,
            sparse=SparseValue(indices=[], values=[]),
            k=3,
        )

        _, kwargs = client.query_points.await_args
        assert len(kwargs["prefetch"]) == 1

    async def test_applies_filter_on_both_prefetches(self) -> None:
        repo, client = make_repo()
        client.query_points = AsyncMock(return_value=MagicMock(points=[]))

        await repo.search_hybrid(
            dense_vector=[0.1] * 4,
            sparse=SparseValue(indices=[1], values=[1.0]),
            k=3,
            filters={"is_parent": False},
        )

        _, kwargs = client.query_points.await_args
        for p in kwargs["prefetch"]:
            assert isinstance(p.filter, models.Filter)

    async def test_prefetch_limit_defaults_to_4x_k(self) -> None:
        repo, client = make_repo()
        client.query_points = AsyncMock(return_value=MagicMock(points=[]))

        await repo.search_hybrid(
            dense_vector=[0.1] * 4,
            sparse=None,
            k=5,
        )

        _, kwargs = client.query_points.await_args
        assert kwargs["prefetch"][0].limit == 20

    async def test_prefetch_limit_override_respected(self) -> None:
        repo, client = make_repo()
        client.query_points = AsyncMock(return_value=MagicMock(points=[]))

        await repo.search_hybrid(
            dense_vector=[0.1] * 4,
            sparse=None,
            k=5,
            prefetch_limit=50,
        )

        _, kwargs = client.query_points.await_args
        assert kwargs["prefetch"][0].limit == 50

    async def test_handles_payload_none_as_empty_dict(self) -> None:
        repo, client = make_repo()
        client.query_points = AsyncMock(
            return_value=MagicMock(
                points=[MagicMock(id="c1", score=0.5, payload=None)],
            )
        )

        hits = await repo.search_hybrid(dense_vector=[0.1] * 4, sparse=None, k=3)

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
        assert hits[0].score == 0.0
        assert hits[0].payload == {"content": "a"}
        assert offset == "next-offset"

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
        assert kwargs["collection_name"] == "docs_v2"
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
