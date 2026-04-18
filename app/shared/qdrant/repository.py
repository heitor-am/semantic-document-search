from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from app.shared.qdrant.collections import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME


@dataclass(frozen=True)
class SparseValue:
    indices: Sequence[int]
    values: Sequence[float]


@dataclass(frozen=True)
class VectorPoint:
    """A point to upsert. `vectors` is a dict of named vectors, where each
    value is either a dense sequence (list[float]) or a SparseValue. Points
    are allowed to ship only some of the named vectors — a chunk whose
    content tokenises to nothing carries only the dense vector."""

    id: str
    vectors: Mapping[str, Sequence[float] | SparseValue]
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class VectorHit:
    id: str
    score: float
    payload: Mapping[str, Any]


class VectorRepository(Protocol):
    """Narrow protocol so the store is swappable (Milvus, Weaviate) and the
    retrieval pipeline can be unit-tested without Qdrant running."""

    async def upsert(self, points: Sequence[VectorPoint]) -> None: ...

    async def search_hybrid(
        self,
        *,
        dense_vector: Sequence[float],
        sparse: SparseValue | None,
        k: int = 10,
        prefetch_limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorHit]:
        """Server-side hybrid query: dense + (optional) sparse prefetches,
        RRF fusion, returns top-k. `prefetch_limit` controls how many
        candidates each prefetch returns before fusion; defaults to 4 * k.
        `sparse=None` collapses to a pure dense query."""
        ...

    async def scroll(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 256,
        offset: Any = None,
    ) -> tuple[list[VectorHit], Any]: ...

    async def delete_by_source(self, source_url: str) -> None: ...


class QdrantRepository:
    """Concrete `VectorRepository` over `AsyncQdrantClient`.

    One instance per collection — the collection name is bound at construction
    so call sites don't have to repeat it.
    """

    def __init__(self, client: AsyncQdrantClient, *, collection: str) -> None:
        self._client = client
        self._collection = collection

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        qdrant_points = [
            models.PointStruct(
                id=p.id,
                vector={name: _to_qdrant_vector(v) for name, v in p.vectors.items()},
                payload=dict(p.payload),
            )
            for p in points
        ]
        await self._client.upsert(collection_name=self._collection, points=qdrant_points)

    async def search_hybrid(
        self,
        *,
        dense_vector: Sequence[float],
        sparse: SparseValue | None,
        k: int = 10,
        prefetch_limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorHit]:
        prefetch_limit = prefetch_limit or (k * 4)
        qdrant_filter = _build_filter(filters) if filters else None

        prefetch: list[models.Prefetch] = [
            models.Prefetch(
                query=list(dense_vector),
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=qdrant_filter,
            ),
        ]
        if sparse is not None and sparse.indices:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(sparse.indices),
                        values=list(sparse.values),
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                )
            )

        response = await self._client.query_points(
            collection_name=self._collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        )
        return [
            VectorHit(id=str(p.id), score=float(p.score), payload=dict(p.payload or {}))
            for p in response.points
        ]

    async def scroll(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 256,
        offset: Any = None,
    ) -> tuple[list[VectorHit], Any]:
        points, next_offset = await self._client.scroll(
            collection_name=self._collection,
            scroll_filter=_build_filter(filters) if filters else None,
            offset=offset,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        hits = [VectorHit(id=str(p.id), score=0.0, payload=dict(p.payload or {})) for p in points]
        return hits, next_offset

    async def delete_by_source(self, source_url: str) -> None:
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_url",
                        match=models.MatchValue(value=source_url),
                    ),
                ],
            ),
        )
        await self._client.delete(collection_name=self._collection, points_selector=selector)


def _to_qdrant_vector(
    value: Sequence[float] | SparseValue,
) -> list[float] | models.SparseVector:
    if isinstance(value, SparseValue):
        return models.SparseVector(indices=list(value.indices), values=list(value.values))
    return list(value)


def _build_filter(filters: Mapping[str, Any]) -> models.Filter:
    """Translate a plain {field: value | [values]} mapping into a Qdrant Filter.

    Lists become MatchAny (OR within one field), scalars become MatchValue.
    All conditions are joined with AND (`must`).
    """
    return models.Filter(must=[_condition(key, value) for key, value in filters.items()])


def _condition(key: str, value: Any) -> models.FieldCondition:
    if isinstance(value, list):
        return models.FieldCondition(key=key, match=models.MatchAny(any=value))
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))
