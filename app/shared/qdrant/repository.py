from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models


@dataclass(frozen=True)
class VectorPoint:
    id: str
    vector: Sequence[float]
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

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorHit]: ...

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
            models.PointStruct(id=p.id, vector=list(p.vector), payload=dict(p.payload))
            for p in points
        ]
        await self._client.upsert(collection_name=self._collection, points=qdrant_points)

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorHit]:
        response = await self._client.query_points(
            collection_name=self._collection,
            query=list(query_vector),
            limit=k,
            query_filter=_build_filter(filters) if filters else None,
            with_payload=True,
        )
        return [
            VectorHit(id=str(p.id), score=float(p.score), payload=dict(p.payload or {}))
            for p in response.points
        ]

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
