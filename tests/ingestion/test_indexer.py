from datetime import UTC, datetime
from typing import Any

import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.indexer import _chunk_to_point, index_chunks
from app.ingestion.schemas import Chunk, SourceDocument
from app.shared.qdrant.collections import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from app.shared.qdrant.repository import SparseValue, VectorPoint


class FakeVectorRepo:
    """In-memory stand-in for VectorRepository — lets us inspect upserts directly."""

    def __init__(self) -> None:
        self.upserts: list[list[VectorPoint]] = []

    async def upsert(self, points):  # type: ignore[no-untyped-def]
        self.upserts.append(list(points))


def make_doc(body: str = "# H\n\nbody content here", **overrides: Any) -> SourceDocument:
    defaults: dict[str, Any] = {
        "source_url": "https://dev.to/user/a-post",
        "source_type": "dev.to",
        "title": "A Post",
        "body_markdown": body,
        "author": "user",
        "published_at": datetime(2026, 1, 1, tzinfo=UTC),
        "tags": ["rag", "python"],
    }
    defaults.update(overrides)
    return SourceDocument(**defaults)


def make_chunk(**overrides: Any) -> Chunk:
    defaults: dict[str, Any] = {
        "chunk_id": "id-0",
        "parent_chunk_id": None,
        "content": "hello",
        "char_count": 5,
        "section_path": ("Intro",),
        "chunk_index": 0,
        "source_url": "https://dev.to/user/a-post",
        "source_type": "dev.to",
        "title": "A Post",
        "author": "user",
        "published_at": datetime(2026, 1, 1, tzinfo=UTC),
        "tags": ("rag",),
    }
    defaults.update(overrides)
    return Chunk(**defaults)


class TestIndexChunks:
    async def test_upserts_every_chunk_with_its_vector(self) -> None:
        chunks = chunk_document(make_doc())
        embeddings = [[0.0] * 4 for _ in chunks]
        repo = FakeVectorRepo()

        count = await index_chunks(chunks, embeddings, vector_repo=repo)

        assert count == len(chunks)
        assert len(repo.upserts) == 1
        assert len(repo.upserts[0]) == len(chunks)

    async def test_empty_chunks_skips_upsert(self) -> None:
        repo = FakeVectorRepo()
        count = await index_chunks([], [], vector_repo=repo)
        assert count == 0
        assert repo.upserts == []

    async def test_rejects_misaligned_lengths(self) -> None:
        repo = FakeVectorRepo()
        with pytest.raises(ValueError, match="mismatch"):
            await index_chunks([make_chunk()], [], vector_repo=repo)

    async def test_points_carry_chunk_ids_verbatim(self) -> None:
        chunks = [
            make_chunk(chunk_id="alpha"),
            make_chunk(chunk_id="beta", chunk_index=1),
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        repo = FakeVectorRepo()

        await index_chunks(chunks, embeddings, vector_repo=repo)

        ids = [p.id for p in repo.upserts[0]]
        assert ids == ["alpha", "beta"]


class TestChunkToPoint:
    def test_payload_flags_parent_chunks(self) -> None:
        parent = make_chunk(parent_chunk_id=None)
        point = _chunk_to_point(parent, [0.0] * 4)
        assert point.payload["is_parent"] is True
        assert point.payload["parent_chunk_id"] is None

    def test_payload_flags_child_chunks(self) -> None:
        child = make_chunk(parent_chunk_id="parent-id")
        point = _chunk_to_point(child, [0.0] * 4)
        assert point.payload["is_parent"] is False
        assert point.payload["parent_chunk_id"] == "parent-id"

    def test_published_at_serialized_as_isoformat(self) -> None:
        point = _chunk_to_point(
            make_chunk(published_at=datetime(2026, 3, 1, 12, 30, tzinfo=UTC)),
            [0.0],
        )
        assert point.payload["published_at"] == "2026-03-01T12:30:00+00:00"

    def test_none_published_at_stays_none(self) -> None:
        point = _chunk_to_point(make_chunk(published_at=None), [0.0])
        assert point.payload["published_at"] is None

    def test_tuple_fields_serialized_as_lists(self) -> None:
        # Qdrant payloads are JSON; tuples would not be JSON-serializable.
        point = _chunk_to_point(
            make_chunk(section_path=("A", "B"), tags=("rag", "ai")),
            [0.0],
        )
        assert point.payload["section_path"] == ["A", "B"]
        assert point.payload["tags"] == ["rag", "ai"]

    def test_id_and_dense_vector_pass_through(self) -> None:
        point = _chunk_to_point(make_chunk(chunk_id="xyz"), [0.5, 0.5, 0.5])
        assert point.id == "xyz"
        assert list(point.vectors[DENSE_VECTOR_NAME]) == [0.5, 0.5, 0.5]

    def test_sparse_bm25_vector_produced_for_tokenisable_content(self) -> None:
        point = _chunk_to_point(make_chunk(content="python async programming"), [0.5])
        assert SPARSE_VECTOR_NAME in point.vectors
        sparse = point.vectors[SPARSE_VECTOR_NAME]
        assert isinstance(sparse, SparseValue)
        assert len(sparse.indices) == 3  # 3 distinct tokens
        assert list(sparse.values) == [1.0, 1.0, 1.0]  # each appears once

    def test_sparse_vector_omitted_for_empty_content(self) -> None:
        # Content with no tokens (pure punctuation) → dense-only point.
        point = _chunk_to_point(make_chunk(content="!!! ???"), [0.5])
        assert SPARSE_VECTOR_NAME not in point.vectors
        assert DENSE_VECTOR_NAME in point.vectors
