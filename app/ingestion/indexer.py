from __future__ import annotations

from collections.abc import Sequence

from app.ingestion.schemas import Chunk
from app.shared.qdrant.repository import VectorPoint, VectorRepository


async def index_chunks(
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    *,
    vector_repo: VectorRepository,
) -> int:
    """Upsert `chunks` with their matching `embeddings` into the vector store.

    The caller is responsible for producing the embeddings (via
    `app.shared.ai.embeddings.embed_texts`) — separating embedding from
    upsert keeps both halves unit-testable with plain in-memory fakes.

    Because chunk_ids are deterministic (`uuid5(url, parent_idx, child_idx)`),
    re-ingesting the same document upserts in place — no duplication.

    Returns the number of points upserted.

    Raises:
        ValueError: if the two sequences don't align 1:1.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}")
    if not chunks:
        return 0

    points = [
        _chunk_to_point(chunk, vector) for chunk, vector in zip(chunks, embeddings, strict=True)
    ]
    await vector_repo.upsert(points)
    return len(points)


def _chunk_to_point(chunk: Chunk, vector: Sequence[float]) -> VectorPoint:
    return VectorPoint(
        id=chunk.chunk_id,
        vector=vector,
        payload={
            "content": chunk.content,
            "char_count": chunk.char_count,
            # Lists for JSON-serializable payload; Qdrant indexes list fields.
            "section_path": list(chunk.section_path),
            "chunk_index": chunk.chunk_index,
            "parent_chunk_id": chunk.parent_chunk_id,
            "is_parent": chunk.parent_chunk_id is None,
            "source_url": chunk.source_url,
            "source_type": chunk.source_type,
            "title": chunk.title,
            "author": chunk.author,
            "published_at": (
                chunk.published_at.isoformat() if chunk.published_at is not None else None
            ),
            "tags": list(chunk.tags),
        },
    )
