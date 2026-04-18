from __future__ import annotations

from collections.abc import Sequence

from app.ingestion.schemas import Chunk
from app.retrieval.sparse_encoder import encode_bm25_sparse
from app.shared.qdrant.collections import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from app.shared.qdrant.repository import SparseValue, VectorPoint, VectorRepository


async def index_chunks(
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    *,
    vector_repo: VectorRepository,
) -> int:
    """Upsert `chunks` into the vector store with both dense and sparse vectors.

    The dense vector comes from the embedder (caller); the sparse BM25
    vector is computed here from the chunk's content. Qdrant applies the
    IDF modifier server-side (collection is configured with
    `modifier=IDF`), so we only ship raw term frequencies.

    Separating embedding from upsert keeps both halves unit-testable with
    plain in-memory fakes. Deterministic chunk_ids (Etapa 5) make upsert
    idempotent end-to-end.

    Returns the number of points upserted.

    Raises:
        ValueError: if `chunks` and `embeddings` don't align 1:1.
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


def _chunk_to_point(chunk: Chunk, dense_vector: Sequence[float]) -> VectorPoint:
    vectors: dict[str, Sequence[float] | SparseValue] = {DENSE_VECTOR_NAME: list(dense_vector)}

    # Sparse BM25: skipped for chunks whose content tokenises to nothing
    # (very short or punctuation-only). Qdrant tolerates points that
    # declare only a subset of the collection's named vectors.
    indices, values = encode_bm25_sparse(chunk.content)
    if indices:
        vectors[SPARSE_VECTOR_NAME] = SparseValue(indices=indices, values=values)

    return VectorPoint(
        id=chunk.chunk_id,
        vectors=vectors,
        payload={
            "content": chunk.content,
            "char_count": chunk.char_count,
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
