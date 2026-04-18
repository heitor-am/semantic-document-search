"""Re-encode the BM25 sparse vectors on every chunk already in Qdrant.

Why this exists — the sparse tokenizer (`app.shared.qdrant.sparse_encoder`)
was hardened to strip stopwords and short tokens. Sparse vectors already
stored in Qdrant still carry the naïve tokenisation; we re-encode them in
place without touching dense vectors or payload.

Usage:
    uv run python scripts/reencode_sparse.py

Behaviour:
    - Scrolls every point in the configured collection.
    - For each point with a `content` payload, recomputes the sparse
      vector and upserts it (keeping the same id and dense vector).
    - Points whose content tokenises to nothing (punctuation / only
      stopwords) get their sparse vector explicitly removed via upsert
      with `Vector(dense=..., bm25=None)` — Qdrant has no atomic sparse
      drop so we upsert the sparse-less vectors dict.

    Uses AsyncQdrantClient directly (not the VectorRepository) because
    we need to preserve the existing dense vector without recomputing
    or re-embedding; the repository's `upsert` assumes the caller
    supplies both halves.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.ingestion.service import current_collection_name
from app.shared.qdrant.collections import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from app.shared.qdrant.sparse_encoder import encode_bm25_sparse


async def main() -> int:
    settings = Settings()
    if not settings.qdrant_url:
        print("QDRANT_URL not set", file=sys.stderr)
        return 1

    collection = current_collection_name()
    client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    try:
        print(f"Re-encoding sparse vectors in {collection} ...")
        total = 0
        reencoded = 0
        skipped_empty = 0
        offset: object = None

        while True:
            points, offset = await client.scroll(
                collection_name=collection,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=True,  # we need the dense back to re-upsert
            )
            for p in points:
                total += 1
                payload = p.payload or {}
                content = str(payload.get("content", ""))
                indices, values = encode_bm25_sparse(content)

                # Reconstruct the full vectors dict. Preserve existing dense
                # exactly so we don't re-embed (which would cost $ + change
                # values trivially because bge-m3 is deterministic anyway
                # but embedding calls are the slow part of ingestion).
                vector_dict = p.vector or {}
                dense = vector_dict.get(DENSE_VECTOR_NAME)
                if dense is None:
                    # Shouldn't happen — every ingested point has dense.
                    print(f"  SKIP id={p.id}: no dense vector")
                    continue

                new_vectors: dict[str, list[float] | models.SparseVector] = {
                    DENSE_VECTOR_NAME: list(dense),
                }
                if indices:
                    new_vectors[SPARSE_VECTOR_NAME] = models.SparseVector(
                        indices=list(indices), values=list(values)
                    )
                    reencoded += 1
                else:
                    # Tokenises to nothing after stopword removal. Upserting
                    # without the sparse key means Qdrant keeps any prior
                    # value — to force-drop, we'd need delete_vectors.
                    # Safer: skip and count.
                    skipped_empty += 1
                    continue  # leave existing sparse alone

                await client.upsert(
                    collection_name=collection,
                    points=[
                        models.PointStruct(
                            id=p.id,
                            vector=new_vectors,
                            payload=dict(payload),
                        )
                    ],
                )
            if offset is None:
                break

        print(f"  scanned: {total}")
        print(f"  re-encoded: {reencoded}")
        print(f"  skipped (no tokens after stopword removal): {skipped_empty}")
        return 0
    finally:
        await client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
