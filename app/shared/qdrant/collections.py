from __future__ import annotations

import re

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

# Dimensionality of the default embedding model (BAAI/bge-m3).
BGE_M3_DIM = 1024

# Payload fields indexed for filtering/faceting at query time.
# Keyword indexes accelerate exact-match filters; `tags` is a list but Qdrant
# indexes each element of an array-valued keyword field.
_INDEXED_PAYLOAD_FIELDS: dict[str, models.PayloadSchemaType] = {
    "source_url": models.PayloadSchemaType.KEYWORD,
    "source_type": models.PayloadSchemaType.KEYWORD,
    "parent_chunk_id": models.PayloadSchemaType.KEYWORD,
    "is_parent": models.PayloadSchemaType.BOOL,
    "tags": models.PayloadSchemaType.KEYWORD,
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(model: str) -> str:
    """Qdrant collection names accept letters, digits, '_' and '-'. Slugify to be safe."""
    return _SLUG_RE.sub("_", model.lower()).strip("_")


def collection_name_for(model: str, version: str) -> str:
    """Build a deterministic, model-scoped collection name.

    Swapping the embedding model means a new collection (not contaminated
    dims), so the model slug is part of the name. `version` is bumped whenever
    the payload schema changes in a breaking way.

        collection_name_for("baai/bge-m3", "v1") -> "documents_baai_bge_m3_v1"
    """
    return f"documents_{_slugify(model)}_{version}"


async def ensure_collection(
    client: AsyncQdrantClient,
    name: str,
    *,
    vector_size: int,
    distance: models.Distance = models.Distance.COSINE,
) -> bool:
    """Create the collection + payload indexes if missing. Returns True if created."""
    if await client.collection_exists(name):
        return False
    await client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=vector_size, distance=distance),
    )
    for field, schema in _INDEXED_PAYLOAD_FIELDS.items():
        await client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=schema,
        )
    return True
