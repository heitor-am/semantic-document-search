from __future__ import annotations

import re

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

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
# Qdrant collection names accept letters, digits, '_' and '-'. Versions come
# from env / config so we validate rather than silently slugify (a typo in
# QDRANT_COLLECTION_VERSION should be loud, not mapped to a neighbour name).
_VERSION_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _slugify(model: str) -> str:
    return _SLUG_RE.sub("_", model.lower()).strip("_")


def collection_name_for(model: str, version: str) -> str:
    """Build a deterministic, model-scoped collection name.

    Swapping the embedding model means a new collection (not contaminated
    dims), so the model slug is part of the name. `version` is bumped whenever
    the payload schema changes in a breaking way.

        collection_name_for("baai/bge-m3", "v1") -> "documents_baai_bge_m3_v1"

    Raises:
        ValueError: if `version` contains characters outside `[A-Za-z0-9_-]`.
    """
    if not _VERSION_RE.fullmatch(version):
        raise ValueError(f"invalid collection version {version!r}: must match [A-Za-z0-9_-]+")
    return f"documents_{_slugify(model)}_{version}"


async def ensure_collection(
    client: AsyncQdrantClient,
    name: str,
    *,
    vector_size: int,
    distance: models.Distance = models.Distance.COSINE,
) -> bool:
    """Create the collection + payload indexes if missing. Returns True if created.

    Idempotent under concurrent callers: if a racing worker creates the
    collection between our `collection_exists` check and `create_collection`,
    the second caller observes the conflict, re-verifies existence, and
    returns False instead of raising.
    """
    if await client.collection_exists(name):
        return False
    try:
        await client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance),
        )
    except (UnexpectedResponse, ValueError):
        # Race: another worker created the collection. If it's really there
        # now, that's fine; otherwise, the failure was something else.
        if await client.collection_exists(name):
            return False
        raise
    for field, schema in _INDEXED_PAYLOAD_FIELDS.items():
        await client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=schema,
        )
    return True
