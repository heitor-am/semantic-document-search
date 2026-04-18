from __future__ import annotations

import re

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

# Named vectors used across the codebase. Keeping these as module-level
# constants means indexer, retrieval, and collection-creation can never
# drift on the spelling.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

# Dimensionality of the default embedding model (BAAI/bge-m3).
BGE_M3_DIM = 1024

# Embedding model → vector dimension. Swapping the embedding model means a
# new Qdrant collection (ADR-006), so adding a model here should be a
# deliberate, reviewed change.
MODEL_VECTOR_SIZES: dict[str, int] = {
    "baai/bge-m3": BGE_M3_DIM,
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
}


def vector_size_for(model: str) -> int:
    """Return the embedding dimension registered for a model.

    Fails loudly on unknown models — silently defaulting would cause a
    runtime dim mismatch on the first upsert, which is a harder bug to
    trace back to a config change.
    """
    key = model.lower()
    if key not in MODEL_VECTOR_SIZES:
        raise ValueError(
            f"unknown embedding model {model!r}; register its dimension in MODEL_VECTOR_SIZES"
        )
    return MODEL_VECTOR_SIZES[key]


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

    The schema provisions two **named vectors** on every point:
      - `dense` (float32) — the chunk embedding used for semantic search.
      - `bm25` (sparse) — the BM25 term-frequency vector; Qdrant applies
        IDF server-side via the `modifier=IDF` setting, so the ingestor
        just ships raw term frequencies.

    Pairing them on the same point means a single server-side hybrid
    query (Query API + FusionQuery) can fuse them in one round-trip,
    instead of the app holding a duplicate BM25 corpus in memory.

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
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(size=vector_size, distance=distance),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
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
