from unittest.mock import AsyncMock

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from app.shared.qdrant.collections import (
    BGE_M3_DIM,
    collection_name_for,
    ensure_collection,
)


def _already_exists_error() -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=409,
        reason_phrase="Conflict",
        content=b"already exists",
        headers=None,
    )


class TestCollectionNameFor:
    def test_slugifies_slashes_and_dashes(self) -> None:
        assert collection_name_for("baai/bge-m3", "v1") == "documents_baai_bge_m3_v1"

    def test_is_stable_across_calls(self) -> None:
        a = collection_name_for("openai/text-embedding-3-small", "v2")
        b = collection_name_for("openai/text-embedding-3-small", "v2")
        assert a == b

    def test_changes_with_model(self) -> None:
        a = collection_name_for("baai/bge-m3", "v1")
        b = collection_name_for("openai/text-embedding-3-small", "v1")
        assert a != b

    def test_changes_with_version(self) -> None:
        assert collection_name_for("baai/bge-m3", "v1") != collection_name_for("baai/bge-m3", "v2")

    def test_lowercases_model(self) -> None:
        # Upper/mixed-case inputs still produce a valid lowercase slug
        assert collection_name_for("BAAI/BGE-m3", "v1") == "documents_baai_bge_m3_v1"

    @pytest.mark.parametrize("bad_version", ["v 1", "v/1", "v.1", "", "v1;drop"])
    def test_rejects_unsafe_version(self, bad_version: str) -> None:
        with pytest.raises(ValueError, match="invalid collection version"):
            collection_name_for("baai/bge-m3", bad_version)


class TestEnsureCollection:
    async def test_creates_collection_when_missing(self) -> None:
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=False)

        created = await ensure_collection(client, "documents_x_v1", vector_size=BGE_M3_DIM)

        assert created is True
        client.create_collection.assert_awaited_once()
        # One payload index call per indexed field
        assert client.create_payload_index.await_count >= 1

    async def test_noop_when_collection_exists(self) -> None:
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=True)

        created = await ensure_collection(client, "documents_x_v1", vector_size=BGE_M3_DIM)

        assert created is False
        client.create_collection.assert_not_awaited()
        client.create_payload_index.assert_not_awaited()

    async def test_passes_vector_size_to_client(self) -> None:
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=False)

        await ensure_collection(client, "documents_x_v1", vector_size=768)

        _, kwargs = client.create_collection.await_args
        assert kwargs["vectors_config"].size == 768

    async def test_treats_concurrent_create_as_noop(self) -> None:
        # First existence check -> False (we try to create);
        # create_collection raises because a racing worker got there first;
        # second existence check -> True (the race winner's create succeeded).
        client = AsyncMock()
        client.collection_exists = AsyncMock(side_effect=[False, True])
        client.create_collection = AsyncMock(side_effect=_already_exists_error())

        created = await ensure_collection(client, "documents_x_v1", vector_size=BGE_M3_DIM)

        assert created is False
        client.create_payload_index.assert_not_awaited()

    async def test_reraises_when_create_fails_for_another_reason(self) -> None:
        # create_collection raises, and the collection really isn't there —
        # we must not swallow the failure.
        client = AsyncMock()
        client.collection_exists = AsyncMock(side_effect=[False, False])
        client.create_collection = AsyncMock(side_effect=_already_exists_error())

        with pytest.raises(UnexpectedResponse):
            await ensure_collection(client, "documents_x_v1", vector_size=BGE_M3_DIM)


@pytest.mark.parametrize(
    "model,version,expected",
    [
        ("baai/bge-m3", "v1", "documents_baai_bge_m3_v1"),
        ("baai/bge-m3", "v2", "documents_baai_bge_m3_v2"),
        ("openai/text-embedding-3-small", "v1", "documents_openai_text_embedding_3_small_v1"),
    ],
)
def test_collection_name_table(model: str, version: str, expected: str) -> None:
    assert collection_name_for(model, version) == expected
