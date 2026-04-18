from unittest.mock import AsyncMock, MagicMock

from app.shared.ai.embeddings import embed_texts


class TestEmbedTexts:
    async def test_empty_input_returns_empty_without_api_call(self) -> None:
        client = MagicMock()
        client.embeddings.create = AsyncMock()

        result = await embed_texts([], client=client)

        assert result == []
        client.embeddings.create.assert_not_awaited()

    async def test_returns_embeddings_from_api_response(self) -> None:
        client = MagicMock()
        client.embeddings.create = AsyncMock(
            return_value=MagicMock(
                data=[
                    MagicMock(embedding=[0.1, 0.2, 0.3]),
                    MagicMock(embedding=[0.4, 0.5, 0.6]),
                ]
            )
        )

        vectors = await embed_texts(["hello", "world"], client=client)

        assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    async def test_uses_default_embedding_model_when_not_overridden(self) -> None:
        client = MagicMock()
        client.embeddings.create = AsyncMock(return_value=MagicMock(data=[]))

        await embed_texts(["text"], client=client)

        _, kwargs = client.embeddings.create.await_args
        assert kwargs["model"]  # non-empty default
        assert kwargs["input"] == ["text"]

    async def test_explicit_model_override_is_passed_through(self) -> None:
        client = MagicMock()
        client.embeddings.create = AsyncMock(return_value=MagicMock(data=[]))

        await embed_texts(["text"], client=client, model="custom/model-v1")

        _, kwargs = client.embeddings.create.await_args
        assert kwargs["model"] == "custom/model-v1"
