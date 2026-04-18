from __future__ import annotations

from collections.abc import Sequence

from openai import AsyncOpenAI

from app.config import get_settings
from app.shared.ai.client import openai_retry


@openai_retry
async def embed_texts(
    texts: Sequence[str],
    *,
    client: AsyncOpenAI,
    model: str | None = None,
) -> list[list[float]]:
    """Embed a batch of texts via OpenRouter.

    The caller owns the AsyncOpenAI client (reuse across calls for connection
    pooling). `model` defaults to `settings.openrouter_embedding_model`.

    Empty input returns empty output without hitting the network.
    """
    if not texts:
        return []
    settings = get_settings()
    response = await client.embeddings.create(
        model=model or settings.openrouter_embedding_model,
        input=list(texts),
    )
    return [item.embedding for item in response.data]
