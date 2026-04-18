from __future__ import annotations

from collections.abc import Sequence

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

# bge-m3 on OpenRouter occasionally returns HTTP 200 with `data=[]` under
# rate-limit pressure instead of a proper 429. The OpenAI SDK surfaces that
# as `ValueError("No embedding data received")`. Retrying catches it —
# widening the openai_retry policy to include ValueError for this call
# specifically (global chat calls shouldn't silently retry ValueErrors).
_embed_retry = retry(
    retry=retry_if_exception_type(
        (APIConnectionError, APITimeoutError, RateLimitError, ValueError)
    ),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=0.8, min=0.5, max=15),
    reraise=True,
)


@_embed_retry
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
