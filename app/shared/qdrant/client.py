from qdrant_client import AsyncQdrantClient

from app.config import get_settings
from app.shared.core.exceptions import VectorStoreUnavailableError


def get_qdrant_client() -> AsyncQdrantClient:
    """Build an AsyncQdrantClient from settings.

    The caller owns the client and is responsible for closing it (Qdrant keeps
    a pooled httpx.AsyncClient under the hood). In the FastAPI lifespan we'll
    share a single instance across requests.
    """
    settings = get_settings()
    if not settings.qdrant_url:
        raise VectorStoreUnavailableError("QDRANT_URL is not configured")
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )
