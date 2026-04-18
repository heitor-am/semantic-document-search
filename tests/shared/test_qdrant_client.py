import pytest
from qdrant_client import AsyncQdrantClient

from app.config import Settings, get_settings
from app.shared.core.exceptions import VectorStoreUnavailableError
from app.shared.qdrant.client import get_qdrant_client


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_raises_when_qdrant_url_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.shared.qdrant.client.get_settings",
        lambda: Settings(qdrant_url=""),
    )
    with pytest.raises(VectorStoreUnavailableError, match="QDRANT_URL"):
        get_qdrant_client()


async def test_builds_client_when_url_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.shared.qdrant.client.get_settings",
        lambda: Settings(qdrant_url="https://localhost:6333", qdrant_api_key="secret"),
    )
    client = get_qdrant_client()
    try:
        assert isinstance(client, AsyncQdrantClient)
    finally:
        await client.close()
