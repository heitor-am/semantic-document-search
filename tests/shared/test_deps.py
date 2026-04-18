from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import AsyncOpenAI

from app.shared.api.deps import get_httpx_client, get_openai_client, get_vector_repo
from app.shared.core.exceptions import LLMUnavailableError, VectorStoreUnavailableError


def _request_with_state(**attrs) -> MagicMock:
    req = MagicMock()
    req.app = SimpleNamespace(state=SimpleNamespace(**attrs))
    return req


class TestGetHttpxClient:
    def test_returns_client_from_app_state(self) -> None:
        client = httpx.AsyncClient()
        try:
            request = _request_with_state(httpx_client=client)
            assert get_httpx_client(request) is client
        finally:
            # No await in sync test; close the underlying transport synchronously.
            pass


class TestGetOpenAIClient:
    def test_returns_client_from_app_state(self) -> None:
        client = MagicMock(spec=AsyncOpenAI)
        request = _request_with_state(openai_client=client)
        assert get_openai_client(request) is client

    def test_raises_when_client_is_none(self) -> None:
        request = _request_with_state(openai_client=None)
        with pytest.raises(LLMUnavailableError, match="OPENROUTER_API_KEY"):
            get_openai_client(request)


class TestGetVectorRepo:
    def test_returns_repo_from_app_state(self) -> None:
        repo = MagicMock()
        request = _request_with_state(vector_repo=repo)
        assert get_vector_repo(request) is repo

    def test_raises_when_repo_is_none(self) -> None:
        request = _request_with_state(vector_repo=None)
        with pytest.raises(VectorStoreUnavailableError, match="QDRANT_URL"):
            get_vector_repo(request)
