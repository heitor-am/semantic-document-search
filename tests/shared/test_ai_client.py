import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.config import get_settings
from app.shared.ai.client import RETRY_EXCEPTIONS, get_openrouter_client
from app.shared.core.exceptions import LLMUnavailableError

FAKE_REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()


class TestGetOpenRouterClient:
    def test_raises_when_api_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        with pytest.raises(LLMUnavailableError, match="OPENROUTER_API_KEY"):
            get_openrouter_client()

    def test_returns_configured_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("OPENROUTER_APP_NAME", "Test App")
        monkeypatch.setenv("OPENROUTER_APP_URL", "https://test.local")

        client = get_openrouter_client()
        assert isinstance(client, AsyncOpenAI)
        assert "openrouter.ai" in str(client.base_url)

    def test_identifying_headers_are_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("OPENROUTER_APP_NAME", "My App")
        monkeypatch.setenv("OPENROUTER_APP_URL", "https://example.com")

        client = get_openrouter_client()
        assert client.default_headers.get("HTTP-Referer") == "https://example.com"
        assert client.default_headers.get("X-Title") == "My App"


class TestRetryConfig:
    def test_retry_exceptions_include_transient_errors(self) -> None:
        assert APIConnectionError in RETRY_EXCEPTIONS
        assert APITimeoutError in RETRY_EXCEPTIONS
        assert RateLimitError in RETRY_EXCEPTIONS

    async def test_retries_up_to_three_times_on_connection_error(self) -> None:
        # Build a fast-wait retry with the same triggers as openai_retry.
        fast_retry = retry(
            retry=retry_if_exception_type(RETRY_EXCEPTIONS),
            stop=stop_after_attempt(3),
            wait=wait_fixed(0),
            reraise=True,
        )
        attempts = {"count": 0}

        @fast_retry
        async def flaky() -> str:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise APIConnectionError(request=FAKE_REQUEST)
            return "ok"

        assert await flaky() == "ok"
        assert attempts["count"] == 3

    async def test_stops_and_reraises_after_three_attempts(self) -> None:
        fast_retry = retry(
            retry=retry_if_exception_type(RETRY_EXCEPTIONS),
            stop=stop_after_attempt(3),
            wait=wait_fixed(0),
            reraise=True,
        )
        attempts = {"count": 0}

        @fast_retry
        async def always_fails() -> None:
            attempts["count"] += 1
            raise APIConnectionError(request=FAKE_REQUEST)

        with pytest.raises(APIConnectionError):
            await always_fails()
        assert attempts["count"] == 3

    async def test_does_not_retry_on_non_transient_error(self) -> None:
        fast_retry = retry(
            retry=retry_if_exception_type(RETRY_EXCEPTIONS),
            stop=stop_after_attempt(3),
            wait=wait_fixed(0),
            reraise=True,
        )
        attempts = {"count": 0}

        @fast_retry
        async def bad_request() -> None:
            attempts["count"] += 1
            raise ValueError("bad input, not transient")

        with pytest.raises(ValueError):
            await bad_request()
        assert attempts["count"] == 1
