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
from app.shared.core.exceptions import LLMUnavailableError

RETRY_EXCEPTIONS = (APIConnectionError, APITimeoutError, RateLimitError)


def get_openrouter_client() -> AsyncOpenAI:
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise LLMUnavailableError("OPENROUTER_API_KEY is not configured")

    return AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers={
            "HTTP-Referer": settings.openrouter_app_url,
            "X-Title": settings.openrouter_app_name,
        },
    )


openai_retry = retry(
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
