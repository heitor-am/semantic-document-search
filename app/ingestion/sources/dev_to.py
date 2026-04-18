from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import httpx

from app.ingestion.parser import normalize_markdown
from app.ingestion.schemas import SourceDocument

DEV_TO_API_BASE = "https://dev.to/api"
SOURCE_TYPE = "dev.to"


class InvalidDevToUrlError(ValueError):
    """Raised when a URL doesn't match the expected dev.to article shape."""


def parse_dev_to_url(url: str) -> str:
    """Given a dev.to article URL, return the 'username/slug' API path."""
    parsed = urlparse(url)

    # Use hostname (strips port) and match exact/subdomain only — 'evildev.to'
    # or similar lookalike domains are rejected even though they end with 'dev.to'.
    hostname = parsed.hostname
    if hostname is None or (hostname != "dev.to" and not hostname.endswith(".dev.to")):
        raise InvalidDevToUrlError(f"not a dev.to URL: {url!r}")

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) != 2:
        raise InvalidDevToUrlError(f"expected /{{username}}/{{slug}} path, got {parsed.path!r}")

    return f"{parts[0]}/{parts[1]}"


def _parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _parse_tags(raw: Any) -> list[str]:
    """Normalize dev.to's `tag_list` to a list of strings.

    The dev.to API is inconsistent: the article detail endpoint returns a
    comma-separated string (`"webdev, javascript, css"`) while the list
    endpoint returns an array. Smoke testing against a real article
    surfaced this — both shapes have to round-trip.
    """
    if not raw:
        return []
    if isinstance(raw, str):
        return [tag.strip() for tag in raw.split(",") if tag.strip()]
    if isinstance(raw, list):
        return [str(tag) for tag in raw]
    return []


async def fetch_dev_to(url: str, *, client: httpx.AsyncClient) -> SourceDocument:
    """Fetch a dev.to article and normalize it to SourceDocument.

    The caller owns the httpx client (reuse across many fetches for connection
    pooling). The function adds no retry logic on its own — wrap with tenacity
    at the service layer when calling it in batch.
    """
    api_path = parse_dev_to_url(url)
    response = await client.get(f"{DEV_TO_API_BASE}/articles/{api_path}")
    response.raise_for_status()
    data = response.json()

    user = data.get("user") or {}

    return SourceDocument(
        source_url=url,
        source_type=SOURCE_TYPE,
        title=data["title"],
        body_markdown=normalize_markdown(data["body_markdown"]),
        author=user.get("username"),
        published_at=_parse_iso(data.get("published_at")),
        tags=_parse_tags(data.get("tag_list")),
        extras={
            "dev_to_id": data.get("id"),
            "reading_time_minutes": data.get("reading_time_minutes"),
            "positive_reactions_count": data.get("positive_reactions_count"),
            "cover_image": data.get("cover_image"),
        },
    )
