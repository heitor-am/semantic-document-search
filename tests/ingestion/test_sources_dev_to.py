from datetime import UTC, datetime

import httpx
import pytest
import respx

from app.ingestion.sources.dev_to import (
    InvalidDevToUrlError,
    _parse_tags,
    fetch_dev_to,
    parse_dev_to_url,
)

SAMPLE_ARTICLE_URL = "https://dev.to/author/a-great-post-abc123"
SAMPLE_API_RESPONSE = {
    "id": 12345,
    "title": "A Great Post",
    "body_markdown": "---\ntitle: A Great Post\n---\n\n# Intro\n\nHello, world!\n\n\n\nExtra blank lines.\n",
    "tag_list": ["python", "rag", "ai"],
    "published_at": "2026-03-15T12:30:00Z",
    "reading_time_minutes": 7,
    "positive_reactions_count": 42,
    "cover_image": "https://dev.to/img/cover.png",
    "user": {"username": "author"},
}


class TestParseDevToUrl:
    def test_extracts_username_and_slug(self) -> None:
        assert parse_dev_to_url(SAMPLE_ARTICLE_URL) == "author/a-great-post-abc123"

    def test_accepts_subdomain_form(self) -> None:
        # Some dev.to articles share subdomain-style URLs; still a match
        assert parse_dev_to_url("https://www.dev.to/user/slug") == "user/slug"

    def test_rejects_non_dev_to_host(self) -> None:
        with pytest.raises(InvalidDevToUrlError, match=r"dev\.to"):
            parse_dev_to_url("https://medium.com/user/post")

    def test_rejects_lookalike_domain_ending_in_dev_to(self) -> None:
        # `evildev.to` must not pass — endswith('dev.to') would accept it
        with pytest.raises(InvalidDevToUrlError):
            parse_dev_to_url("https://evildev.to/user/slug")

    def test_accepts_explicit_port(self) -> None:
        # hostname() strips the port, so dev.to:443 is still a valid host
        assert parse_dev_to_url("https://dev.to:443/user/slug") == "user/slug"

    def test_rejects_wrong_path_depth(self) -> None:
        with pytest.raises(InvalidDevToUrlError, match="path"):
            parse_dev_to_url("https://dev.to/just-one-segment")

    def test_rejects_path_with_extra_segments(self) -> None:
        with pytest.raises(InvalidDevToUrlError):
            parse_dev_to_url("https://dev.to/user/slug/comments")


class TestFetchDevTo:
    @respx.mock
    async def test_returns_source_document_with_canonical_fields(self) -> None:
        respx.get("https://dev.to/api/articles/author/a-great-post-abc123").mock(
            return_value=httpx.Response(200, json=SAMPLE_API_RESPONSE)
        )

        async with httpx.AsyncClient() as client:
            doc = await fetch_dev_to(SAMPLE_ARTICLE_URL, client=client)

        assert doc.source_url == SAMPLE_ARTICLE_URL
        assert doc.source_type == "dev.to"
        assert doc.title == "A Great Post"
        assert doc.author == "author"
        assert doc.published_at == datetime(2026, 3, 15, 12, 30, tzinfo=UTC)
        assert doc.tags == ["python", "rag", "ai"]

    @respx.mock
    async def test_normalizes_body_markdown_on_fetch(self) -> None:
        respx.get("https://dev.to/api/articles/author/a-great-post-abc123").mock(
            return_value=httpx.Response(200, json=SAMPLE_API_RESPONSE)
        )

        async with httpx.AsyncClient() as client:
            doc = await fetch_dev_to(SAMPLE_ARTICLE_URL, client=client)

        # Front-matter stripped, excess blank lines collapsed, no trailing whitespace
        assert doc.body_markdown == "# Intro\n\nHello, world!\n\nExtra blank lines."

    @respx.mock
    async def test_captures_source_specific_fields_in_extras(self) -> None:
        respx.get("https://dev.to/api/articles/author/a-great-post-abc123").mock(
            return_value=httpx.Response(200, json=SAMPLE_API_RESPONSE)
        )

        async with httpx.AsyncClient() as client:
            doc = await fetch_dev_to(SAMPLE_ARTICLE_URL, client=client)

        assert doc.extras["dev_to_id"] == 12345
        assert doc.extras["reading_time_minutes"] == 7
        assert doc.extras["positive_reactions_count"] == 42

    @respx.mock
    async def test_missing_optional_fields_produce_nulls(self) -> None:
        minimal = {
            "title": "Just a title",
            "body_markdown": "body",
            # everything else omitted
        }
        respx.get("https://dev.to/api/articles/author/a-great-post-abc123").mock(
            return_value=httpx.Response(200, json=minimal)
        )

        async with httpx.AsyncClient() as client:
            doc = await fetch_dev_to(SAMPLE_ARTICLE_URL, client=client)

        assert doc.author is None
        assert doc.published_at is None
        assert doc.tags == []

    @respx.mock
    async def test_accepts_csv_string_for_tag_list(self) -> None:
        # dev.to's article detail endpoint returns tag_list as a CSV string,
        # not a list. A real article (Sylwia Lask's overengineering post)
        # surfaced this during smoke testing.
        csv_response = {**SAMPLE_API_RESPONSE, "tag_list": "webdev, javascript, css, browser"}
        respx.get("https://dev.to/api/articles/author/a-great-post-abc123").mock(
            return_value=httpx.Response(200, json=csv_response)
        )

        async with httpx.AsyncClient() as client:
            doc = await fetch_dev_to(SAMPLE_ARTICLE_URL, client=client)

        assert doc.tags == ["webdev", "javascript", "css", "browser"]

    @respx.mock
    async def test_propagates_http_errors(self) -> None:
        respx.get("https://dev.to/api/articles/author/a-great-post-abc123").mock(
            return_value=httpx.Response(404, json={"error": "Not Found"})
        )

        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await fetch_dev_to(SAMPLE_ARTICLE_URL, client=client)


class TestParseTags:
    def test_csv_string_splits_and_strips(self) -> None:
        assert _parse_tags("webdev, javascript, css, browser") == [
            "webdev",
            "javascript",
            "css",
            "browser",
        ]

    def test_list_is_passed_through_as_strings(self) -> None:
        assert _parse_tags(["python", "rag"]) == ["python", "rag"]

    def test_empty_string_becomes_empty_list(self) -> None:
        assert _parse_tags("") == []

    def test_none_becomes_empty_list(self) -> None:
        assert _parse_tags(None) == []

    def test_unexpected_shape_falls_back_to_empty(self) -> None:
        assert _parse_tags({"unexpected": "dict"}) == []

    def test_single_tag_csv_works(self) -> None:
        assert _parse_tags("python") == ["python"]

    def test_trailing_comma_is_ignored(self) -> None:
        assert _parse_tags("python,rag,") == ["python", "rag"]
