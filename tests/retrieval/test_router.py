"""Integration-style tests for the /search endpoint via TestClient."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.main import create_app
from app.shared.api.deps import get_httpx_client, get_openai_client, get_vector_repo
from app.shared.qdrant.repository import VectorHit

RERANK_URL = "https://openrouter.ai/api/v1/rerank"


def make_fake_openai(vector: list[float] | None = None) -> MagicMock:
    client = MagicMock()
    vec = vector if vector is not None else [0.1] * 4

    async def create(**kwargs: Any) -> MagicMock:
        return MagicMock(data=[MagicMock(embedding=vec)])

    client.embeddings.create = AsyncMock(side_effect=create)
    return client


def make_fake_vector_repo(hits: list[VectorHit]) -> MagicMock:
    repo = MagicMock()
    repo.search_hybrid = AsyncMock(return_value=hits)
    return repo


@pytest.fixture
async def search_client(monkeypatch) -> AsyncIterator[dict[str, Any]]:
    """TestClient wired to fake Qdrant and OpenAI, with a live httpx
    AsyncClient so respx can intercept the /rerank call."""
    hits = [
        VectorHit(
            id="c1",
            score=0.9,
            payload={
                "content": "python async programming",
                "title": "Async",
                "source_url": "https://dev.to/a/p",
                "source_type": "dev.to",
                "section_path": ["Intro"],
                "parent_chunk_id": "parent-1",
                "author": "alice",
                "tags": ["python"],
                "is_parent": False,
            },
        ),
        VectorHit(
            id="c2",
            score=0.7,
            payload={
                "content": "javascript event loop",
                "title": "JS Loop",
                "source_url": "https://dev.to/b/q",
                "source_type": "dev.to",
                "section_path": [],
                "parent_chunk_id": "parent-2",
                "author": "bob",
                "tags": ["javascript"],
                "is_parent": False,
            },
        ),
    ]
    vector_repo = make_fake_vector_repo(hits)
    openai_client = make_fake_openai()
    httpx_client = httpx.AsyncClient()

    app = create_app()
    app.dependency_overrides[get_vector_repo] = lambda: vector_repo
    app.dependency_overrides[get_openai_client] = lambda: openai_client
    app.dependency_overrides[get_httpx_client] = lambda: httpx_client

    try:
        with TestClient(app) as client:
            yield {"client": client, "vector_repo": vector_repo, "openai_client": openai_client}
    finally:
        await httpx_client.aclose()


class TestSearch:
    def test_missing_query_returns_422(self, search_client) -> None:
        response = search_client["client"].get("/search")
        assert response.status_code == 422

    def test_empty_query_string_returns_422(self, search_client) -> None:
        response = search_client["client"].get("/search", params={"q": ""})
        assert response.status_code == 422

    def test_hybrid_strategy_returns_results_from_vector_repo(self, search_client) -> None:
        response = search_client["client"].get(
            "/search", params={"q": "python", "strategy": "hybrid"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["query"] == "python"
        assert body["strategy"] == "hybrid"
        assert len(body["results"]) == 2
        assert body["results"][0]["chunk_id"] == "c1"
        # Parent-child stage preserves dedup (different parents here, so all kept)
        assert {r["chunk_id"] for r in body["results"]} == {"c1", "c2"}

    def test_hybrid_strategy_surfaces_metadata(self, search_client) -> None:
        response = search_client["client"].get(
            "/search", params={"q": "python", "strategy": "hybrid"}
        )
        hit = response.json()["results"][0]
        assert hit["title"] == "Async"
        assert hit["section_path"] == ["Intro"]
        assert hit["tags"] == ["python"]
        assert hit["parent_chunk_id"] == "parent-1"
        assert hit["author"] == "alice"

    @respx.mock
    def test_hybrid_rerank_strategy_calls_rerank_endpoint(self, search_client) -> None:
        respx.post(RERANK_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 1, "relevance_score": 0.95, "document": {"text": "b"}},
                        {"index": 0, "relevance_score": 0.45, "document": {"text": "a"}},
                    ],
                    "usage": {"cost": 0.001, "search_units": 1, "total_tokens": 50},
                },
            )
        )

        response = search_client["client"].get(
            "/search", params={"q": "python", "strategy": "hybrid_rerank"}
        )
        assert response.status_code == 200
        body = response.json()
        # Rerank reordered: c2 now first.
        assert body["results"][0]["chunk_id"] == "c2"
        assert body["results"][0]["score"] == 0.95

    @respx.mock
    def test_rerank_failure_degrades_gracefully(self, search_client) -> None:
        respx.post(RERANK_URL).mock(return_value=httpx.Response(503, text="upstream down"))

        response = search_client["client"].get(
            "/search", params={"q": "python", "strategy": "hybrid_rerank"}
        )
        assert response.status_code == 200
        body = response.json()
        # Hybrid results still come back (c1 then c2 by RRF score order).
        assert body["results"][0]["chunk_id"] == "c1"
        assert len(body["warnings"]) == 1
        assert "reranker" in body["warnings"][0]

    def test_top_k_caps_results(self, search_client) -> None:
        response = search_client["client"].get(
            "/search", params={"q": "python", "top_k": 1, "strategy": "hybrid"}
        )
        assert response.status_code == 200
        assert len(response.json()["results"]) == 1

    def test_min_score_filters_low_scores(self, search_client) -> None:
        response = search_client["client"].get(
            "/search", params={"q": "python", "min_score": 0.95, "strategy": "hybrid"}
        )
        # Both RRF scores in the test fixture are below 0.95, so zero results.
        assert response.json()["results"] == []

    def test_default_strategy_is_hybrid_rerank(self, search_client) -> None:
        # No strategy param → defaults to hybrid_rerank. Without respx mocking
        # the rerank endpoint the call will fail, but gracefully (optional
        # stage) and the response still returns 200 with warnings.
        response = search_client["client"].get("/search", params={"q": "python"})
        body = response.json()
        assert response.status_code == 200
        assert body["strategy"] == "hybrid_rerank"

    def test_invalid_strategy_returns_422(self, search_client) -> None:
        response = search_client["client"].get(
            "/search", params={"q": "python", "strategy": "invalid"}
        )
        assert response.status_code == 422
