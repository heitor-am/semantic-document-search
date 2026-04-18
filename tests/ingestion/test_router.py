from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.main import create_app
from app.shared.api.deps import (
    get_db,
    get_httpx_client,
    get_openai_client,
    get_vector_repo,
)
from app.shared.db.database import Base


@pytest.fixture
async def test_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def test_session_maker(test_engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(test_engine, expire_on_commit=False)


class FakeVectorRepo:
    def __init__(self) -> None:
        self.upserts: list[Any] = []

    async def upsert(self, points: Any) -> None:
        self.upserts.append(list(points))


def make_fake_openai(vector_size: int = 1024) -> MagicMock:
    client = MagicMock()

    async def create(**kwargs: Any) -> MagicMock:
        inputs = kwargs["input"]
        return MagicMock(data=[MagicMock(embedding=[0.0] * vector_size) for _ in inputs])

    client.embeddings.create = AsyncMock(side_effect=create)
    return client


@pytest.fixture
async def client_bundle(test_session_maker, monkeypatch) -> AsyncIterator[dict[str, Any]]:
    """Spin up a TestClient with fake Qdrant/OpenAI and an in-memory DB.

    Returns a bag with `client`, `vector_repo`, `openai_client` so each test
    can poke at them. Background tasks run inside TestClient — when the call
    returns, the ingestion has already executed end-to-end.

    One httpx.AsyncClient is created per fixture and closed at teardown so
    the transport is never leaked.
    """
    vector_repo = FakeVectorRepo()
    openai_client = make_fake_openai()
    httpx_client = httpx.AsyncClient()

    # Point the router's BackgroundTasks session_maker at the test DB so the
    # task running after the response uses the same :memory: SQLite.
    monkeypatch.setattr("app.ingestion.router.SessionLocal", test_session_maker)

    app = create_app()

    async def override_db() -> AsyncIterator[AsyncSession]:
        async with test_session_maker() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_vector_repo] = lambda: vector_repo
    app.dependency_overrides[get_openai_client] = lambda: openai_client
    app.dependency_overrides[get_httpx_client] = lambda: httpx_client

    try:
        with TestClient(app) as client:
            yield {
                "client": client,
                "vector_repo": vector_repo,
                "openai_client": openai_client,
            }
    finally:
        await httpx_client.aclose()


SAMPLE_URL = "https://dev.to/author/great-post-abc"
SAMPLE_DEV_TO_RESPONSE = {
    "id": 42,
    "title": "Great Post",
    "body_markdown": "# Intro\n\nHello, world!\n\n## Body\n\nMore text.",
    "tag_list": ["python"],
    "published_at": "2026-03-15T12:30:00Z",
    "reading_time_minutes": 3,
    "positive_reactions_count": 7,
    "user": {"username": "author"},
}


class TestPostIngest:
    def test_new_url_returns_202_with_pending_job(self, client_bundle, respx_mock) -> None:
        respx_mock.get("https://dev.to/api/articles/author/great-post-abc").respond(
            200, json=SAMPLE_DEV_TO_RESPONSE
        )

        response = client_bundle["client"].post("/ingest", json={"source_url": SAMPLE_URL})

        assert response.status_code == 202
        body = response.json()
        assert body["source_url"] == SAMPLE_URL
        # The response is serialized before the background task runs, so the
        # job is still in its initial state here. Completion is verified via
        # the follow-up GET tests below.
        assert body["state"] == "pending"
        assert body["job_id"]

    def test_second_post_same_url_is_idempotent(self, client_bundle, respx_mock) -> None:
        respx_mock.get("https://dev.to/api/articles/author/great-post-abc").respond(
            200, json=SAMPLE_DEV_TO_RESPONSE
        )

        first = client_bundle["client"].post("/ingest", json={"source_url": SAMPLE_URL})
        second = client_bundle["client"].post("/ingest", json={"source_url": SAMPLE_URL})

        assert first.json()["job_id"] == second.json()["job_id"]
        # Only the first call actually upserted; the second short-circuits because
        # the job is already COMPLETED.
        assert len(client_bundle["vector_repo"].upserts) == 1

    def test_invalid_url_returns_422(self, client_bundle) -> None:
        response = client_bundle["client"].post("/ingest", json={"source_url": "not-a-url"})
        assert response.status_code == 422

    def test_non_dev_to_url_fails_job(self, client_bundle) -> None:
        # The URL parses as http but isn't dev.to — fetcher raises
        # InvalidDevToUrlError; the job lands in FAILED.
        response = client_bundle["client"].post(
            "/ingest", json={"source_url": "https://medium.com/u/post"}
        )
        assert response.status_code == 202
        # Second call — now the existing job is FAILED, so a retry is scheduled;
        # still fails for the same reason.
        job_id = response.json()["job_id"]
        detail = client_bundle["client"].get(f"/ingest/jobs/{job_id}")
        assert detail.json()["state"] == "failed"

    def test_retry_of_failed_job_clears_stale_error(self, client_bundle, respx_mock) -> None:
        # First POST: non-dev.to URL lands the job in FAILED with an error.
        fail_url = "https://medium.com/u/post"
        first = client_bundle["client"].post("/ingest", json={"source_url": fail_url})
        job_id = first.json()["job_id"]
        assert client_bundle["client"].get(f"/ingest/jobs/{job_id}").json()["error"]

        # Second POST: retry is scheduled. In the transient 202 response the
        # job is back at PENDING and the stale error from the previous failure
        # must be gone — before this fix, record_transition preserved it.
        retry = client_bundle["client"].post("/ingest", json={"source_url": fail_url})
        assert retry.status_code == 202
        retry_body = retry.json()
        assert retry_body["state"] == "pending"
        assert retry_body["error"] is None


class TestGetJob:
    def test_unknown_job_returns_404(self, client_bundle) -> None:
        response = client_bundle["client"].get("/ingest/jobs/deadbeef")
        assert response.status_code == 404
        problem = response.json()
        assert problem["code"] == "JOB_NOT_FOUND"

    def test_known_job_returns_full_history(self, client_bundle, respx_mock) -> None:
        respx_mock.get("https://dev.to/api/articles/author/great-post-abc").respond(
            200, json=SAMPLE_DEV_TO_RESPONSE
        )

        post = client_bundle["client"].post("/ingest", json={"source_url": SAMPLE_URL})
        job_id = post.json()["job_id"]

        detail = client_bundle["client"].get(f"/ingest/jobs/{job_id}")

        assert detail.status_code == 200
        payload = detail.json()
        assert payload["job_id"] == job_id
        assert payload["state"] == "completed"
        assert len(payload["history"]) >= 6  # 6 FSM transitions


class TestListJobs:
    def test_empty_list(self, client_bundle) -> None:
        response = client_bundle["client"].get("/ingest/jobs")
        assert response.status_code == 200
        assert response.json() == []

    def test_lists_created_jobs(self, client_bundle, respx_mock) -> None:
        respx_mock.get("https://dev.to/api/articles/author/great-post-abc").respond(
            200, json=SAMPLE_DEV_TO_RESPONSE
        )

        client_bundle["client"].post("/ingest", json={"source_url": SAMPLE_URL})

        response = client_bundle["client"].get("/ingest/jobs")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) == 1
        assert jobs[0]["source_url"] == SAMPLE_URL

    def test_filter_by_state(self, client_bundle, respx_mock) -> None:
        respx_mock.get("https://dev.to/api/articles/author/great-post-abc").respond(
            200, json=SAMPLE_DEV_TO_RESPONSE
        )

        client_bundle["client"].post("/ingest", json={"source_url": SAMPLE_URL})

        completed = client_bundle["client"].get("/ingest/jobs", params={"state": "completed"})
        pending = client_bundle["client"].get("/ingest/jobs", params={"state": "pending"})

        assert len(completed.json()) == 1
        assert len(pending.json()) == 0
