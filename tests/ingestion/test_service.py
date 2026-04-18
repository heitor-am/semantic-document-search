from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.ingestion.repository import JobRepository, job_repository
from app.ingestion.schemas import SourceDocument
from app.ingestion.service import (
    current_collection_name,
    deterministic_job_id,
    run_ingestion,
)
from app.ingestion.state import JobState
from app.shared.db.database import Base


@pytest.fixture
async def session_maker() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield maker
    finally:
        await engine.dispose()


@pytest.fixture
def repo() -> JobRepository:
    return job_repository


@pytest.fixture
def source_doc() -> SourceDocument:
    return SourceDocument(
        source_url="https://dev.to/user/a-post",
        source_type="dev.to",
        title="A Post",
        body_markdown="# Intro\n\nHello, world!\n\n## Details\n\nMore text.",
        author="user",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        tags=["rag"],
    )


class FakeVectorRepo:
    def __init__(self) -> None:
        self.upserts: list[Any] = []

    async def upsert(self, points: Any) -> None:
        self.upserts.append(list(points))


def make_fake_openai(vector_size: int = 1024) -> MagicMock:
    """AsyncOpenAI stand-in whose embeddings.create returns a deterministic
    fixed-size vector per input text."""
    client = MagicMock()

    async def create(**kwargs: Any) -> MagicMock:
        inputs = kwargs["input"]
        return MagicMock(data=[MagicMock(embedding=[0.0] * vector_size) for _ in inputs])

    client.embeddings.create = AsyncMock(side_effect=create)
    return client


class TestDeterministicJobId:
    def test_stable_across_calls(self) -> None:
        url = "https://dev.to/user/a-post"
        assert deterministic_job_id(url) == deterministic_job_id(url)

    def test_differs_between_urls(self) -> None:
        a = deterministic_job_id("https://dev.to/a/p1")
        b = deterministic_job_id("https://dev.to/b/p2")
        assert a != b

    def test_fixed_length(self) -> None:
        assert len(deterministic_job_id("https://dev.to/u/post")) == 16


class TestCurrentCollectionName:
    def test_returns_valid_collection_name(self) -> None:
        name = current_collection_name()
        assert name.startswith("documents_")
        # v2 is the current default (named-vector schema); assert the
        # shape, not the literal version, so bumps don't require churn
        # across unrelated tests.
        assert name.split("_")[-1].startswith("v")


class TestRunIngestion:
    async def test_happy_path_transitions_to_completed(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        repo: JobRepository,
        source_doc: SourceDocument,
    ) -> None:
        async with session_maker() as db:
            await repo.create(db, job_id="job-1", source_url=source_doc.source_url)

        vector_repo = FakeVectorRepo()
        openai_client = make_fake_openai()
        httpx_client = MagicMock()

        async def fake_fetcher(url: str, client: Any) -> SourceDocument:
            return source_doc

        await run_ingestion(
            job_id="job-1",
            source_url=source_doc.source_url,
            vector_repo=vector_repo,  # type: ignore[arg-type]
            httpx_client=httpx_client,
            openai_client=openai_client,
            session_maker=session_maker,
            repo=repo,
            fetcher=fake_fetcher,
        )

        async with session_maker() as db:
            job = await repo.get(db, "job-1")
            assert job is not None
            assert job.state == JobState.COMPLETED
            assert job.error is None
            transitions = await repo.get_transitions(db, "job-1")
            # 6 triggers: fetch, parse, chunk, embed, index, complete
            assert len(transitions) == 6
            assert transitions[-1].to_state == JobState.COMPLETED

        assert len(vector_repo.upserts) == 1
        assert len(vector_repo.upserts[0]) > 0

    async def test_failure_during_fetch_persists_error(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        repo: JobRepository,
        source_doc: SourceDocument,
    ) -> None:
        async with session_maker() as db:
            await repo.create(db, job_id="job-2", source_url=source_doc.source_url)

        async def failing_fetcher(url: str, client: Any) -> SourceDocument:
            raise RuntimeError("network blew up")

        await run_ingestion(
            job_id="job-2",
            source_url=source_doc.source_url,
            vector_repo=FakeVectorRepo(),  # type: ignore[arg-type]
            httpx_client=MagicMock(),
            openai_client=make_fake_openai(),
            session_maker=session_maker,
            repo=repo,
            fetcher=failing_fetcher,
        )

        async with session_maker() as db:
            job = await repo.get(db, "job-2")
            assert job is not None
            assert job.state == JobState.FAILED
            assert job.error is not None
            assert "network blew up" in job.error
            transitions = await repo.get_transitions(db, "job-2")
            # fetch transition, then FAILED
            assert transitions[-1].to_state == JobState.FAILED
            assert transitions[-1].error is not None

    async def test_failure_during_embed_still_records_progress(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        repo: JobRepository,
        source_doc: SourceDocument,
    ) -> None:
        async with session_maker() as db:
            await repo.create(db, job_id="job-3", source_url=source_doc.source_url)

        async def fake_fetcher(url: str, client: Any) -> SourceDocument:
            return source_doc

        openai_client = MagicMock()
        openai_client.embeddings.create = AsyncMock(side_effect=RuntimeError("LLM down"))

        await run_ingestion(
            job_id="job-3",
            source_url=source_doc.source_url,
            vector_repo=FakeVectorRepo(),  # type: ignore[arg-type]
            httpx_client=MagicMock(),
            openai_client=openai_client,
            session_maker=session_maker,
            repo=repo,
            fetcher=fake_fetcher,
        )

        async with session_maker() as db:
            job = await repo.get(db, "job-3")
            assert job is not None
            assert job.state == JobState.FAILED
            transitions = await repo.get_transitions(db, "job-3")
            states = [t.to_state for t in transitions]
            # The pipeline made it past fetch, parse, chunk, and transitioned
            # into embed before failing — those earlier states should be in
            # the audit trail.
            assert JobState.FETCHING in states
            assert JobState.CHUNKING in states
            assert JobState.EMBEDDING in states
            assert states[-1] == JobState.FAILED
