import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.repository import JobRepository
from app.ingestion.state import JobState


@pytest.fixture
def repo() -> JobRepository:
    return JobRepository()


async def _create_job(
    repo: JobRepository,
    db: AsyncSession,
    *,
    job_id: str = "job-1",
    source_url: str = "https://example.com/post",
) -> None:
    await repo.create(db, job_id=job_id, source_url=source_url)


class TestCreate:
    async def test_creates_job_with_pending_state(
        self, repo: JobRepository, db_session: AsyncSession
    ) -> None:
        job = await repo.create(db_session, job_id="abc123", source_url="https://example.com")
        assert job.id == "abc123"
        assert job.state == JobState.PENDING
        assert job.error is None
        assert job.created_at is not None

    async def test_create_sets_timestamps(
        self, repo: JobRepository, db_session: AsyncSession
    ) -> None:
        job = await repo.create(db_session, job_id="abc", source_url="https://x")
        assert job.created_at == job.updated_at


class TestGet:
    async def test_returns_job_by_id(self, repo: JobRepository, db_session: AsyncSession) -> None:
        await _create_job(repo, db_session, job_id="abc")
        found = await repo.get(db_session, "abc")
        assert found is not None
        assert found.id == "abc"

    async def test_returns_none_for_missing(
        self, repo: JobRepository, db_session: AsyncSession
    ) -> None:
        assert await repo.get(db_session, "does-not-exist") is None


class TestList:
    async def test_returns_all_unfiltered(
        self, repo: JobRepository, db_session: AsyncSession
    ) -> None:
        await _create_job(repo, db_session, job_id="a", source_url="https://a")
        await _create_job(repo, db_session, job_id="b", source_url="https://b")

        jobs = await repo.list(db_session)
        assert len(jobs) == 2

    async def test_filters_by_state(self, repo: JobRepository, db_session: AsyncSession) -> None:
        await _create_job(repo, db_session, job_id="a")
        await _create_job(repo, db_session, job_id="b")
        await repo.record_transition(
            db_session,
            job_id="b",
            from_state=JobState.PENDING,
            to_state=JobState.FETCHING,
        )

        pending = await repo.list(db_session, state=JobState.PENDING)
        fetching = await repo.list(db_session, state=JobState.FETCHING)

        assert [j.id for j in pending] == ["a"]
        assert [j.id for j in fetching] == ["b"]

    async def test_pagination(self, repo: JobRepository, db_session: AsyncSession) -> None:
        for i in range(5):
            await _create_job(repo, db_session, job_id=f"job-{i}", source_url=f"https://x/{i}")

        page = await repo.list(db_session, skip=2, limit=2)
        assert len(page) == 2


class TestRecordTransition:
    async def test_appends_transition_and_updates_job_state(
        self, repo: JobRepository, db_session: AsyncSession
    ) -> None:
        await _create_job(repo, db_session, job_id="j")

        await repo.record_transition(
            db_session,
            job_id="j",
            from_state=JobState.PENDING,
            to_state=JobState.FETCHING,
            duration_ms=12.5,
        )

        job = await repo.get(db_session, "j")
        assert job is not None
        assert job.state == JobState.FETCHING

        transitions = await repo.get_transitions(db_session, "j")
        assert len(transitions) == 1
        t = transitions[0]
        assert t.from_state == JobState.PENDING
        assert t.to_state == JobState.FETCHING
        assert t.duration_ms == 12.5

    async def test_records_error_on_failure_transition(
        self, repo: JobRepository, db_session: AsyncSession
    ) -> None:
        await _create_job(repo, db_session, job_id="j")
        await repo.record_transition(
            db_session,
            job_id="j",
            from_state=JobState.FETCHING,
            to_state=JobState.FAILED,
            error="network timeout",
        )

        job = await repo.get(db_session, "j")
        assert job is not None
        assert job.state == JobState.FAILED
        assert job.error == "network timeout"

        transitions = await repo.get_transitions(db_session, "j")
        assert transitions[-1].error == "network timeout"

    async def test_transitions_are_chronological(
        self, repo: JobRepository, db_session: AsyncSession
    ) -> None:
        await _create_job(repo, db_session, job_id="j")

        for from_s, to_s in [
            (JobState.PENDING, JobState.FETCHING),
            (JobState.FETCHING, JobState.PARSING),
            (JobState.PARSING, JobState.CHUNKING),
        ]:
            await repo.record_transition(db_session, job_id="j", from_state=from_s, to_state=to_s)

        transitions = await repo.get_transitions(db_session, "j")
        assert [t.to_state for t in transitions] == [
            JobState.FETCHING,
            JobState.PARSING,
            JobState.CHUNKING,
        ]


class TestDelete:
    async def test_deletes_job_and_cascades_transitions(
        self, repo: JobRepository, db_session: AsyncSession
    ) -> None:
        await _create_job(repo, db_session, job_id="j")
        await repo.record_transition(
            db_session,
            job_id="j",
            from_state=JobState.PENDING,
            to_state=JobState.FETCHING,
        )

        result = await repo.delete(db_session, "j")
        assert result is True
        assert await repo.get(db_session, "j") is None
        assert await repo.get_transitions(db_session, "j") == []

    async def test_returns_false_for_missing(
        self, repo: JobRepository, db_session: AsyncSession
    ) -> None:
        assert await repo.delete(db_session, "ghost") is False
