from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.ingestion.schemas import IngestJobRead, IngestRequest, JobTransitionRead
from app.ingestion.state import JobState


class TestIngestRequest:
    def test_valid_http_url(self) -> None:
        req = IngestRequest(source_url="https://dev.to/author/post")  # type: ignore[arg-type]
        assert str(req.source_url).startswith("https://dev.to")

    def test_rejects_non_url(self) -> None:
        with pytest.raises(ValidationError):
            IngestRequest.model_validate({"source_url": "not-a-url"})


class TestJobTransitionRead:
    def test_roundtrip_from_tuple(self) -> None:
        transition = JobTransitionRead(state=JobState.FETCHING, at=datetime.now(UTC))
        assert transition.state == JobState.FETCHING


class TestIngestJobRead:
    def test_minimal_payload(self) -> None:
        job = IngestJobRead(
            job_id="abc",
            source_url="https://example.com",
            state=JobState.PENDING,
            history=[JobTransitionRead(state=JobState.PENDING, at=datetime.now(UTC))],
        )
        assert job.job_id == "abc"
        assert job.error is None
        assert len(job.history) == 1

    def test_with_error(self) -> None:
        job = IngestJobRead(
            job_id="abc",
            source_url="https://example.com",
            state=JobState.FAILED,
            error="network timeout",
            history=[
                JobTransitionRead(state=JobState.PENDING, at=datetime.now(UTC)),
                JobTransitionRead(state=JobState.FETCHING, at=datetime.now(UTC)),
                JobTransitionRead(state=JobState.FAILED, at=datetime.now(UTC)),
            ],
        )
        assert job.state == JobState.FAILED
        assert job.error == "network timeout"
