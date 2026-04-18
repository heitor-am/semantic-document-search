import pytest
from transitions import MachineError

from app.ingestion.state import IngestJob, JobState


def _make_job() -> IngestJob:
    return IngestJob(job_id="test-job-1", source_url="https://example.com/post")


class TestInitialState:
    def test_defaults_to_pending(self) -> None:
        job = _make_job()
        assert job.current_state == JobState.PENDING

    def test_history_starts_with_initial_state(self) -> None:
        job = _make_job()
        assert len(job.history) == 1
        assert job.history[0][0] == JobState.PENDING

    def test_explicit_initial_state_is_honored(self) -> None:
        job = IngestJob("j", "https://x", initial_state=JobState.FAILED)
        assert job.current_state == JobState.FAILED

    def test_error_is_none_initially(self) -> None:
        assert _make_job().error is None


class TestHappyPath:
    def test_full_pipeline_transitions(self) -> None:
        job = _make_job()
        job.start_fetch()
        assert job.current_state == JobState.FETCHING
        job.start_parse()
        assert job.current_state == JobState.PARSING
        job.start_chunk()
        assert job.current_state == JobState.CHUNKING
        job.start_embed()
        assert job.current_state == JobState.EMBEDDING
        job.start_index()
        assert job.current_state == JobState.INDEXING
        job.complete()
        assert job.current_state == JobState.COMPLETED

    def test_happy_path_records_every_state_in_history(self) -> None:
        job = _make_job()
        job.start_fetch()
        job.start_parse()
        job.start_chunk()
        job.start_embed()
        job.start_index()
        job.complete()

        states = [s for s, _ in job.history]
        assert states == [
            JobState.PENDING,
            JobState.FETCHING,
            JobState.PARSING,
            JobState.CHUNKING,
            JobState.EMBEDDING,
            JobState.INDEXING,
            JobState.COMPLETED,
        ]


class TestInvalidTransitions:
    def test_cannot_skip_from_pending_to_indexing(self) -> None:
        job = _make_job()
        with pytest.raises(MachineError):
            job.start_index()

    def test_cannot_complete_from_pending(self) -> None:
        job = _make_job()
        with pytest.raises(MachineError):
            job.complete()

    def test_cannot_retry_from_pending(self) -> None:
        # retry only valid from FAILED
        job = _make_job()
        with pytest.raises(MachineError):
            job.retry()

    def test_completed_is_terminal(self) -> None:
        job = _make_job()
        for trigger in (
            "start_fetch",
            "start_parse",
            "start_chunk",
            "start_embed",
            "start_index",
            "complete",
        ):
            getattr(job, trigger)()
        # All transitions out of COMPLETED must fail
        for trigger in ("start_fetch", "start_parse", "complete", "fail", "retry"):
            with pytest.raises(MachineError):
                getattr(job, trigger)()


class TestFailurePath:
    def test_can_fail_from_any_non_terminal_state(self) -> None:
        states_that_can_fail = [
            JobState.PENDING,
            JobState.FETCHING,
            JobState.PARSING,
            JobState.CHUNKING,
            JobState.EMBEDDING,
            JobState.INDEXING,
        ]
        for start in states_that_can_fail:
            job = IngestJob("j", "https://x", initial_state=start)
            job.fail()
            assert job.current_state == JobState.FAILED

    def test_cannot_fail_from_completed(self) -> None:
        job = _make_job()
        for trigger in (
            "start_fetch",
            "start_parse",
            "start_chunk",
            "start_embed",
            "start_index",
            "complete",
        ):
            getattr(job, trigger)()
        with pytest.raises(MachineError):
            job.fail()

    def test_cannot_fail_from_failed(self) -> None:
        job = IngestJob("j", "https://x", initial_state=JobState.FAILED)
        with pytest.raises(MachineError):
            job.fail()


class TestRetry:
    def test_failed_can_retry_to_pending(self) -> None:
        job = _make_job()
        job.start_fetch()
        job.fail()
        assert job.current_state == JobState.FAILED
        job.retry()
        assert job.current_state == JobState.PENDING

    def test_retry_cycle_preserves_history(self) -> None:
        job = _make_job()
        job.start_fetch()
        job.fail()
        job.retry()
        states = [s for s, _ in job.history]
        assert JobState.FAILED in states
        assert states.count(JobState.PENDING) == 2  # initial + after retry


class TestHistoryAndLogging:
    def test_each_transition_adds_history_entry(self) -> None:
        job = _make_job()
        initial_len = len(job.history)
        job.start_fetch()
        assert len(job.history) == initial_len + 1
        job.start_parse()
        assert len(job.history) == initial_len + 2

    def test_history_entries_are_in_chronological_order(self) -> None:
        job = _make_job()
        job.start_fetch()
        job.start_parse()
        timestamps = [t for _, t in job.history]
        assert timestamps == sorted(timestamps)


class TestStateDiagram:
    def test_exports_png_when_graphviz_available(self, tmp_path: object) -> None:
        pytest.importorskip("graphviz")
        from app.ingestion.state import get_state_diagram

        output = tmp_path / "fsm.png"  # type: ignore[operator]
        try:
            result = get_state_diagram(output)
        except (ImportError, FileNotFoundError) as e:
            pytest.skip(f"graphviz binary not available: {e}")

        assert result.exists()
        assert result.stat().st_size > 0
