from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog
from transitions import Machine

logger = structlog.get_logger()


class JobState(StrEnum):
    PENDING = "pending"
    FETCHING = "fetching"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


# Declarative transitions. Invalid transitions (e.g. pending → indexing)
# are impossible by construction — transitions library raises MachineError.
TRANSITIONS: list[dict[str, Any]] = [
    {"trigger": "start_fetch", "source": JobState.PENDING.value, "dest": JobState.FETCHING.value},
    {"trigger": "start_parse", "source": JobState.FETCHING.value, "dest": JobState.PARSING.value},
    {"trigger": "start_chunk", "source": JobState.PARSING.value, "dest": JobState.CHUNKING.value},
    {"trigger": "start_embed", "source": JobState.CHUNKING.value, "dest": JobState.EMBEDDING.value},
    {"trigger": "start_index", "source": JobState.EMBEDDING.value, "dest": JobState.INDEXING.value},
    {"trigger": "complete", "source": JobState.INDEXING.value, "dest": JobState.COMPLETED.value},
    # Any non-terminal state can fail; only failed can retry.
    {
        "trigger": "fail",
        "source": [
            JobState.PENDING.value,
            JobState.FETCHING.value,
            JobState.PARSING.value,
            JobState.CHUNKING.value,
            JobState.EMBEDDING.value,
            JobState.INDEXING.value,
        ],
        "dest": JobState.FAILED.value,
    },
    {"trigger": "retry", "source": JobState.FAILED.value, "dest": JobState.PENDING.value},
]


class IngestJob:
    """Finite state machine wrapper around an ingestion job.

    States progress linearly through the pipeline; any non-terminal state can
    transition to FAILED, and FAILED can be retried back to PENDING. COMPLETED
    is a terminal absorbing state.
    """

    state: str  # attribute provided by the transitions Machine at runtime

    def __init__(
        self,
        job_id: str,
        source_url: str,
        initial_state: JobState = JobState.PENDING,
    ) -> None:
        self.job_id = job_id
        self.source_url = source_url
        self.error: str | None = None
        self.history: list[tuple[JobState, datetime]] = []
        self._last_transition_at: datetime | None = None

        self.machine = Machine(
            model=self,
            states=[s.value for s in JobState],
            transitions=TRANSITIONS,
            initial=initial_state.value,
            after_state_change="_record_transition",
            send_event=False,
        )

        # Record the initial state so history is always non-empty.
        now = datetime.now(UTC)
        self._last_transition_at = now
        self.history.append((initial_state, now))

    @property
    def current_state(self) -> JobState:
        return JobState(self.state)

    def _record_transition(self) -> None:
        now = datetime.now(UTC)
        duration_ms: float | None = None
        if self._last_transition_at is not None:
            duration_ms = round((now - self._last_transition_at).total_seconds() * 1000, 2)

        current = self.current_state
        self.history.append((current, now))

        logger.info(
            "ingestion.transition",
            job_id=self.job_id,
            state=current.value,
            duration_ms=duration_ms,
        )
        self._last_transition_at = now


def get_state_diagram(output_path: str | Path) -> Path:  # pragma: no cover
    """Export the FSM graph to a PNG file.

    Requires the `graphviz` Python package plus the system `dot` binary.
    Run via `make diagram-states`. Excluded from coverage because it's a
    dev utility — the FSM logic itself is fully covered by unit tests.
    """
    from transitions.extensions import GraphMachine

    class _DiagramModel:
        pass

    model = _DiagramModel()
    GraphMachine(
        model=model,
        states=[s.value for s in JobState],
        transitions=TRANSITIONS,
        initial=JobState.PENDING.value,
        title="Ingestion Pipeline State Machine",
        show_conditions=False,
        show_auto_transitions=False,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.get_graph().draw(str(output), prog="dot", format="png")  # type: ignore[attr-defined]
    return output
