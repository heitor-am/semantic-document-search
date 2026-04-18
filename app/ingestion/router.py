from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Query, status

from app.ingestion.models import IngestJob as IngestJobModel
from app.ingestion.models import JobTransition
from app.ingestion.repository import job_repository
from app.ingestion.schemas import IngestJobRead, IngestRequest, JobTransitionRead
from app.ingestion.service import deterministic_job_id, run_ingestion
from app.ingestion.state import JobState
from app.shared.api.deps import DbDep, HttpxDep, OpenAIDep, VectorRepoDep
from app.shared.core.exceptions import JobNotFoundError
from app.shared.db.database import SessionLocal

router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Non-terminal states — a job in one of these is already being worked on;
# re-POSTing the same URL should observe it, not spawn a duplicate pipeline.
_IN_PROGRESS_STATES = {
    JobState.PENDING,
    JobState.FETCHING,
    JobState.PARSING,
    JobState.CHUNKING,
    JobState.EMBEDDING,
    JobState.INDEXING,
}


@router.post("", response_model=IngestJobRead, status_code=status.HTTP_202_ACCEPTED)
async def start_ingestion(
    payload: IngestRequest,
    background_tasks: BackgroundTasks,
    db: DbDep,
    vector_repo: VectorRepoDep,
    httpx_client: HttpxDep,
    openai_client: OpenAIDep,
) -> IngestJobRead:
    """Kick off (or observe) an ingestion job for a source URL.

    The job_id is a deterministic hash of the URL, so re-POSTing is idempotent:
    - If the job is COMPLETED: return it as-is, no new work.
    - If the job is in-progress: return it as-is, don't spawn a duplicate.
    - If the job FAILED: record a retry transition and spawn a fresh run.
    - Otherwise: create the job and spawn the run.
    """
    source_url = str(payload.source_url)
    job_id = deterministic_job_id(source_url)

    existing = await job_repository.get(db, job_id)

    if existing is not None:
        if existing.state == JobState.COMPLETED:
            return _to_read(existing, await job_repository.get_transitions(db, job_id))
        if existing.state in _IN_PROGRESS_STATES:
            return _to_read(existing, await job_repository.get_transitions(db, job_id))
        # FAILED — reset so the fresh IngestJobFSM(initial=PENDING) matches
        # the DB state, then re-run.
        await job_repository.record_transition(
            db,
            job_id=job_id,
            from_state=existing.state,
            to_state=JobState.PENDING,
        )
        await db.refresh(existing)
        job = existing
    else:
        job = await job_repository.create(db, job_id=job_id, source_url=source_url)

    background_tasks.add_task(
        run_ingestion,
        job_id=job_id,
        source_url=source_url,
        vector_repo=vector_repo,
        httpx_client=httpx_client,
        openai_client=openai_client,
        session_maker=SessionLocal,
    )

    transitions = await job_repository.get_transitions(db, job_id)
    return _to_read(job, transitions)


@router.get("/jobs/{job_id}", response_model=IngestJobRead)
async def get_job(job_id: str, db: DbDep) -> IngestJobRead:
    job = await job_repository.get(db, job_id)
    if job is None:
        raise JobNotFoundError(f"job {job_id} not found")
    transitions = await job_repository.get_transitions(db, job_id)
    return _to_read(job, transitions)


@router.get("/jobs", response_model=list[IngestJobRead])
async def list_jobs(
    db: DbDep,
    state: Annotated[JobState | None, Query(description="Filter by job state")] = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[IngestJobRead]:
    jobs = await job_repository.list(db, state=state, skip=skip, limit=limit)
    # List view skips the transition history to keep the payload small.
    return [_to_read(j, []) for j in jobs]


def _to_read(job: IngestJobModel, transitions: Sequence[JobTransition]) -> IngestJobRead:
    return IngestJobRead(
        job_id=job.id,
        source_url=job.source_url,
        state=job.state,
        error=job.error,
        history=[JobTransitionRead(state=t.to_state, at=t.at) for t in transitions],
    )
