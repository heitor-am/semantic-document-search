from collections.abc import Sequence
from typing import List  # noqa: UP035  (builtin shadowed by the list method)

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.models import IngestJob, JobTransition
from app.ingestion.state import JobState


class JobRepository:
    async def create(
        self,
        db: AsyncSession,
        *,
        job_id: str,
        source_url: str,
    ) -> IngestJob:
        job = IngestJob(id=job_id, source_url=source_url, state=JobState.PENDING)
        db.add(job)
        await db.commit()
        await db.refresh(job)
        return job

    async def get(self, db: AsyncSession, job_id: str) -> IngestJob | None:
        return await db.get(IngestJob, job_id)

    async def list(
        self,
        db: AsyncSession,
        *,
        state: JobState | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> List[IngestJob]:  # noqa: UP006  (list method shadowing)
        stmt = select(IngestJob)
        if state is not None:
            stmt = stmt.where(IngestJob.state == state)
        stmt = stmt.order_by(IngestJob.updated_at.desc()).offset(skip).limit(limit)

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def record_transition(
        self,
        db: AsyncSession,
        *,
        job_id: str,
        from_state: JobState | None,
        to_state: JobState,
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> JobTransition:
        transition = JobTransition(
            job_id=job_id,
            from_state=from_state,
            to_state=to_state,
            duration_ms=duration_ms,
            error=error,
        )
        db.add(transition)

        job = await db.get(IngestJob, job_id)
        if job is not None:
            job.state = to_state
            if error is not None:
                job.error = error

        await db.commit()
        await db.refresh(transition)
        return transition

    async def get_transitions(self, db: AsyncSession, job_id: str) -> Sequence[JobTransition]:
        stmt = (
            select(JobTransition).where(JobTransition.job_id == job_id).order_by(JobTransition.at)
        )
        result = await db.execute(stmt)
        return result.scalars().all()

    async def delete(self, db: AsyncSession, job_id: str) -> bool:
        job = await db.get(IngestJob, job_id)
        if job is None:
            return False
        await db.delete(job)
        await db.commit()
        return True


job_repository = JobRepository()
