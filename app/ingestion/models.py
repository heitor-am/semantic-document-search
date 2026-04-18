from datetime import datetime

from sqlalchemy import Enum as SAEnum
from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.ingestion.state import JobState
from app.shared.db.database import Base

# Single SAEnum instance reused across columns so SQLAlchemy doesn't try to
# register the enum type twice.
_JOB_STATE_ENUM = SAEnum(JobState, name="job_state", native_enum=False)


class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_url: Mapped[str] = mapped_column(String(2000), nullable=False, index=True)
    state: Mapped[JobState] = mapped_column(
        _JOB_STATE_ENUM,
        nullable=False,
        default=JobState.PENDING,
        server_default=JobState.PENDING.value,
    )
    error: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    transitions: Mapped[list["JobTransition"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="JobTransition.at",
    )

    def __repr__(self) -> str:
        return f"<IngestJob id={self.id!r} state={self.state}>"


class JobTransition(Base):
    __tablename__ = "job_transitions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("ingest_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_state: Mapped[JobState | None] = mapped_column(_JOB_STATE_ENUM, nullable=True)
    to_state: Mapped[JobState] = mapped_column(_JOB_STATE_ENUM, nullable=False)
    at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(String(2000), nullable=True)

    job: Mapped["IngestJob"] = relationship(back_populates="transitions")

    def __repr__(self) -> str:
        return f"<JobTransition job_id={self.job_id!r} {self.from_state} → {self.to_state}>"
