from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.ingestion.state import JobState


class IngestRequest(BaseModel):
    source_url: HttpUrl = Field(description="URL of the article to ingest (e.g. a dev.to post)")


class JobTransitionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    state: JobState
    at: datetime


class IngestJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: str
    source_url: str
    state: JobState
    error: str | None = None
    history: list[JobTransitionRead]
