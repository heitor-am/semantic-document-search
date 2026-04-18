from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.ingestion.state import JobState


class SourceDocument(BaseModel):
    """Canonical document model. Sources (dev.to, wikipedia, pdf, ...) normalize
    into this at the fetcher boundary so chunker/indexer stay source-agnostic.
    """

    model_config = ConfigDict(frozen=True)

    source_url: str
    source_type: str = Field(description="Identifier for the source: 'dev.to', 'wikipedia', ...")
    title: str
    body_markdown: str
    author: str | None = None
    published_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific fields that don't fit the common taxonomy.",
    )


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
