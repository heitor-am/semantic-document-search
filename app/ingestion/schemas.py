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


class Chunk(BaseModel):
    """Parent or child chunk produced by hierarchical splitting.

    Parents preserve header sections; children are the embeddable units.
    SourceDocument metadata is denormalized so retrieval can filter/rank
    without a join back to the document table.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    parent_chunk_id: str | None = Field(
        default=None,
        description="None if this chunk is itself a parent; otherwise the parent's chunk_id.",
    )
    content: str
    char_count: int
    section_path: list[str] = Field(
        default_factory=list,
        description="Ancestor headings (h1 → h3) for the section this chunk belongs to.",
    )
    chunk_index: int = Field(
        description=(
            "For a parent: its 0-based position among all parents in the document. "
            "For a child: its 0-based position within the parent."
        ),
    )

    source_url: str
    source_type: str
    title: str
    author: str | None = None
    published_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)


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
