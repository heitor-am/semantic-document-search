"""Response schemas for the /search endpoint."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.context import Candidate
from app.retrieval.service import Strategy


class SearchHit(BaseModel):
    """One result row surfaced by /search.

    `score` is the score from the *last* stage that touched this
    candidate — so for `hybrid_rerank`, it's the rerank relevance;
    for plain `hybrid`, it's the RRF score. The scales aren't
    comparable across strategies (documented in the README /
    OpenAPI), which is fine because clients use scores to order
    *within* a response, not *across* responses.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    score: float
    content: str
    title: str
    source_url: str
    source_type: str
    section_path: list[str] = Field(default_factory=list)
    parent_chunk_id: str | None = None
    author: str | None = None
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def from_candidate(cls, c: Candidate) -> SearchHit:
        p = c.payload
        return cls(
            chunk_id=c.chunk_id,
            score=c.score,
            content=str(p.get("content", "")),
            title=str(p.get("title", "")),
            source_url=str(p.get("source_url", "")),
            source_type=str(p.get("source_type", "")),
            section_path=list(p.get("section_path") or []),
            parent_chunk_id=(
                str(p["parent_chunk_id"]) if p.get("parent_chunk_id") is not None else None
            ),
            author=(str(p["author"]) if p.get("author") is not None else None),
            tags=list(p.get("tags") or []),
        )


class SearchResponse(BaseModel):
    query: str
    strategy: Strategy
    top_k: int
    results: list[SearchHit]
    # Non-fatal stage failures from the pipeline (e.g. reranker timeout):
    # the response still comes back, this field tells the client that
    # a stage was skipped.
    warnings: list[str] = Field(default_factory=list)
