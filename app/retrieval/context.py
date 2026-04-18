from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.retrieval.pipeline import StageError


@dataclass(frozen=True)
class Candidate:
    """A chunk surfaced by retrieval, with its score and full payload.

    Stages refine the population of candidates (dense/sparse produce them,
    fusion merges them, rerank reorders, parent-child expands context), so
    the payload has to carry enough metadata — section_path, parent_chunk_id,
    title, tags — to round-trip through every stage without extra lookups.
    """

    chunk_id: str
    score: float
    payload: Mapping[str, Any]

    @property
    def content(self) -> str:
        return str(self.payload.get("content", ""))

    @property
    def parent_chunk_id(self) -> str | None:
        raw = self.payload.get("parent_chunk_id")
        return None if raw is None else str(raw)


@dataclass
class Context:
    """Mutable state carried through every retrieval stage.

    `state` is an intentionally-generic key-value bag so stages can stash
    intermediate results (e.g. `state["dense"]` vs `state["sparse"]`)
    without bloating this class with stage-specific fields. `results` is
    the final ranked list that the endpoint returns; the terminal stage
    populates it.
    """

    query: str
    top_k: int = 10
    min_score: float = 0.0
    state: dict[str, Any] = field(default_factory=dict)
    results: list[Candidate] = field(default_factory=list)
    errors: list[StageError] = field(default_factory=list)
