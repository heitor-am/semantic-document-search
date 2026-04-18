"""In-memory BM25 index over the child chunks stored in Qdrant.

Why not persist to disk / Redis? At MVP scale the corpus is <1 000 chunks; a
scan of the Qdrant payload is a few hundred ms at most. Rebuilding on boot
keeps the moving parts to zero. Documented upgrade path (ADR-007): persist to
pickle + invalidate on ingest when corpus outgrows this.

Why only children? Parents exist for context expansion (the parent-child
stage fetches them by id after retrieval). Indexing them would dilute BM25
scoring and duplicate content — the child is literally a substring of its
parent, so a query that matches the child will also match the parent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from rank_bm25 import BM25Okapi

from app.retrieval.context import Candidate

logger = structlog.get_logger()

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens.

    Good enough for English technical prose (the bulk of dev.to). For
    multilingual corpora we'd swap in spaCy or a unicode-aware splitter;
    left as-is until the eval shows it's the bottleneck.
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _Entry:
    chunk_id: str
    payload: dict[str, Any]


@dataclass
class BM25Index:
    """Rebuildable in-memory BM25Okapi index over child chunks.

    One instance lives on app.state; the retrieval Pipeline's sparse stage
    borrows it. Not thread-safe to rebuild while a search is in flight — but
    BM25Okapi.get_scores is read-only, so concurrent *searches* are fine.
    """

    _bm25: BM25Okapi | None = None
    _entries: list[_Entry] = field(default_factory=list)
    # Tokenised corpus kept alongside BM25Okapi so we can filter on
    # query-token overlap explicitly. Needed because BM25's IDF can go
    # negative on tiny corpora (term appears in most docs), which means
    # filtering by `score > 0` would drop legitimate matches.
    _corpus: list[list[str]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self._entries)

    async def rebuild(
        self,
        client: AsyncQdrantClient,
        collection: str,
        *,
        batch_size: int = 256,
    ) -> int:
        """Scroll every child chunk out of Qdrant and rebuild the index.

        Returns the number of entries indexed.
        """
        corpus: list[list[str]] = []
        entries: list[_Entry] = []
        next_offset: Any = None

        children_only = models.Filter(
            must=[
                models.FieldCondition(key="is_parent", match=models.MatchValue(value=False)),
            ],
        )

        while True:
            points, next_offset = await client.scroll(
                collection_name=collection,
                scroll_filter=children_only,
                offset=next_offset,
                limit=batch_size,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = dict(p.payload or {})
                tokens = tokenize(str(payload.get("content", "")))
                if not tokens:
                    continue
                corpus.append(tokens)
                entries.append(_Entry(chunk_id=str(p.id), payload=payload))
            if next_offset is None:
                break

        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._entries = entries
        self._corpus = corpus
        logger.info("bm25.rebuilt", collection=collection, entries=len(entries))
        return len(entries)

    def search(self, query: str, *, k: int) -> list[Candidate]:
        """Return the top-k children by BM25 score.

        Empty query (no tokens after normalization) or empty index yields []
        instead of raising — downstream fusion handles the empty branch.
        Docs without any query-token overlap are excluded even if BM25 would
        assign them a non-zero (possibly negative) score.
        """
        if self._bm25 is None or not self._entries:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_set = set(query_tokens)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            (
                (float(scores[i]), i)
                for i in range(len(self._entries))
                if query_set.intersection(self._corpus[i])
            ),
            reverse=True,
        )
        return [
            Candidate(
                chunk_id=self._entries[i].chunk_id,
                score=score,
                payload=self._entries[i].payload,
            )
            for score, i in ranked[:k]
        ]

    def seed(self, entries: Sequence[tuple[str, dict[str, Any]]]) -> None:
        """Test helper: populate the index directly without Qdrant.

        `entries` is a sequence of (chunk_id, payload) where payload["content"]
        is the text used for BM25 scoring.
        """
        corpus: list[list[str]] = []
        self._entries = []
        for chunk_id, payload in entries:
            tokens = tokenize(str(payload.get("content", "")))
            if not tokens:
                continue
            corpus.append(tokens)
            self._entries.append(_Entry(chunk_id=chunk_id, payload=dict(payload)))
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._corpus = corpus
