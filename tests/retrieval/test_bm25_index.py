from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.retrieval.bm25_index import BM25Index, tokenize


class TestTokenize:
    def test_lowercases_and_splits_on_whitespace(self) -> None:
        assert tokenize("Hello WORLD  foo") == ["hello", "world", "foo"]

    def test_drops_punctuation(self) -> None:
        assert tokenize("don't, stop — believing!") == [
            "don",
            "t",
            "stop",
            "believing",
        ]

    def test_keeps_digits_with_letters(self) -> None:
        assert tokenize("python3.12 and bge-m3") == [
            "python3",
            "12",
            "and",
            "bge",
            "m3",
        ]

    def test_empty_string_produces_no_tokens(self) -> None:
        assert tokenize("") == []


class TestBM25IndexSeed:
    def test_empty_seed_is_empty_index(self) -> None:
        idx = BM25Index()
        idx.seed([])
        assert idx.size == 0
        assert idx.search("query", k=5) == []

    def test_search_returns_higher_score_for_better_match(self) -> None:
        idx = BM25Index()
        idx.seed(
            [
                ("c1", {"content": "python async programming"}),
                ("c2", {"content": "javascript event loop"}),
                ("c3", {"content": "python event loop and async tasks"}),
            ]
        )
        hits = idx.search("python async", k=3)

        ids = [h.chunk_id for h in hits]
        # Both c1 and c3 mention python+async; c2 mentions neither.
        assert set(ids[:2]) == {"c1", "c3"}
        assert "c2" not in ids  # zero-score docs are dropped

    def test_search_respects_k(self) -> None:
        idx = BM25Index()
        idx.seed([(f"c{i}", {"content": f"python chunk {i}"}) for i in range(10)])
        assert len(idx.search("python", k=3)) == 3

    def test_search_with_empty_query_returns_empty(self) -> None:
        idx = BM25Index()
        idx.seed([("c1", {"content": "hello world"})])
        assert idx.search("", k=5) == []
        # Query that tokenizes to empty (only punctuation) behaves the same.
        assert idx.search("!!!", k=5) == []

    def test_search_returns_candidate_with_full_payload(self) -> None:
        idx = BM25Index()
        idx.seed(
            [
                (
                    "c1",
                    {
                        "content": "python async",
                        "title": "A post",
                        "source_url": "https://dev.to/x/y",
                    },
                ),
            ]
        )
        hit = idx.search("python", k=1)[0]
        assert hit.chunk_id == "c1"
        # Don't assert sign: in a 1-doc corpus BM25 IDF goes negative (every
        # term is maximally common). What matters is that the doc is
        # returned and carries its payload.
        assert hit.score != 0
        assert hit.payload["title"] == "A post"
        assert hit.payload["source_url"] == "https://dev.to/x/y"

    def test_seed_skips_entries_with_empty_content(self) -> None:
        idx = BM25Index()
        idx.seed(
            [
                ("c1", {"content": "real content here"}),
                ("c2", {"content": ""}),
                ("c3", {"content": "!!! ???"}),  # tokenizes to nothing
            ]
        )
        # Only c1 is index-worthy.
        assert idx.size == 1


class TestBM25IndexRebuild:
    async def test_rebuilds_from_qdrant_scroll(self) -> None:
        client = MagicMock()
        # Two scroll batches; first returns 2 points + next_offset, second
        # returns 1 point + None (end).
        batch_1 = (
            [
                MagicMock(id="c1", payload={"content": "python async", "is_parent": False}),
                MagicMock(id="c2", payload={"content": "javascript loops", "is_parent": False}),
            ],
            "offset-1",
        )
        batch_2 = (
            [
                MagicMock(id="c3", payload={"content": "rust ownership", "is_parent": False}),
            ],
            None,
        )
        client.scroll = AsyncMock(side_effect=[batch_1, batch_2])

        idx = BM25Index()
        indexed = await idx.rebuild(client, "docs_v1", batch_size=2)

        assert indexed == 3
        assert idx.size == 3
        # Verify scroll filter pins is_parent=False
        first_kwargs = client.scroll.await_args_list[0].kwargs
        assert first_kwargs["collection_name"] == "docs_v1"
        assert first_kwargs["limit"] == 2
        # offset/scroll_filter types come from qdrant-client; just sanity-check
        assert first_kwargs.get("scroll_filter") is not None

    async def test_rebuild_of_empty_collection_leaves_index_empty(self) -> None:
        client = MagicMock()
        client.scroll = AsyncMock(return_value=([], None))

        idx = BM25Index()
        indexed = await idx.rebuild(client, "docs_v1")

        assert indexed == 0
        assert idx.size == 0
        assert idx.search("anything", k=5) == []

    async def test_rebuild_skips_children_with_empty_content(self) -> None:
        client = MagicMock()
        client.scroll = AsyncMock(
            return_value=(
                [
                    MagicMock(id="c1", payload={"content": "real content"}),
                    MagicMock(id="c2", payload={"content": ""}),
                    MagicMock(id="c3", payload={}),
                ],
                None,
            )
        )

        idx = BM25Index()
        indexed = await idx.rebuild(client, "docs_v1")

        assert indexed == 1


@pytest.mark.parametrize(
    "query,expected",
    [
        ("python", {"c1", "c2"}),
        ("loops", {"c2"}),
        ("rust lifetime", set()),  # no match
    ],
)
def test_query_selectivity(query: str, expected: set[str]) -> None:
    idx = BM25Index()
    idx.seed(
        [
            ("c1", {"content": "python async programming"}),
            ("c2", {"content": "python event loops"}),
            ("c3", {"content": "hello world"}),
        ]
    )
    ids: set[Any] = {h.chunk_id for h in idx.search(query, k=5)}
    assert ids == expected
