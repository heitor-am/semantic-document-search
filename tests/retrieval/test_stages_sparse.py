from __future__ import annotations

from app.retrieval.bm25_index import BM25Index
from app.retrieval.context import Context
from app.retrieval.stages.sparse import SparseSearchStage


def make_index() -> BM25Index:
    idx = BM25Index()
    idx.seed(
        [
            ("c1", {"content": "python async programming"}),
            ("c2", {"content": "javascript event loop"}),
            ("c3", {"content": "python event loops and tasks"}),
        ]
    )
    return idx


class TestSparseSearchStage:
    async def test_writes_candidates_to_state_sparse(self) -> None:
        stage = SparseSearchStage(make_index())
        ctx = await stage.run(Context(query="python"))
        ids = {c.chunk_id for c in ctx.state["sparse"]}
        assert ids == {"c1", "c3"}  # javascript doc scored 0, excluded

    async def test_fetches_top_k_times_multiplier(self) -> None:
        idx = BM25Index()
        idx.seed([(f"c{i}", {"content": f"python tip {i}"}) for i in range(10)])

        stage = SparseSearchStage(idx, fetch_multiplier=2)
        ctx = await stage.run(Context(query="python", top_k=3))

        # 3 * 2 = 6
        assert len(ctx.state["sparse"]) == 6

    async def test_empty_index_returns_empty_list(self) -> None:
        stage = SparseSearchStage(BM25Index())
        ctx = await stage.run(Context(query="anything"))
        assert ctx.state["sparse"] == []

    async def test_query_with_no_matches_returns_empty(self) -> None:
        stage = SparseSearchStage(make_index())
        ctx = await stage.run(Context(query="rust lifetime"))
        assert ctx.state["sparse"] == []

    async def test_candidates_carry_payload(self) -> None:
        idx = BM25Index()
        idx.seed(
            [
                (
                    "c1",
                    {
                        "content": "python tips",
                        "title": "Python Tips",
                        "source_url": "https://dev.to/x/y",
                    },
                ),
            ]
        )
        stage = SparseSearchStage(idx)
        ctx = await stage.run(Context(query="python"))
        hit = ctx.state["sparse"][0]
        assert hit.payload["title"] == "Python Tips"
