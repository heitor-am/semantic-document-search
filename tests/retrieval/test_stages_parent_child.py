from __future__ import annotations

from app.retrieval.context import Candidate, Context
from app.retrieval.stages.parent_child import ParentChildStage


def cand(chunk_id: str, score: float, parent_id: str | None) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        score=score,
        payload={"content": chunk_id, "parent_chunk_id": parent_id},
    )


class TestParentChildStage:
    async def test_empty_results_is_noop(self) -> None:
        ctx = Context(query="q", top_k=5)
        ctx = await ParentChildStage().run(ctx)
        assert ctx.results == []

    async def test_keeps_highest_scoring_child_per_parent(self) -> None:
        ctx = Context(query="q", top_k=10)
        ctx.results = [
            cand("c1a", 0.9, "p1"),
            cand("c1b", 0.7, "p1"),
            cand("c2a", 0.8, "p2"),
            cand("c1c", 0.6, "p1"),
        ]
        ctx = await ParentChildStage().run(ctx)

        ids = [c.chunk_id for c in ctx.results]
        # c1a wins p1 (0.9 > 0.7, 0.6); c2a wins p2
        assert set(ids) == {"c1a", "c2a"}

    async def test_preserves_score_ordering_after_dedup(self) -> None:
        ctx = Context(query="q", top_k=10)
        ctx.results = [
            cand("c1", 0.6, "p1"),
            cand("c2", 0.9, "p2"),
            cand("c3", 0.7, "p3"),
        ]
        ctx = await ParentChildStage().run(ctx)
        scores = [c.score for c in ctx.results]
        assert scores == sorted(scores, reverse=True)

    async def test_orphans_without_parent_are_kept(self) -> None:
        ctx = Context(query="q", top_k=10)
        ctx.results = [
            cand("parent-chunk", 0.8, None),  # itself a parent
            cand("child-1", 0.7, "parent-chunk"),
            cand("child-2", 0.6, "another-parent"),
        ]
        ctx = await ParentChildStage().run(ctx)

        ids = {c.chunk_id for c in ctx.results}
        # The parent-chunk (orphan = None) stays; child-1 and child-2 too
        # (they target different parents).
        assert "parent-chunk" in ids
        assert len(ids) == 3

    async def test_applies_top_k_after_dedup(self) -> None:
        ctx = Context(query="q", top_k=2)
        ctx.results = [
            cand("a", 0.9, "p1"),
            cand("b", 0.8, "p2"),
            cand("c", 0.7, "p3"),
            cand("d", 0.6, "p4"),
        ]
        ctx = await ParentChildStage().run(ctx)
        assert len(ctx.results) == 2
        assert [c.chunk_id for c in ctx.results] == ["a", "b"]

    async def test_stores_result_in_state(self) -> None:
        ctx = Context(query="q", top_k=5)
        ctx.results = [cand("a", 0.5, "p")]
        ctx = await ParentChildStage().run(ctx)
        assert ctx.state["parent_child"] == ctx.results
