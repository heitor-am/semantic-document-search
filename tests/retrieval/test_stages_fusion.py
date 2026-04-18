from __future__ import annotations

from app.retrieval.context import Candidate, Context
from app.retrieval.stages.fusion import RRFFusionStage


def cand(chunk_id: str, score: float = 0.0, **payload: object) -> Candidate:
    return Candidate(chunk_id=chunk_id, score=score, payload={**payload})


class TestRRFFusionStage:
    async def test_merges_disjoint_lists(self) -> None:
        ctx = Context(query="q", top_k=10)
        ctx.state["dense"] = [cand("d1"), cand("d2")]
        ctx.state["sparse"] = [cand("s1"), cand("s2")]

        ctx = await RRFFusionStage().run(ctx)

        ids = [c.chunk_id for c in ctx.state["fused"]]
        assert set(ids) == {"d1", "d2", "s1", "s2"}

    async def test_document_in_both_lists_scores_higher(self) -> None:
        # c1 is rank 0 in both lists -> 2 * 1/(60+1) ~= 0.0328
        # c2 is rank 1 in dense only -> 1/(60+2) ~= 0.0161
        ctx = Context(query="q", top_k=10)
        ctx.state["dense"] = [cand("c1"), cand("c2")]
        ctx.state["sparse"] = [cand("c1"), cand("c3")]

        ctx = await RRFFusionStage(k=60).run(ctx)

        fused = ctx.state["fused"]
        assert fused[0].chunk_id == "c1"
        assert fused[0].score > fused[1].score

    async def test_rrf_score_formula(self) -> None:
        # Single source, single doc at rank 0 → score = 1/(60+1)
        ctx = Context(query="q", top_k=10)
        ctx.state["dense"] = [cand("c1")]

        ctx = await RRFFusionStage(k=60).run(ctx)

        assert ctx.state["fused"][0].score == pytest.approx(1 / 61)

    async def test_writes_top_k_to_results(self) -> None:
        ctx = Context(query="q", top_k=2)
        ctx.state["dense"] = [cand(f"c{i}") for i in range(10)]

        ctx = await RRFFusionStage().run(ctx)

        # Results is limited to top_k; fused keeps more for rerank slack
        assert len(ctx.results) == 2
        assert len(ctx.state["fused"]) >= 2

    async def test_empty_sources_produces_empty_fused(self) -> None:
        ctx = Context(query="q", top_k=5)
        ctx = await RRFFusionStage().run(ctx)
        assert ctx.state["fused"] == []
        assert ctx.results == []

    async def test_missing_source_keys_are_tolerated(self) -> None:
        # Source declared but key not populated (e.g. stage skipped)
        ctx = Context(query="q", top_k=5)
        ctx.state["dense"] = [cand("c1"), cand("c2")]
        # No "sparse" key at all
        ctx = await RRFFusionStage().run(ctx)
        assert {c.chunk_id for c in ctx.state["fused"]} == {"c1", "c2"}

    async def test_custom_sources_list_is_honoured(self) -> None:
        ctx = Context(query="q", top_k=5)
        ctx.state["a"] = [cand("x"), cand("y")]
        ctx.state["b"] = [cand("y"), cand("z")]

        ctx = await RRFFusionStage(sources=("a", "b")).run(ctx)
        # y is in both → highest score
        assert ctx.state["fused"][0].chunk_id == "y"

    async def test_payload_preserved_from_first_source(self) -> None:
        # Same chunk_id appears in both lists with different payload; the
        # first source wins (dense's Qdrant payload is richer than BM25's).
        ctx = Context(query="q", top_k=5)
        ctx.state["dense"] = [cand("c1", content="DENSE", title="T")]
        ctx.state["sparse"] = [cand("c1", content="SPARSE")]

        ctx = await RRFFusionStage().run(ctx)
        assert ctx.state["fused"][0].payload["content"] == "DENSE"
        assert ctx.state["fused"][0].payload["title"] == "T"


# Imported late so the import doesn't leak into module-level test discovery
import pytest  # noqa: E402
