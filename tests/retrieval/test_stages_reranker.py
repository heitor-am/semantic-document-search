from __future__ import annotations

import httpx
import pytest
import respx

from app.retrieval.context import Candidate, Context
from app.retrieval.pipeline import StageError
from app.retrieval.stages.reranker import RerankerStage

RERANK_URL = "https://openrouter.ai/api/v1/rerank"


def make_stage(client: httpx.AsyncClient) -> RerankerStage:
    return RerankerStage(
        httpx_client=client,
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
        model="cohere/rerank-v3.5",
        app_url="http://localhost:8000",
        app_name="Semantic Document Search",
    )


def make_ctx(*chunks: tuple[str, float, str]) -> Context:
    ctx = Context(query="python async", top_k=5)
    ctx.results = [
        Candidate(chunk_id=cid, score=score, payload={"content": content})
        for cid, score, content in chunks
    ]
    return ctx


class TestRerankerStage:
    @respx.mock
    async def test_reorders_results_by_relevance_score(self) -> None:
        respx.post(RERANK_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "r-1",
                    "model": "cohere/rerank-v3.5",
                    "provider": "cohere",
                    "results": [
                        # Reverse order — index 2 is most relevant, then 0, then 1.
                        {"index": 2, "relevance_score": 0.95, "document": {"text": "c"}},
                        {"index": 0, "relevance_score": 0.60, "document": {"text": "a"}},
                        {"index": 1, "relevance_score": 0.30, "document": {"text": "b"}},
                    ],
                    "usage": {"cost": 0.001, "search_units": 1, "total_tokens": 100},
                },
            )
        )

        async with httpx.AsyncClient() as client:
            stage = make_stage(client)
            ctx = make_ctx(
                ("c0", 0.5, "a"),
                ("c1", 0.4, "b"),
                ("c2", 0.3, "c"),
            )

            ctx = await stage.run(ctx)

        assert [c.chunk_id for c in ctx.results] == ["c2", "c0", "c1"]
        assert ctx.results[0].score == 0.95
        assert ctx.results[1].score == 0.60

    @respx.mock
    async def test_sends_expected_request_body(self) -> None:
        route = respx.post(RERANK_URL).mock(return_value=httpx.Response(200, json={"results": []}))

        async with httpx.AsyncClient() as client:
            stage = make_stage(client)
            # 3 docs, top_k=3 → no clamp needed, top_n in the wire body is 3.
            ctx = make_ctx(("c0", 0.5, "alpha"), ("c1", 0.4, "beta"), ("c2", 0.3, "gamma"))
            ctx.top_k = 3
            await stage.run(ctx)

        assert route.called
        body = route.calls.last.request.content
        import json as _json

        payload = _json.loads(body)
        assert payload["query"] == "python async"
        assert payload["documents"] == ["alpha", "beta", "gamma"]
        assert payload["model"] == "cohere/rerank-v3.5"
        assert payload["top_n"] == 3
        assert route.calls.last.request.headers["Authorization"] == "Bearer sk-test"

    async def test_empty_results_is_noop(self) -> None:
        async with httpx.AsyncClient() as client:
            stage = make_stage(client)
            ctx = Context(query="q", top_k=5)

            ctx = await stage.run(ctx)

        assert ctx.results == []

    @respx.mock
    async def test_http_error_raises_stage_error(self) -> None:
        respx.post(RERANK_URL).mock(return_value=httpx.Response(503, text="upstream down"))

        async with httpx.AsyncClient() as client:
            stage = make_stage(client)
            ctx = make_ctx(("c0", 0.5, "a"))

            with pytest.raises(StageError) as exc_info:
                await stage.run(ctx)

        assert exc_info.value.stage_name == "reranker"
        assert isinstance(exc_info.value.cause, httpx.HTTPStatusError)

    @respx.mock
    async def test_network_error_raises_stage_error(self) -> None:
        respx.post(RERANK_URL).mock(side_effect=httpx.ConnectError("no route"))

        async with httpx.AsyncClient() as client:
            stage = make_stage(client)
            ctx = make_ctx(("c0", 0.5, "a"))

            with pytest.raises(StageError) as exc_info:
                await stage.run(ctx)

        assert exc_info.value.stage_name == "reranker"

    def test_marked_optional(self) -> None:
        # optional is a class attribute — no event loop needed.
        assert RerankerStage.optional is True

    @respx.mock
    async def test_clamps_top_n_to_document_count(self) -> None:
        # ctx.top_k=10 but only 2 docs in results → top_n should be 2.
        route = respx.post(RERANK_URL).mock(return_value=httpx.Response(200, json={"results": []}))
        async with httpx.AsyncClient() as client:
            stage = make_stage(client)
            ctx = make_ctx(("c0", 0.5, "a"), ("c1", 0.4, "b"))
            ctx.top_k = 10

            await stage.run(ctx)

        import json as _json

        payload = _json.loads(route.calls.last.request.content)
        assert payload["top_n"] == 2

    @respx.mock
    async def test_reranked_candidates_get_copied_payloads(self) -> None:
        # Mutating a reranked candidate's payload must not leak into the
        # original, keeping state isolation consistent with HybridSearchStage.
        respx.post(RERANK_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 0, "relevance_score": 0.9, "document": {"text": "a"}},
                    ],
                },
            )
        )
        async with httpx.AsyncClient() as client:
            stage = make_stage(client)
            ctx = make_ctx(("c0", 0.5, "a"))
            original_payload = ctx.results[0].payload

            ctx = await stage.run(ctx)

            # Reranked candidate has its own payload dict
            assert ctx.results[0].payload is not original_payload
