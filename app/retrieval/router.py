"""GET /search — the v0.2.0 gate endpoint."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, status

from app.retrieval.context import Context
from app.retrieval.schemas import SearchHit, SearchResponse
from app.retrieval.service import DEFAULT_STRATEGY, Strategy, build_pipeline
from app.shared.api.deps import HttpxDep, OpenAIDep, VectorRepoDep
from app.shared.schemas.problem import ProblemDetails

_PROBLEM_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_422_UNPROCESSABLE_ENTITY: {
        "model": ProblemDetails,
        "content": {"application/problem+json": {}},
        "description": "Validation error",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ProblemDetails,
        "content": {"application/problem+json": {}},
        "description": "LLM or vector store backend is not configured",
    },
}

router = APIRouter(prefix="/search", tags=["retrieval"], responses=_PROBLEM_RESPONSES)


@router.get("", response_model=SearchResponse)
async def search(
    vector_repo: VectorRepoDep,
    httpx_client: HttpxDep,
    openai_client: OpenAIDep,
    q: Annotated[str, Query(min_length=1, description="Query text")],
    strategy: Annotated[
        Strategy,
        Query(
            description="Pipeline preset: 'hybrid' (dense+BM25+RRF) or 'hybrid_rerank' (+ reranker)"
        ),
    ] = DEFAULT_STRATEGY,
    top_k: Annotated[int, Query(ge=1, le=50, description="Max results")] = 10,
    min_score: Annotated[
        float,
        Query(ge=0.0, description="Drop results below this score (post-pipeline filter)"),
    ] = 0.0,
) -> SearchResponse:
    """Semantic search over ingested documents."""
    pipeline = build_pipeline(
        strategy,
        vector_repo=vector_repo,
        openai_client=openai_client,
        httpx_client=httpx_client,
    )
    ctx = Context(query=q, top_k=top_k, min_score=min_score)
    ctx = await pipeline.run(ctx)

    hits = [SearchHit.from_candidate(c) for c in ctx.results if c.score >= min_score]
    warnings = [str(e) for e in ctx.errors]

    return SearchResponse(
        query=q,
        strategy=strategy,
        top_k=top_k,
        results=hits,
        warnings=warnings,
    )
