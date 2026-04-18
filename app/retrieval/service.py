"""Retrieval service: composes the pipeline from a strategy name.

Strategies map directly to stage compositions. Keeping the mapping in
one place means the router doesn't have to know which stages exist,
and the eval framework (Etapa 12) can run the same strategies it sees
in prod without reaching into stage wiring.

    HYBRID          → HybridSearchStage  + ParentChildStage
    HYBRID_RERANK   → HybridSearchStage  + RerankerStage + ParentChildStage
"""

from __future__ import annotations

from enum import StrEnum

import httpx
from openai import AsyncOpenAI

from app.config import Settings, get_settings
from app.retrieval.pipeline import Pipeline, Stage
from app.retrieval.stages.hybrid import HybridSearchStage
from app.retrieval.stages.parent_child import ParentChildStage
from app.retrieval.stages.reranker import RerankerStage
from app.shared.qdrant.repository import VectorRepository


class Strategy(StrEnum):
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


DEFAULT_STRATEGY = Strategy.HYBRID_RERANK


def build_pipeline(
    strategy: Strategy,
    *,
    vector_repo: VectorRepository,
    openai_client: AsyncOpenAI,
    httpx_client: httpx.AsyncClient,
    settings: Settings | None = None,
) -> Pipeline:
    s = settings or get_settings()

    stages: list[Stage] = [
        HybridSearchStage(
            vector_repo=vector_repo,
            openai_client=openai_client,
            embedding_model=s.openrouter_embedding_model,
        ),
    ]

    if strategy == Strategy.HYBRID_RERANK:
        stages.append(
            RerankerStage(
                httpx_client=httpx_client,
                api_key=s.openrouter_api_key,
                base_url=s.openrouter_base_url,
                model=s.openrouter_rerank_model,
                app_url=s.openrouter_app_url,
                app_name=s.openrouter_app_name,
            )
        )

    # Dedup is useful regardless of reranker; top-k of hybrid can easily
    # include 3-4 children of the same parent.
    stages.append(ParentChildStage())

    return Pipeline(stages)
