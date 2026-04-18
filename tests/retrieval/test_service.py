from __future__ import annotations

from unittest.mock import MagicMock

import httpx

from app.config import Settings
from app.retrieval.service import Strategy, build_pipeline
from app.retrieval.stages.hybrid import HybridSearchStage
from app.retrieval.stages.parent_child import ParentChildStage
from app.retrieval.stages.reranker import RerankerStage


def _settings() -> Settings:
    return Settings(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_embedding_model="baai/bge-m3",
        openrouter_rerank_model="cohere/rerank-v3.5",
        openrouter_app_url="http://localhost:8000",
        openrouter_app_name="Semantic Document Search",
    )


class TestBuildPipeline:
    def test_hybrid_strategy_has_two_stages(self) -> None:
        pipe = build_pipeline(
            Strategy.HYBRID,
            vector_repo=MagicMock(),
            openai_client=MagicMock(),
            httpx_client=MagicMock(spec=httpx.AsyncClient),
            settings=_settings(),
        )
        names = [s.name for s in pipe.stages]
        assert names == ["hybrid", "parent_child"]

    def test_hybrid_rerank_strategy_inserts_reranker_before_parent_child(self) -> None:
        pipe = build_pipeline(
            Strategy.HYBRID_RERANK,
            vector_repo=MagicMock(),
            openai_client=MagicMock(),
            httpx_client=MagicMock(spec=httpx.AsyncClient),
            settings=_settings(),
        )
        names = [s.name for s in pipe.stages]
        assert names == ["hybrid", "reranker", "parent_child"]

    def test_hybrid_uses_configured_embedding_model(self) -> None:
        pipe = build_pipeline(
            Strategy.HYBRID,
            vector_repo=MagicMock(),
            openai_client=MagicMock(),
            httpx_client=MagicMock(spec=httpx.AsyncClient),
            settings=_settings(),
        )
        hybrid = next(s for s in pipe.stages if isinstance(s, HybridSearchStage))
        assert hybrid._embedding_model == "baai/bge-m3"

    def test_reranker_uses_configured_model(self) -> None:
        pipe = build_pipeline(
            Strategy.HYBRID_RERANK,
            vector_repo=MagicMock(),
            openai_client=MagicMock(),
            httpx_client=MagicMock(spec=httpx.AsyncClient),
            settings=_settings(),
        )
        reranker = next(s for s in pipe.stages if isinstance(s, RerankerStage))
        assert reranker._model == "cohere/rerank-v3.5"

    def test_parent_child_is_terminal_stage(self) -> None:
        for strategy in Strategy:
            pipe = build_pipeline(
                strategy,
                vector_repo=MagicMock(),
                openai_client=MagicMock(),
                httpx_client=MagicMock(spec=httpx.AsyncClient),
                settings=_settings(),
            )
            assert isinstance(pipe.stages[-1], ParentChildStage)
