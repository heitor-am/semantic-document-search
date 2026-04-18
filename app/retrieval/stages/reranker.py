"""Cross-encoder reranker stage via OpenRouter's rerank endpoint.

OpenRouter proxies Cohere's rerank API (same path, same auth) so the
existing OPENROUTER_API_KEY is enough — no separate Cohere account.
Default model is `cohere/rerank-v3.5` ($0.001/search). The endpoint is
distinct from chat/embeddings (custom rerank API, not OpenAI-spec), so
we call it with httpx directly.

Marked optional: if OpenRouter rate-limits, times out, or the model is
unavailable, the hybrid-fusion result already in ctx.results is what
the endpoint returns. The pipeline records the StageError on
ctx.errors so the response can surface that reranking was skipped.
"""

from __future__ import annotations

import httpx

from app.retrieval.context import Candidate, Context
from app.retrieval.pipeline import Stage, StageError


class RerankerStage(Stage):
    name = "reranker"
    optional = True

    def __init__(
        self,
        *,
        httpx_client: httpx.AsyncClient,
        api_key: str,
        base_url: str,
        model: str,
        app_url: str,
        app_name: str,
        timeout: float = 10.0,
    ) -> None:
        self._client = httpx_client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._app_url = app_url
        self._app_name = app_name
        self._timeout = timeout

    async def run(self, ctx: Context) -> Context:
        if not ctx.results:
            return ctx

        documents = [c.content for c in ctx.results]
        try:
            response = await self._client.post(
                f"{self._base_url}/rerank",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "HTTP-Referer": self._app_url,
                    "X-Title": self._app_name,
                },
                json={
                    "query": ctx.query,
                    "documents": documents,
                    "model": self._model,
                    "top_n": ctx.top_k,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            raise StageError(self.name, cause=exc) from exc

        reordered = []
        for item in data.get("results", []):
            idx = item["index"]
            original = ctx.results[idx]
            reordered.append(
                Candidate(
                    chunk_id=original.chunk_id,
                    score=float(item["relevance_score"]),
                    payload=original.payload,
                )
            )
        ctx.results = reordered
        ctx.state["reranked"] = reordered
        return ctx
