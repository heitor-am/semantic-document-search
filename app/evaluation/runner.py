"""Evaluation runner — replays golden queries against each retrieval
strategy and prints a comparative table.

Invoked via `make eval` or `python -m app.evaluation.runner`. Requires
the app to be running (it hits GET /search the same way a real client
would, so the numbers reflect end-to-end production behaviour including
reranker latency, fusion, payload deserialization).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx
import yaml

from app.evaluation.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank
from app.evaluation.schemas import GoldenQuery, GoldenSet, QueryResult, StrategySummary
from app.retrieval.service import Strategy

DEFAULT_APP_URL = "http://127.0.0.1:8000"
DEFAULT_QUERIES_PATH = Path(__file__).parent / "queries.yaml"
DEFAULT_STRATEGIES: tuple[Strategy, ...] = (
    Strategy.DENSE_ONLY,
    Strategy.HYBRID,
    Strategy.HYBRID_RERANK,
)
TOP_K = 10  # fetch 10 so recall@5 and recall@10 both land in the same call


def load_queries(path: Path) -> list[GoldenQuery]:
    with path.open() as f:
        raw = yaml.safe_load(f)
    return GoldenSet.model_validate(raw).queries


async def run_one(
    client: httpx.AsyncClient,
    app_url: str,
    query: GoldenQuery,
    strategy: Strategy,
) -> QueryResult | None:
    """Runs one (query, strategy) pair through /search. Returns None on a
    persistent backend failure so the rest of the suite can continue — the
    caller notes it as a warning in the summary."""
    try:
        response = await client.get(
            f"{app_url}/search",
            params={"q": query.query, "strategy": strategy.value, "top_k": TOP_K},
            timeout=60.0,
        )
        response.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        print(f"  [{query.id} / {strategy.value}] backend error: {exc}")
        return None
    body = response.json()
    # Dedup by source_url preserving rank order: /search returns chunk-
    # level hits, but eval is document-level (recall = "did we surface
    # the article?"). Without dedup, DCG inflates because the same
    # relevant URL scores multiple times.
    retrieved_urls: list[str] = []
    seen: set[str] = set()
    for r in body["results"]:
        url = r["source_url"]
        if url not in seen:
            seen.add(url)
            retrieved_urls.append(url)
    relevant = query.expected_source_urls

    return QueryResult(
        query_id=query.id,
        query=query.query,
        strategy=strategy,
        recall_at_5=recall_at_k(retrieved_urls, relevant, k=5),
        recall_at_10=recall_at_k(retrieved_urls, relevant, k=10),
        mrr=reciprocal_rank(retrieved_urls, relevant),
        ndcg_at_5=ndcg_at_k(retrieved_urls, relevant, k=5),
        ndcg_at_10=ndcg_at_k(retrieved_urls, relevant, k=10),
        warnings=list(body.get("warnings") or []),
    )


def summarise(results: Sequence[QueryResult], strategy: Strategy) -> StrategySummary:
    per_strategy = [r for r in results if r.strategy == strategy]
    return StrategySummary(
        strategy=strategy,
        mean_recall_at_5=mean(r.recall_at_5 for r in per_strategy),
        mean_recall_at_10=mean(r.recall_at_10 for r in per_strategy),
        mean_mrr=mean(r.mrr for r in per_strategy),
        mean_ndcg_at_5=mean(r.ndcg_at_5 for r in per_strategy),
        mean_ndcg_at_10=mean(r.ndcg_at_10 for r in per_strategy),
        query_count=len(per_strategy),
    )


def print_table(summaries: Sequence[StrategySummary]) -> None:
    header = (
        f"{'STRATEGY':<20} {'R@5':<8} {'R@10':<8} {'MRR':<8} {'NDCG@5':<9} {'NDCG@10':<9} {'N'}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.strategy.value:<20} "
            f"{s.mean_recall_at_5:<8.3f} "
            f"{s.mean_recall_at_10:<8.3f} "
            f"{s.mean_mrr:<8.3f} "
            f"{s.mean_ndcg_at_5:<9.3f} "
            f"{s.mean_ndcg_at_10:<9.3f} "
            f"{s.query_count}"
        )
    print("=" * len(header))


def print_per_query(results: Sequence[QueryResult]) -> None:
    print()
    print("Per-query breakdown:")
    print()
    # group by query_id
    by_query: dict[str, list[QueryResult]] = {}
    for r in results:
        by_query.setdefault(r.query_id, []).append(r)
    for qid, rows in by_query.items():
        query_text = rows[0].query
        print(f"  {qid}: {query_text!r}")
        for r in rows:
            print(
                f"    {r.strategy.value:<16} R@5={r.recall_at_5:.2f} "
                f"MRR={r.mrr:.2f} NDCG@5={r.ndcg_at_5:.2f}"
            )


async def run(app_url: str, queries_path: Path, strategies: Sequence[Strategy]) -> int:
    queries = load_queries(queries_path)
    if not queries:
        print(f"No queries in {queries_path}", file=sys.stderr)
        return 1

    print(f"Running {len(queries)} queries x {len(strategies)} strategies against {app_url}")

    results: list[QueryResult] = []
    skipped: list[tuple[str, str]] = []
    # Throttle: OpenRouter's bge-m3 free tier rate-limits on sustained
    # request rate and occasionally returns HTTP 200 with `data=[]`
    # instead of a 429. Pausing between strategies gives the limiter
    # headroom; 18 queries * 3 strategies = 54 calls, the extra time is
    # seconds. Individual failures (after embed_texts's 6 retries) are
    # skipped, not fatal.
    async with httpx.AsyncClient() as client:
        for q in queries:
            for strategy in strategies:
                r = await run_one(client, app_url, q, strategy)
                if r is None:
                    skipped.append((q.id, strategy.value))
                else:
                    results.append(r)
                await asyncio.sleep(0.3)

    summaries = [summarise(results, s) for s in strategies]
    print()
    print_table(summaries)
    print_per_query(results)
    if skipped:
        print()
        print(f"Skipped {len(skipped)} (query, strategy) pairs due to backend errors:")
        for qid, strat in skipped:
            print(f"  - {qid} / {strat}")

    # Regression tripwire: WARN when a query's recall@5 is below the
    # declared floor for any strategy.
    regressions: list[str] = []
    for r in results:
        q = next(x for x in queries if x.id == r.query_id)
        if q.min_recall_at_5 is not None and r.recall_at_5 < q.min_recall_at_5:
            regressions.append(
                f"  {r.query_id} [{r.strategy.value}]: recall@5={r.recall_at_5:.2f} "
                f"< floor {q.min_recall_at_5:.2f}"
            )
    if regressions:
        print()
        print("REGRESSIONS (recall@5 below declared floor):")
        for line in regressions:
            print(line)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the retrieval evaluation suite.")
    parser.add_argument("--app-url", default=DEFAULT_APP_URL)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    args = parser.parse_args()
    return asyncio.run(run(args.app_url, args.queries, DEFAULT_STRATEGIES))


if __name__ == "__main__":
    sys.exit(main())
