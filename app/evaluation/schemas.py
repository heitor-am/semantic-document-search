"""Pydantic types for the golden-set YAML and the runner output."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.retrieval.service import Strategy


class GoldenQuery(BaseModel):
    """One entry in queries.yaml.

    `expected_source_urls` declares which documents *should* appear in the
    top results. Source URL is used rather than chunk id because the
    golden set is authored once and chunk ids are derived values — if the
    chunker changes, URL-based golden sets don't churn.
    """

    id: str
    query: str
    description: str = ""
    expected_source_urls: list[str] = Field(default_factory=list)
    # Optional floor: runner prints a WARN when this query's recall@5
    # falls below. Used as a regression tripwire after collection changes.
    min_recall_at_5: float | None = None


class GoldenSet(BaseModel):
    """Top-level shape of queries.yaml."""

    queries: list[GoldenQuery]


class QueryResult(BaseModel):
    query_id: str
    query: str
    strategy: Strategy
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    warnings: list[str] = Field(default_factory=list)


class StrategySummary(BaseModel):
    strategy: Strategy
    mean_recall_at_5: float
    mean_recall_at_10: float
    mean_mrr: float
    mean_ndcg_at_5: float
    mean_ndcg_at_10: float
    query_count: int
