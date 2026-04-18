"""Parent-child post-processing: dedup children sharing a parent.

Hierarchical chunking (Etapa 5) produces fine-grained children for
matching precision. Search often returns several children of the same
parent with similar scores — same topic, slightly different slices.
This stage keeps the single best child per parent, preserving score
order, so the caller gets a deduplicated top-k.

Parent content expansion (replacing / augmenting child content with
the full parent section) is deferred: the plan leaves "parent chunks
for context" as a client-side concern, and the payload already carries
`parent_chunk_id` + `section_path` so clients that want context can
fetch the parent on demand.
"""

from __future__ import annotations

from app.retrieval.context import Candidate, Context
from app.retrieval.pipeline import Stage


class ParentChildStage(Stage):
    name = "parent_child"
    optional = False  # cheap; no external calls

    async def run(self, ctx: Context) -> Context:
        if not ctx.results:
            return ctx

        best_per_parent: dict[str, Candidate] = {}
        orphans: list[Candidate] = []  # no parent_chunk_id (e.g. itself a parent)

        for cand in ctx.results:
            parent_id = cand.parent_chunk_id
            if parent_id is None:
                orphans.append(cand)
                continue
            existing = best_per_parent.get(parent_id)
            if existing is None or cand.score > existing.score:
                best_per_parent[parent_id] = cand

        deduped = list(best_per_parent.values()) + orphans
        deduped.sort(key=lambda c: c.score, reverse=True)
        ctx.results = deduped[: ctx.top_k]
        ctx.state["parent_child"] = ctx.results
        return ctx
