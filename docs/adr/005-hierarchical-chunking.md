# 005 — Hierarchical chunking (parent / child)

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** @heitor-am

## Context

Chunk size is a precision/context trade-off:

- **Small chunks** (~256 tokens) — high matching precision, embeddings stay focused, but a hit gives the LLM (or the user) very little surrounding context.
- **Large chunks** (~1024+ tokens) — lots of context, but embeddings dilute (a 1024-token chunk averages many topics into one vector), so retrieval misses the *exact* match.

Naïve fixed-size splitting also breaks markdown semantics — a `## Section` heading lands halfway through a chunk, the next chunk has no idea what section it's in.

## Decision

Two-level split, semantic-first then size-bounded:

1. **Parent split:** `langchain-text-splitters.MarkdownHeaderTextSplitter` on `##` and `###`. Each parent ≈ a section, ~1024 tokens typical; oversized parents are recursively split. `section_path` (the heading ancestry) is preserved in the payload.
2. **Child split:** for each parent, `RecursiveCharacterTextSplitter` produces children of ~256 tokens with 12.5% overlap. Children inherit the parent's section_path.

What we store in Qdrant:

| Chunk type | `parent_chunk_id` | Embedded? | Used for |
|---|---|---|---|
| Parent | `None` | Yes | Direct hits + context resolution |
| Child  | UUID of parent | Yes | Precise matching |

Retrieval matches against children for precision; the `parent_child` post-stage dedupes children sharing a parent (best score wins). Parent content expansion is left to the client — the payload carries `parent_chunk_id` so the client can fetch the parent on demand.

## Consequences

**Positive:**
- Precision (small embeddings) without sacrificing recoverable context (parent_id in payload).
- Markdown structure preserved via the header splitter — section paths survive into the payload and are useful for filtering.
- The `parent_child` dedup stage prevents the result list from being dominated by 5 children of the same section.

**Negative:**
- ~5× more points in Qdrant per document compared to flat chunking. Acceptable at this corpus size; storage is cheap.
- Two splitter passes per ingest — measurable but not a bottleneck (chunking is dwarfed by embedding cost).

**Trade-offs accepted:**
- Parent expansion is *client-side*, not server-side. The plan called this out: the API exposes `parent_chunk_id`; clients that want context fetch it themselves. Server-side expansion would require a second Qdrant trip and an opinion on "how much context is enough."

## Alternatives considered

- **Flat fixed-size chunks** — rejected: loses markdown structure and forces a hard size compromise.
- **Semantic chunking via embedding similarity** (`SemanticSplitter`) — interesting; rejected for the demo because it adds an embedding pass to the ingestion path and the gain over header-aware splitting is marginal on already-structured content (dev.to posts).
- **Sliding window with no parent concept** — equivalent to children-only; loses the dedup heuristic.

## References

- `app/ingestion/chunker.py`
- `app/retrieval/stages/parent_child.py`
- LangChain text splitters: https://python.langchain.com/docs/modules/data_connection/document_transformers/
