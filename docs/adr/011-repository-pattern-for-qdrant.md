# 011 — Repository pattern for Qdrant

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** @heitor-am

## Context

Two pressures push us to *not* call `qdrant-client` directly from the indexer / retrieval stages:

1. **Tests.** Indexer / retrieval logic (chunk → point conversion, RRF prefetch shape, parent-child resolution) is rich enough to warrant unit tests. Hitting a real Qdrant from unit tests is slow and flaky; mocking `qdrant-client` directly couples every test to its (often-changing) call surface.
2. **Vendor swap.** The PRD lists "Milvus / Weaviate" as plausible alternatives. The longer the project goes with `qdrant-client` calls scattered across the codebase, the harder that swap becomes — even though we're not planning to do it, the shape of the abstraction tells the reader where the seam is.

The trap is over-abstracting: a "VectorStore" interface that tries to be neutral across Qdrant / Milvus / Weaviate ends up being the lowest common denominator (no metadata filters, no hybrid, no payloads), at which point we've thrown away the reason we picked Qdrant in the first place.

## Decision

A `VectorRepository` Protocol exposes only the operations the app actually uses, in terms of small types we own:

```python
class VectorRepository(Protocol):
    async def ensure_collection(self, ...) -> None: ...
    async def upsert(self, points: Iterable[VectorPoint]) -> None: ...
    async def query(self, request: QueryRequest) -> list[Candidate]: ...
    async def scroll(self, ...) -> AsyncIterator[VectorPoint]: ...
```

`VectorPoint`, `SparseValue`, `QueryRequest`, `Candidate` are **our types**, not Qdrant's. Conversion to/from `qdrant-client` types happens in **one place** (`QdrantVectorRepository`).

Tests use a `FakeVectorRepo` that holds upserts in a list and answers `query` from in-memory state. Real Qdrant is hit only by the `smoke` make target and the eval harness.

## Consequences

**Positive:**
- Indexer and retrieval tests run in milliseconds; no Qdrant container required.
- The Protocol surface is the documentation: "here's what this app needs from a vector store." A reader doesn't have to grep `qdrant-client` calls to find out.
- `qdrant-client` version bumps touch one file.
- The retrieval pipeline (`app/retrieval/`) imports nothing from `qdrant-client` directly.

**Negative:**
- One more layer of types to maintain.
- New Qdrant features (e.g., quantization config) require touching the Protocol *and* the implementation. Acceptable — that's the cost of the seam.

**Trade-offs accepted:**
- We don't try to make `VectorRepository` portable across vector stores. The Protocol's shape is "things a Qdrant-style hybrid store does"; a Pinecone implementation would either fit or not. That's deliberate — the abstraction serves testability first, vendor portability second.

## Alternatives considered

- **Direct `qdrant-client` use** — rejected for the test-speed reason above.
- **Generic `VectorStore` abstraction over multiple vendors** — rejected: lowest-common-denominator interfaces give up on the very features we picked Qdrant for.
- **`Mock(spec=AsyncQdrantClient)` everywhere** — rejected: the mock surface drifts every time `qdrant-client` updates.

## References

- `app/shared/qdrant/repository.py`
- `app/shared/qdrant/collections.py`
- `tests/ingestion/test_indexer.py` (the `FakeVectorRepo`)
