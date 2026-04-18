from datetime import UTC, datetime
from typing import Any

from app.ingestion.chunker import chunk_document
from app.ingestion.schemas import SourceDocument


def make_doc(body: str, **overrides: Any) -> SourceDocument:
    defaults: dict[str, Any] = {
        "source_url": "https://dev.to/user/a-post",
        "source_type": "dev.to",
        "title": "A Post",
        "body_markdown": body,
        "author": "user",
        "published_at": datetime(2026, 1, 1, tzinfo=UTC),
        "tags": ["rag", "python"],
    }
    defaults.update(overrides)
    return SourceDocument(**defaults)


class TestChunkDocument:
    def test_empty_body_produces_no_chunks(self) -> None:
        assert chunk_document(make_doc("")) == []

    def test_whitespace_only_body_produces_no_chunks(self) -> None:
        assert chunk_document(make_doc("   \n\n   ")) == []

    def test_returns_parent_and_children(self) -> None:
        body = "# Title\n\nA short paragraph that fits in a single child chunk."
        chunks = chunk_document(
            make_doc(body),
            parent_chunk_size=500,
            child_chunk_size=200,
            child_chunk_overlap=20,
        )

        parents = [c for c in chunks if c.parent_chunk_id is None]
        children = [c for c in chunks if c.parent_chunk_id is not None]

        assert len(parents) == 1
        assert len(children) >= 1
        assert all(c.parent_chunk_id == parents[0].chunk_id for c in children)

    def test_section_path_captures_heading_hierarchy(self) -> None:
        body = (
            "# Top\n\nLead paragraph.\n\n## Middle\n\nSub paragraph.\n\n### Deep\n\nDeep paragraph."
        )
        chunks = chunk_document(
            make_doc(body),
            parent_chunk_size=500,
            child_chunk_size=200,
            child_chunk_overlap=20,
        )

        paths = {tuple(c.section_path) for c in chunks if c.parent_chunk_id is None}
        assert ("Top",) in paths
        assert ("Top", "Middle") in paths
        assert ("Top", "Middle", "Deep") in paths

    def test_child_inherits_section_path_from_its_parent(self) -> None:
        body = "# A\n\ncontent of A\n\n## B\n\ncontent of B"
        chunks = chunk_document(
            make_doc(body),
            parent_chunk_size=500,
            child_chunk_size=200,
            child_chunk_overlap=20,
        )
        parents = {c.chunk_id: c for c in chunks if c.parent_chunk_id is None}
        for child in (c for c in chunks if c.parent_chunk_id is not None):
            assert child.section_path == parents[child.parent_chunk_id].section_path

    def test_doc_without_headers_produces_one_parent(self) -> None:
        chunks = chunk_document(
            make_doc("Just a plain paragraph. No headers anywhere."),
            parent_chunk_size=500,
            child_chunk_size=200,
            child_chunk_overlap=20,
        )
        parents = [c for c in chunks if c.parent_chunk_id is None]
        assert len(parents) == 1
        assert parents[0].section_path == []

    def test_chunks_carry_document_metadata(self) -> None:
        doc = make_doc("# H\n\nbody")
        for chunk in chunk_document(doc):
            assert chunk.source_url == doc.source_url
            assert chunk.source_type == doc.source_type
            assert chunk.title == doc.title
            assert chunk.author == doc.author
            assert chunk.published_at == doc.published_at
            assert chunk.tags == list(doc.tags)

    def test_char_count_matches_content_length(self) -> None:
        for chunk in chunk_document(make_doc("# H\n\nsome content here")):
            assert chunk.char_count == len(chunk.content)

    def test_chunk_ids_are_deterministic_across_runs(self) -> None:
        doc = make_doc("# H\n\nbody content here")
        ids_first = [c.chunk_id for c in chunk_document(doc)]
        ids_second = [c.chunk_id for c in chunk_document(doc)]
        assert ids_first == ids_second

    def test_chunk_ids_differ_for_different_urls(self) -> None:
        body = "# H\n\nbody"
        ids_a = {
            c.chunk_id for c in chunk_document(make_doc(body, source_url="https://dev.to/a/p1"))
        }
        ids_b = {
            c.chunk_id for c in chunk_document(make_doc(body, source_url="https://dev.to/b/p2"))
        }
        assert ids_a.isdisjoint(ids_b)

    def test_long_parent_gets_multiple_children(self) -> None:
        body = "# Big\n\n" + ("word " * 120)
        chunks = chunk_document(
            make_doc(body),
            parent_chunk_size=2000,
            child_chunk_size=80,
            child_chunk_overlap=10,
        )
        children = [c for c in chunks if c.parent_chunk_id is not None]
        assert len(children) > 1
        # Indices per parent are sequential from 0
        assert [c.chunk_index for c in children] == list(range(len(children)))

    def test_child_chunks_respect_size_budget(self) -> None:
        body = "# Big\n\n" + ("word " * 300)
        chunks = chunk_document(
            make_doc(body),
            parent_chunk_size=5000,
            child_chunk_size=100,
            child_chunk_overlap=10,
        )
        for child in (c for c in chunks if c.parent_chunk_id is not None):
            # Splitter may slightly exceed the budget on unsplittable tokens,
            # but for space-separated prose it stays well under.
            assert child.char_count <= 150

    def test_oversized_section_is_split_into_multiple_parents(self) -> None:
        body = "# Giant\n\n" + ("sentence. " * 250)
        chunks = chunk_document(
            make_doc(body),
            parent_chunk_size=500,
            child_chunk_size=100,
            child_chunk_overlap=10,
        )
        parents = [c for c in chunks if c.parent_chunk_id is None]
        assert len(parents) > 1
        for parent in parents:
            assert parent.char_count <= 500
            assert parent.section_path == ["Giant"]

    def test_parent_indexes_are_sequential(self) -> None:
        body = "# A\n\nfirst\n\n## B\n\nsecond\n\n## C\n\nthird"
        chunks = chunk_document(
            make_doc(body),
            parent_chunk_size=500,
            child_chunk_size=200,
            child_chunk_overlap=20,
        )
        parents = [c for c in chunks if c.parent_chunk_id is None]
        assert [p.chunk_index for p in parents] == list(range(len(parents)))
