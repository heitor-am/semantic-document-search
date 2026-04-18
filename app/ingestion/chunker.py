from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.ingestion.schemas import Chunk, SourceDocument

DEFAULT_PARENT_CHUNK_SIZE = 4096
DEFAULT_CHILD_CHUNK_SIZE = 1024
DEFAULT_CHILD_CHUNK_OVERLAP = 128

_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def _make_chunk_id(source_url: str, parent_index: int, child_index: int | None) -> str:
    """Deterministic UUID5 per (url, parent_index, child_index).

    child_index=None flags a parent chunk. Re-chunking the same document yields
    the same IDs, which makes upsert into Qdrant idempotent.
    """
    suffix = "-" if child_index is None else str(child_index)
    return str(uuid5(NAMESPACE_URL, f"{source_url}#p{parent_index}#c{suffix}"))


def _section_path(md_metadata: dict[str, str]) -> tuple[str, ...]:
    return tuple(md_metadata[h] for h in ("h1", "h2", "h3") if h in md_metadata)


def chunk_document(
    doc: SourceDocument,
    *,
    parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    child_chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a document into hierarchical parent/child chunks.

    Stage 1 — header split (h1/h2/h3) so each parent carries a coherent
    section with its ancestor path. Oversized sections are further split so
    no parent exceeds `parent_chunk_size`.

    Stage 2 — each parent is re-split into overlapping child chunks that the
    embedder consumes. Children inherit `section_path` and point back to
    their parent via `parent_chunk_id`.

    Sizes are in characters; the plan speaks in tokens (~1024 / ~256) which
    we approximate at ~4 chars/token. Callers can override the defaults.

    Raises:
        ValueError: if size kwargs are not positive, overlap is negative, or
            overlap is not strictly smaller than `child_chunk_size`.
    """
    if parent_chunk_size <= 0 or child_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if child_chunk_overlap < 0:
        raise ValueError("child_chunk_overlap must be non-negative")
    if child_chunk_overlap >= child_chunk_size:
        raise ValueError("child_chunk_overlap must be smaller than child_chunk_size")

    if not doc.body_markdown.strip():
        return []

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    # Defensive: splitter returns >=1 section for any non-empty input in practice.
    sections = header_splitter.split_text(doc.body_markdown)
    if not sections:  # pragma: no cover
        return []

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=0,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )

    chunks: list[Chunk] = []
    parent_index = 0
    for section in sections:
        path = _section_path(section.metadata)
        for parent_text in parent_splitter.split_text(section.page_content):
            parent_id = _make_chunk_id(doc.source_url, parent_index, None)
            chunks.append(
                _build_chunk(
                    doc,
                    chunk_id=parent_id,
                    parent_chunk_id=None,
                    content=parent_text,
                    section_path=path,
                    chunk_index=parent_index,
                )
            )
            for child_index, child_text in enumerate(child_splitter.split_text(parent_text)):
                chunks.append(
                    _build_chunk(
                        doc,
                        chunk_id=_make_chunk_id(doc.source_url, parent_index, child_index),
                        parent_chunk_id=parent_id,
                        content=child_text,
                        section_path=path,
                        chunk_index=child_index,
                    )
                )
            parent_index += 1

    return chunks


def _build_chunk(
    doc: SourceDocument,
    *,
    chunk_id: str,
    parent_chunk_id: str | None,
    content: str,
    section_path: tuple[str, ...],
    chunk_index: int,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        parent_chunk_id=parent_chunk_id,
        content=content,
        char_count=len(content),
        section_path=section_path,
        chunk_index=chunk_index,
        source_url=doc.source_url,
        source_type=doc.source_type,
        title=doc.title,
        author=doc.author,
        published_at=doc.published_at,
        tags=tuple(doc.tags),
    )
