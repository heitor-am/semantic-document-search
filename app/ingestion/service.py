from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Awaitable, Callable

import httpx
import structlog
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.ingestion.chunker import chunk_document
from app.ingestion.indexer import index_chunks
from app.ingestion.repository import JobRepository, job_repository
from app.ingestion.schemas import SourceDocument
from app.ingestion.sources.dev_to import fetch_dev_to
from app.ingestion.state import IngestJob as IngestJobFSM
from app.ingestion.state import JobState
from app.shared.ai.embeddings import embed_texts
from app.shared.qdrant.repository import VectorRepository

logger = structlog.get_logger()

# MVP supports one source. The dispatcher is a function pointer so tests
# can swap in a fake fetcher without patching module globals.
SourceFetcher = Callable[[str, httpx.AsyncClient], Awaitable[SourceDocument]]


def deterministic_job_id(source_url: str) -> str:
    """Hash the URL into a stable 16-hex-char job_id.

    Re-POSTing the same URL yields the same id — the router observes the
    existing job and short-circuits instead of creating a duplicate.
    """
    return hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]


async def _default_fetcher(url: str, client: httpx.AsyncClient) -> SourceDocument:
    return await fetch_dev_to(url, client=client)


async def run_ingestion(
    *,
    job_id: str,
    source_url: str,
    vector_repo: VectorRepository,
    httpx_client: httpx.AsyncClient,
    openai_client: AsyncOpenAI,
    session_maker: async_sessionmaker[AsyncSession],
    repo: JobRepository | None = None,
    fetcher: SourceFetcher = _default_fetcher,
) -> None:
    """End-to-end ingestion: fetch → parse → chunk → embed → index.

    Every FSM transition is persisted via the job repository so an operator
    can inspect where a crashed job got stuck. On any exception we transition
    to FAILED and store the error message on the job row.

    Runs from a FastAPI BackgroundTask, which fires after the HTTP response,
    so the request-scoped DB session is already gone — we open our own via
    `session_maker`.
    """
    repo = repo or job_repository
    fsm = IngestJobFSM(job_id, source_url)
    log = logger.bind(job_id=job_id, source_url=source_url)
    log.info("ingestion.start")

    async def _transition(trigger: str, db: AsyncSession) -> None:
        previous = fsm.current_state
        getattr(fsm, trigger)()
        await repo.record_transition(
            db,
            job_id=job_id,
            from_state=previous,
            to_state=fsm.current_state,
        )

    try:
        async with session_maker() as db:
            await _transition("start_fetch", db)
            doc = await fetcher(source_url, httpx_client)

            await _transition("start_parse", db)
            # The source fetcher already normalized the markdown into a
            # SourceDocument; parse is a dedicated FSM stage for auditability,
            # not because additional work is needed here.

            await _transition("start_chunk", db)
            chunks = chunk_document(doc)
            log.info("ingestion.chunked", chunk_count=len(chunks))

            await _transition("start_embed", db)
            embeddings = await embed_texts([c.content for c in chunks], client=openai_client)
            log.info("ingestion.embedded", vector_count=len(embeddings))

            await _transition("start_index", db)
            upserted = await index_chunks(chunks, embeddings, vector_repo=vector_repo)
            log.info("ingestion.indexed", upserted=upserted)

            await _transition("complete", db)
            log.info("ingestion.completed")
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        log.exception("ingestion.failed", error=error_msg)
        previous_state = fsm.current_state
        with contextlib.suppress(Exception):
            # Already in FAILED or COMPLETED would raise — still persist the error below.
            # `fail` is injected by the transitions Machine at runtime; mypy can't see it.
            fsm.fail()  # type: ignore[attr-defined]
        async with session_maker() as db:
            await repo.record_transition(
                db,
                job_id=job_id,
                from_state=previous_state,
                to_state=JobState.FAILED,
                error=error_msg,
            )


def current_collection_name() -> str:
    """Build the Qdrant collection name from current settings.

    Centralized so the lifespan (which calls ensure_collection) and tests use
    the exact same name the repository will query later.
    """
    settings = get_settings()
    from app.shared.qdrant.collections import collection_name_for

    return collection_name_for(
        settings.openrouter_embedding_model,
        settings.qdrant_collection_version,
    )
