from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from scalar_fastapi import get_scalar_api_reference

from app import __version__
from app.config import get_settings
from app.ingestion import router as ingestion_router
from app.ingestion.service import current_collection_name
from app.shared.ai.client import get_openrouter_client
from app.shared.api.routers import health
from app.shared.core.exceptions import AppError, app_error_handler, validation_exception_handler
from app.shared.core.logging import RequestIdMiddleware, configure_logging
from app.shared.qdrant.client import get_qdrant_client
from app.shared.qdrant.collections import ensure_collection, vector_size_for
from app.shared.qdrant.repository import QdrantRepository

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    configure_logging()
    settings = get_settings()

    # Shared httpx client for source fetchers (dev.to, etc.)
    app.state.httpx_client = httpx.AsyncClient(timeout=30.0)

    # Qdrant is optional in local dev. When unset, ingestion endpoints return
    # 503 via get_vector_repo; the rest of the app (health, landing) still
    # boots normally.
    app.state.qdrant_client = None
    app.state.vector_repo = None
    if settings.qdrant_url:
        qdrant = get_qdrant_client()
        collection = current_collection_name()
        vector_size = vector_size_for(settings.openrouter_embedding_model)
        await ensure_collection(qdrant, collection, vector_size=vector_size)
        app.state.qdrant_client = qdrant
        app.state.vector_repo = QdrantRepository(qdrant, collection=collection)

    # OpenRouter client — also optional locally.
    app.state.openai_client = get_openrouter_client() if settings.openrouter_api_key else None

    try:
        yield
    finally:
        await app.state.httpx_client.aclose()
        if app.state.qdrant_client is not None:
            await app.state.qdrant_client.close()
        if app.state.openai_client is not None:
            await app.state.openai_client.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Semantic Document Search",
        description=(
            "Production-grade RAG search: package-by-feature, FSM-driven ingestion, "
            "functional retrieval pipeline with hybrid (BM25 + dense) + rerank + parent-child."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )

    app.add_middleware(RequestIdMiddleware)

    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]

    app.include_router(health.router)
    app.include_router(ingestion_router.router)

    @app.get("/", include_in_schema=False)
    async def landing() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/docs", include_in_schema=False)
    async def scalar_docs():  # type: ignore[no-untyped-def]
        return get_scalar_api_reference(
            openapi_url=app.openapi_url or "/openapi.json",
            title=f"{app.title} — Reference",
        )

    return app


app = create_app()
