from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from scalar_fastapi import get_scalar_api_reference

from app import __version__
from app.shared.api.routers import health
from app.shared.core.exceptions import AppError, app_error_handler, validation_exception_handler
from app.shared.core.logging import RequestIdMiddleware, configure_logging

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    configure_logging()
    yield


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
