import time

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from app import __version__
from app.config import get_settings
from app.shared.api.deps import DbDep

router = APIRouter(tags=["health"])

_STARTED_AT = time.time()


class HealthResponse(BaseModel):
    status: str
    version: str
    commit: str
    uptime_seconds: float
    environment: str
    db: str


@router.get("/health", response_model=HealthResponse)
async def health(db: DbDep) -> HealthResponse:
    settings = get_settings()

    db_status = "ok"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        version=__version__,
        commit=settings.git_sha,
        uptime_seconds=round(time.time() - _STARTED_AT, 2),
        environment=settings.environment,
        db=db_status,
    )
