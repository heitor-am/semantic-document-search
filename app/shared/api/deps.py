from typing import Annotated

import httpx
from fastapi import Depends, Request
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.core.exceptions import LLMUnavailableError, VectorStoreUnavailableError
from app.shared.db.database import get_db
from app.shared.qdrant.repository import VectorRepository


def get_httpx_client(request: Request) -> httpx.AsyncClient:
    """Shared httpx client from app.state (created in lifespan)."""
    return request.app.state.httpx_client  # type: ignore[no-any-return]


def get_openai_client(request: Request) -> AsyncOpenAI:
    client = request.app.state.openai_client
    if client is None:
        raise LLMUnavailableError("OPENROUTER_API_KEY is not configured")
    return client  # type: ignore[no-any-return]


def get_vector_repo(request: Request) -> VectorRepository:
    repo = request.app.state.vector_repo
    if repo is None:
        raise VectorStoreUnavailableError("QDRANT_URL is not configured")
    return repo  # type: ignore[no-any-return]


DbDep = Annotated[AsyncSession, Depends(get_db)]
HttpxDep = Annotated[httpx.AsyncClient, Depends(get_httpx_client)]
OpenAIDep = Annotated[AsyncOpenAI, Depends(get_openai_client)]
VectorRepoDep = Annotated[VectorRepository, Depends(get_vector_repo)]
