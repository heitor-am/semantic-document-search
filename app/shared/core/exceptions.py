from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.shared.schemas.problem import ProblemDetails


class AppError(Exception):
    status_code: int = 500
    code: str = "INTERNAL_ERROR"
    title: str = "Internal Error"

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class JobNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "JOB_NOT_FOUND"
    title = "Ingestion Job Not Found"


class InvalidTransitionError(AppError):
    status_code = status.HTTP_409_CONFLICT
    code = "INVALID_TRANSITION"
    title = "Invalid State Transition"


class LLMUnavailableError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "LLM_UNAVAILABLE"
    title = "LLM Service Unavailable"


class VectorStoreUnavailableError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "VECTOR_STORE_UNAVAILABLE"
    title = "Vector Store Unavailable"


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    problem = ProblemDetails(
        type=f"{request.base_url}errors/{exc.code.lower().replace('_', '-')}",
        title=exc.title,
        status=exc.status_code,
        detail=exc.detail,
        instance=str(request.url.path),
        code=exc.code,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=problem.model_dump(),
        media_type="application/problem+json",
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    problem = ProblemDetails(
        type=f"{request.base_url}errors/validation-error",
        title="Validation Error",
        status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()),
        instance=str(request.url.path),
        code="VALIDATION_ERROR",
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=problem.model_dump(),
        media_type="application/problem+json",
    )
