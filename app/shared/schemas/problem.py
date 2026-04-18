from pydantic import BaseModel


class ProblemDetails(BaseModel):
    """RFC 7807 Problem Details for HTTP APIs."""

    type: str
    title: str
    status: int
    detail: str
    instance: str
    code: str
