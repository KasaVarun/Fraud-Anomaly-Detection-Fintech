"""Consistent API error responses."""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class AppError(Exception):
    """Expected application error safe to disclose to clients."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


_ERROR_DESCRIPTIONS = {
    401: "A valid bearer token is required.",
    403: "The authenticated principal lacks the required permission.",
    404: "Published evidence was not found.",
    422: "Request parameters are invalid.",
    500: "An unexpected error occurred.",
    503: "A required evidence dependency is unavailable.",
}


def documented_error_responses(*status_codes: int) -> dict[int | str, dict[str, Any]]:
    """Describe the API's sanitized error envelope in generated OpenAPI."""

    return {
        status_code: {
            "model": ErrorResponse,
            "description": _ERROR_DESCRIPTIONS[status_code],
        }
        for status_code in status_codes
    }


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        del exc
        body = ErrorResponse(
            error=ErrorBody(
                code="invalid_request",
                message="Request parameters are invalid.",
                request_id=_request_id(request),
            )
        )
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        body = ErrorResponse(
            error=ErrorBody(
                code=exc.code,
                message=exc.message,
                request_id=_request_id(request),
                details=exc.details,
            )
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_request_error", extra={"request_id": _request_id(request)})
        body = ErrorResponse(
            error=ErrorBody(
                code="internal_error",
                message="An unexpected error occurred.",
                request_id=_request_id(request),
            )
        )
        return JSONResponse(status_code=500, content=body.model_dump(mode="json"))
