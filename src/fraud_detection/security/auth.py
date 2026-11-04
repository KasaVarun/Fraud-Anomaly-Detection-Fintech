"""Microsoft Entra ID bearer-token validation for deployed ``/v1`` routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import jwt
from fastapi import Request
from fastapi.responses import JSONResponse
from jwt import PyJWKClient
from jwt.exceptions import PyJWTError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from fraud_detection.api.errors import ErrorBody, ErrorResponse
from fraud_detection.config import Settings


class TokenValidator(Protocol):
    def validate(self, token: str) -> dict[str, Any]: ...


class AuthenticationFailure(Exception):
    """Sanitized authentication/authorization failure."""

    def __init__(self, *, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class EntraJwtValidator:
    """Validate an Entra v2 access token against configured tenant and application lineage."""

    def __init__(
        self,
        settings: Settings,
        *,
        signing_key_resolver: Callable[[str], Any] | None = None,
    ) -> None:
        if settings.auth_mode != "entra":
            raise ValueError("Entra validator requires auth_mode=entra")
        assert settings.entra_jwks_url is not None
        self._settings = settings
        self._jwks_client = PyJWKClient(
            settings.entra_jwks_url,
            cache_keys=True,
            lifespan=settings.auth_jwks_cache_seconds,
            timeout=settings.auth_request_timeout_seconds,
        )
        self._signing_key_resolver = signing_key_resolver

    def _signing_key(self, token: str) -> Any:
        if self._signing_key_resolver is not None:
            return self._signing_key_resolver(token)
        return self._jwks_client.get_signing_key_from_jwt(token).key

    def validate(self, token: str) -> dict[str, Any]:
        settings = self._settings
        assert settings.entra_audience is not None
        assert settings.entra_issuer is not None
        assert settings.entra_tenant_id is not None
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                self._signing_key(token),
                algorithms=["RS256"],
                audience=settings.entra_audience,
                issuer=settings.entra_issuer,
                options={"require": ["exp", "iat", "iss", "aud", "tid"]},
            )
        except (PyJWTError, ValueError, TypeError) as exc:
            raise AuthenticationFailure(
                status_code=401,
                code="invalid_token",
                message="The bearer token is invalid or expired.",
            ) from exc
        if claims.get("tid") != settings.entra_tenant_id:
            raise AuthenticationFailure(
                status_code=401,
                code="invalid_tenant",
                message="The bearer token tenant is not authorized.",
            )
        scopes = set(str(claims.get("scp", "")).split())
        roles = claims.get("roles", [])
        if isinstance(roles, list):
            scopes.update(str(role) for role in roles)
        if settings.entra_required_scope not in scopes:
            raise AuthenticationFailure(
                status_code=403,
                code="insufficient_scope",
                message="The bearer token lacks the required investigator permission.",
            )
        return claims


class V1AuthenticationMiddleware(BaseHTTPMiddleware):
    """Keep health/OpenAPI anonymous while enforcing authentication on every ``/v1`` route."""

    def __init__(
        self,
        app: Any,
        *,
        settings: Settings,
        validator: TokenValidator | None = None,
    ) -> None:
        super().__init__(app)
        self._settings = settings
        self._validator = validator or (
            EntraJwtValidator(settings) if settings.auth_mode == "entra" else None
        )

    @staticmethod
    def _error(request: Request, failure: AuthenticationFailure) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        body = ErrorResponse(
            error=ErrorBody(
                code=failure.code,
                message=failure.message,
                request_id=request_id,
            )
        )
        return JSONResponse(
            status_code=failure.status_code,
            content=body.model_dump(mode="json"),
            headers={"WWW-Authenticate": "Bearer"},
        )

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not request.url.path.startswith("/v1"):
            return await call_next(request)
        if self._settings.auth_mode == "local_bypass":
            return await call_next(request)
        authorization = request.headers.get("authorization", "")
        scheme, separator, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not separator or not token or len(token) > 16_384:
            return self._error(
                request,
                AuthenticationFailure(
                    status_code=401,
                    code="authentication_required",
                    message="A valid bearer token is required.",
                ),
            )
        assert self._validator is not None
        try:
            claims = self._validator.validate(token)
        except AuthenticationFailure as failure:
            return self._error(request, failure)
        request.state.authenticated_principal = {
            "oid": claims.get("oid"),
            "tid": claims.get("tid"),
        }
        return await call_next(request)
