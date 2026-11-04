"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from fraud_detection.api.container import AppContainer, build_container
from fraud_detection.api.errors import install_error_handlers
from fraud_detection.api.middleware import request_context_middleware
from fraud_detection.api.routes.dashboard import router as dashboard_router
from fraud_detection.api.routes.health import router as health_router
from fraud_detection.api.routes.investigations import router as investigation_router
from fraud_detection.api.routes.meta import router as meta_router
from fraud_detection.config import Settings, get_settings
from fraud_detection.logging import configure_logging
from fraud_detection.security.auth import TokenValidator, V1AuthenticationMiddleware


def create_app(
    settings: Settings | None = None,
    *,
    container: AppContainer | None = None,
    auth_validator: TokenValidator | None = None,
) -> FastAPI:
    """Build an application with replaceable dependencies for tests and later adapters."""

    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        description="Read-only fraud investigation API foundation.",
    )
    app.state.container = container or build_container(resolved_settings)
    app.add_middleware(
        V1AuthenticationMiddleware,
        settings=resolved_settings,
        validator=auth_validator,
    )
    if resolved_settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_allowed_origins),
            allow_credentials=False,
            allow_methods=["GET"],
            allow_headers=["Accept", "Content-Type", "X-Request-ID"],
        )
    app.middleware("http")(request_context_middleware)
    install_error_handlers(app)
    app.include_router(health_router)
    app.include_router(investigation_router)
    app.include_router(dashboard_router)
    app.include_router(meta_router)

    def deployment_openapi() -> dict[str, object]:
        if app.openapi_schema is not None:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        components = schema.setdefault("components", {})
        assert isinstance(components, dict)
        schemes = components.setdefault("securitySchemes", {})
        assert isinstance(schemes, dict)
        schemes["entraOpenId"] = {
            "type": "openIdConnect",
            "openIdConnectUrl": (
                "https://login.microsoftonline.com/{tenant-id}/v2.0/"
                ".well-known/openid-configuration"
            ),
            "description": "Gate B replaces the tenant placeholder and grants Fraud.Investigate.",
        }
        paths = schema.get("paths", {})
        assert isinstance(paths, dict)
        for path, operations in paths.items():
            if str(path).startswith("/v1") and isinstance(operations, dict):
                for operation in operations.values():
                    if isinstance(operation, dict):
                        operation["security"] = [{"entraOpenId": []}]
        app.openapi_schema = schema
        return schema

    app.openapi = deployment_openapi  # type: ignore[method-assign]
    return app


app = create_app()
