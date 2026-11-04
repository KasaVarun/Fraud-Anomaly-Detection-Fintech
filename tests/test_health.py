"""Health endpoint contract tests."""

from dataclasses import replace

from fastapi.testclient import TestClient

from fraud_detection.api.app import create_app
from fraud_detection.api.container import build_container
from fraud_detection.config import Settings
from fraud_detection.ports.health import DependencyHealth, DependencyState
from fraud_detection.ports.unavailable import UnavailableAdapter
from fraud_detection.reconciliation.service import ReadinessService


class HealthyProbe:
    def __init__(self, name: str) -> None:
        self.name = name

    async def readiness(self) -> DependencyHealth:
        return DependencyHealth(
            name=self.name,
            state=DependencyState.HEALTHY,
            required=True,
            detail="health check passed",
        )


class UnavailableProbe:
    async def readiness(self) -> DependencyHealth:
        return DependencyHealth(
            name="postgresql",
            state=DependencyState.UNAVAILABLE,
            required=True,
            detail="connection refused",
        )


def test_liveness_is_independent_of_external_stores(client: TestClient) -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}
    assert response.headers["x-request-id"]


def test_default_readiness_reports_unconfigured_dependencies(client: TestClient) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["ready"] is True
    assert {dependency["state"] for dependency in payload["dependencies"]} == {"not_configured"}
    assert {dependency["name"] for dependency in payload["dependencies"]} == {
        "postgresql",
        "cosmos_nosql_raw",
        "cosmos_nosql_embeddings",
        "cosmos_gremlin",
        "model_artifacts",
        "batch_reconciliation",
    }


def test_readiness_returns_healthy_injected_skeleton(settings: Settings) -> None:
    container = build_container(settings)
    probes = [HealthyProbe("postgresql"), HealthyProbe("cosmos_nosql_raw")]
    container = replace(container, readiness=ReadinessService(probes))

    with TestClient(create_app(settings, container=container)) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert [item["state"] for item in response.json()["dependencies"]] == [
        "healthy",
        "healthy",
    ]


def test_readiness_returns_503_for_required_unavailable_dependency(settings: Settings) -> None:
    container = build_container(settings)
    container = replace(container, readiness=ReadinessService([UnavailableProbe()]))

    with TestClient(create_app(settings, container=container)) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "ready": False,
        "dependencies": [
            {
                "name": "postgresql",
                "state": "unavailable",
                "required": True,
                "detail": "connection refused",
            }
        ],
    }


def test_configured_store_is_not_reported_connected() -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        postgres_dsn="postgresql+psycopg://local:local@localhost/db",
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    postgres = next(
        item for item in response.json()["dependencies"] if item["name"] == "postgresql"
    )
    assert postgres["state"] == "unavailable"
    assert postgres["required"] is True
    assert "local:local" not in response.text


def test_openapi_documents_health_routes(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/health/live" in response.json()["paths"]
    assert "/health/ready" in response.json()["paths"]


def test_publication_reconciliation_remains_a_phase_eight_dependency(settings: Settings) -> None:
    container = build_container(settings)

    assert isinstance(container.reconciliation, UnavailableAdapter)
    assert container.reconciliation.phase == "Phase 8"
