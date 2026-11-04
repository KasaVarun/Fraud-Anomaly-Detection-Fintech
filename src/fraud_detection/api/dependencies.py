"""FastAPI dependency providers."""

from fastapi import Request

from fraud_detection.api.container import AppContainer
from fraud_detection.investigator.service import InvestigatorService
from fraud_detection.reconciliation.service import ReadinessService


def get_container(request: Request) -> AppContainer:
    return request.app.state.container  # type: ignore[no-any-return]


def get_readiness_service(request: Request) -> ReadinessService:
    return get_container(request).readiness


def get_investigator_service(request: Request) -> InvestigatorService:
    return get_container(request).investigator
