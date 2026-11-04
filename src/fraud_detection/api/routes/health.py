"""Liveness and readiness routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, ConfigDict

from fraud_detection.api.dependencies import get_readiness_service
from fraud_detection.ports.health import DependencyHealth
from fraud_detection.reconciliation.service import ReadinessService

router = APIRouter(prefix="/health", tags=["health"])


class LivenessResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    ready: bool
    dependencies: list[DependencyHealth]


@router.get("/live", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """Report that the API process can serve requests."""

    return LivenessResponse(status="alive")


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
)
async def readiness(
    response: Response,
    service: Annotated[ReadinessService, Depends(get_readiness_service)],
) -> ReadinessResponse:
    """Report sanitized availability of configured application dependencies."""

    result = await service.check()
    if not result.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(**result.model_dump())
