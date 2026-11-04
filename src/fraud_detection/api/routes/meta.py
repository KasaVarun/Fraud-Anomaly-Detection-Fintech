"""Sanitized evidence freshness route."""

from typing import Annotated

from fastapi import APIRouter, Depends

from fraud_detection.api.dependencies import get_investigator_service
from fraud_detection.api.errors import AppError, documented_error_responses
from fraud_detection.investigator.contracts import EvidenceEnvelope
from fraud_detection.investigator.service import (
    InvestigatorService,
    RelationalEvidenceUnavailable,
)

router = APIRouter(prefix="/v1/meta", tags=["metadata"])


@router.get(
    "/freshness",
    response_model=EvidenceEnvelope,
    summary="Get sanitized evidence freshness",
    description="Reports published cutoff, pending counts, active lineage, and dependency states.",
    operation_id="getEvidenceFreshness",
    responses=documented_error_responses(401, 403, 500, 503),
)
async def freshness(
    service: Annotated[InvestigatorService, Depends(get_investigator_service)],
) -> EvidenceEnvelope:
    try:
        return await service.freshness()
    except RelationalEvidenceUnavailable as exc:
        raise AppError(
            "service_unavailable",
            "Published relational evidence is unavailable.",
            status_code=503,
        ) from exc
