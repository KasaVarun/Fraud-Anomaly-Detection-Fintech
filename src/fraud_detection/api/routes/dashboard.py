"""Published-only bounded dashboard-support routes."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from fraud_detection.api.dependencies import get_investigator_service
from fraud_detection.api.errors import documented_error_responses
from fraud_detection.api.routes.investigations import _safe_call
from fraud_detection.investigator.contracts import EvidenceEnvelope
from fraud_detection.investigator.service import InvestigatorService

router = APIRouter(prefix="/v1/dashboard", tags=["dashboard"])


@router.get(
    "/summary",
    response_model=EvidenceEnvelope,
    summary="Get published transaction and score aggregates",
    description=(
        "Returns bounded published-only aggregates without source-label performance metrics."
    ),
    operation_id="getPublishedDashboardSummary",
    responses=documented_error_responses(401, 403, 422, 500, 503),
)
def dashboard_summary(
    service: Annotated[InvestigatorService, Depends(get_investigator_service)],
    start_step: Annotated[int | None, Query(ge=0, le=1_000_000)] = None,
    end_step: Annotated[int | None, Query(ge=0, le=1_000_000)] = None,
    transaction_type: Annotated[
        Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"] | None,
        Query(),
    ] = None,
    flagged: bool | None = None,
) -> EvidenceEnvelope:
    if start_step is not None and end_step is not None and start_step > end_step:
        from fraud_detection.api.errors import AppError

        raise AppError("invalid_request", "Request parameters are invalid.", status_code=422)
    return _safe_call(
        lambda: service.dashboard_summary(
            {
                "start_step": start_step,
                "end_step": end_step,
                "transaction_type": transaction_type,
                "flagged": flagged,
            }
        )
    )


@router.get(
    "/flags",
    response_model=EvidenceEnvelope,
    summary="List published fraud flags",
    description="Returns deterministic cursor pagination over locked-model published flags.",
    operation_id="listPublishedFraudFlags",
    responses=documented_error_responses(401, 403, 422, 500, 503),
)
def fraud_flags(
    service: Annotated[InvestigatorService, Depends(get_investigator_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    review_status: Annotated[Literal["open", "reviewed"] | None, Query()] = None,
    minimum_score: Annotated[float | None, Query(ge=0, le=1)] = None,
    transaction_type: Annotated[
        Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"] | None,
        Query(),
    ] = None,
    cursor: Annotated[str | None, Query(max_length=1024)] = None,
) -> EvidenceEnvelope:
    return _safe_call(
        lambda: service.flags(
            {
                "review_status": review_status,
                "minimum_score": minimum_score,
                "transaction_type": transaction_type,
            },
            limit=limit,
            cursor=cursor,
        )
    )
