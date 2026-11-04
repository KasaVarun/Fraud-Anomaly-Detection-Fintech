"""Published-only read-only investigator routes."""

from collections.abc import Callable
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Path, Query

from fraud_detection.api.dependencies import get_investigator_service
from fraud_detection.api.errors import AppError, documented_error_responses
from fraud_detection.investigator.contracts import EvidenceEnvelope
from fraud_detection.investigator.cursor import InvalidCursorError
from fraud_detection.investigator.service import (
    InvestigatorService,
    PublishedEvidenceNotFound,
    RelationalEvidenceUnavailable,
)

router = APIRouter(prefix="/v1/investigations", tags=["investigations"])
Identifier = Annotated[str, Path(min_length=1, max_length=256)]


def _safe_call(operation: Callable[[], EvidenceEnvelope]) -> EvidenceEnvelope:
    try:
        return operation()
    except PublishedEvidenceNotFound as exc:
        raise AppError(
            "not_found",
            "Published evidence was not found.",
            status_code=404,
        ) from exc
    except RelationalEvidenceUnavailable as exc:
        raise AppError(
            "service_unavailable",
            "Published relational evidence is unavailable.",
            status_code=503,
        ) from exc
    except (InvalidCursorError, ValueError) as exc:
        raise AppError(
            "invalid_request", "Request parameters are invalid.", status_code=422
        ) from exc


@router.get(
    "/transactions/{transaction_id}",
    response_model=EvidenceEnvelope,
    summary="Get unified published transaction evidence",
    description=(
        "Returns published relational evidence and explicit optional raw, graph, and vector states."
    ),
    operation_id="getTransactionEvidence",
    responses=documented_error_responses(401, 403, 404, 422, 500, 503),
)
def transaction_evidence(
    transaction_id: Identifier,
    service: Annotated[InvestigatorService, Depends(get_investigator_service)],
) -> EvidenceEnvelope:
    return _safe_call(lambda: service.transaction(transaction_id))


@router.get(
    "/accounts/{account_id}/related",
    response_model=EvidenceEnvelope,
    summary="Get bounded related-account evidence",
    description=(
        "Returns published-snapshot relationships with bounded parameterized graph traversal."
    ),
    operation_id="getRelatedAccountEvidence",
    responses=documented_error_responses(401, 403, 404, 422, 500, 503),
)
def related_accounts(
    account_id: Identifier,
    service: Annotated[InvestigatorService, Depends(get_investigator_service)],
    relationship_type: Annotated[
        Literal["all", "money_flow", "shared_device", "shared_ip"], Query()
    ] = "all",
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    minimum_evidence_weight: Annotated[float, Query(ge=0, le=1_000_000)] = 0,
    snapshot_id: Annotated[str | None, Query(min_length=1, max_length=256)] = None,
    cursor: Annotated[str | None, Query(max_length=1024)] = None,
) -> EvidenceEnvelope:
    return _safe_call(
        lambda: service.related_accounts(
            account_id,
            relationship_type=relationship_type,
            limit=limit,
            minimum_weight=minimum_evidence_weight,
            snapshot_id=snapshot_id,
            cursor=cursor,
        )
    )


@router.get(
    "/transactions/{transaction_id}/similar",
    response_model=EvidenceEnvelope,
    summary="Get bounded similar-behavior evidence",
    description="Returns compatible earlier-reference cosine distances, never fraud labels.",
    operation_id="getSimilarBehaviorEvidence",
    responses=documented_error_responses(401, 403, 404, 422, 500, 503),
)
def similar_behavior(
    transaction_id: Identifier,
    service: Annotated[InvestigatorService, Depends(get_investigator_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    maximum_cosine_distance: Annotated[float | None, Query(ge=0, le=2)] = None,
) -> EvidenceEnvelope:
    return _safe_call(
        lambda: service.similar(
            transaction_id,
            limit=limit,
            maximum_distance=maximum_cosine_distance,
        )
    )
