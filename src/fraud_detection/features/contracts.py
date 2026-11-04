"""Phase 3 leakage boundary for future predictive feature assembly."""

from typing import Final

SAFE_INITIAL_PREDICTOR_FIELDS: Final = frozenset({"step", "transaction_type", "amount"})
FORBIDDEN_PREDICTOR_FIELDS: Final = frozenset(
    {
        "isFraud",
        "isFlaggedFraud",
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "future_events",
        "future_graph_edges",
        "id",
        "sender_id",
        "receiver_id",
        "forensic_balances",
        "source_truth",
    }
)


def validate_predictor_field_contract(fields: set[str]) -> None:
    forbidden = fields & FORBIDDEN_PREDICTOR_FIELDS
    if forbidden:
        raise ValueError(f"forbidden predictive fields: {', '.join(sorted(forbidden))}")
    unsupported = fields - SAFE_INITIAL_PREDICTOR_FIELDS
    if unsupported:
        raise ValueError(f"unsupported predictive fields: {', '.join(sorted(unsupported))}")
