"""Leakage boundary tests without implementing feature engineering."""

import pytest

from fraud_detection.features import (
    SAFE_INITIAL_PREDICTOR_FIELDS,
    validate_predictor_field_contract,
)


def test_safe_predictive_contract_excludes_labels_accounts_balances_and_future_data() -> None:
    validate_predictor_field_contract(set(SAFE_INITIAL_PREDICTOR_FIELDS))

    for forbidden in [
        "isFraud",
        "isFlaggedFraud",
        "nameOrig",
        "oldbalanceOrg",
        "future_events",
        "future_graph_edges",
    ]:
        with pytest.raises(ValueError, match="forbidden predictive fields"):
            validate_predictor_field_contract({forbidden})
