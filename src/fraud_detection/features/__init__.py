"""Leakage-safe feature engineering boundary."""

from fraud_detection.features.behavioral import (
    BehavioralTemplate,
    build_behavioral_templates,
    template_digest,
)
from fraud_detection.features.contracts import (
    FORBIDDEN_PREDICTOR_FIELDS,
    SAFE_INITIAL_PREDICTOR_FIELDS,
    validate_predictor_field_contract,
)

__all__ = [
    "FORBIDDEN_PREDICTOR_FIELDS",
    "SAFE_INITIAL_PREDICTOR_FIELDS",
    "BehavioralTemplate",
    "build_behavioral_templates",
    "template_digest",
    "validate_predictor_field_contract",
]
