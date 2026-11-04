"""Raw-document construction and leakage contract tests."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from fraud_detection.features import validate_predictor_field_contract
from fraud_detection.persistence.document_vector.models import (
    FORBIDDEN_RAW_PREDICTIVE_FIELDS,
    RAW_DOCUMENT_SCHEMA_VERSION,
    RAW_PARTITION_KEY_PATH,
    RAW_PROJECTION_VERSION,
)
from fraud_detection.persistence.document_vector.projection import build_raw_document


def sample_row() -> dict[str, object]:
    return {
        "transaction_id": "paysim:sha:7",
        "nameOrig": "C1",
        "nameDest": "M2",
        "source_row": 7,
        "step": 3,
        "type": "PAYMENT",
        "amount": Decimal("10.25"),
        "data_split": "train",
        "isFraud": 1,
        "isFlaggedFraud": 0,
        "oldbalanceOrg": Decimal("100.00"),
        "newbalanceOrig": Decimal("89.75"),
        "oldbalanceDest": Decimal("0.00"),
        "newbalanceDest": Decimal("10.25"),
    }


def postgres_row() -> dict[str, object]:
    return {
        "transaction_id": "paysim:sha:7",
        "batch_id": "batch-1",
        "source_row": 7,
        "sender_id": "C1",
        "receiver_id": "M2",
        "step": 3,
        "transaction_type": "PAYMENT",
        "amount": Decimal("10.25"),
        "data_split": "train",
        "is_fraud": True,
        "source_flag": False,
    }


@dataclass(frozen=True)
class ManifestFixture:
    source_filename: str = "paysim.csv"
    source_sha256: str = "a" * 64
    sampling_seed: int = 73750
    enrichment_seed: int = 73751


def manifest_fixture() -> ManifestFixture:
    return ManifestFixture()


def test_document_is_deterministic_and_partitioned_by_sender() -> None:
    projected_at = datetime(2026, 10, 31, tzinfo=UTC)
    first = build_raw_document(
        sample_row(), postgres_row(), manifest_fixture(), "batch-1", projected_at
    )
    second = build_raw_document(
        sample_row(), postgres_row(), manifest_fixture(), "batch-1", projected_at
    )
    later = build_raw_document(
        sample_row(),
        postgres_row(),
        manifest_fixture(),
        "batch-1",
        projected_at + timedelta(seconds=1),
    )

    assert first == second
    assert first.id == "paysim:sha:7"
    assert first.sender_id == "C1"
    assert RAW_PARTITION_KEY_PATH == "/sender_id"
    assert first.document_schema_version == RAW_DOCUMENT_SCHEMA_VERSION
    assert first.projection_version == RAW_PROJECTION_VERSION
    assert first.content_digest == later.content_digest
    assert first.immutable_payload() == later.immutable_payload()


def test_source_row_mismatch_is_rejected() -> None:
    stored = postgres_row()
    stored["source_row"] = 8

    with pytest.raises(Exception, match="source_row"):
        build_raw_document(sample_row(), stored, manifest_fixture(), "batch-1", datetime.now(UTC))


def test_raw_forensic_and_truth_fields_are_forbidden_predictive_inputs() -> None:
    assert "source_truth.is_fraud" in FORBIDDEN_RAW_PREDICTIVE_FIELDS
    assert "forensic_balances.oldbalanceOrg" in FORBIDDEN_RAW_PREDICTIVE_FIELDS
    for field in ["id", "sender_id", "receiver_id", "forensic_balances", "source_truth"]:
        with pytest.raises(ValueError, match="forbidden predictive fields"):
            validate_predictor_field_contract({field})
