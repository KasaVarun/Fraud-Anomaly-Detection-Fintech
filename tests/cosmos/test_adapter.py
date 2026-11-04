"""Controlled SDK tests for the production Cosmos raw adapter."""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from azure.cosmos.exceptions import CosmosHttpResponseError

from fraud_detection.config import Settings
from fraud_detection.persistence.document_vector.adapter import CosmosRawDocumentAdapter
from fraud_detection.persistence.document_vector.errors import (
    CosmosAdapterError,
    CosmosConfigurationError,
    CosmosDocumentConflictError,
)
from fraud_detection.persistence.document_vector.models import content_digest
from fraud_detection.ports.health import DependencyState
from tests.cosmos.fakes import FakeClient, FakeContainer, FakeDatabase


def cosmos_settings(*, cosmos_max_retries: int = 2) -> Settings:
    return Settings(
        _env_file=None,
        cosmos_nosql_endpoint="https://secret-account.documents.azure.com",
        cosmos_nosql_key="super-secret-key",
        cosmos_nosql_database="fraud-db",
        cosmos_raw_container="transactions_raw",
        cosmos_max_retries=cosmos_max_retries,
    )


def raw_payload(identifier: str = "tx-1", sender: str = "C1") -> dict[str, object]:
    payload: dict[str, object] = {
        "id": identifier,
        "sender_id": sender,
        "receiver_id": "M1",
        "batch_id": "batch-1",
        "source_row": 1,
        "step": 1,
        "data_split": "train",
        "transaction_type": "PAYMENT",
        "amount": "10.00",
        "forensic_balances": {
            "oldbalanceOrg": "20.00",
            "newbalanceOrig": "10.00",
            "oldbalanceDest": "0.00",
            "newbalanceDest": "10.00",
            "predictive_use": "forbidden",
        },
        "source_truth": {
            "is_fraud": False,
            "source_flag": False,
            "provenance": "PaySim:ealaxi/paysim1",
            "predictive_use": "forbidden",
        },
        "source_provenance": {
            "source_filename": "paysim.csv",
            "source_sha256": "a" * 64,
            "sampling_seed": 73750,
            "enrichment_seed": 73751,
        },
        "forbidden_predictive_fields": [
            "id",
            "sender_id",
            "receiver_id",
            "forensic_balances.oldbalanceOrg",
            "forensic_balances.newbalanceOrig",
            "forensic_balances.oldbalanceDest",
            "forensic_balances.newbalanceDest",
            "source_truth.is_fraud",
            "source_truth.source_flag",
        ],
        "document_schema_version": "1.0.0",
        "projection_version": "raw-paysim-v1",
        "projected_at": datetime(2026, 10, 31, tzinfo=UTC).isoformat(),
    }
    payload["content_digest"] = content_digest(payload)
    return payload


def adapter_fixture(
    *, cosmos_max_retries: int = 2
) -> tuple[CosmosRawDocumentAdapter, FakeDatabase, FakeContainer]:
    container = FakeContainer()
    database = FakeDatabase(container)
    adapter = CosmosRawDocumentAdapter(
        cosmos_settings(cosmos_max_retries=cosmos_max_retries),
        client=FakeClient(database),
    )
    return adapter, database, container


def test_container_validation_and_readiness_states() -> None:
    adapter, database, container = adapter_fixture(cosmos_max_retries=0)
    adapter.validate_container()
    assert __import__("asyncio").run(adapter.readiness()).state is DependencyState.HEALTHY

    container.properties["partitionKey"] = {"paths": ["/wrong"]}
    with pytest.raises(CosmosConfigurationError, match="/sender_id"):
        adapter.validate_container()
    assert __import__("asyncio").run(adapter.readiness()).state is DependencyState.UNAVAILABLE

    database.failure = CosmosHttpResponseError(  # type: ignore[no-untyped-call]
        status_code=503, message="unreachable"
    )
    assert __import__("asyncio").run(adapter.readiness()).state is DependencyState.UNAVAILABLE


def test_point_reads_idempotent_creates_and_conflicts() -> None:
    adapter, _, container = adapter_fixture()
    payload = raw_payload()

    assert adapter.read_document("tx-1", "C1") is None
    assert adapter.upsert_document(payload) == "created"
    assert adapter.upsert_document(payload) == "existing"
    assert adapter.read_document("tx-1", "C1") is not None

    conflicting = raw_payload()
    conflicting["amount"] = "99.00"
    conflicting["content_digest"] = content_digest(conflicting)
    with pytest.raises(CosmosDocumentConflictError, match="conflicting immutable content"):
        adapter.upsert_document(conflicting)
    assert len(container.documents) == 1


def test_pagination_enumerates_all_documents() -> None:
    adapter, _, _ = adapter_fixture()
    for index in range(5):
        adapter.upsert_document(raw_payload(f"tx-{index}", f"C{index}"))

    documents = list(adapter.iter_batch_documents("batch-1", page_size=2))

    assert {document["id"] for document in documents} == {f"tx-{index}" for index in range(5)}


def test_cross_partition_id_lookup_is_chunked_parameterized_and_paginated() -> None:
    adapter, _, container = adapter_fixture()
    for index in range(7):
        adapter.upsert_document(raw_payload(f"tx-{index}", f"unexpected-partition-{index}"))

    documents = list(
        adapter.iter_documents_by_ids(
            [f"tx-{index}" for index in range(7)] + ["absent-id"],
            chunk_size=3,
            page_size=2,
        )
    )

    assert {document["id"] for document in documents} == {f"tx-{index}" for index in range(7)}
    assert container.query_calls >= 5
    for query, parameters in container.queries:
        assert query == "SELECT c.id FROM c WHERE ARRAY_CONTAINS(@transaction_ids, c.id)"
        assert "tx-0" not in query
        assert parameters[0]["name"] == "@transaction_ids"
        assert len(parameters[0]["value"]) <= 3


def test_transient_failures_retry_and_errors_are_secret_safe() -> None:
    adapter, _, container = adapter_fixture()
    payload = raw_payload()
    container.documents[("tx-1", "C1")] = payload
    container.read_failures.extend(
        [
            CosmosHttpResponseError(  # type: ignore[no-untyped-call]
                status_code=429, message="retry"
            ),
            CosmosHttpResponseError(  # type: ignore[no-untyped-call]
                status_code=503, message="retry"
            ),
        ]
    )
    with patch("fraud_detection.persistence.document_vector.adapter.time.sleep") as sleep:
        assert adapter.read_document("tx-1", "C1") is not None
        assert sleep.call_count == 2

    failing, database, _ = adapter_fixture(cosmos_max_retries=0)
    database.failure = CosmosHttpResponseError(  # type: ignore[no-untyped-call]
        status_code=401,
        message="super-secret-key https://secret-account.documents.azure.com",
    )
    with pytest.raises(CosmosAdapterError) as captured:
        failing.validate_container()
    message = str(captured.value)
    assert "super-secret-key" not in message
    assert "secret-account" not in message


def test_incomplete_configuration_is_rejected_and_secrets_are_masked() -> None:
    settings = Settings(
        _env_file=None,
        cosmos_nosql_endpoint="https://account.documents.azure.com",
        cosmos_nosql_key="key-secret",
    )
    assert settings.document_vector_configured is False
    assert "key-secret" not in repr(settings)
    with pytest.raises(CosmosConfigurationError, match="settings are incomplete"):
        CosmosRawDocumentAdapter(settings, client=object())
