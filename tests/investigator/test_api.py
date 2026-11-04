"""Controlled Phase 9 endpoint and partial-evidence tests."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fraud_detection.api.app import create_app
from fraud_detection.api.container import build_container
from fraud_detection.config import Settings
from fraud_detection.features.behavioral import build_behavioral_templates
from fraud_detection.investigator.service import InvestigatorService
from fraud_detection.persistence.document_vector.embedding_generation import (
    build_embedding_document,
)
from fraud_detection.persistence.document_vector.embedding_models import (
    EMBEDDING_MODEL_REVISION,
    embedding_content_digest,
)
from fraud_detection.persistence.document_vector.models import (
    RawTransactionDocument,
    content_digest,
)
from fraud_detection.persistence.graph.projection import build_graph_evidence
from fraud_detection.reconciliation.service import ReadinessService
from tests.graph.helpers import raw_document
from tests.investigator.fakes import FakeGraph, FakeRaw, FakeRelational, FakeVectors
from tests.vector.fakes import FakeEncoder


def _row(*, flagged: bool = True) -> dict[str, Any]:
    return {
        "transaction_id": "tx-1",
        "batch_id": "published-batch",
        "source_row": 1,
        "sender_id": "C1",
        "receiver_id": "M1",
        "step": 10,
        "transaction_type": "PAYMENT",
        "amount": "10.00",
        "data_split": "test",
        "batch_source_sha256": "a" * 64,
        "batch_sample_seed": 73750,
        "batch_enrichment_seed": 73751,
        "batch_split_policy": "whole-step-60-20-20-v1",
        "sender_merchant_id": None,
        "receiver_merchant_id": "M1",
        "score_id": "score-1",
        "fraud_probability": "0.99",
        "vector_distance": 0.2,
        "input_version": "input-v1",
        "snapshot_id": "snapshot-1",
        "scored_at": datetime(2026, 10, 31, tzinfo=UTC),
        "model_version_id": "model:lightgbm_combined:1e9a3fc0a52dbac541bf4f92",
        "feature_set": "combined",
        "training_cutoff_step": 281,
        "threshold": "0.9496",
        "embedding_version": "test-embedding-v1",
        "artifact_uri": "trusted",
        "flag_id": "flag-1" if flagged else None,
        "flag_reason": "threshold decision" if flagged else None,
        "review_status": "open" if flagged else None,
        "snapshot_cutoff_step": 9,
        "snapshot_algorithm_version": "graph-v1",
    }


def _client(
    relational: FakeRelational,
    *,
    raw: FakeRaw | None = None,
    graph: FakeGraph | None = None,
    vectors: FakeVectors | None = None,
) -> TestClient:
    settings = Settings(_env_file=None, environment="test")
    container = build_container(settings)
    readiness = ReadinessService([relational])
    investigator = InvestigatorService(relational, readiness, raw=raw, graph=graph, vectors=vectors)
    container = replace(container, readiness=readiness, investigator=investigator)
    return TestClient(create_app(settings, container=container))


def _evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    raw = raw_document(
        transaction_id="tx-1",
        sender_id="C1",
        receiver_id="M1",
        transaction_type="PAYMENT",
        step=10,
    )
    raw_payload = raw.cosmos_payload()
    raw_payload["batch_id"] = "published-batch"
    raw_payload["data_split"] = "test"
    raw_payload["content_digest"] = content_digest(raw_payload)
    matching_raw = RawTransactionDocument.model_validate(raw_payload)
    template = build_behavioral_templates([matching_raw])[0]
    encoder = FakeEncoder()
    settings = Settings(_env_file=None, embedding_version="test-embedding-v1")
    embedding = build_embedding_document(
        template,
        encoder.encode([template.text], batch_size=1)[0],
        encoder,
        settings,
        datetime(2026, 10, 31, tzinfo=UTC),
    )
    embedding_payload = embedding.payload()
    embedding_payload["model_revision"] = EMBEDDING_MODEL_REVISION
    embedding_payload["content_digest"] = embedding_content_digest(embedding_payload)
    return raw_payload, embedding_payload


def test_complete_flagged_transaction_evidence_excludes_source_labels() -> None:
    raw_payload, embedding = _evidence()
    relational = FakeRelational(_row(flagged=True))
    graph = FakeGraph()
    matching_raw = RawTransactionDocument.model_validate(raw_payload)
    graph.transaction_values = [
        edge.payload() for edge in build_graph_evidence(matching_raw, shard_count=64).edges
    ]
    with _client(
        relational,
        raw=FakeRaw(raw_payload),
        graph=graph,
        vectors=FakeVectors(embedding),
    ) as client:
        response = client.get(
            "/v1/investigations/transactions/tx-1", headers={"x-request-id": "request-1"}
        )
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "request-1"
    payload = response.json()
    assert payload["partial_evidence"] is False
    assert payload["data"]["prediction"]["at_or_above_threshold"] is True
    assert payload["data"]["fraud_flag"]["proof_of_fraud"] is False
    assert "is_fraud" not in response.text
    assert "source_flag" not in response.text
    assert relational.query_count <= 4
    assert graph.calls == 1


@pytest.mark.parametrize("transaction_id", ["unknown", "unpublished"])
def test_unknown_and_unpublished_transactions_share_safe_404(transaction_id: str) -> None:
    with _client(FakeRelational(None)) as client:
        response = client.get(f"/v1/investigations/transactions/{transaction_id}")
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Published evidence was not found."


def test_partial_transaction_evidence_distinguishes_missing_and_unavailable() -> None:
    raw_payload, _ = _evidence()
    raw = FakeRaw(raw_payload)
    raw.fail = True
    graph = FakeGraph()
    with _client(FakeRelational(_row(flagged=False)), raw=raw, graph=graph) as client:
        response = client.get("/v1/investigations/transactions/tx-1")
    payload = response.json()
    assert response.status_code == 200
    assert payload["partial_evidence"] is True
    assert payload["data"]["raw_evidence"]["status"] == "unavailable"
    assert payload["data"]["graph_evidence"]["status"] == "not_found"
    assert payload["data"]["vector_evidence"]["status"] == "not_configured"
    assert payload["data"]["fraud_flag"] is None
    assert "secret raw endpoint" not in response.text

    with _client(FakeRelational(_row()), raw=FakeRaw(None)) as client:
        missing = client.get("/v1/investigations/transactions/tx-1")
    assert missing.json()["data"]["raw_evidence"]["status"] == "not_found"


def test_embedding_missing_and_version_mismatch_are_explicit() -> None:
    _, embedding = _evidence()
    mismatch = dict(embedding)
    mismatch["embedding_version"] = "wrong-version"
    from fraud_detection.persistence.document_vector.embedding_models import (
        embedding_content_digest,
    )

    mismatch["content_digest"] = embedding_content_digest(mismatch)
    for vectors, expected in [
        (FakeVectors(None), "not_found"),
        (FakeVectors(mismatch), "version_mismatch"),
    ]:
        with _client(FakeRelational(_row()), vectors=vectors) as client:
            response = client.get("/v1/investigations/transactions/tx-1")
        assert response.json()["data"]["vector_evidence"]["status"] == expected


def test_related_accounts_are_bounded_paginated_and_snapshot_correct() -> None:
    relational = FakeRelational(_row())
    graph = FakeGraph()
    graph.related_values = [
        {
            "related_account_id": f"C{index + 1}",
            "relationship_type": "all",
            "transaction_count": index,
            "shared_device_count": 1,
            "shared_ip_count": 0,
            "evidence_weight": float(index) + 0.5,
        }
        for index in range(1, 5)
    ]
    with _client(relational, graph=graph) as client:
        first = client.get(
            "/v1/investigations/accounts/C1/related",
            params={"limit": 2, "minimum_evidence_weight": 1},
        )
        cursor = first.json()["next_cursor"]
        second = client.get(
            "/v1/investigations/accounts/C1/related",
            params={"limit": 2, "minimum_evidence_weight": 1, "cursor": cursor},
        )
        malformed = client.get("/v1/investigations/accounts/C1/related", params={"cursor": "bad"})
        too_large = client.get("/v1/investigations/accounts/C1/related", params={"limit": 101})
    first_ids = [
        item["related_account_id"] for item in first.json()["data"]["relationships"]["data"]
    ]
    second_ids = [
        item["related_account_id"] for item in second.json()["data"]["relationships"]["data"]
    ]
    assert first_ids == ["C2", "C3"]
    assert second_ids == ["C4", "C5"]
    assert not set(first_ids) & set(second_ids)
    assert graph.last_cutoff == 10
    assert graph.calls == 2
    assert malformed.status_code == 422
    assert too_large.status_code == 422
    assert "not proof" in first.text


def test_related_graph_failure_is_partial_not_clean_negative() -> None:
    graph = FakeGraph()
    graph.fail = True
    with _client(FakeRelational(_row()), graph=graph) as client:
        response = client.get("/v1/investigations/accounts/C1/related")
    assert response.status_code == 200
    assert response.json()["partial_evidence"] is True
    assert response.json()["data"]["relationships"]["status"] == "unavailable"

    with _client(FakeRelational(None), graph=FakeGraph()) as client:
        unpublished = client.get("/v1/investigations/accounts/C1/related")
    assert unpublished.status_code == 404


def test_similar_behavior_filters_self_distance_and_versions_without_labels() -> None:
    _, embedding = _evidence()
    vectors = FakeVectors(embedding)
    vectors.results = [
        {
            "transaction_id": "tx-near",
            "step": 1,
            "distance": 0.2,
            "embedding_version": "test-embedding-v1",
            "template_version": "behavior-template-v1",
            "model_revision": EMBEDDING_MODEL_REVISION,
            "source_version": "raw-v1",
        },
        {
            "transaction_id": "tx-far",
            "step": 1,
            "distance": 0.8,
            "embedding_version": "test-embedding-v1",
            "template_version": "behavior-template-v1",
            "model_revision": EMBEDDING_MODEL_REVISION,
        },
    ]
    with _client(FakeRelational(_row()), vectors=vectors) as client:
        response = client.get(
            "/v1/investigations/transactions/tx-1/similar",
            params={"limit": 3, "maximum_cosine_distance": 0.5},
        )
        too_large = client.get(
            "/v1/investigations/transactions/tx-1/similar", params={"limit": 101}
        )
    matches = response.json()["data"]["matches"]["data"]
    assert [item["transaction_id"] for item in matches] == ["tx-near"]
    assert "fraud_probability" not in response.text
    assert "is_fraud" not in response.text
    assert vectors.calls == 2
    assert too_large.status_code == 422


def test_invalid_vector_and_vector_timeout_are_explicitly_unavailable() -> None:
    _, embedding = _evidence()
    invalid = dict(embedding)
    invalid["vector"] = [float("nan")] * 384
    vectors = FakeVectors(invalid)
    with _client(FakeRelational(_row()), vectors=vectors) as client:
        invalid_response = client.get("/v1/investigations/transactions/tx-1/similar")
    assert invalid_response.json()["data"]["matches"]["status"] == "unavailable"

    vectors = FakeVectors(embedding)
    vectors.fail = True
    with _client(FakeRelational(_row()), vectors=vectors) as client:
        timeout_response = client.get("/v1/investigations/transactions/tx-1/similar")
    assert timeout_response.json()["data"]["matches"]["status"] == "unavailable"
    assert "secret vector endpoint" not in timeout_response.text


def test_dashboard_summary_flags_cursor_filters_and_query_budgets() -> None:
    relational = FakeRelational(_row())
    now = datetime(2026, 10, 31, tzinfo=UTC)
    relational.flag_rows = [
        {
            **_row(),
            "transaction_id": f"tx-{index}",
            "score_id": f"score-{index}",
            "step": index,
            "scored_at": now,
        }
        for index in range(1, 4)
    ]
    with _client(relational) as client:
        summary = client.get(
            "/v1/dashboard/summary",
            params={"start_step": 1, "end_step": 10, "transaction_type": "PAYMENT"},
        )
        first = client.get("/v1/dashboard/flags", params={"limit": 2})
        second = client.get(
            "/v1/dashboard/flags",
            params={"limit": 2, "cursor": first.json()["next_cursor"]},
        )
        malformed = client.get("/v1/dashboard/flags", params={"cursor": "bad"})
        incompatible = client.get(
            "/v1/dashboard/flags",
            params={
                "limit": 2,
                "review_status": "reviewed",
                "cursor": first.json()["next_cursor"],
            },
        )
    assert summary.json()["data"]["transaction_count"] == 3
    assert summary.json()["data"]["flag_rate"] == pytest.approx(1 / 3)
    assert "is_fraud" not in summary.text
    assert len(first.json()["data"]["items"]) == 2
    assert len(second.json()["data"]["items"]) == 1
    assert malformed.status_code == incompatible.status_code == 422
    assert relational.query_count <= 9


def test_postgresql_unavailable_returns_sanitized_503() -> None:
    relational = FakeRelational(_row())
    relational.raise_unavailable = True
    with _client(relational) as client:
        response = client.get("/v1/dashboard/summary")
    assert response.status_code == 503
    assert "secret database endpoint" not in response.text


def test_openapi_describes_phase9_routes_and_request_id_is_sanitized() -> None:
    with _client(FakeRelational(None)) as client:
        response = client.get("/openapi.json")
        request = client.get(
            "/v1/dashboard/summary", headers={"x-request-id": "invalid request id with spaces"}
        )
    paths = response.json()["paths"]
    assert "/v1/investigations/transactions/{transaction_id}" in paths
    assert "/v1/investigations/accounts/{account_id}/related" in paths
    assert "/v1/investigations/transactions/{transaction_id}/similar" in paths
    assert "/v1/dashboard/summary" in paths
    assert "/v1/dashboard/flags" in paths
    assert "/v1/meta/freshness" in paths
    assert request.headers["x-request-id"] != "invalid request id with spaces"


def test_cors_is_restrictive_and_allowlisted() -> None:
    settings = Settings(
        _env_file=None,
        cors_allowed_origins=("https://investigator.example",),
    )
    relational = FakeRelational(None)
    container = build_container(settings)
    readiness = ReadinessService([relational])
    container = replace(
        container,
        readiness=readiness,
        investigator=InvestigatorService(relational, readiness),
    )
    with TestClient(create_app(settings, container=container)) as client:
        allowed = client.options(
            "/v1/dashboard/summary",
            headers={
                "origin": "https://investigator.example",
                "access-control-request-method": "GET",
            },
        )
        denied = client.options(
            "/v1/dashboard/summary",
            headers={
                "origin": "https://not-allowed.example",
                "access-control-request-method": "GET",
            },
        )
    assert allowed.headers["access-control-allow-origin"] == "https://investigator.example"
    assert "access-control-allow-origin" not in denied.headers


def test_investigator_repository_source_does_not_reference_protected_labels() -> None:
    source = (
        Path(__file__).parents[2] / "src/fraud_detection/investigator/repository.py"
    ).read_text(encoding="utf-8")
    forbidden = ("transaction_label", "is_fraud", "source_flag")
    assert all(value not in source for value in forbidden)
