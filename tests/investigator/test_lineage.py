"""Adversarial cross-store lineage and protected-field regression tests."""

from copy import deepcopy

import pytest

from fraud_detection.features.behavioral import template_digest
from fraud_detection.persistence.document_vector.embedding_models import embedding_content_digest
from fraud_detection.persistence.document_vector.models import content_digest
from fraud_detection.persistence.graph.models import graph_digest
from tests.investigator.fakes import FakeGraph, FakeRaw, FakeRelational, FakeVectors
from tests.investigator.test_api import _client, _evidence, _row


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_id", "wrong-batch"),
        ("source_row", 999),
        ("sender_id", "wrong-sender"),
        ("receiver_id", "wrong-receiver"),
        ("step", 999),
        ("transaction_type", "DEBIT"),
        ("amount", "999.00"),
        ("data_split", "validation"),
    ],
)
def test_valid_but_wrong_raw_lineage_is_not_returned(field: str, value: object) -> None:
    raw, _ = _evidence()
    raw[field] = value
    raw["content_digest"] = content_digest(raw)
    with _client(FakeRelational(_row()), raw=FakeRaw(raw)) as client:
        response = client.get("/v1/investigations/transactions/tx-1")
    section = response.json()["data"]["raw_evidence"]
    assert section["status"] == "version_mismatch"
    assert section["data"] is None
    assert str(value) not in str(section)


def test_wrong_raw_source_hash_and_malformed_digest_are_rejected() -> None:
    raw, _ = _evidence()
    wrong_hash = deepcopy(raw)
    wrong_hash["source_provenance"]["source_sha256"] = "b" * 64
    wrong_hash["content_digest"] = content_digest(wrong_hash)
    malformed = deepcopy(raw)
    malformed["content_digest"] = "0" * 64
    for payload, expected in [(wrong_hash, "version_mismatch"), (malformed, "unavailable")]:
        with _client(FakeRelational(_row()), raw=FakeRaw(payload)) as client:
            response = client.get("/v1/investigations/transactions/tx-1")
        section = response.json()["data"]["raw_evidence"]
        assert section["status"] == expected
        assert section["data"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transaction_id", "other-transaction"),
        ("raw_id", "other-raw"),
        ("raw_pk", "other-partition"),
        ("sender_id", "other-sender"),
        ("batch_id", "other-batch"),
        ("step", 99),
        ("data_split", "validation"),
        ("embedding_version", "other-embedding"),
        ("model_id", "other-model"),
        ("model_revision", "other-revision"),
    ],
)
def test_embedding_lineage_mismatch_is_rejected_before_vector_search(
    field: str, value: object
) -> None:
    _, embedding = _evidence()
    embedding[field] = value
    embedding["content_digest"] = embedding_content_digest(embedding)
    vectors = FakeVectors(embedding)
    with _client(FakeRelational(_row()), vectors=vectors) as client:
        response = client.get("/v1/investigations/transactions/tx-1/similar")
    assert response.json()["data"]["matches"]["status"] == "version_mismatch"
    assert vectors.calls == 1


def test_embedding_template_and_reference_eligibility_mismatch_are_rejected() -> None:
    _, embedding = _evidence()
    wrong_template = deepcopy(embedding)
    wrong_template["template_version"] = "other-template"
    wrong_template["template_digest"] = template_digest(
        wrong_template["template_text"],
        wrong_template["template_version"],
        wrong_template["amount_band_version"],
    )
    wrong_template["content_digest"] = embedding_content_digest(wrong_template)
    wrong_reference = deepcopy(embedding)
    wrong_reference["reference_eligible"] = True
    wrong_reference["content_digest"] = embedding_content_digest(wrong_reference)
    for payload in (wrong_template, wrong_reference):
        vectors = FakeVectors(payload)
        with _client(FakeRelational(_row()), vectors=vectors) as client:
            response = client.get("/v1/investigations/transactions/tx-1/similar")
        assert response.json()["data"]["matches"]["status"] in {
            "version_mismatch",
            "unavailable",
        }
        assert vectors.calls == 1


def _graph_values() -> list[dict[str, object]]:
    raw, _ = _evidence()
    from fraud_detection.persistence.document_vector.models import RawTransactionDocument
    from fraud_detection.persistence.graph.projection import build_graph_evidence

    document = RawTransactionDocument.model_validate(raw)
    return [edge.payload() for edge in build_graph_evidence(document, shard_count=64).edges]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transaction_id", "other-transaction"),
        ("batch_id", "other-batch"),
        ("step", 999),
        ("data_split", "validation"),
        ("transaction_type", "DEBIT"),
        ("amount", "999.00"),
        ("out_id", "other-endpoint"),
    ],
)
def test_valid_but_wrong_graph_edge_is_version_mismatch(field: str, value: object) -> None:
    values = _graph_values()
    target = 0 if field in {"amount", "out_id"} else 1
    values[target][field] = value
    values[target]["content_digest"] = graph_digest(values[target])
    graph = FakeGraph()
    graph.transaction_values = values
    with _client(FakeRelational(_row()), graph=graph) as client:
        response = client.get("/v1/investigations/transactions/tx-1")
    section = response.json()["data"]["graph_evidence"]
    assert section["status"] == "version_mismatch"
    assert section["data"] is None


def test_graph_missing_duplicate_or_orphan_edges_are_rejected() -> None:
    exact = _graph_values()
    cases = [exact[:2], [*exact, exact[0]], [*exact, {**exact[0], "id": "edge:orphan"}]]
    cases[2][-1]["content_digest"] = graph_digest(cases[2][-1])
    for values in cases:
        graph = FakeGraph()
        graph.transaction_values = values
        with _client(FakeRelational(_row()), graph=graph) as client:
            response = client.get("/v1/investigations/transactions/tx-1")
        assert response.json()["data"]["graph_evidence"]["status"] in {
            "version_mismatch",
            "unavailable",
        }


def test_protected_or_malformed_graph_properties_never_reach_response() -> None:
    protected = _graph_values()
    protected[0]["is_fraud"] = True
    malformed = _graph_values()
    malformed[0]["content_digest"] = "0" * 64
    for values in (protected, malformed):
        graph = FakeGraph()
        graph.transaction_values = values
        with _client(FakeRelational(_row()), graph=graph) as client:
            response = client.get("/v1/investigations/transactions/tx-1")
        section = response.json()["data"]["graph_evidence"]
        assert section["status"] == "unavailable"
        assert section["data"] is None
        assert "is_fraud" not in response.text


def test_exact_three_edge_graph_evidence_succeeds() -> None:
    graph = FakeGraph()
    graph.transaction_values = _graph_values()
    with _client(FakeRelational(_row()), graph=graph) as client:
        response = client.get("/v1/investigations/transactions/tx-1")
    section = response.json()["data"]["graph_evidence"]
    assert section["status"] == "available"
    assert len(section["data"]) == 3
    assert all(
        set(item)
        <= {
            "edge_id",
            "label",
            "kind",
            "out_id",
            "in_id",
            "transaction_id",
            "step",
            "amount",
            "synthetic",
            "source_version",
            "graph_schema_version",
            "projection_version",
        }
        for item in section["data"]
    )


@pytest.mark.parametrize(
    "value",
    [
        {
            "related_account_id": "C1",
            "relationship_type": "all",
            "transaction_count": 1,
            "shared_device_count": 0,
            "shared_ip_count": 0,
            "evidence_weight": 1.0,
        },
        {
            "related_account_id": "C2",
            "relationship_type": "all",
            "transaction_count": -1,
            "shared_device_count": 0,
            "shared_ip_count": 0,
            "evidence_weight": 0.0,
        },
        {
            "related_account_id": "C2",
            "relationship_type": "all",
            "transaction_count": 1,
            "shared_device_count": 1,
            "shared_ip_count": 1,
            "evidence_weight": 99.0,
        },
        {
            "related_account_id": "C2",
            "relationship_type": "all",
            "transaction_count": 1,
            "shared_device_count": 0,
            "shared_ip_count": 0,
            "evidence_weight": 1.0,
            "is_fraud": True,
        },
    ],
)
def test_invalid_related_account_results_are_unavailable_and_not_exposed(
    value: dict[str, object],
) -> None:
    graph = FakeGraph()
    graph.related_values = [value]
    with _client(FakeRelational(_row()), graph=graph) as client:
        response = client.get("/v1/investigations/accounts/C1/related")
    section = response.json()["data"]["relationships"]
    assert section["status"] == "unavailable"
    assert section["data"] is None
    assert "is_fraud" not in response.text


@pytest.mark.parametrize(
    "result",
    [
        {"transaction_id": "tx-1", "step": 1, "distance": 0.1},
        {"transaction_id": "tx-future", "step": 10, "distance": 0.1},
        {"transaction_id": "tx-distance", "step": 1, "distance": 2.1},
        {"transaction_id": "tx-protected", "step": 1, "distance": 0.1, "source_flag": True},
    ],
)
def test_invalid_similarity_results_make_section_unavailable(result: dict[str, object]) -> None:
    _, embedding = _evidence()
    result.update(
        {
            "embedding_version": "test-embedding-v1",
            "template_version": "behavior-template-v1",
            "model_revision": embedding["model_revision"],
            "source_version": "raw-paysim-v1",
        }
    )
    vectors = FakeVectors(embedding)
    vectors.results = [result]
    with _client(FakeRelational(_row()), vectors=vectors) as client:
        response = client.get("/v1/investigations/transactions/tx-1/similar")
    assert response.json()["data"]["matches"]["status"] == "unavailable"


def test_recursive_phase9_responses_exclude_protected_source_fields() -> None:
    raw, embedding = _evidence()
    graph = FakeGraph()
    graph.transaction_values = _graph_values()
    vectors = FakeVectors(embedding)
    vectors.results = [
        {
            "transaction_id": "tx-earlier",
            "step": 1,
            "distance": 0.2,
            "embedding_version": "test-embedding-v1",
            "template_version": "behavior-template-v1",
            "model_revision": embedding["model_revision"],
            "source_version": "raw-paysim-v1",
        }
    ]
    with _client(FakeRelational(_row()), raw=FakeRaw(raw), graph=graph, vectors=vectors) as client:
        responses = [
            client.get("/v1/investigations/transactions/tx-1").json(),
            client.get("/v1/investigations/transactions/tx-1/similar").json(),
            client.get("/v1/dashboard/summary").json(),
            client.get("/v1/dashboard/flags").json(),
        ]
    forbidden = {
        "is_fraud",
        "isFraud",
        "source_flag",
        "isFlaggedFraud",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {item for child in value.values() for item in keys(child)}
        if isinstance(value, list):
            return {item for child in value for item in keys(child)}
        return set()

    assert all(not (keys(response) & forbidden) for response in responses)
