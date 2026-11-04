"""Controlled production Gremlin client tests."""

from concurrent.futures import TimeoutError as FutureTimeoutError
from unittest.mock import patch

import pytest

from fraud_detection.config import Settings
from fraud_detection.persistence.graph.adapter import CosmosGremlinAdapter
from fraud_detection.persistence.graph.errors import (
    GremlinAdapterError,
    GremlinConfigurationError,
    GremlinEvidenceConflictError,
)
from fraud_detection.persistence.graph.projection import build_graph_evidence
from fraud_detection.ports.health import DependencyState
from tests.graph.fakes import ScriptedGremlinClient
from tests.graph.helpers import raw_document


def graph_settings(*, retries: int = 1) -> Settings:
    return Settings(
        _env_file=None,
        gremlin_endpoint="wss://secret.gremlin.cosmos.azure.com:443/",
        gremlin_username="/dbs/fraud/colls/evidence",
        gremlin_database="fraud",
        gremlin_graph="evidence",
        gremlin_key="secret-gremlin-key",
        gremlin_max_retries=retries,
    )


def test_readiness_retry_and_secret_safe_errors() -> None:
    client = ScriptedGremlinClient([[0]])
    adapter = CosmosGremlinAdapter(graph_settings(), client=client)
    assert __import__("asyncio").run(adapter.readiness()).state is DependencyState.HEALTHY

    retry_client = ScriptedGremlinClient([FutureTimeoutError(), [0]])
    retry_adapter = CosmosGremlinAdapter(graph_settings(), client=retry_client)
    with patch("fraud_detection.persistence.graph.adapter.time.sleep") as sleep:
        retry_adapter.validate_graph()
        assert sleep.call_count == 1

    failing = CosmosGremlinAdapter(
        graph_settings(retries=0), client=ScriptedGremlinClient([FutureTimeoutError()])
    )
    with pytest.raises(GremlinAdapterError) as captured:
        failing.validate_graph()
    assert "secret-gremlin-key" not in str(captured.value)
    assert "secret.gremlin" not in str(captured.value)


def test_parameterized_idempotent_vertex_and_edge_creation() -> None:
    evidence = build_graph_evidence(raw_document(), shard_count=64)
    vertex = evidence.vertices[0]
    edge = evidence.edges[0]
    client = ScriptedGremlinClient(
        [
            [],
            [],
            [
                {
                    "id": vertex.id,
                    "label": vertex.label,
                    "properties": {"content_digest": [vertex.content_digest]},
                }
            ],
            [],
            [],
        ]
    )
    adapter = CosmosGremlinAdapter(graph_settings(), client=client)

    assert adapter.upsert_vertex(vertex.payload()) == "created"
    assert adapter.upsert_vertex(vertex.payload()) == "existing"
    assert adapter.upsert_edge(edge.payload()) == "created"
    for query, bindings in client.submissions:
        assert vertex.id not in query
        assert edge.id not in query
        assert isinstance(bindings, dict)


def test_immutable_vertex_conflict_is_rejected() -> None:
    vertex = build_graph_evidence(raw_document(), shard_count=64).vertices[0]
    client = ScriptedGremlinClient(
        [[{"id": vertex.id, "label": vertex.label, "properties": {"content_digest": ["other"]}}]]
    )
    adapter = CosmosGremlinAdapter(graph_settings(), client=client)

    with pytest.raises(GremlinEvidenceConflictError, match="conflicting immutable evidence"):
        adapter.upsert_vertex(vertex.payload())


def test_bounded_pagination_and_transaction_id_chunks() -> None:
    client = ScriptedGremlinClient(
        [
            [{"id": "e1"}, {"id": "e2"}],
            [{"id": "e3"}],
            [{"id": "v1", "transaction_id": "tx1"}],
            [],
            [{"id": "e1", "transaction_id": "tx1"}],
            [],
            [],
            [],
        ]
    )
    adapter = CosmosGremlinAdapter(graph_settings(), client=client)

    assert len(list(adapter.iter_transaction_edges("batch", page_size=2))) == 3
    found = list(
        adapter.iter_elements_by_transaction_ids(["tx1", "tx2", "tx3"], chunk_size=2, page_size=2)
    )
    assert {item["transaction_id"] for item in found} == {"tx1"}
    for query, bindings in client.submissions:
        assert "tx1" not in query
        if "transaction_ids" in bindings:
            assert len(bindings["transaction_ids"]) <= 2


def test_incomplete_configuration_is_not_healthy() -> None:
    settings = Settings(
        _env_file=None,
        gremlin_endpoint="wss://example.gremlin.cosmos.azure.com:443/",
        gremlin_key="masked-key",
    )
    assert settings.graph_configured is False
    assert "masked-key" not in repr(settings)
    with pytest.raises(GremlinConfigurationError, match="settings are incomplete"):
        CosmosGremlinAdapter(settings, client=object())


def test_investigator_traversals_are_parameterized_bounded_and_weighted() -> None:
    client = ScriptedGremlinClient(
        [
            [
                {
                    "related_account_id": "C2",
                    "evidence_type": "money_flow",
                    "artifact_id": "edge-1",
                },
                {
                    "related_account_id": "C2",
                    "evidence_type": "shared_device",
                    "artifact_id": "device-1",
                },
                {
                    "related_account_id": "C2",
                    "evidence_type": "shared_ip",
                    "artifact_id": "ip-1",
                },
            ],
            [{"id": "edge-1", "label": "PAYS", "out_id": "C1", "in_id": "C2"}],
        ]
    )
    adapter = CosmosGremlinAdapter(graph_settings(), client=client)

    related = adapter.related_accounts(
        "C1",
        relationship_type="all",
        cutoff_step=10,
        minimum_weight=0,
        offset=0,
        limit=5,
    )
    transaction = adapter.transaction_evidence("tx-1", limit=3)

    assert related == [
        {
            "related_account_id": "C2",
            "relationship_type": "all",
            "transaction_count": 1,
            "shared_device_count": 1,
            "shared_ip_count": 1,
            "evidence_weight": 1.75,
        }
    ]
    assert transaction[0]["id"] == "edge-1"
    for query, bindings in client.submissions:
        assert "C1" not in query
        assert "tx-1" not in query
        assert isinstance(bindings, dict)
    assert client.submissions[0][1]["scan_limit"] <= 5_000
