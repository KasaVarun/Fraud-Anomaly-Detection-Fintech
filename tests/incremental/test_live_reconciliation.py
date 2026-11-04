"""Exact controlled live-store reconciliation tests."""

from datetime import UTC, datetime

import pytest

from fraud_detection.config import Settings
from fraud_detection.features.behavioral import build_behavioral_templates
from fraud_detection.incremental.contracts import TestChunk as ChunkContract
from fraud_detection.incremental.reconciliation import (
    ExpectedLiveEvidence,
    LiveStores,
    verify_live_stores,
)
from fraud_detection.incremental.workflow import project_expected_live_evidence
from fraud_detection.persistence.document_vector.embedding_generation import (
    build_embedding_document,
)
from fraud_detection.persistence.document_vector.models import (
    RawTransactionDocument,
    content_digest,
)
from fraud_detection.persistence.graph.models import GraphEdge, graph_digest
from fraud_detection.persistence.graph.projection import build_graph_evidence
from tests.cosmos.fakes import FakeRawStore
from tests.graph.fakes import FakeGraphStore
from tests.graph.helpers import raw_document
from tests.vector.fakes import FakeEmbeddingStore, FakeEncoder


def _fixture() -> tuple[
    Settings, ChunkContract, ExpectedLiveEvidence, FakeRawStore, FakeEmbeddingStore, FakeGraphStore
]:
    settings = Settings(
        _env_file=None,
        cosmos_nosql_endpoint="https://example.documents.azure.com",
        cosmos_nosql_key="secret",
        cosmos_nosql_database="db",
        gremlin_endpoint="wss://example.gremlin.cosmos.azure.com",
        gremlin_username="/dbs/db/colls/graph",
        gremlin_database="db",
        gremlin_graph="graph",
        gremlin_key="secret",
        gremlin_partition_shards=8,
        embedding_version="test-embedding-v1",
    )
    payload = raw_document(transaction_id="tx-live", source_row=7).cosmos_payload()
    payload["batch_id"] = "batch-live"
    payload["data_split"] = "test"
    payload["content_digest"] = content_digest(payload)
    raw = RawTransactionDocument.model_validate(payload)
    template = build_behavioral_templates([raw])[0]
    encoder = FakeEncoder()
    embedding = build_embedding_document(
        template,
        encoder.encode([template.text], batch_size=1)[0],
        encoder,
        settings,
        datetime(2026, 10, 31, tzinfo=UTC),
    )
    graph = build_graph_evidence(raw, shard_count=8)
    expected = ExpectedLiveEvidence(
        raw_documents=(raw,),
        embeddings=(embedding,),
        vertices=graph.vertices,
        edges=graph.edges,
    )
    raw_store = FakeRawStore()
    embedding_store = FakeEmbeddingStore()
    graph_store = FakeGraphStore()
    raw_store.upsert_document(raw.cosmos_payload())
    embedding_store.upsert_embedding(embedding.payload())
    for vertex in graph.vertices:
        graph_store.upsert_vertex(vertex.payload())
    for edge in graph.edges:
        graph_store.upsert_edge(edge.payload())
    chunk = ChunkContract(
        index=0,
        start_step=355,
        end_step=378,
        row_count=1,
        chunk_id="chunk-live",
        batch_id="batch-live",
        row_digest="a" * 64,
    )
    return settings, chunk, expected, raw_store, embedding_store, graph_store


def test_complete_controlled_live_content_reconciles() -> None:
    settings, chunk, expected, raw, embeddings, graph = _fixture()
    verify_live_stores(chunk, expected, LiveStores(raw, embeddings, graph), settings)


@pytest.mark.parametrize(
    "mutation",
    ["raw-content", "duplicate-embedding", "orphan-edge", "missing-vertex"],
)
def test_altered_duplicate_or_orphaned_live_content_is_rejected(mutation: str) -> None:
    settings, chunk, expected, raw, embeddings, graph = _fixture()
    if mutation == "raw-content":
        altered = dict(raw.documents[0])
        altered["amount"] = "999.00"
        altered["content_digest"] = content_digest(altered)
        raw.documents[0] = RawTransactionDocument.model_validate(altered).cosmos_payload()
    elif mutation == "duplicate-embedding":
        embeddings.documents.append(dict(embeddings.documents[0]))
    elif mutation == "orphan-edge":
        orphan = dict(graph.edges[0])
        orphan["id"] = "edge:orphan"
        orphan["content_digest"] = graph_digest(orphan)
        graph.edges.append(GraphEdge.model_validate(orphan).payload())
    else:
        graph.vertices.pop()
    with pytest.raises(ValueError, match=r"duplicate|immutable content|reconcile"):
        verify_live_stores(chunk, expected, LiveStores(raw, embeddings, graph), settings)


def test_live_validation_failure_is_propagated(monkeypatch: pytest.MonkeyPatch) -> None:
    settings, chunk, expected, raw, embeddings, graph = _fixture()

    def unavailable() -> None:
        raise OSError("controlled network failure")

    monkeypatch.setattr(raw, "validate_container", unavailable)
    with pytest.raises(OSError, match="network failure"):
        verify_live_stores(chunk, expected, LiveStores(raw, embeddings, graph), settings)


@pytest.mark.parametrize("failure_stage", ["raw", "graph", "embedding"])
def test_live_projection_stage_failure_resumes_without_duplicates(failure_stage: str) -> None:
    _, chunk, expected, _, _, _ = _fixture()
    raw = FakeRawStore()
    embeddings = FakeEmbeddingStore()
    graph = FakeGraphStore()
    if failure_stage == "raw":
        raw.fail_after = 0
    elif failure_stage == "graph":
        graph.fail_after = 0
    else:
        embeddings.fail_after = 0
    with pytest.raises(RuntimeError, match="controlled"):
        project_expected_live_evidence(chunk, expected, LiveStores(raw, embeddings, graph))
    raw.fail_after = graph.fail_after = embeddings.fail_after = None
    project_expected_live_evidence(chunk, expected, LiveStores(raw, embeddings, graph))
    project_expected_live_evidence(chunk, expected, LiveStores(raw, embeddings, graph))
    assert len(raw.documents) == 1
    assert len(embeddings.documents) == 1
    assert len(graph.vertices) == len(expected.vertices)
    assert len(graph.edges) == 3


@pytest.mark.parametrize("stage", ["live_raw", "live_graph", "live_embedding"])
def test_interruption_after_live_stage_resumes_safely(stage: str) -> None:
    _, chunk, expected, _, _, _ = _fixture()
    raw = FakeRawStore()
    embeddings = FakeEmbeddingStore()
    graph = FakeGraphStore()

    def interrupt(current: str) -> None:
        if current == stage:
            raise RuntimeError("controlled stage interruption")

    with pytest.raises(RuntimeError, match="stage interruption"):
        project_expected_live_evidence(
            chunk,
            expected,
            LiveStores(raw, embeddings, graph),
            stage_hook=interrupt,
        )
    project_expected_live_evidence(chunk, expected, LiveStores(raw, embeddings, graph))
    assert len(raw.documents) == 1
    assert len(embeddings.documents) == 1
    assert len(graph.edges) == 3
