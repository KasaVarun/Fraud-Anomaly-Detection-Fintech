"""Exact raw-document-to-Gremlin graph evidence reconciliation."""

from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import Engine, create_engine

from fraud_detection.ingestion.errors import ReconciliationError
from fraud_detection.ingestion.loader import initial_batch_id
from fraud_detection.ingestion.manifest import (
    sample_path_from_manifest,
    validate_manifest_artifacts,
)
from fraud_detection.persistence.document_vector.models import RawTransactionDocument
from fraud_detection.persistence.document_vector.projection import initial_batch_status
from fraud_detection.persistence.graph.models import GraphEdge, GraphVertex
from fraud_detection.persistence.graph.projection import build_graph_evidence
from fraud_detection.ports.services import (
    GraphEvidencePersistencePort,
    RawDocumentPersistencePort,
)


class GraphIssueSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    sample_ids: list[str]


class GraphReconciliationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_id: str
    batch_status: str
    expected_transactions: int
    raw_document_count: int
    money_edge_count: int
    graph_vertex_count: int
    graph_edge_count: int
    missing: GraphIssueSummary
    orphaned: GraphIssueSummary
    duplicate: GraphIssueSummary
    conflicting: GraphIssueSummary
    invalid: GraphIssueSummary
    reserved_test: GraphIssueSummary
    reconciled: bool


def _summary(values: set[str], limit: int = 100) -> GraphIssueSummary:
    return GraphIssueSummary(count=len(values), sample_ids=sorted(values)[:limit])


def reconcile_graph_projection(
    source_path: Path,
    manifest_path: Path,
    database_url: str,
    raw_store: RawDocumentPersistencePort,
    graph_store: GraphEvidencePersistencePort,
    *,
    shard_count: int,
    chunk_size: int,
    page_size: int,
) -> GraphReconciliationReport:
    manifest = validate_manifest_artifacts(source_path, manifest_path)
    batch_id = initial_batch_id(manifest)
    expected_count = (
        manifest.split("train").actual_count + manifest.split("validation").actual_count
    )
    raw_store.validate_container()
    graph_store.validate_graph()

    expected_vertices: dict[str, GraphVertex] = {}
    expected_edges: dict[str, GraphEdge] = {}
    raw_ids: set[str] = set()
    raw_count = 0
    invalid: set[str] = set()
    duplicate: set[str] = set()
    for payload in raw_store.iter_batch_documents(batch_id, page_size=page_size):
        raw_count += 1
        identifier = str(payload.get("id", "<missing-id>"))
        if identifier in raw_ids:
            duplicate.add(f"raw:{identifier}")
        raw_ids.add(identifier)
        try:
            document = RawTransactionDocument.model_validate(dict(payload))
        except ValidationError:
            invalid.add(f"raw:{identifier}")
            continue
        evidence = build_graph_evidence(document, shard_count=shard_count)
        for vertex in evidence.vertices:
            previous = expected_vertices.get(vertex.id)
            if previous is not None and previous.content_digest != vertex.content_digest:
                invalid.add(f"expected-vertex:{vertex.id}")
            expected_vertices[vertex.id] = vertex
        expected_edges.update({edge.id: edge for edge in evidence.edges})

    seen_edges: set[str] = set()
    orphaned: set[str] = set()
    conflicting: set[str] = set()
    money_transactions: set[str] = set()
    graph_edge_count = 0
    for payload in graph_store.iter_batch_edges(batch_id, page_size=page_size):
        graph_edge_count += 1
        identifier = str(payload.get("id", "<missing-id>"))
        if identifier in seen_edges:
            duplicate.add(f"edge:{identifier}")
        seen_edges.add(identifier)
        expected = expected_edges.get(identifier)
        if expected is None:
            orphaned.add(f"edge:{identifier}")
        try:
            edge = GraphEdge.model_validate(dict(payload))
        except ValidationError:
            invalid.add(f"edge:{identifier}")
            continue
        if expected is None:
            continue
        if edge.content_digest != expected.content_digest or edge.payload() != expected.payload():
            conflicting.add(f"edge:{identifier}")
            continue
        if edge.kind == "money_flow":
            money_transactions.add(edge.transaction_id)

    missing = {f"edge:{identifier}" for identifier in set(expected_edges) - seen_edges}
    seen_vertices: set[str] = set()
    graph_vertex_count = 0
    for payload in graph_store.iter_vertices_by_ids(
        sorted(expected_vertices), chunk_size=chunk_size, page_size=page_size
    ):
        graph_vertex_count += 1
        identifier = str(payload.get("id", "<missing-id>"))
        if identifier in seen_vertices:
            duplicate.add(f"vertex:{identifier}")
        seen_vertices.add(identifier)
        expected_vertex = expected_vertices.get(identifier)
        try:
            vertex = GraphVertex.model_validate(dict(payload))
        except ValidationError:
            invalid.add(f"vertex:{identifier}")
            continue
        if expected_vertex is None:
            orphaned.add(f"vertex:{identifier}")
        elif (
            vertex.content_digest != expected_vertex.content_digest
            or vertex.payload() != expected_vertex.payload()
        ):
            conflicting.add(f"vertex:{identifier}")
    missing.update(f"vertex:{identifier}" for identifier in set(expected_vertices) - seen_vertices)

    sample = pl.read_parquet(sample_path_from_manifest(manifest, manifest_path))
    reserved_ids = set(
        str(value)
        for value in sample.filter(pl.col("data_split") == "test")["transaction_id"].to_list()
    )
    reserved_test = {
        str(payload.get("transaction_id", "<missing-id>"))
        for payload in graph_store.iter_elements_by_transaction_ids(
            sorted(reserved_ids), chunk_size=chunk_size, page_size=page_size
        )
        if str(payload.get("transaction_id", "<missing-id>")) in reserved_ids
    }
    missing.update(f"money:{identifier}" for identifier in raw_ids - money_transactions)

    engine: Engine = create_engine(database_url, hide_parameters=True)
    try:
        batch_status = initial_batch_status(engine, batch_id)
    finally:
        engine.dispose()
    reconciled = all(
        [
            batch_status == "pending",
            raw_count == expected_count,
            len(money_transactions) == expected_count,
            not missing,
            not orphaned,
            not duplicate,
            not conflicting,
            not invalid,
            not reserved_test,
        ]
    )
    report = GraphReconciliationReport(
        batch_id=batch_id,
        batch_status=batch_status,
        expected_transactions=expected_count,
        raw_document_count=raw_count,
        money_edge_count=len(money_transactions),
        graph_vertex_count=graph_vertex_count,
        graph_edge_count=graph_edge_count,
        missing=_summary(missing),
        orphaned=_summary(orphaned),
        duplicate=_summary(duplicate),
        conflicting=_summary(conflicting),
        invalid=_summary(invalid),
        reserved_test=_summary(reserved_test),
        reconciled=reconciled,
    )
    if not report.reconciled:
        raise ReconciliationError(report.model_dump_json())
    return report
