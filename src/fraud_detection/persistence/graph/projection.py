"""Deterministic one-event Gremlin evidence projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from fraud_detection.persistence.document_vector.models import RawTransactionDocument
from fraud_detection.persistence.graph.models import (
    ENRICHMENT_ALGORITHM_VERSION,
    GRAPH_PROJECTION_VERSION,
    GRAPH_SCHEMA_VERSION,
    PARTITION_ALGORITHM_VERSION,
    GraphEdge,
    GraphEvidenceBundle,
    GraphVertex,
    deterministic_artifact_id,
    graph_digest,
    money_edge_label,
    partition_key,
)
from fraud_detection.ports.services import (
    GraphEvidencePersistencePort,
    RawDocumentPersistencePort,
)


class GraphProjectionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_id: str
    batch_status: str
    expected_transactions: int
    processed_transactions: int
    created_vertices: int
    existing_vertices: int
    created_edges: int
    existing_edges: int


def _vertex(
    *,
    vertex_id: str,
    label: Literal["account", "merchant", "device", "ip"],
    kind: Literal["customer", "merchant", "device", "ip"],
    shard_count: int,
    synthetic: bool,
    account_id: str | None = None,
    enrichment_seed: int | None = None,
) -> GraphVertex:
    payload: dict[str, Any] = {
        "id": vertex_id,
        "pk": partition_key(vertex_id, shard_count),
        "label": label,
        "kind": kind,
        "account_id": account_id,
        "synthetic": synthetic,
        "enrichment_seed": enrichment_seed,
        "enrichment_algorithm_version": (ENRICHMENT_ALGORITHM_VERSION if synthetic else None),
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "projection_version": GRAPH_PROJECTION_VERSION,
        "partition_algorithm_version": PARTITION_ALGORITHM_VERSION,
    }
    payload["content_digest"] = graph_digest(payload)
    return GraphVertex.model_validate(payload)


def _edge(
    *,
    edge_id: str,
    label: Literal["PAYS", "TRANSFER_TO", "USES_DEVICE", "USES_IP"],
    out_vertex: GraphVertex,
    in_vertex: GraphVertex,
    document: RawTransactionDocument,
    kind: Literal["money_flow", "device_observation", "ip_observation"],
    synthetic: bool,
) -> GraphEdge:
    payload: dict[str, Any] = {
        "id": edge_id,
        "label": label,
        "out_id": out_vertex.id,
        "out_pk": out_vertex.pk,
        "in_id": in_vertex.id,
        "in_pk": in_vertex.pk,
        "transaction_id": document.id,
        "step": document.step,
        "batch_id": document.batch_id,
        "data_split": document.data_split,
        "transaction_type": document.transaction_type,
        "amount": document.amount if kind == "money_flow" else None,
        "kind": kind,
        "synthetic": synthetic,
        "source_version": document.projection_version,
        "provenance": (
            f"{document.source_provenance.source_filename}:"
            f"{document.source_provenance.source_sha256}"
        ),
        "enrichment_seed": document.source_provenance.enrichment_seed if synthetic else None,
        "enrichment_algorithm_version": (ENRICHMENT_ALGORITHM_VERSION if synthetic else None),
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "projection_version": GRAPH_PROJECTION_VERSION,
    }
    payload["content_digest"] = graph_digest(payload)
    return GraphEdge.model_validate(payload)


def build_graph_evidence(
    document: RawTransactionDocument,
    *,
    shard_count: int,
) -> GraphEvidenceBundle:
    sender_kind: Literal["customer", "merchant"] = (
        "merchant" if document.sender_id.startswith("M") else "customer"
    )
    receiver_kind: Literal["customer", "merchant"] = (
        "merchant" if document.receiver_id.startswith("M") else "customer"
    )
    sender = _vertex(
        vertex_id=document.sender_id,
        label="merchant" if sender_kind == "merchant" else "account",
        kind=sender_kind,
        shard_count=shard_count,
        synthetic=False,
        account_id=document.sender_id,
    )
    receiver = _vertex(
        vertex_id=document.receiver_id,
        label="merchant" if receiver_kind == "merchant" else "account",
        kind=receiver_kind,
        shard_count=shard_count,
        synthetic=False,
        account_id=document.receiver_id,
    )
    device_id = deterministic_artifact_id(
        "device",
        transaction_id=document.id,
        source_row=document.source_row,
        step=document.step,
        seed=document.source_provenance.enrichment_seed,
    )
    ip_id = deterministic_artifact_id(
        "ip",
        transaction_id=document.id,
        source_row=document.source_row,
        step=document.step,
        seed=document.source_provenance.enrichment_seed,
    )
    device = _vertex(
        vertex_id=device_id,
        label="device",
        kind="device",
        shard_count=shard_count,
        synthetic=True,
        enrichment_seed=document.source_provenance.enrichment_seed,
    )
    ip = _vertex(
        vertex_id=ip_id,
        label="ip",
        kind="ip",
        shard_count=shard_count,
        synthetic=True,
        enrichment_seed=document.source_provenance.enrichment_seed,
    )
    money_label = money_edge_label(document.transaction_type)
    edges = (
        _edge(
            edge_id=f"edge:{money_label}:{document.id}",
            label=money_label,
            out_vertex=sender,
            in_vertex=receiver,
            document=document,
            kind="money_flow",
            synthetic=False,
        ),
        _edge(
            edge_id=f"edge:USES_DEVICE:{document.id}",
            label="USES_DEVICE",
            out_vertex=sender,
            in_vertex=device,
            document=document,
            kind="device_observation",
            synthetic=True,
        ),
        _edge(
            edge_id=f"edge:USES_IP:{document.id}",
            label="USES_IP",
            out_vertex=sender,
            in_vertex=ip,
            document=document,
            kind="ip_observation",
            synthetic=True,
        ),
    )
    return GraphEvidenceBundle(vertices=(sender, receiver, device, ip), edges=edges)


def project_raw_document(
    document_payload: dict[str, Any],
    store: GraphEvidencePersistencePort,
    *,
    shard_count: int,
) -> tuple[int, int, int, int]:
    document = RawTransactionDocument.model_validate(document_payload)
    evidence = build_graph_evidence(document, shard_count=shard_count)
    created_vertices = existing_vertices = created_edges = existing_edges = 0
    for vertex in evidence.vertices:
        outcome = store.upsert_vertex(vertex.payload())
        created_vertices += outcome == "created"
        existing_vertices += outcome == "existing"
    for edge in evidence.edges:
        outcome = store.upsert_edge(edge.payload())
        created_edges += outcome == "created"
        existing_edges += outcome == "existing"
    return created_vertices, existing_vertices, created_edges, existing_edges


def project_initial_graph(
    source_path: Path,
    manifest_path: Path,
    database_url: str,
    raw_store: RawDocumentPersistencePort,
    graph_store: GraphEvidencePersistencePort,
    *,
    shard_count: int,
    page_size: int,
) -> GraphProjectionResult:
    from sqlalchemy import create_engine

    from fraud_detection.ingestion.loader import initial_batch_id
    from fraud_detection.ingestion.manifest import validate_manifest_artifacts
    from fraud_detection.persistence.document_vector.projection import initial_batch_status

    manifest = validate_manifest_artifacts(source_path, manifest_path)
    batch_id = initial_batch_id(manifest)
    expected = manifest.split("train").actual_count + manifest.split("validation").actual_count
    engine = create_engine(database_url, hide_parameters=True)
    try:
        batch_status = initial_batch_status(engine, batch_id)
    finally:
        engine.dispose()
    if batch_status != "pending":
        raise ValueError("initial graph projection requires a pending ingestion batch")
    raw_store.validate_container()
    graph_store.validate_graph()
    created_vertices = existing_vertices = created_edges = existing_edges = processed = 0
    for raw_payload in raw_store.iter_batch_documents(batch_id, page_size=page_size):
        document = RawTransactionDocument.model_validate(dict(raw_payload))
        if document.data_split not in {"train", "validation"}:
            raise ValueError("initial graph projection cannot include test evidence")
        counts = project_raw_document(
            document.cosmos_payload(), graph_store, shard_count=shard_count
        )
        created_vertices += counts[0]
        existing_vertices += counts[1]
        created_edges += counts[2]
        existing_edges += counts[3]
        processed += 1
    if processed != expected:
        raise ValueError(f"raw document count {processed} does not equal expected {expected}")
    return GraphProjectionResult(
        batch_id=batch_id,
        batch_status=batch_status,
        expected_transactions=expected,
        processed_transactions=processed,
        created_vertices=created_vertices,
        existing_vertices=existing_vertices,
        created_edges=created_edges,
        existing_edges=existing_edges,
    )
