"""Phase 8 reconciliation, pooled evaluation, and transactional publication gates."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sqlalchemy import Engine, create_engine, func, select, update

from fraud_detection.config import Settings
from fraud_detection.ingestion.hashing import sha256_file
from fraud_detection.persistence.document_vector.adapter import CosmosRawDocumentAdapter
from fraud_detection.persistence.document_vector.artifacts import (
    iter_embedding_artifact,
    iter_template_artifact,
)
from fraud_detection.persistence.document_vector.embedding_adapter import CosmosEmbeddingAdapter
from fraud_detection.persistence.document_vector.embedding_models import EmbeddingDocument
from fraud_detection.persistence.document_vector.models import RawTransactionDocument
from fraud_detection.persistence.graph.adapter import CosmosGremlinAdapter
from fraud_detection.persistence.graph.models import GraphEdge, GraphVertex
from fraud_detection.persistence.graph.projection import build_graph_evidence
from fraud_detection.persistence.relational import Base
from fraud_detection.ports.services import (
    EmbeddingPersistencePort,
    GraphEvidencePersistencePort,
    RawDocumentPersistencePort,
)

from .contracts import (
    LOCKED_MODEL_ID,
    LOCKED_THRESHOLD,
    ChunkPlan,
    PooledEvaluationResult,
    PublicationResult,
    ReconciliationResult,
    TestChunk,
)
from .evidence import validate_offline_evidence_release
from .scoring import locked_metrics

ingestion_batch = Base.metadata.tables["ingestion_batch"]
transaction = Base.metadata.tables["transaction"]
transaction_label = Base.metadata.tables["transaction_label"]
graph_snapshot = Base.metadata.tables["graph_snapshot"]
transaction_score = Base.metadata.tables["transaction_score"]
fraud_flag = Base.metadata.tables["fraud_flag"]
model_evaluation = Base.metadata.tables["model_evaluation"]


@dataclass(frozen=True)
class LiveStores:
    raw: RawDocumentPersistencePort
    embeddings: EmbeddingPersistencePort
    graph: GraphEvidencePersistencePort


@dataclass(frozen=True)
class ExpectedLiveEvidence:
    raw_documents: tuple[RawTransactionDocument, ...]
    embeddings: tuple[EmbeddingDocument, ...]
    vertices: tuple[GraphVertex, ...]
    edges: tuple[GraphEdge, ...]


def _missing_stores(settings: Settings) -> tuple[str, ...]:
    missing: list[str] = []
    if not settings.document_vector_configured:
        missing.append("cosmos_nosql_raw")
    if not settings.embedding_configured:
        missing.append("cosmos_nosql_embeddings")
    if not settings.graph_configured:
        missing.append("cosmos_gremlin")
    return tuple(missing)


def _offline_evidence_counts(
    chunk: TestChunk, expected_ids: set[str], settings: Settings
) -> dict[str, int]:
    manifest_path = settings.incremental_data_root / "offline_evidence_manifest_v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_offline_evidence_release(
        settings.incremental_data_root,
        settings.embedding_local_artifact,
        expected_test_count=int(manifest["test_count"]),
        source_sha256=str(manifest["source_sha256"]),
        sample_sha256=str(manifest["sample_sha256"]),
        graph_shard_count=settings.gremlin_partition_shards,
    )
    raw_payloads = [
        json.loads(str(value))
        for value in pl.read_parquet(Path(str(manifest["raw_artifact"])))["payload"].to_list()
    ]
    raw_documents = [
        RawTransactionDocument.model_validate(payload)
        for payload in raw_payloads
        if payload.get("batch_id") == chunk.batch_id
    ]
    raw_ids = {document.id for document in raw_documents}
    edge_payloads = [
        json.loads(str(value))
        for value in pl.read_parquet(Path(str(manifest["graph_edge_artifact"])))[
            "payload"
        ].to_list()
    ]
    batch_edges = [
        payload for payload in edge_payloads if payload.get("batch_id") == chunk.batch_id
    ]
    edge_ids = {str(payload["transaction_id"]) for payload in batch_edges}
    embedding_ids = {
        document.transaction_id
        for document in iter_embedding_artifact(Path(str(manifest["embedding_artifact"])))
        if document.batch_id == chunk.batch_id
    }
    template_ids = {
        template.transaction_id
        for template in iter_template_artifact(Path(str(manifest["template_artifact"])))
        if template.batch_id == chunk.batch_id
    }
    vertex_ids = set(
        str(value)
        for value in pl.read_parquet(Path(str(manifest["graph_vertex_artifact"])))["id"].to_list()
    )
    required_vertices = {
        vertex.id
        for document in raw_documents
        for vertex in build_graph_evidence(
            document, shard_count=settings.gremlin_partition_shards
        ).vertices
    }
    if (
        raw_ids != expected_ids
        or edge_ids != expected_ids
        or len(batch_edges) != 3 * len(expected_ids)
        or embedding_ids != expected_ids
        or template_ids != expected_ids
        or not required_vertices.issubset(vertex_ids)
    ):
        raise ValueError("offline evidence transaction identities do not reconcile")
    return {
        "raw_documents": len(raw_ids),
        "graph_vertices_required": len(required_vertices),
        "graph_edges": len(batch_edges),
        "templates": len(template_ids),
        "embeddings": len(embedding_ids),
        "features": len(expected_ids),
    }


def reconcile_chunk(
    database_url: str,
    feature_artifact: Path,
    chunk: TestChunk,
    settings: Settings,
    *,
    verify_offline_artifacts: bool = True,
) -> ReconciliationResult:
    feature_ids = set(
        str(value)
        for value in (
            pl.scan_parquet(feature_artifact)
            .filter(pl.col("batch_id") == chunk.batch_id)
            .select("transaction_id")
            .collect(engine="streaming")["transaction_id"]
            .to_list()
        )
    )
    engine: Engine = create_engine(database_url, hide_parameters=True)
    try:
        with engine.connect() as connection:
            transaction_rows: list[Any] = list(
                connection.execute(
                    select(transaction.c.transaction_id).where(
                        transaction.c.batch_id == chunk.batch_id
                    )
                )
                .scalars()
                .all()
            )
            transaction_ids = {str(value) for value in transaction_rows}
            label_count = int(
                connection.scalar(
                    select(func.count())
                    .select_from(transaction_label)
                    .join(transaction)
                    .where(transaction.c.batch_id == chunk.batch_id)
                )
                or 0
            )
            scores = (
                connection.execute(
                    select(
                        transaction_score.c.score_id,
                        transaction_score.c.transaction_id,
                        transaction_score.c.fraud_probability,
                        transaction_score.c.snapshot_id,
                        graph_snapshot.c.cutoff_step,
                        transaction.c.step,
                    )
                    .join(transaction)
                    .join(graph_snapshot)
                    .where(transaction.c.batch_id == chunk.batch_id)
                    .where(transaction_score.c.model_version_id == LOCKED_MODEL_ID)
                )
                .mappings()
                .all()
            )
            qualifying_ids = {
                str(row["score_id"])
                for row in scores
                if float(row["fraud_probability"]) >= float(LOCKED_THRESHOLD)
            }
            if any(int(row["cutoff_step"]) != int(row["step"]) - 1 for row in scores):
                raise ValueError("score graph snapshot cutoff is not strictly prior")
            flag_rows: list[Any] = list(
                connection.execute(
                    select(fraud_flag.c.score_id)
                    .select_from(fraud_flag)
                    .join(
                        transaction_score,
                        transaction_score.c.score_id == fraud_flag.c.score_id,
                    )
                    .join(
                        transaction,
                        transaction.c.transaction_id == transaction_score.c.transaction_id,
                    )
                    .where(transaction.c.batch_id == chunk.batch_id)
                    .where(transaction_score.c.model_version_id == LOCKED_MODEL_ID)
                )
                .scalars()
                .all()
            )
            flag_ids = {str(value) for value in flag_rows}
            evaluation_count = int(
                connection.scalar(
                    select(func.count())
                    .select_from(model_evaluation)
                    .where(
                        model_evaluation.c.batch_id == chunk.batch_id,
                        model_evaluation.c.model_version_id == LOCKED_MODEL_ID,
                        model_evaluation.c.split_name == "test",
                    )
                )
                or 0
            )
            snapshot_count = int(
                connection.scalar(
                    select(func.count())
                    .select_from(graph_snapshot)
                    .where(graph_snapshot.c.batch_id == chunk.batch_id)
                )
                or 0
            )
        evidence_counts = (
            _offline_evidence_counts(chunk, transaction_ids, settings)
            if verify_offline_artifacts
            else {"features": len(feature_ids)}
        )
        offline_reconciled = all(
            [
                len(transaction_ids) == chunk.row_count,
                label_count == chunk.row_count,
                len(scores) == chunk.row_count,
                transaction_ids == feature_ids,
                flag_ids == qualifying_ids,
                evaluation_count == 1,
                snapshot_count
                == len(
                    set(
                        int(value)
                        for value in (
                            pl.scan_parquet(feature_artifact)
                            .filter(pl.col("batch_id") == chunk.batch_id)
                            .select("step")
                            .collect(engine="streaming")["step"]
                            .to_list()
                        )
                    )
                ),
            ]
        )
        missing = _missing_stores(settings)
        reason = None
        if not offline_reconciled:
            reason = "offline_reconciliation_failed"
        elif missing:
            reason = "live_stores_not_configured:" + ",".join(missing)
        return ReconciliationResult(
            batch_id=chunk.batch_id,
            expected_count=chunk.row_count,
            transaction_count=len(transaction_ids),
            label_count=label_count,
            score_count=len(scores),
            qualifying_score_count=len(qualifying_ids),
            flag_count=len(flag_ids),
            evaluation_count=evaluation_count,
            snapshot_count=snapshot_count,
            offline_evidence_counts=evidence_counts,
            missing_live_stores=missing,
            offline_reconciled=offline_reconciled,
            publication_eligible=offline_reconciled and not missing,
            blocking_reason=reason,
        )
    finally:
        engine.dispose()


def expected_live_evidence(settings: Settings, chunk: TestChunk) -> ExpectedLiveEvidence:
    manifest = json.loads(
        (settings.incremental_data_root / "offline_evidence_manifest_v2.json").read_text(
            encoding="utf-8"
        )
    )
    raw_documents = tuple(
        RawTransactionDocument.model_validate(json.loads(str(value)))
        for value in pl.read_parquet(Path(str(manifest["raw_artifact"])))["payload"].to_list()
        if json.loads(str(value)).get("batch_id") == chunk.batch_id
    )
    embeddings = tuple(
        value
        for value in iter_embedding_artifact(Path(str(manifest["embedding_artifact"])))
        if value.batch_id == chunk.batch_id
    )
    vertices_by_id: dict[str, GraphVertex] = {}
    edges: list[GraphEdge] = []
    for document in raw_documents:
        bundle = build_graph_evidence(document, shard_count=settings.gremlin_partition_shards)
        for vertex in bundle.vertices:
            previous = vertices_by_id.setdefault(vertex.id, vertex)
            if previous.payload() != vertex.payload():
                raise ValueError("expected graph vertex immutable content conflicts")
        edges.extend(bundle.edges)
    return ExpectedLiveEvidence(
        raw_documents=raw_documents,
        embeddings=embeddings,
        vertices=tuple(vertices_by_id.values()),
        edges=tuple(edges),
    )


def _exact_records(
    name: str,
    expected: dict[str, dict[str, Any]],
    actual_values: list[dict[str, Any]],
) -> None:
    actual: dict[str, dict[str, Any]] = {}
    for value in actual_values:
        identifier = str(value["id"])
        if identifier in actual:
            raise ValueError(f"duplicate live {name} identity {identifier}")
        actual[identifier] = value
    if actual != expected:
        raise ValueError(f"live {name} immutable content does not reconcile")


def verify_live_stores(
    chunk: TestChunk,
    expected: ExpectedLiveEvidence,
    stores: LiveStores,
    settings: Settings,
) -> None:
    stores.raw.validate_container()
    stores.embeddings.validate_embedding_container()
    stores.graph.validate_graph()
    expected_raw = {value.id: value.cosmos_payload() for value in expected.raw_documents}
    expected_embeddings = {value.id: value.payload() for value in expected.embeddings}
    expected_edges = {value.id: value.payload() for value in expected.edges}
    expected_vertices = {value.id: value.payload() for value in expected.vertices}
    if (
        len(expected_raw) != len(expected.raw_documents)
        or len(expected_embeddings) != len(expected.embeddings)
        or len(expected_edges) != len(expected.edges)
        or len(expected_vertices) != len(expected.vertices)
    ):
        raise ValueError("expected live evidence contains duplicate identities")
    actual_raw = [
        RawTransactionDocument.model_validate(dict(value)).cosmos_payload()
        for value in stores.raw.iter_batch_documents(
            chunk.batch_id, page_size=settings.cosmos_page_size
        )
    ]
    _exact_records("raw document", expected_raw, actual_raw)

    actual_embeddings = [
        EmbeddingDocument.model_validate(dict(value)).payload()
        for value in stores.embeddings.iter_embedding_batch(
            chunk.batch_id, page_size=settings.embedding_page_size
        )
    ]
    _exact_records("embedding", expected_embeddings, actual_embeddings)

    actual_edges = [
        GraphEdge.model_validate(dict(value)).payload()
        for value in stores.graph.iter_batch_edges(
            chunk.batch_id, page_size=settings.cosmos_page_size
        )
    ]
    _exact_records("graph edge", expected_edges, actual_edges)

    actual_vertices = [
        GraphVertex.model_validate(dict(value)).payload()
        for value in stores.graph.iter_vertices_by_ids(
            sorted(expected_vertices),
            chunk_size=settings.gremlin_projection_batch_size,
            page_size=settings.cosmos_page_size,
        )
    ]
    _exact_records("graph vertex", expected_vertices, actual_vertices)


def attempt_publication(
    database_url: str,
    feature_artifact: Path,
    plan: ChunkPlan,
    chunk: TestChunk,
    settings: Settings,
    *,
    verify_offline_artifacts: bool = True,
    live_stores: LiveStores | None = None,
    expected_evidence: ExpectedLiveEvidence | None = None,
    transition_hook: Callable[[], None] | None = None,
) -> PublicationResult:
    dependency_engine: Engine = create_engine(database_url, hide_parameters=True)
    current_status: str | None = None
    try:
        with dependency_engine.connect() as connection:
            current_value = connection.scalar(
                select(ingestion_batch.c.status).where(ingestion_batch.c.batch_id == chunk.batch_id)
            )
            current_status = str(current_value) if current_value is not None else None
            initial_status = connection.scalar(
                select(ingestion_batch.c.status).where(
                    ingestion_batch.c.batch_id == plan.initial_batch_id
                )
            )
            if initial_status != "published":
                return PublicationResult(
                    batch_id=chunk.batch_id,
                    status="pending",
                    published=False,
                    reason="initial_batch_not_published",
                )
            if chunk.index:
                previous_statuses: list[Any] = list(
                    connection.execute(
                        select(ingestion_batch.c.status).where(
                            ingestion_batch.c.batch_id.in_(
                                [value.batch_id for value in plan.chunks[: chunk.index]]
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                if len(previous_statuses) != chunk.index or any(
                    value != "published" for value in previous_statuses
                ):
                    return PublicationResult(
                        batch_id=chunk.batch_id,
                        status="pending",
                        published=False,
                        reason="earlier_chunk_not_published",
                    )
    finally:
        dependency_engine.dispose()
    report = reconcile_chunk(
        database_url,
        feature_artifact,
        chunk,
        settings,
        verify_offline_artifacts=verify_offline_artifacts,
    )
    if not report.publication_eligible:
        return PublicationResult(
            batch_id=chunk.batch_id,
            status="published" if current_status == "published" else "pending",
            published=current_status == "published",
            reason=report.blocking_reason,
        )
    engine: Engine = create_engine(database_url, hide_parameters=True)
    production_graph: CosmosGremlinAdapter | None = None
    try:
        with engine.begin() as connection:
            resolved_evidence = expected_evidence or expected_live_evidence(settings, chunk)
            database_values: list[Any] = list(
                connection.execute(
                    select(transaction.c.transaction_id).where(
                        transaction.c.batch_id == chunk.batch_id
                    )
                ).scalars()
            )
            database_ids = {str(value) for value in database_values}
            if {value.id for value in resolved_evidence.raw_documents} != database_ids:
                raise ValueError("expected live evidence differs from PostgreSQL transactions")
            if live_stores is None:
                raw = CosmosRawDocumentAdapter(settings)
                embeddings = CosmosEmbeddingAdapter(settings)
                production_graph = CosmosGremlinAdapter(settings)
                resolved_stores = LiveStores(raw, embeddings, production_graph)
            else:
                resolved_stores = live_stores
            verify_live_stores(chunk, resolved_evidence, resolved_stores, settings)
            current = connection.scalar(
                select(ingestion_batch.c.status).where(ingestion_batch.c.batch_id == chunk.batch_id)
            )
            if current not in {"pending", "published"}:
                raise ValueError("batch is not eligible for publication transition")
            if current == "pending":
                if transition_hook:
                    transition_hook()
                connection.execute(
                    update(ingestion_batch)
                    .where(
                        ingestion_batch.c.batch_id == chunk.batch_id,
                        ingestion_batch.c.status == "pending",
                    )
                    .values(status="published")
                )
        return PublicationResult(
            batch_id=chunk.batch_id, status="published", published=True, reason=None
        )
    finally:
        engine.dispose()
        if production_graph is not None:
            production_graph.close()


def evaluate_pooled_locked_test(
    database_url: str,
    plan: ChunkPlan,
    output_path: Path,
) -> PooledEvaluationResult:
    engine: Engine = create_engine(database_url, hide_parameters=True)
    try:
        with engine.connect() as connection:
            rows = (
                connection.execute(
                    select(
                        transaction.c.batch_id,
                        transaction_score.c.transaction_id,
                        transaction_score.c.fraud_probability,
                        transaction_label.c.is_fraud,
                    )
                    .join(transaction_score)
                    .join(transaction_label)
                    .where(transaction.c.batch_id.in_([chunk.batch_id for chunk in plan.chunks]))
                    .where(transaction_score.c.model_version_id == LOCKED_MODEL_ID)
                    .order_by(transaction.c.step, transaction_score.c.transaction_id)
                )
                .mappings()
                .all()
            )
    finally:
        engine.dispose()
    if len(rows) != plan.test_row_count:
        raise ValueError("pooled locked-test evaluation requires all 40,000 immutable scores")
    labels = np.asarray([bool(row["is_fraud"]) for row in rows], dtype=np.int8)
    scores = np.asarray([float(row["fraud_probability"]) for row in rows], dtype=np.float64)
    metrics = locked_metrics(labels, scores)
    per_chunk: list[dict[str, object]] = []
    for chunk in plan.chunks:
        positions = [index for index, row in enumerate(rows) if row["batch_id"] == chunk.batch_id]
        chunk_metrics = locked_metrics(labels[positions], scores[positions])
        per_chunk.append(
            {
                "index": chunk.index,
                "batch_id": chunk.batch_id,
                "start_step": chunk.start_step,
                "end_step": chunk.end_step,
                **chunk_metrics.model_dump(mode="json"),
            }
        )
    payload = {
        "evaluation_version": "phase8-locked-test-evaluation-v1",
        "model_version_id": LOCKED_MODEL_ID,
        "threshold": str(LOCKED_THRESHOLD),
        "chunk_policy": plan.policy,
        "source_sha256": plan.source_sha256,
        "sample_sha256": plan.sample_sha256,
        "metrics": metrics.model_dump(mode="json"),
        "per_chunk": per_chunk,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_text(encoding="utf-8") != encoded:
        raise ValueError("pooled locked-test evaluation artifact conflicts with existing content")
    if not output_path.exists():
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output_path)
    return PooledEvaluationResult(
        model_version_id=LOCKED_MODEL_ID,
        threshold=LOCKED_THRESHOLD,
        chunk_count=len(plan.chunks),
        metrics=metrics,
        per_chunk=tuple(per_chunk),
        artifact_path=str(output_path),
        artifact_sha256=sha256_file(output_path),
    )
