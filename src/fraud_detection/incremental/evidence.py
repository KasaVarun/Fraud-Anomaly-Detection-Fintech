"""Deterministic local Phase 8 evidence and causal feature artifacts."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from fraud_detection.config import Settings
from fraud_detection.features.behavioral import iter_behavioral_templates_sorted
from fraud_detection.ingestion.constants import PROVENANCE
from fraud_detection.ingestion.hashing import sha256_file
from fraud_detection.ingestion.loader import initial_batch_id
from fraud_detection.ingestion.manifest import read_manifest, sample_path_from_manifest
from fraud_detection.modeling.contracts import (
    ALL_FEATURES,
    PHASE6_LOGICAL_EMBEDDING_DIGEST,
)
from fraud_detection.modeling.features import (
    _tabular_features,
    build_causal_graph_feature_rows,
    feature_logical_digest,
)
from fraud_detection.persistence.document_vector.artifacts import (
    embedding_artifact_logical_digest,
    iter_embedding_artifact,
    iter_template_artifact,
    write_embedding_artifact,
    write_template_artifact,
)
from fraud_detection.persistence.document_vector.embedding_generation import (
    iter_embedding_documents,
)
from fraud_detection.persistence.document_vector.encoder import BehavioralEncoder, MiniLMEncoder
from fraud_detection.persistence.document_vector.models import (
    FORBIDDEN_RAW_PREDICTIVE_FIELDS,
    RAW_DOCUMENT_SCHEMA_VERSION,
    RAW_PROJECTION_VERSION,
    RawTransactionDocument,
    content_digest,
)
from fraud_detection.persistence.graph.models import GraphEdge, GraphVertex
from fraud_detection.persistence.graph.projection import build_graph_evidence

from .contracts import (
    LOCKED_FEATURE_VERSION,
    PHASE8_INPUT_VERSION,
    OfflineEvidenceResult,
)
from .planning import plan_test_chunks


def _decimal(value: object) -> str:
    return format(Decimal(str(value)), "f")


def _raw_document(
    row: dict[str, Any],
    *,
    batch_id: str,
    manifest: Any,
    projected_at: datetime,
) -> RawTransactionDocument:
    payload: dict[str, Any] = {
        "id": str(row["transaction_id"]),
        "sender_id": str(row["nameOrig"]),
        "receiver_id": str(row["nameDest"]),
        "batch_id": batch_id,
        "source_row": int(row["source_row"]),
        "step": int(row["step"]),
        "data_split": str(row["data_split"]),
        "transaction_type": str(row["type"]),
        "amount": _decimal(row["amount"]),
        "forensic_balances": {
            "oldbalanceOrg": _decimal(row["oldbalanceOrg"]),
            "newbalanceOrig": _decimal(row["newbalanceOrig"]),
            "oldbalanceDest": _decimal(row["oldbalanceDest"]),
            "newbalanceDest": _decimal(row["newbalanceDest"]),
            "predictive_use": "forbidden",
        },
        "source_truth": {
            "is_fraud": bool(row["isFraud"]),
            "source_flag": bool(row["isFlaggedFraud"]),
            "provenance": PROVENANCE,
            "predictive_use": "forbidden",
        },
        "source_provenance": {
            "source_filename": manifest.source_filename,
            "source_sha256": manifest.source_sha256,
            "sampling_seed": manifest.sampling_seed,
            "enrichment_seed": manifest.enrichment_seed,
        },
        "forbidden_predictive_fields": FORBIDDEN_RAW_PREDICTIVE_FIELDS,
        "document_schema_version": RAW_DOCUMENT_SCHEMA_VERSION,
        "projection_version": RAW_PROJECTION_VERSION,
        "projected_at": projected_at,
    }
    payload["content_digest"] = content_digest(payload)
    return RawTransactionDocument.model_validate(payload)


def _write_payloads(path: Path, payloads: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "id": [str(payload["id"]) for payload in payloads],
            "payload": [
                json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
                for payload in payloads
            ],
        }
    ).sort("id")
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(path)
    return sha256_file(path)


def _test_vector_features(
    training_embedding_path: Path,
    test_embedding_path: Path,
    *,
    batch_size: int = 512,
) -> dict[str, dict[str, float]]:
    training = pq.read_table(
        training_embedding_path,
        columns=["transaction_id", "reference_eligible", "data_split", "vector"],
    ).combine_chunks()
    eligible = np.asarray(training.column("reference_eligible").to_pylist(), dtype=bool)
    splits = np.asarray(training.column("data_split").to_pylist(), dtype=object)
    if not np.all(splits[eligible] == "train"):
        raise ValueError("vector reference pool contains non-training evidence")
    training_values = training.column("vector").chunk(0).values.to_numpy(zero_copy_only=False)
    training_vectors = np.asarray(training_values, dtype=np.float32).reshape(len(eligible), 384)
    reference_vectors = training_vectors[eligible]

    test = pq.read_table(
        test_embedding_path,
        columns=["transaction_id", "reference_eligible", "data_split", "vector"],
    ).combine_chunks()
    if any(bool(value) for value in test.column("reference_eligible").to_pylist()):
        raise ValueError("test embeddings cannot be similarity references")
    if set(str(value) for value in test.column("data_split").to_pylist()) != {"test"}:
        raise ValueError("Phase 8 embedding artifact must contain test rows only")
    identifiers = [str(value) for value in test.column("transaction_id").to_pylist()]
    test_values = test.column("vector").chunk(0).values.to_numpy(zero_copy_only=False)
    test_vectors = np.asarray(test_values, dtype=np.float32).reshape(len(identifiers), 384)
    output: dict[str, dict[str, float]] = {}
    for start in range(0, len(identifiers), batch_size):
        query = test_vectors[start : start + batch_size]
        similarities = query @ reference_vectors.T
        distances = np.maximum(0.0, 1.0 - np.max(similarities, axis=1))
        for identifier, distance in zip(
            identifiers[start : start + batch_size], distances, strict=True
        ):
            output[identifier] = {
                "nearest_earlier_training_fraud_cosine_distance": float(distance),
                "missing_earlier_training_fraud_reference": 0.0,
            }
    return output


def _build_offline_evidence_unpublished(
    source_path: Path,
    manifest_path: Path,
    training_embedding_path: Path,
    data_root: Path,
    settings: Settings,
    *,
    encoder: BehavioralEncoder | None = None,
    expected_sample_count: int = 200_000,
    expected_test_count: int = 40_000,
) -> OfflineEvidenceResult:
    started = time.perf_counter()
    manifest = read_manifest(manifest_path)
    sample = pl.read_parquet(sample_path_from_manifest(manifest, manifest_path)).sort(
        ["step", "transaction_id"]
    )
    if sample.height != expected_sample_count:
        raise ValueError("Phase 8 evidence preparation sample count is incorrect")
    plan = plan_test_chunks(
        source_path,
        manifest_path,
        chunk_steps=settings.incremental_chunk_steps,
        expected_test_count=expected_test_count,
    )
    batch_by_step = {
        step: chunk.batch_id
        for chunk in plan.chunks
        for step in range(chunk.start_step, chunk.end_step + 1)
    }
    initial_batch = initial_batch_id(manifest)
    documents: list[RawTransactionDocument] = []
    test_documents: list[RawTransactionDocument] = []
    for row in sample.iter_rows(named=True):
        split = str(row["data_split"])
        batch_id = initial_batch if split != "test" else batch_by_step[int(row["step"])]
        document = _raw_document(
            row, batch_id=batch_id, manifest=manifest, projected_at=manifest.created_at
        )
        documents.append(document)
        if split == "test":
            test_documents.append(document)
    raw_path = data_root / "test_raw_documents_v1.parquet"
    raw_sha = _write_payloads(raw_path, [document.cosmos_payload() for document in test_documents])

    vertices: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for document in test_documents:
        bundle = build_graph_evidence(document, shard_count=settings.gremlin_partition_shards)
        vertices.extend(vertex.payload() for vertex in bundle.vertices)
        edges.extend(edge.payload() for edge in bundle.edges)
    vertex_path = data_root / "test_graph_vertices_v1.parquet"
    edge_path = data_root / "test_graph_edges_v1.parquet"
    vertex_sha = _write_payloads(vertex_path, vertices)
    edge_sha = _write_payloads(edge_path, edges)
    del vertices, edges

    template_path = data_root / "test_behavior_templates_v1.parquet"
    templates = (
        template
        for template in iter_behavioral_templates_sorted(documents)
        if template.data_split == "test"
    )
    template_count, template_sha, _ = write_template_artifact(
        template_path, templates, batch_size=settings.embedding_batch_size
    )
    if template_count != expected_test_count:
        raise ValueError("Phase 8 template count differs from the test plan")
    resolved_encoder: BehavioralEncoder = encoder or cast(
        BehavioralEncoder,
        MiniLMEncoder(cache_directory=settings.embedding_model_cache),
    )
    embedding_path = data_root / "test_behavior_embeddings_v1.parquet"
    embedding_documents = iter_embedding_documents(
        iter_template_artifact(template_path),
        resolved_encoder,
        settings,
        projected_at=manifest.created_at,
    )
    (
        embedding_count,
        reference_count,
        embedding_sha,
        _,
        embedding_logical_digest,
    ) = write_embedding_artifact(
        embedding_path,
        embedding_documents,
        batch_size=settings.embedding_batch_size,
    )
    if embedding_count != expected_test_count or reference_count:
        raise ValueError("Phase 8 test embeddings are incomplete or reference-eligible")

    rows = sample.to_dicts()
    graph = build_causal_graph_feature_rows(rows, enrichment_seed=manifest.enrichment_seed)
    vector = _test_vector_features(training_embedding_path, embedding_path)
    test_rows = [row for row in rows if str(row["data_split"]) == "test"]
    feature_records: list[dict[str, Any]] = []
    for row in test_rows:
        identifier = str(row["transaction_id"])
        record: dict[str, Any] = {
            "transaction_id": identifier,
            "batch_id": batch_by_step[int(row["step"])],
            "step": int(row["step"]),
            "data_split": "test",
        }
        record.update(_tabular_features(row))
        record.update(graph[identifier])
        record.update(vector[identifier])
        feature_records.append(record)
    feature_frame = (
        pl.DataFrame(feature_records)
        .select(["transaction_id", "batch_id", "step", "data_split", *ALL_FEATURES])
        .sort(["step", "transaction_id"])
    )
    if (
        feature_frame.height != expected_test_count
        or not np.isfinite(feature_frame.select(ALL_FEATURES).to_numpy()).all()
    ):
        raise ValueError("Phase 8 feature artifact is incomplete or non-finite")
    feature_path = data_root / "locked_test_features_v2.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.write_parquet(feature_path, compression="zstd")
    feature_sha = sha256_file(feature_path)
    feature_digest = feature_logical_digest(feature_frame)
    metadata = {
        "input_version": PHASE8_INPUT_VERSION,
        "feature_version": LOCKED_FEATURE_VERSION,
        "ordered_features": ALL_FEATURES,
        "feature_sha256": feature_sha,
        "feature_logical_digest": feature_digest,
        "embedding_sha256": embedding_sha,
        "embedding_logical_digest": embedding_logical_digest,
        "phase6_training_embedding_sha256": sha256_file(training_embedding_path),
        "source_sha256": manifest.source_sha256,
        "sample_sha256": manifest.sample_sha256,
        "test_count": expected_test_count,
        "test_labels_used_for_features": False,
    }
    (data_root / "locked_test_features_v2.metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    immutable_manifest = {
        "version": "phase8-offline-evidence-v1",
        "source_sha256": manifest.source_sha256,
        "sample_sha256": manifest.sample_sha256,
        "raw_artifact": str(raw_path),
        "raw_sha256": raw_sha,
        "graph_vertex_artifact": str(vertex_path),
        "graph_vertex_sha256": vertex_sha,
        "graph_edge_artifact": str(edge_path),
        "graph_edge_sha256": edge_sha,
        "template_artifact": str(template_path),
        "template_sha256": template_sha,
        "embedding_artifact": str(embedding_path),
        "embedding_sha256": embedding_sha,
        "embedding_logical_digest": embedding_logical_digest,
        "feature_artifact": str(feature_path),
        "feature_sha256": feature_sha,
        "feature_logical_digest": feature_digest,
        "feature_version": LOCKED_FEATURE_VERSION,
        "test_count": expected_test_count,
    }
    manifest_artifact = data_root / "offline_evidence_manifest_v1.json"
    manifest_encoded = json.dumps(immutable_manifest, sort_keys=True, separators=(",", ":"))
    if (
        manifest_artifact.exists()
        and manifest_artifact.read_text(encoding="utf-8") != manifest_encoded
    ):
        raise ValueError("offline evidence manifest conflicts with existing artifacts")
    if not manifest_artifact.exists():
        manifest_artifact.write_text(manifest_encoded, encoding="utf-8")
    return OfflineEvidenceResult(
        raw_artifact=str(raw_path),
        raw_sha256=raw_sha,
        graph_vertex_artifact=str(vertex_path),
        graph_vertex_sha256=vertex_sha,
        graph_edge_artifact=str(edge_path),
        graph_edge_sha256=edge_sha,
        template_artifact=str(template_path),
        template_sha256=template_sha,
        embedding_artifact=str(embedding_path),
        embedding_sha256=embedding_sha,
        embedding_logical_digest=embedding_artifact_logical_digest(embedding_path),
        feature_artifact=str(feature_path),
        feature_sha256=feature_sha,
        feature_logical_digest=feature_digest,
        manifest_artifact=str(manifest_artifact),
        manifest_sha256=sha256_file(manifest_artifact),
        test_count=expected_test_count,
        feature_version=LOCKED_FEATURE_VERSION,
        build_seconds=time.perf_counter() - started,
    )


_ARTIFACT_NAMES = (
    "test_raw_documents_v1.parquet",
    "test_graph_vertices_v1.parquet",
    "test_graph_edges_v1.parquet",
    "test_behavior_templates_v1.parquet",
    "test_behavior_embeddings_v1.parquet",
    "locked_test_features_v2.parquet",
    "locked_test_features_v2.metadata.json",
)
_V1_MANIFEST_NAME = "offline_evidence_manifest_v1.json"
_V2_METADATA_NAME = "locked_test_features_v2.metadata.v2.json"
_V2_MANIFEST_NAME = "offline_evidence_manifest_v2.json"


def _json_payloads(path: Path) -> list[dict[str, Any]]:
    frame = pl.read_parquet(path)
    if frame.columns != ["id", "payload"]:
        raise ValueError(f"offline artifact schema is invalid: {path.name}")
    return [json.loads(str(value)) for value in frame["payload"].to_list()]


def _validate_v1_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "version",
        "source_sha256",
        "sample_sha256",
        "raw_artifact",
        "raw_sha256",
        "graph_vertex_artifact",
        "graph_vertex_sha256",
        "graph_edge_artifact",
        "graph_edge_sha256",
        "template_artifact",
        "template_sha256",
        "embedding_artifact",
        "embedding_sha256",
        "embedding_logical_digest",
        "feature_artifact",
        "feature_sha256",
        "feature_logical_digest",
        "feature_version",
        "test_count",
    }
    if set(payload) != required or payload["version"] != "phase8-offline-evidence-v1":
        raise ValueError("historical offline evidence manifest contract is invalid")
    for artifact_field, hash_field in (
        ("raw_artifact", "raw_sha256"),
        ("graph_vertex_artifact", "graph_vertex_sha256"),
        ("graph_edge_artifact", "graph_edge_sha256"),
        ("template_artifact", "template_sha256"),
        ("embedding_artifact", "embedding_sha256"),
        ("feature_artifact", "feature_sha256"),
    ):
        artifact = Path(str(payload[artifact_field]))
        if not artifact.is_file() or sha256_file(artifact) != payload[hash_field]:
            raise ValueError(f"historical offline artifact conflicts on {artifact_field}")
    return cast(dict[str, Any], payload)


def _authoritative_metadata(
    root: Path,
    training_embedding_path: Path,
    *,
    source_sha256: str,
    sample_sha256: str,
    expected_test_count: int,
) -> dict[str, Any]:
    feature_path = root / "locked_test_features_v2.parquet"
    embedding_path = root / "test_behavior_embeddings_v1.parquet"
    legacy_path = root / "locked_test_features_v2.metadata.json"
    if not legacy_path.is_file():
        raise ValueError("legacy feature metadata is missing")
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    feature_frame = pl.read_parquet(feature_path).sort(["step", "transaction_id"])
    feature_digest = feature_logical_digest(feature_frame)
    embedding_digest = embedding_artifact_logical_digest(embedding_path)
    training_digest = embedding_artifact_logical_digest(training_embedding_path)
    metadata = {
        "version": "phase8-locked-test-feature-metadata-v2",
        "input_version": PHASE8_INPUT_VERSION,
        "feature_version": LOCKED_FEATURE_VERSION,
        "ordered_features": list(ALL_FEATURES),
        "feature_sha256": sha256_file(feature_path),
        "feature_logical_digest": feature_digest,
        "test_count": expected_test_count,
        "source_sha256": source_sha256,
        "sample_sha256": sample_sha256,
        "test_embedding_sha256": sha256_file(embedding_path),
        "test_embedding_logical_digest": embedding_digest,
        "phase6_training_embedding_sha256": sha256_file(training_embedding_path),
        "phase6_training_embedding_logical_digest": training_digest,
        "test_labels_used_for_features": False,
    }
    legacy_expected = {
        "input_version": metadata["input_version"],
        "feature_version": metadata["feature_version"],
        "ordered_features": metadata["ordered_features"],
        "feature_sha256": metadata["feature_sha256"],
        "feature_logical_digest": metadata["feature_logical_digest"],
        "embedding_sha256": metadata["test_embedding_sha256"],
        "embedding_logical_digest": metadata["test_embedding_logical_digest"],
        "phase6_training_embedding_sha256": metadata["phase6_training_embedding_sha256"],
        "source_sha256": source_sha256,
        "sample_sha256": sample_sha256,
        "test_count": expected_test_count,
        "test_labels_used_for_features": False,
    }
    if legacy != legacy_expected:
        raise ValueError("legacy feature metadata conflicts with immutable evidence")
    if training_digest != PHASE6_LOGICAL_EMBEDDING_DIGEST:
        raise ValueError("Phase 6 training embedding logical digest is invalid")
    if (
        feature_frame.height != expected_test_count
        or feature_frame["transaction_id"].n_unique() != expected_test_count
        or set(feature_frame["data_split"].to_list()) != {"test"}
        or feature_frame.columns
        != ["transaction_id", "batch_id", "step", "data_split", *ALL_FEATURES]
        or not np.isfinite(feature_frame.select(ALL_FEATURES).to_numpy()).all()
    ):
        raise ValueError("locked test feature contract is invalid")
    return metadata


def _validate_contract_artifacts(
    root: Path,
    *,
    expected_test_count: int,
    source_sha256: str,
    graph_shard_count: int,
) -> set[str]:
    raw_documents = [
        RawTransactionDocument.model_validate(payload)
        for payload in _json_payloads(root / "test_raw_documents_v1.parquet")
    ]
    raw_ids = [document.id for document in raw_documents]
    if (
        len(raw_documents) != expected_test_count
        or len(set(raw_ids)) != expected_test_count
        or any(document.data_split != "test" for document in raw_documents)
        or any(
            document.source_provenance.source_sha256 != source_sha256 for document in raw_documents
        )
    ):
        raise ValueError("raw offline evidence contract is invalid")

    actual_vertices: dict[str, dict[str, Any]] = {}
    for payload in _json_payloads(root / "test_graph_vertices_v1.parquet"):
        vertex = GraphVertex.model_validate(payload)
        previous = actual_vertices.setdefault(vertex.id, vertex.payload())
        if previous != vertex.payload():
            raise ValueError("duplicate graph vertex has conflicting immutable content")
    edge_payloads = _json_payloads(root / "test_graph_edges_v1.parquet")
    edges = [GraphEdge.model_validate(payload) for payload in edge_payloads]
    if len(edges) != 3 * expected_test_count or len({edge.id for edge in edges}) != len(edges):
        raise ValueError("graph edge count or identity contract is invalid")
    expected_vertices: dict[str, dict[str, Any]] = {}
    expected_edges: dict[str, dict[str, Any]] = {}
    for document in raw_documents:
        bundle = build_graph_evidence(document, shard_count=graph_shard_count)
        for vertex in bundle.vertices:
            previous = expected_vertices.setdefault(vertex.id, vertex.payload())
            if previous != vertex.payload():
                raise ValueError("expected graph vertex content conflicts")
        for edge in bundle.edges:
            expected_edges[edge.id] = edge.payload()
    if (
        actual_vertices != expected_vertices
        or {edge.id: edge.payload() for edge in edges} != expected_edges
    ):
        raise ValueError("graph offline evidence differs from deterministic projection")

    templates = list(iter_template_artifact(root / "test_behavior_templates_v1.parquet"))
    embeddings = list(iter_embedding_artifact(root / "test_behavior_embeddings_v1.parquet"))
    template_ids = [template.transaction_id for template in templates]
    embedding_ids = [embedding.transaction_id for embedding in embeddings]
    if (
        len(templates) != expected_test_count
        or len(set(template_ids)) != expected_test_count
        or len(embeddings) != expected_test_count
        or len(set(embedding_ids)) != expected_test_count
        or any(template.data_split != "test" for template in templates)
        or any(
            embedding.data_split != "test" or embedding.reference_eligible
            for embedding in embeddings
        )
        or set(raw_ids) != set(template_ids)
        or set(raw_ids) != set(embedding_ids)
    ):
        raise ValueError("template or embedding offline evidence contract is invalid")
    return set(raw_ids)


def _v2_manifest_payload(
    root: Path,
    historical_manifest: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    *,
    source_sha256: str,
    sample_sha256: str,
    expected_test_count: int,
) -> dict[str, Any]:
    return {
        "version": "phase8-offline-evidence-v2",
        "source_sha256": source_sha256,
        "sample_sha256": sample_sha256,
        "test_count": expected_test_count,
        "historical_manifest_v1": str(historical_manifest),
        "historical_manifest_v1_sha256": sha256_file(historical_manifest),
        "raw_artifact": str(root / "test_raw_documents_v1.parquet"),
        "raw_sha256": sha256_file(root / "test_raw_documents_v1.parquet"),
        "graph_vertex_artifact": str(root / "test_graph_vertices_v1.parquet"),
        "graph_vertex_sha256": sha256_file(root / "test_graph_vertices_v1.parquet"),
        "graph_edge_artifact": str(root / "test_graph_edges_v1.parquet"),
        "graph_edge_sha256": sha256_file(root / "test_graph_edges_v1.parquet"),
        "template_artifact": str(root / "test_behavior_templates_v1.parquet"),
        "template_sha256": sha256_file(root / "test_behavior_templates_v1.parquet"),
        "embedding_artifact": str(root / "test_behavior_embeddings_v1.parquet"),
        "embedding_sha256": metadata["test_embedding_sha256"],
        "embedding_logical_digest": metadata["test_embedding_logical_digest"],
        "feature_artifact": str(root / "locked_test_features_v2.parquet"),
        "feature_sha256": metadata["feature_sha256"],
        "feature_logical_digest": metadata["feature_logical_digest"],
        "feature_metadata_artifact": str(metadata_path),
        "feature_metadata_sha256": sha256_file(metadata_path),
        "feature_version": LOCKED_FEATURE_VERSION,
        "input_version": PHASE8_INPUT_VERSION,
        "ordered_features": list(ALL_FEATURES),
        "phase6_training_embedding_sha256": metadata["phase6_training_embedding_sha256"],
        "phase6_training_embedding_logical_digest": metadata[
            "phase6_training_embedding_logical_digest"
        ],
        "test_labels_used_for_features": False,
    }


def validate_offline_evidence_release(
    data_root: Path,
    training_embedding_path: Path,
    *,
    expected_test_count: int,
    source_sha256: str,
    sample_sha256: str,
    graph_shard_count: int,
) -> OfflineEvidenceResult:
    manifest_path = data_root / _V2_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_path = data_root / _V2_METADATA_NAME
    metadata = _authoritative_metadata(
        data_root,
        training_embedding_path,
        source_sha256=source_sha256,
        sample_sha256=sample_sha256,
        expected_test_count=expected_test_count,
    )
    expected_manifest = _v2_manifest_payload(
        data_root,
        data_root / _V1_MANIFEST_NAME,
        metadata_path,
        metadata,
        source_sha256=source_sha256,
        sample_sha256=sample_sha256,
        expected_test_count=expected_test_count,
    )
    if manifest != expected_manifest:
        raise ValueError("authoritative offline evidence manifest conflicts with artifacts")
    stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if stored_metadata != metadata:
        raise ValueError("authoritative feature metadata conflicts with feature evidence")
    historical = _validate_v1_manifest(data_root / _V1_MANIFEST_NAME)
    if (
        historical["source_sha256"] != source_sha256
        or historical["sample_sha256"] != sample_sha256
        or int(historical["test_count"]) != expected_test_count
    ):
        raise ValueError("historical manifest lineage conflicts with authoritative manifest")
    _validate_contract_artifacts(
        data_root,
        expected_test_count=expected_test_count,
        source_sha256=source_sha256,
        graph_shard_count=graph_shard_count,
    )
    return OfflineEvidenceResult(
        raw_artifact=str(data_root / "test_raw_documents_v1.parquet"),
        raw_sha256=str(manifest["raw_sha256"]),
        graph_vertex_artifact=str(data_root / "test_graph_vertices_v1.parquet"),
        graph_vertex_sha256=str(manifest["graph_vertex_sha256"]),
        graph_edge_artifact=str(data_root / "test_graph_edges_v1.parquet"),
        graph_edge_sha256=str(manifest["graph_edge_sha256"]),
        template_artifact=str(data_root / "test_behavior_templates_v1.parquet"),
        template_sha256=str(manifest["template_sha256"]),
        embedding_artifact=str(data_root / "test_behavior_embeddings_v1.parquet"),
        embedding_sha256=str(manifest["embedding_sha256"]),
        embedding_logical_digest=str(manifest["embedding_logical_digest"]),
        feature_artifact=str(data_root / "locked_test_features_v2.parquet"),
        feature_sha256=str(manifest["feature_sha256"]),
        feature_logical_digest=str(manifest["feature_logical_digest"]),
        manifest_artifact=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        test_count=expected_test_count,
        feature_version=LOCKED_FEATURE_VERSION,
        build_seconds=0.0,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.rename(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _adopt_existing_v1(
    data_root: Path,
    training_embedding_path: Path,
    *,
    source_sha256: str,
    sample_sha256: str,
    expected_test_count: int,
    graph_shard_count: int,
) -> OfflineEvidenceResult:
    historical = data_root / _V1_MANIFEST_NAME
    _validate_v1_manifest(historical)
    _validate_contract_artifacts(
        data_root,
        expected_test_count=expected_test_count,
        source_sha256=source_sha256,
        graph_shard_count=graph_shard_count,
    )
    metadata = _authoritative_metadata(
        data_root,
        training_embedding_path,
        source_sha256=source_sha256,
        sample_sha256=sample_sha256,
        expected_test_count=expected_test_count,
    )
    metadata_path = data_root / _V2_METADATA_NAME
    if metadata_path.exists():
        raise ValueError("unexpected authoritative feature metadata exists without manifest")
    _write_json_atomic(metadata_path, metadata)
    manifest_path = data_root / _V2_MANIFEST_NAME
    payload = _v2_manifest_payload(
        data_root,
        historical,
        metadata_path,
        metadata,
        source_sha256=source_sha256,
        sample_sha256=sample_sha256,
        expected_test_count=expected_test_count,
    )
    _write_json_atomic(manifest_path, payload)
    return validate_offline_evidence_release(
        data_root,
        training_embedding_path,
        expected_test_count=expected_test_count,
        source_sha256=source_sha256,
        sample_sha256=sample_sha256,
        graph_shard_count=graph_shard_count,
    )


def prepare_offline_evidence(
    source_path: Path,
    manifest_path: Path,
    training_embedding_path: Path,
    data_root: Path,
    settings: Settings,
    *,
    encoder: BehavioralEncoder | None = None,
    expected_sample_count: int = 200_000,
    expected_test_count: int = 40_000,
) -> OfflineEvidenceResult:
    manifest = read_manifest(manifest_path)
    authoritative = data_root / _V2_MANIFEST_NAME
    historical = data_root / _V1_MANIFEST_NAME
    if authoritative.exists():
        return validate_offline_evidence_release(
            data_root,
            training_embedding_path,
            expected_test_count=expected_test_count,
            source_sha256=manifest.source_sha256,
            sample_sha256=manifest.sample_sha256,
            graph_shard_count=settings.gremlin_partition_shards,
        )
    if historical.exists():
        return _adopt_existing_v1(
            data_root,
            training_embedding_path,
            source_sha256=manifest.source_sha256,
            sample_sha256=manifest.sample_sha256,
            expected_test_count=expected_test_count,
            graph_shard_count=settings.gremlin_partition_shards,
        )
    unexpected = [
        name for name in (*_ARTIFACT_NAMES, _V2_METADATA_NAME) if (data_root / name).exists()
    ]
    if unexpected:
        raise ValueError(
            "unexpected offline artifact paths exist without a manifest: "
            + ",".join(sorted(unexpected))
        )
    data_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{data_root.name}.stage-", dir=data_root.parent))
    try:
        staged = _build_offline_evidence_unpublished(
            source_path,
            manifest_path,
            training_embedding_path,
            stage,
            settings,
            encoder=encoder,
            expected_sample_count=expected_sample_count,
            expected_test_count=expected_test_count,
        )
        _validate_v1_manifest(Path(staged.manifest_artifact))
        _validate_contract_artifacts(
            stage,
            expected_test_count=expected_test_count,
            source_sha256=manifest.source_sha256,
            graph_shard_count=settings.gremlin_partition_shards,
        )
        data_root.mkdir(parents=True, exist_ok=False)
        for name in _ARTIFACT_NAMES:
            (stage / name).rename(data_root / name)
        v1_payload = json.loads((stage / _V1_MANIFEST_NAME).read_text(encoding="utf-8"))
        for field in (
            "raw_artifact",
            "graph_vertex_artifact",
            "graph_edge_artifact",
            "template_artifact",
            "embedding_artifact",
            "feature_artifact",
        ):
            v1_payload[field] = str(data_root / Path(str(v1_payload[field])).name)
        _write_json_atomic(data_root / _V1_MANIFEST_NAME, v1_payload)
        return _adopt_existing_v1(
            data_root,
            training_embedding_path,
            source_sha256=manifest.source_sha256,
            sample_sha256=manifest.sample_sha256,
            expected_test_count=expected_test_count,
            graph_shard_count=settings.gremlin_partition_shards,
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)
