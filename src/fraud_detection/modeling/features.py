"""Causal train/validation feature artifact construction."""

from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sqlalchemy import Engine, create_engine, select

from fraud_detection.ingestion.hashing import canonical_json_bytes, sha256_file
from fraud_detection.ingestion.loader import initial_batch_id
from fraud_detection.ingestion.manifest import (
    read_manifest,
    sample_path_from_manifest,
)
from fraud_detection.persistence.document_vector.artifacts import (
    embedding_artifact_logical_digest,
)
from fraud_detection.persistence.graph.analysis import (
    SHARED_DEVICE_WEIGHT,
    SHARED_IP_WEIGHT,
    TRANSACTION_WEIGHT,
)
from fraud_detection.persistence.graph.models import deterministic_artifact_id
from fraud_detection.persistence.relational import Base

from .contracts import (
    ALL_FEATURES,
    COMBINED_FEATURE_VERSION,
    GRAPH_FEATURE_VERSION,
    PHASE6_LOGICAL_EMBEDDING_DIGEST,
    TABULAR_FEATURE_VERSION,
    TEST_MIN_STEP,
    TRAIN_MAX_STEP,
    TRAIN_MIN_STEP,
    VALIDATION_MAX_STEP,
    VALIDATION_MIN_STEP,
    VECTOR_FEATURE_VERSION,
    FeatureArtifactResult,
    validate_predictor_allowlist,
)

transaction = Base.metadata.tables["transaction"]
transaction_label = Base.metadata.tables["transaction_label"]


def _tabular_features(row: Mapping[str, Any]) -> dict[str, float]:
    transaction_type = str(row["type"]).upper()
    step = int(row["step"])
    return {
        "amount_log1p": math.log1p(float(row["amount"])),
        "simulation_hour": float((step - 1) % 24),
        "simulation_day": float((step - 1) // 24),
        "type_cash_in": float(transaction_type == "CASH_IN"),
        "type_cash_out": float(transaction_type == "CASH_OUT"),
        "type_debit": float(transaction_type == "DEBIT"),
        "type_payment": float(transaction_type == "PAYMENT"),
        "type_transfer": float(transaction_type == "TRANSFER"),
    }


def _pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


@dataclass
class CausalGraphState:
    """Incremental state equivalent to the Phase 5 historical account graph."""

    neighbors: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    weighted_degree: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    transaction_count: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    pair_devices: dict[tuple[str, str], set[str]] = field(default_factory=lambda: defaultdict(set))
    pair_ips: dict[tuple[str, str], set[str]] = field(default_factory=lambda: defaultdict(set))
    device_accounts: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    ip_accounts: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    nodes: set[str] = field(default_factory=set)

    def feature_values(
        self, sender: str, receiver: str, device_id: str, ip_id: str
    ) -> dict[str, float]:
        node_count = len(self.nodes)
        denominator = node_count - 1 if node_count >= 2 else 1
        pair = _pair(sender, receiver)
        return {
            "sender_prior_degree_centrality": (
                len(self.neighbors[sender]) / denominator if node_count >= 2 else 0.0
            ),
            "receiver_prior_degree_centrality": (
                len(self.neighbors[receiver]) / denominator if node_count >= 2 else 0.0
            ),
            "sender_prior_weighted_degree": self.weighted_degree[sender],
            "receiver_prior_weighted_degree": self.weighted_degree[receiver],
            "prior_sender_receiver_transaction_count": float(
                self.transaction_count[pair] if sender != receiver else 0
            ),
            "prior_device_account_count": float(len(self.device_accounts[device_id])),
            "prior_ip_account_count": float(len(self.ip_accounts[ip_id])),
            "sender_is_new": float(sender not in self.nodes),
            "receiver_is_new": float(receiver not in self.nodes),
        }

    def _add_pair_evidence(
        self,
        left: str,
        right: str,
        artifact_id: str,
        evidence: dict[tuple[str, str], set[str]],
        weight: float,
    ) -> None:
        if left == right:
            return
        pair = _pair(left, right)
        if artifact_id in evidence[pair]:
            return
        evidence[pair].add(artifact_id)
        self.neighbors[left].add(right)
        self.neighbors[right].add(left)
        self.weighted_degree[left] += weight
        self.weighted_degree[right] += weight

    def _add_artifact_observation(
        self,
        account: str,
        artifact_id: str,
        accounts: dict[str, set[str]],
        evidence: dict[tuple[str, str], set[str]],
        weight: float,
    ) -> None:
        if account in accounts[artifact_id]:
            return
        for other in sorted(accounts[artifact_id]):
            self._add_pair_evidence(account, other, artifact_id, evidence, weight)
        accounts[artifact_id].add(account)

    def add_transaction(self, sender: str, receiver: str, device_id: str, ip_id: str) -> None:
        self.nodes.update((sender, receiver))
        if sender != receiver:
            pair = _pair(sender, receiver)
            self.transaction_count[pair] += 1
            self.neighbors[sender].add(receiver)
            self.neighbors[receiver].add(sender)
            self.weighted_degree[sender] += TRANSACTION_WEIGHT
            self.weighted_degree[receiver] += TRANSACTION_WEIGHT
        self._add_artifact_observation(
            sender,
            device_id,
            self.device_accounts,
            self.pair_devices,
            SHARED_DEVICE_WEIGHT,
        )
        self._add_artifact_observation(
            sender,
            ip_id,
            self.ip_accounts,
            self.pair_ips,
            SHARED_IP_WEIGHT,
        )


def build_causal_graph_feature_rows(
    rows: Iterable[Mapping[str, Any]], *, enrichment_seed: int
) -> dict[str, dict[str, float]]:
    """Build strict-prior graph features with whole-step isolation."""
    ordered = sorted(rows, key=lambda row: (int(row["step"]), str(row["transaction_id"])))
    state = CausalGraphState()
    output: dict[str, dict[str, float]] = {}
    start = 0
    while start < len(ordered):
        step = int(ordered[start]["step"])
        end = start
        while end < len(ordered) and int(ordered[end]["step"]) == step:
            end += 1
        group = ordered[start:end]
        artifacts: list[tuple[str, str, str]] = []
        for row in group:
            transaction_id = str(row["transaction_id"])
            sender = str(row["nameOrig"])
            receiver = str(row["nameDest"])
            device_id = deterministic_artifact_id(
                "device",
                transaction_id=transaction_id,
                source_row=int(row["source_row"]),
                step=step,
                seed=enrichment_seed,
            )
            ip_id = deterministic_artifact_id(
                "ip",
                transaction_id=transaction_id,
                source_row=int(row["source_row"]),
                step=step,
                seed=enrichment_seed,
            )
            output[transaction_id] = state.feature_values(sender, receiver, device_id, ip_id)
            artifacts.append((sender, device_id, ip_id))
        for row, (sender, device_id, ip_id) in zip(group, artifacts, strict=True):
            receiver = str(row["nameDest"])
            state.add_transaction(sender, receiver, device_id, ip_id)
        start = end
    return output


def _embedding_arrays(
    path: Path,
) -> tuple[list[str], np.ndarray[Any, np.dtype[np.float32]], dict[str, list[Any]]]:
    table = pq.read_table(
        path,
        columns=[
            "transaction_id",
            "step",
            "data_split",
            "reference_eligible",
            "embedding_version",
            "template_version",
            "model_revision",
            "vector",
        ],
    ).combine_chunks()
    identifiers = [str(value) for value in table.column("transaction_id").to_pylist()]
    vector_values = table.column("vector").chunk(0).values.to_numpy(zero_copy_only=False)
    vectors = np.asarray(vector_values, dtype=np.float32).reshape(len(identifiers), 384)
    metadata = {
        name: table.column(name).to_pylist()
        for name in (
            "step",
            "data_split",
            "reference_eligible",
            "embedding_version",
            "template_version",
            "model_revision",
        )
    }
    return identifiers, vectors, metadata


def build_vector_feature_rows(
    rows: Iterable[Mapping[str, Any]],
    embedding_path: Path,
    *,
    expected_logical_digest: str = PHASE6_LOGICAL_EMBEDDING_DIGEST,
    query_batch_size: int = 512,
) -> dict[str, dict[str, float]]:
    if embedding_artifact_logical_digest(embedding_path) != expected_logical_digest:
        raise ValueError("embedding logical digest does not match the approved Phase 6 artifact")
    ordered = sorted(rows, key=lambda row: (int(row["step"]), str(row["transaction_id"])))
    identifiers, vectors, metadata = _embedding_arrays(embedding_path)
    index = {identifier: position for position, identifier in enumerate(identifiers)}
    expected_ids = {str(row["transaction_id"]) for row in ordered}
    if set(index) != expected_ids or len(index) != len(identifiers):
        raise ValueError("embedding and feature transaction-ID sets differ")
    for row in ordered:
        position = index[str(row["transaction_id"])]
        if int(metadata["step"][position]) != int(row["step"]) or str(
            metadata["data_split"][position]
        ) != str(row["data_split"]):
            raise ValueError("embedding step or split differs from validated feature input")
        expected_reference = str(row["data_split"]) == "train" and bool(row["isFraud"])
        if bool(metadata["reference_eligible"][position]) != expected_reference:
            raise ValueError("embedding reference eligibility differs from protected source truth")

    output: dict[str, dict[str, float]] = {}
    reference_positions: list[int] = []
    start = 0
    while start < len(ordered):
        step = int(ordered[start]["step"])
        end = start
        while end < len(ordered) and int(ordered[end]["step"]) == step:
            end += 1
        group = ordered[start:end]
        group_positions = [index[str(row["transaction_id"])] for row in group]
        if not reference_positions:
            for row in group:
                output[str(row["transaction_id"])] = {
                    "nearest_earlier_training_fraud_cosine_distance": 1.0,
                    "missing_earlier_training_fraud_reference": 1.0,
                }
        else:
            reference_matrix = vectors[np.asarray(reference_positions)]
            for offset in range(0, len(group), query_batch_size):
                positions = group_positions[offset : offset + query_batch_size]
                similarities = vectors[np.asarray(positions)] @ reference_matrix.T
                distances = np.maximum(0.0, 1.0 - np.max(similarities, axis=1))
                for row, distance in zip(
                    group[offset : offset + query_batch_size], distances, strict=True
                ):
                    output[str(row["transaction_id"])] = {
                        "nearest_earlier_training_fraud_cosine_distance": float(distance),
                        "missing_earlier_training_fraud_reference": 0.0,
                    }
        reference_positions.extend(
            position
            for position in group_positions
            if bool(metadata["reference_eligible"][position])
        )
        start = end
    return output


def _validate_postgres(rows: list[dict[str, Any]], database_url: str, batch_id: str) -> None:
    engine: Engine = create_engine(database_url, hide_parameters=True)
    statement = (
        select(
            transaction.c.transaction_id,
            transaction.c.batch_id,
            transaction.c.step,
            transaction.c.transaction_type,
            transaction.c.amount,
            transaction.c.data_split,
            transaction_label.c.is_fraud,
        )
        .join(transaction_label)
        .where(transaction.c.batch_id == batch_id)
    )
    try:
        with engine.connect() as connection:
            stored = {
                str(value["transaction_id"]): value
                for value in connection.execute(statement).mappings()
            }
    finally:
        engine.dispose()
    expected = {str(row["transaction_id"]): row for row in rows}
    if set(stored) != set(expected):
        raise ValueError("PostgreSQL and feature transaction-ID sets differ")
    for identifier, row in expected.items():
        actual = stored[identifier]
        if any(
            (
                str(actual["batch_id"]) != batch_id,
                int(actual["step"]) != int(row["step"]),
                str(actual["transaction_type"]) != str(row["type"]),
                float(actual["amount"]) != float(row["amount"]),
                str(actual["data_split"]) != str(row["data_split"]),
                bool(actual["is_fraud"]) != bool(row["isFraud"]),
            )
        ):
            raise ValueError(f"PostgreSQL feature source mismatch for {identifier}")


def feature_logical_digest(frame: pl.DataFrame) -> str:
    ordered_columns = list(frame.columns)
    digest = hashlib.sha256(canonical_json_bytes({"columns": ordered_columns}))
    for row in frame.iter_rows(named=True):
        digest.update(b"\n")
        digest.update(canonical_json_bytes(row))
    return digest.hexdigest()


def build_feature_artifact(
    source_path: Path,
    manifest_path: Path,
    embedding_path: Path,
    database_url: str,
    output_path: Path,
    *,
    expected_train_count: int = 120_000,
    expected_validation_count: int = 40_000,
) -> FeatureArtifactResult:
    started = time.perf_counter()
    validate_predictor_allowlist(ALL_FEATURES)
    manifest = read_manifest(manifest_path)
    if source_path.name != manifest.source_filename:
        raise ValueError("source filename differs from the approved manifest")
    if source_path.stat().st_size != manifest.source_byte_size:
        raise ValueError("source size differs from the approved manifest")
    if sha256_file(source_path) != manifest.source_sha256:
        raise ValueError("source hash differs from the approved manifest")
    sample_path = sample_path_from_manifest(manifest, manifest_path)
    if sha256_file(sample_path) != manifest.sample_sha256:
        raise ValueError("sample hash differs from the approved manifest")
    if (
        manifest.split("train").minimum_step != TRAIN_MIN_STEP
        or manifest.split("train").maximum_step != TRAIN_MAX_STEP
        or manifest.split("validation").minimum_step != VALIDATION_MIN_STEP
        or manifest.split("validation").maximum_step != VALIDATION_MAX_STEP
        or manifest.split("test").minimum_step != TEST_MIN_STEP
    ):
        raise ValueError("manifest split boundaries differ from the approved Phase 7 contract")
    sample = (
        pl.scan_parquet(sample_path)
        .filter(pl.col("data_split").is_in(["train", "validation"]))
        .sort(["step", "transaction_id"])
        .collect(engine="streaming")
    )
    if (
        sample.height != expected_train_count + expected_validation_count
        or sample.filter(pl.col("data_split") == "train").height != expected_train_count
        or sample.filter(pl.col("data_split") == "validation").height != expected_validation_count
        or sample["transaction_id"].n_unique() != sample.height
    ):
        raise ValueError("Phase 7 feature input has incorrect train/validation identities")
    if sample.filter(pl.col("step") >= TEST_MIN_STEP).height:
        raise ValueError("Phase 7 feature input cannot read reserved test steps")
    rows = sample.to_dicts()
    batch_id = initial_batch_id(manifest)
    _validate_postgres(rows, database_url, batch_id)
    graph = build_causal_graph_feature_rows(rows, enrichment_seed=manifest.enrichment_seed)
    vector = build_vector_feature_rows(rows, embedding_path)
    records: list[dict[str, Any]] = []
    for row in rows:
        identifier = str(row["transaction_id"])
        record: dict[str, Any] = {
            "transaction_id": identifier,
            "batch_id": batch_id,
            "step": int(row["step"]),
            "data_split": str(row["data_split"]),
            "is_fraud": bool(row["isFraud"]),
        }
        record.update(_tabular_features(row))
        record.update(graph[identifier])
        record.update(vector[identifier])
        records.append(record)
    ordered_columns = [
        "transaction_id",
        "batch_id",
        "step",
        "data_split",
        "is_fraud",
        *ALL_FEATURES,
    ]
    frame = pl.DataFrame(records).select(ordered_columns).sort(["step", "transaction_id"])
    numeric = frame.select(ALL_FEATURES).to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError("feature artifact contains NaN or infinite predictor values")
    logical_digest = feature_logical_digest(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(output_path)
    feature_types = {name: str(frame.schema[name]) for name in ALL_FEATURES}
    missing_rules = {
        "nearest_earlier_training_fraud_cosine_distance": "1.0 when no eligible prior reference",
        "missing_earlier_training_fraud_reference": "1.0 iff no eligible prior reference",
        "all_other_features": "zero-valued counts/indicators; no nulls permitted",
    }
    result = FeatureArtifactResult(
        artifact_path=str(output_path),
        metadata_path=str(output_path.with_suffix(".metadata.json")),
        physical_sha256=sha256_file(output_path),
        logical_digest=logical_digest,
        byte_size=output_path.stat().st_size,
        row_count=frame.height,
        train_count=frame.filter(pl.col("data_split") == "train").height,
        validation_count=frame.filter(pl.col("data_split") == "validation").height,
        test_count=0,
        ordered_features=ALL_FEATURES,
        feature_types=feature_types,
        missing_value_rules=missing_rules,
        tabular_version=TABULAR_FEATURE_VERSION,
        graph_version=GRAPH_FEATURE_VERSION,
        vector_version=VECTOR_FEATURE_VERSION,
        combined_version=COMBINED_FEATURE_VERSION,
        embedding_logical_digest=PHASE6_LOGICAL_EMBEDDING_DIGEST,
        source_sha256=manifest.source_sha256,
        sample_sha256=manifest.sample_sha256,
        build_seconds=time.perf_counter() - started,
    )
    metadata_path = Path(result.metadata_path)
    metadata_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def read_and_validate_feature_artifact(
    artifact_path: Path,
    metadata_path: Path,
    *,
    expected_train_count: int = 120_000,
    expected_validation_count: int = 40_000,
) -> tuple[pl.DataFrame, FeatureArtifactResult]:
    result = FeatureArtifactResult.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(artifact_path) != result.physical_sha256:
        raise ValueError("feature artifact physical hash mismatch")
    frame = pl.read_parquet(artifact_path).sort(["step", "transaction_id"])
    if feature_logical_digest(frame) != result.logical_digest:
        raise ValueError("feature artifact logical digest mismatch")
    if (
        frame.height != expected_train_count + expected_validation_count
        or frame.filter(pl.col("data_split") == "train").height != expected_train_count
        or frame.filter(pl.col("data_split") == "validation").height != expected_validation_count
        or frame.filter(pl.col("data_split") == "test").height
    ):
        raise ValueError("feature artifact must contain train/validation rows only")
    validate_predictor_allowlist(result.ordered_features)
    return frame, result
