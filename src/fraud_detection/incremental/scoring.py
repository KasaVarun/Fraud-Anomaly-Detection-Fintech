"""Locked-model Phase 8 scoring, flags, snapshots, and evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, confusion_matrix
from sqlalchemy import Connection, Engine, create_engine, func, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert

from fraud_detection.ingestion.hashing import canonical_json_bytes, sha256_file
from fraud_detection.modeling.artifacts import load_trusted_model_artifact
from fraud_detection.modeling.contracts import ALL_FEATURES
from fraud_detection.modeling.features import feature_logical_digest
from fraud_detection.persistence.document_vector.artifacts import (
    embedding_artifact_logical_digest,
)
from fraud_detection.persistence.relational import Base

from .contracts import (
    LOCKED_FEATURE_LOGICAL_DIGEST,
    LOCKED_FEATURE_VERSION,
    LOCKED_METADATA_SHA256,
    LOCKED_MODEL_ID,
    LOCKED_MODEL_SHA256,
    LOCKED_THRESHOLD,
    LOCKED_TRAINING_CUTOFF,
    PHASE8_EVALUATION_VERSION,
    PHASE8_INPUT_VERSION,
    PHASE8_SNAPSHOT_VERSION,
    ChunkPlan,
    ChunkRunResult,
    LockedMetrics,
    TestChunk,
)
from .loader import load_test_chunk
from .planning import reserved_test_rows, rows_for_chunk

ingestion_batch = Base.metadata.tables["ingestion_batch"]
transaction = Base.metadata.tables["transaction"]
transaction_label = Base.metadata.tables["transaction_label"]
model_version = Base.metadata.tables["model_version"]
graph_snapshot = Base.metadata.tables["graph_snapshot"]
transaction_score = Base.metadata.tables["transaction_score"]
fraud_flag = Base.metadata.tables["fraud_flag"]
model_evaluation = Base.metadata.tables["model_evaluation"]

StageHook = Callable[[str], None]


def _validated_scoring_metadata(
    feature_artifact: Path,
    feature_metadata_path: Path,
    plan: ChunkPlan,
) -> dict[str, Any]:
    manifest_path = feature_metadata_path.parent / "offline_evidence_manifest_v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(feature_metadata_path.read_text(encoding="utf-8"))
    if (
        manifest.get("version") != "phase8-offline-evidence-v2"
        or Path(str(manifest.get("feature_metadata_artifact"))) != feature_metadata_path
        or sha256_file(feature_metadata_path) != manifest.get("feature_metadata_sha256")
        or Path(str(manifest.get("feature_artifact"))) != feature_artifact
    ):
        raise ValueError("authoritative feature metadata is not anchored by the evidence manifest")
    frame = pl.read_parquet(feature_artifact).sort(["step", "transaction_id"])
    feature_digest = feature_logical_digest(frame)
    test_embedding_path = Path(str(manifest["embedding_artifact"]))
    required = {
        "version": "phase8-locked-test-feature-metadata-v2",
        "input_version": PHASE8_INPUT_VERSION,
        "feature_version": LOCKED_FEATURE_VERSION,
        "ordered_features": list(ALL_FEATURES),
        "feature_sha256": sha256_file(feature_artifact),
        "feature_logical_digest": feature_digest,
        "test_count": plan.test_row_count,
        "source_sha256": plan.source_sha256,
        "sample_sha256": plan.sample_sha256,
        "test_embedding_sha256": sha256_file(test_embedding_path),
        "test_embedding_logical_digest": embedding_artifact_logical_digest(test_embedding_path),
        "phase6_training_embedding_sha256": (
            "32f79267f8d65077983adf3735c94541c1993929212108be8d819ced1b5c84a0"
        ),
        "phase6_training_embedding_logical_digest": (
            "50c904d3a82dc916eb389196ce35e8becb2b3bef747ec79cd91f893503d1f767"
        ),
        "test_labels_used_for_features": False,
    }
    if metadata != required:
        raise ValueError("authoritative locked-test feature metadata is invalid")
    return cast(dict[str, Any], metadata)


def locked_metrics(
    labels: np.ndarray[Any, np.dtype[np.int8]], scores: np.ndarray[Any, np.dtype[np.float64]]
) -> LockedMetrics:
    predictions = scores >= float(LOCKED_THRESHOLD)
    tn, fp, fn, tp = (
        int(value) for value in confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    predicted = int(predictions.sum())
    return LockedMetrics(
        row_count=len(labels),
        fraud_count=int(labels.sum()),
        nonfraud_count=int(len(labels) - labels.sum()),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=float(average_precision_score(labels, scores)),
        predicted_fraud_count=predicted,
        flag_rate=predicted / len(labels) if len(labels) else 0.0,
        score_minimum=float(scores.min()) if len(scores) else 0.0,
        score_maximum=float(scores.max()) if len(scores) else 0.0,
        score_mean=float(scores.mean()) if len(scores) else 0.0,
    )


def _hash_id(prefix: str, payload: dict[str, object]) -> str:
    return f"{prefix}:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def snapshot_id(chunk: TestChunk, transaction_step: int, source_sha256: str) -> str:
    return _hash_id(
        "snapshot",
        {
            "batch_id": chunk.batch_id,
            "latest_evidence_step": transaction_step - 1,
            "transaction_step": transaction_step,
            "source_sha256": source_sha256,
            "feature_version": LOCKED_FEATURE_VERSION,
            "snapshot_version": PHASE8_SNAPSHOT_VERSION,
        },
    )


def input_version(test_feature_digest: str, transaction_step: int) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "phase8_input_version": PHASE8_INPUT_VERSION,
                "test_feature_digest": test_feature_digest,
                "feature_version": LOCKED_FEATURE_VERSION,
                "transaction_step": transaction_step,
            }
        )
    ).hexdigest()
    return f"{PHASE8_INPUT_VERSION}:{digest[:32]}"


def score_id(transaction_id: str, input_version_value: str) -> str:
    return _hash_id(
        "score",
        {
            "transaction_id": transaction_id,
            "model_version_id": LOCKED_MODEL_ID,
            "input_version": input_version_value,
        },
    )


def flag_id(score_id_value: str) -> str:
    return _hash_id("flag", {"score_id": score_id_value, "threshold": str(LOCKED_THRESHOLD)})


def _insert_verify(
    connection: Connection,
    table: Any,
    record: dict[str, Any],
    key: str,
    immutable_fields: tuple[str, ...],
) -> None:
    connection.execute(
        postgres_insert(table).values(record).on_conflict_do_nothing(index_elements=[key])
    )
    stored = connection.execute(select(table).where(table.c[key] == record[key])).mappings().one()
    for field in immutable_fields:
        if isinstance(record[field], Decimal | float):
            differs = Decimal(str(stored[field])) != Decimal(str(record[field]))
        else:
            differs = stored[field] != record[field]
        if differs:
            raise ValueError(f"immutable {table.name} conflicts on {field}")


def _verify_locked_model(database_url: str, artifact_root: Path) -> tuple[Any, dict[str, Any]]:
    model, metadata = load_trusted_model_artifact(
        artifact_root,
        LOCKED_MODEL_ID,
        expected_feature_logical_digest=LOCKED_FEATURE_LOGICAL_DIGEST,
        expected_model_sha256=LOCKED_MODEL_SHA256,
        expected_metadata_sha256=LOCKED_METADATA_SHA256,
    )
    required = {
        "model_version_id": LOCKED_MODEL_ID,
        "feature_logical_digest": LOCKED_FEATURE_LOGICAL_DIGEST,
        "threshold": str(LOCKED_THRESHOLD),
        "training_steps": f"1-{LOCKED_TRAINING_CUTOFF}",
        "embedding_logical_digest": (
            "50c904d3a82dc916eb389196ce35e8becb2b3bef747ec79cd91f893503d1f767"
        ),
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("locked model metadata lineage is invalid")
    if tuple(metadata.get("ordered_features", ())) != ALL_FEATURES:
        raise ValueError("locked model feature ordering is invalid")
    if metadata.get("feature_versions", {}).get("combined") != LOCKED_FEATURE_VERSION:
        raise ValueError("locked model combined feature version is invalid")
    engine: Engine = create_engine(database_url, hide_parameters=True)
    try:
        with engine.connect() as connection:
            stored = (
                connection.execute(
                    select(model_version).where(model_version.c.model_version_id == LOCKED_MODEL_ID)
                )
                .mappings()
                .one()
            )
        if (
            Decimal(str(stored["threshold"])) != LOCKED_THRESHOLD
            or int(stored["training_cutoff_step"]) != LOCKED_TRAINING_CUTOFF
            or str(stored["feature_set"]) != "combined"
        ):
            raise ValueError("locked PostgreSQL model record conflicts with release")
    finally:
        engine.dispose()
    return model, metadata


def _assert_previous_chunks_complete(
    connection: Connection, plan: ChunkPlan, chunk: TestChunk
) -> None:
    for previous in plan.chunks[: chunk.index]:
        scored = int(
            connection.scalar(
                select(func.count())
                .select_from(transaction_score)
                .join(transaction)
                .where(transaction.c.batch_id == previous.batch_id)
                .where(transaction_score.c.model_version_id == LOCKED_MODEL_ID)
            )
            or 0
        )
        evaluated = int(
            connection.scalar(
                select(func.count())
                .select_from(model_evaluation)
                .where(
                    model_evaluation.c.batch_id == previous.batch_id,
                    model_evaluation.c.model_version_id == LOCKED_MODEL_ID,
                    model_evaluation.c.split_name == "test",
                )
            )
            or 0
        )
        if scored != previous.row_count or evaluated != 1:
            raise ValueError(
                f"earlier chunk {previous.index} must complete before chunk {chunk.index}"
            )


def run_offline_chunk(
    source_path: Path,
    manifest_path: Path,
    feature_artifact: Path,
    feature_metadata_path: Path,
    database_url: str,
    artifact_root: Path,
    plan: ChunkPlan,
    chunk: TestChunk,
    *,
    stage_hook: StageHook | None = None,
    expected_test_count: int = 40_000,
    verify_authoritative_metadata: bool = True,
) -> ChunkRunResult:
    all_test_rows, manifest = reserved_test_rows(
        source_path, manifest_path, expected_test_count=expected_test_count
    )
    metadata = (
        _validated_scoring_metadata(feature_artifact, feature_metadata_path, plan)
        if verify_authoritative_metadata
        else json.loads(feature_metadata_path.read_text(encoding="utf-8"))
    )
    source_rows = rows_for_chunk(all_test_rows, chunk)
    transaction_count, label_count, _ = load_test_chunk(database_url, source_rows, chunk, manifest)
    if stage_hook:
        stage_hook("ingestion")
    if sha256_file(feature_artifact) != metadata.get("feature_sha256"):
        raise ValueError("locked test feature artifact hash mismatch")
    if metadata.get("feature_version") != LOCKED_FEATURE_VERSION:
        raise ValueError("locked test feature version mismatch")
    test_feature_digest = str(metadata["feature_logical_digest"])
    frame = (
        pl.scan_parquet(feature_artifact)
        .filter(pl.col("batch_id") == chunk.batch_id)
        .sort(["step", "transaction_id"])
        .collect(engine="streaming")
    )
    if frame.height != chunk.row_count or frame.filter(pl.col("data_split") != "test").height:
        raise ValueError("locked test feature chunk is incomplete")
    model, _ = _verify_locked_model(database_url, artifact_root)
    matrix = frame.select(ALL_FEATURES).to_pandas()
    probabilities = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    probability_by_id = dict(zip(frame["transaction_id"].to_list(), probabilities, strict=True))
    engine: Engine = create_engine(database_url, hide_parameters=True)
    try:
        with engine.begin() as connection:
            _assert_previous_chunks_complete(connection, plan, chunk)
            snapshot_by_step: dict[int, str] = {}
            for step in sorted(int(value) for value in frame["step"].unique().to_list()):
                identifier = snapshot_id(chunk, step, plan.source_sha256)
                snapshot_by_step[step] = identifier
                record = {
                    "snapshot_id": identifier,
                    "batch_id": chunk.batch_id,
                    "cutoff_step": step - 1,
                    "algorithm_version": (
                        f"{PHASE8_SNAPSHOT_VERSION};evidence_step_lt={step};"
                        f"feature_version={LOCKED_FEATURE_VERSION};source={plan.source_sha256}"
                    ),
                    "computed_at": datetime.now(UTC),
                }
                _insert_verify(
                    connection,
                    graph_snapshot,
                    record,
                    "snapshot_id",
                    ("batch_id", "cutoff_step", "algorithm_version"),
                )
            if stage_hook:
                stage_hook("snapshots")
            score_records: list[dict[str, Any]] = []
            for row in frame.iter_rows(named=True):
                version = input_version(test_feature_digest, int(row["step"]))
                identifier = score_id(str(row["transaction_id"]), version)
                probability = Decimal(str(probability_by_id[str(row["transaction_id"])])).quantize(
                    Decimal("0.0000001"), rounding=ROUND_HALF_UP
                )
                record = {
                    "score_id": identifier,
                    "transaction_id": str(row["transaction_id"]),
                    "model_version_id": LOCKED_MODEL_ID,
                    "snapshot_id": snapshot_by_step[int(row["step"])],
                    "input_version": version,
                    "fraud_probability": probability,
                    "vector_distance": float(row["nearest_earlier_training_fraud_cosine_distance"]),
                    "scored_at": datetime.now(UTC),
                }
                _insert_verify(
                    connection,
                    transaction_score,
                    record,
                    "score_id",
                    (
                        "transaction_id",
                        "model_version_id",
                        "snapshot_id",
                        "input_version",
                        "fraud_probability",
                        "vector_distance",
                    ),
                )
                score_records.append(record)
            if stage_hook:
                stage_hook("scores")
            for record in score_records:
                if Decimal(str(record["fraud_probability"])) < LOCKED_THRESHOLD:
                    continue
                flag_record = {
                    "flag_id": flag_id(str(record["score_id"])),
                    "score_id": record["score_id"],
                    "reason": (
                        f"model={LOCKED_MODEL_ID};threshold={LOCKED_THRESHOLD};"
                        "decision=probability_at_or_above_threshold"
                    ),
                    "review_status": "open",
                }
                _insert_verify(
                    connection,
                    fraud_flag,
                    flag_record,
                    "flag_id",
                    ("score_id", "reason", "review_status"),
                )
            if stage_hook:
                stage_hook("flags")
            joined = (
                connection.execute(
                    select(
                        transaction_score.c.transaction_id,
                        transaction_score.c.fraud_probability,
                        transaction_label.c.is_fraud,
                    )
                    .join(
                        transaction,
                        transaction.c.transaction_id == transaction_score.c.transaction_id,
                    )
                    .join(transaction_label)
                    .where(transaction.c.batch_id == chunk.batch_id)
                    .where(transaction_score.c.model_version_id == LOCKED_MODEL_ID)
                    .order_by(transaction_score.c.transaction_id)
                )
                .mappings()
                .all()
            )
            labels = np.asarray([bool(row["is_fraud"]) for row in joined], dtype=np.int8)
            stored_scores = np.asarray(
                [float(row["fraud_probability"]) for row in joined], dtype=np.float64
            )
            metrics = locked_metrics(labels, stored_scores)
            evaluation_id = _hash_id(
                "evaluation",
                {
                    "version": PHASE8_EVALUATION_VERSION,
                    "model_version_id": LOCKED_MODEL_ID,
                    "batch_id": chunk.batch_id,
                    "split": "test",
                },
            )
            evaluation = {
                "evaluation_id": evaluation_id,
                "model_version_id": LOCKED_MODEL_ID,
                "batch_id": chunk.batch_id,
                "split_name": "test",
                "tp": metrics.tp,
                "fp": metrics.fp,
                "tn": metrics.tn,
                "fn": metrics.fn,
                "precision": Decimal(str(metrics.precision)),
                "recall": Decimal(str(metrics.recall)),
                "f1": Decimal(str(metrics.f1)),
                "pr_auc": Decimal(str(metrics.pr_auc)),
            }
            _insert_verify(
                connection,
                model_evaluation,
                evaluation,
                "evaluation_id",
                (
                    "model_version_id",
                    "batch_id",
                    "split_name",
                    "tp",
                    "fp",
                    "tn",
                    "fn",
                    "precision",
                    "recall",
                    "f1",
                    "pr_auc",
                ),
            )
            if stage_hook:
                stage_hook("evaluation")
            score_count = int(
                connection.scalar(
                    select(func.count())
                    .select_from(transaction_score)
                    .join(transaction)
                    .where(transaction.c.batch_id == chunk.batch_id)
                    .where(transaction_score.c.model_version_id == LOCKED_MODEL_ID)
                )
                or 0
            )
            flag_count = int(
                connection.scalar(
                    select(func.count())
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
            evaluation_count = int(
                connection.scalar(
                    select(func.count())
                    .select_from(model_evaluation)
                    .where(
                        model_evaluation.c.batch_id == chunk.batch_id,
                        model_evaluation.c.model_version_id == LOCKED_MODEL_ID,
                    )
                )
                or 0
            )
            status = str(
                connection.scalar(
                    select(ingestion_batch.c.status).where(
                        ingestion_batch.c.batch_id == chunk.batch_id
                    )
                )
            )
        return ChunkRunResult(
            chunk=chunk,
            batch_status=status,
            transaction_count=transaction_count,
            label_count=label_count,
            score_count=score_count,
            flag_count=flag_count,
            snapshot_count=snapshot_count,
            evaluation_count=evaluation_count,
            metrics=metrics,
            live_projection=False,
        )
    finally:
        engine.dispose()
