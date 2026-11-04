"""Locked Phase 8 lineage and result contracts."""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict

LOCKED_MODEL_ID = "model:lightgbm_combined:1e9a3fc0a52dbac541bf4f92"
LOCKED_MODEL_SHA256 = "4893e1a67ea0c8eb290a71332264842c001a879d2487a1ba08578cf21dfda498"
LOCKED_METADATA_SHA256 = "e5be2044c14f0a5381907f4fac9b1136be8a807b835f2d4c700c8d639d0a0cc0"
LOCKED_FEATURE_LOGICAL_DIGEST = "57606ca504fd0b5d50a16e66ee56a2c16c949563df07fc51be34b1bed84d7190"
LOCKED_FEATURE_VERSION = "tabular-graph-vector-v2"
LOCKED_THRESHOLD = Decimal("0.9496")
LOCKED_TRAINING_CUTOFF = 281
PHASE8_INPUT_VERSION = "phase8-locked-test-input-v1"
PHASE8_SNAPSHOT_VERSION = "phase8-causal-graph-snapshot-v1"
PHASE8_EVALUATION_VERSION = "phase8-locked-test-evaluation-v1"


class TestChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    start_step: int
    end_step: int
    row_count: int
    chunk_id: str
    batch_id: str
    row_digest: str


class ChunkPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    policy: str
    chunk_steps: int
    source_sha256: str
    sample_sha256: str
    selected_test_row_digest: str
    initial_batch_id: str
    test_row_count: int
    chunks: tuple[TestChunk, ...]


class LockedMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    row_count: int
    fraud_count: int
    nonfraud_count: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    pr_auc: float
    predicted_fraud_count: int
    flag_rate: float
    score_minimum: float
    score_maximum: float
    score_mean: float


class ChunkRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: TestChunk
    batch_status: str
    transaction_count: int
    label_count: int
    score_count: int
    flag_count: int
    snapshot_count: int
    evaluation_count: int
    metrics: LockedMetrics
    live_projection: bool


class ReconciliationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_id: str
    expected_count: int
    transaction_count: int
    label_count: int
    score_count: int
    qualifying_score_count: int
    flag_count: int
    evaluation_count: int
    snapshot_count: int
    offline_evidence_counts: dict[str, int]
    missing_live_stores: tuple[str, ...]
    offline_reconciled: bool
    publication_eligible: bool
    blocking_reason: str | None


class PublicationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_id: str
    status: Literal["published", "pending"]
    published: bool
    reason: str | None


class PooledEvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_version_id: str
    threshold: Decimal
    chunk_count: int
    metrics: LockedMetrics
    per_chunk: tuple[dict[str, object], ...]
    artifact_path: str
    artifact_sha256: str


class OfflineEvidenceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw_artifact: str
    raw_sha256: str
    graph_vertex_artifact: str
    graph_vertex_sha256: str
    graph_edge_artifact: str
    graph_edge_sha256: str
    template_artifact: str
    template_sha256: str
    embedding_artifact: str
    embedding_sha256: str
    embedding_logical_digest: str
    feature_artifact: str
    feature_sha256: str
    feature_logical_digest: str
    manifest_artifact: str
    manifest_sha256: str
    test_count: int
    feature_version: str
    build_seconds: float
