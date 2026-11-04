"""Versioned leakage-safe Phase 7 feature and evaluation contracts."""

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

TABULAR_FEATURE_VERSION = "paysim-safe-tabular-v1"
GRAPH_FEATURE_VERSION = "causal-graph-prior-step-v2"
VECTOR_FEATURE_VERSION = "minilm-earlier-training-fraud-distance-v1"
COMBINED_FEATURE_VERSION = "tabular-graph-vector-v2"
FEATURE_ARTIFACT_SCHEMA_VERSION = "1.0.0"
MODEL_RELEASE_SCHEMA_VERSION = "1.0.0"
PHASE6_LOGICAL_EMBEDDING_DIGEST = "50c904d3a82dc916eb389196ce35e8becb2b3bef747ec79cd91f893503d1f767"
TRAIN_MIN_STEP = 1
TRAIN_MAX_STEP = 281
VALIDATION_MIN_STEP = 282
VALIDATION_MAX_STEP = 354
TEST_MIN_STEP = 355
TEST_MAX_STEP = 743
MODEL_RANDOM_SEED = 73_752
TUNING_CUTOFF_STEP = 224

TABULAR_FEATURES = (
    "amount_log1p",
    "simulation_hour",
    "simulation_day",
    "type_cash_in",
    "type_cash_out",
    "type_debit",
    "type_payment",
    "type_transfer",
)
GRAPH_FEATURES = (
    "sender_prior_degree_centrality",
    "receiver_prior_degree_centrality",
    "sender_prior_weighted_degree",
    "receiver_prior_weighted_degree",
    "prior_sender_receiver_transaction_count",
    "prior_device_account_count",
    "prior_ip_account_count",
    "sender_is_new",
    "receiver_is_new",
)
VECTOR_FEATURES = (
    "nearest_earlier_training_fraud_cosine_distance",
    "missing_earlier_training_fraud_reference",
)
ALL_FEATURES = TABULAR_FEATURES + GRAPH_FEATURES + VECTOR_FEATURES
NON_PREDICTIVE_COLUMNS = (
    "transaction_id",
    "batch_id",
    "step",
    "data_split",
    "is_fraud",
)
FORBIDDEN_MODEL_FIELDS = frozenset(
    {
        "transaction_id",
        "batch_id",
        "data_split",
        "is_fraud",
        "isFraud",
        "isFlaggedFraud",
        "source_flag",
        "sender_id",
        "receiver_id",
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "reference_eligible",
        "community_id",
    }
)

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "tabular": TABULAR_FEATURES,
    "tabular_graph": TABULAR_FEATURES + GRAPH_FEATURES,
    "tabular_vector": TABULAR_FEATURES + VECTOR_FEATURES,
    "combined": ALL_FEATURES,
}


def validate_predictor_allowlist(features: tuple[str, ...]) -> None:
    if len(features) != len(set(features)):
        raise ValueError("predictor allowlist contains duplicates")
    forbidden = set(features) & FORBIDDEN_MODEL_FIELDS
    if forbidden:
        raise ValueError(f"forbidden predictor fields: {', '.join(sorted(forbidden))}")
    unsupported = set(features) - set(ALL_FEATURES)
    if unsupported:
        raise ValueError(f"unsupported predictor fields: {', '.join(sorted(unsupported))}")


class FeatureArtifactResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_path: str
    metadata_path: str
    physical_sha256: str
    logical_digest: str
    byte_size: int
    row_count: int
    train_count: int
    validation_count: int
    test_count: Literal[0]
    ordered_features: tuple[str, ...]
    feature_types: dict[str, str]
    missing_value_rules: dict[str, str]
    tabular_version: str
    graph_version: str
    vector_version: str
    combined_version: str
    embedding_logical_digest: str
    source_sha256: str
    sample_sha256: str
    build_seconds: float


class ValidationMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    split_name: Literal["validation"] = "validation"
    sample_count: int
    fraud_count: int
    fraud_prevalence: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    pr_auc: float
    threshold: Decimal
    training_seconds: float
    inference_seconds: float


class CandidateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    model_version_id: str
    feature_family: str
    ordered_features: tuple[str, ...]
    hyperparameters: dict[str, Any]
    tuning_scores: list[float]
    search_budget: int
    internal_fit_class_weight_ratio: float
    final_class_weight_ratio: float
    metrics: ValidationMetrics
    artifact_directory: str
    model_artifact_sha256: str
    metadata_sha256: str


class TrainingRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_id: str
    batch_status: Literal["pending"]
    feature_physical_sha256: str
    feature_logical_digest: str
    internal_fit_steps: str
    internal_tuning_steps: str
    internal_fit_class_weight_ratio: float
    final_class_weight_ratio: float
    candidates: list[CandidateResult]
    locked_combined_threshold: Decimal
    model_version_count: int
    validation_evaluation_count: int
    transaction_score_count: int
    fraud_flag_count: int
    total_seconds: float
