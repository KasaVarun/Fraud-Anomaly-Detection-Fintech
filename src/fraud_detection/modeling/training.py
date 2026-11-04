"""Deterministic chronological Phase 7 candidate training and release."""

from __future__ import annotations

import hashlib
import platform
import time
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import polars as pl
import sklearn
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.ingestion.hashing import canonical_json_bytes
from fraud_detection.persistence.document_vector.embedding_models import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
)

from .artifacts import write_trusted_model_artifact
from .contracts import (
    FEATURE_FAMILIES,
    MODEL_RANDOM_SEED,
    MODEL_RELEASE_SCHEMA_VERSION,
    PHASE6_LOGICAL_EMBEDDING_DIGEST,
    TRAIN_MAX_STEP,
    TUNING_CUTOFF_STEP,
    CandidateResult,
    TrainingRunResult,
    ValidationMetrics,
    validate_predictor_allowlist,
)
from .features import read_and_validate_feature_artifact
from .persistence import persist_model_release_records

LIGHTGBM_SEARCH_SPACE: tuple[dict[str, Any], ...] = (
    {"learning_rate": 0.08, "max_depth": -1, "n_estimators": 100, "num_leaves": 15},
    {"learning_rate": 0.05, "max_depth": -1, "n_estimators": 160, "num_leaves": 31},
    {"learning_rate": 0.05, "max_depth": 8, "n_estimators": 120, "num_leaves": 63},
)


def chronological_tuning_masks(
    steps: np.ndarray[Any, np.dtype[np.integer[Any]]],
) -> tuple[np.ndarray[Any, np.dtype[np.bool_]], np.ndarray[Any, np.dtype[np.bool_]]]:
    fit = steps <= TUNING_CUTOFF_STEP
    tune = (steps > TUNING_CUTOFF_STEP) & (steps <= TRAIN_MAX_STEP)
    if not fit.any() or not tune.any() or np.intersect1d(steps[fit], steps[tune]).size:
        raise ValueError("internal tuning split must contain disjoint whole steps")
    return fit, tune


def class_weight_ratio(labels: np.ndarray[Any, np.dtype[np.integer[Any]]]) -> float:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0:
        raise ValueError("training labels contain no fraud examples")
    return negatives / positives


def select_quantized_f1_threshold(
    labels: np.ndarray[Any, np.dtype[np.integer[Any]]],
    probabilities: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> Decimal:
    candidates = sorted({float(value) for value in probabilities}, reverse=True)
    best: tuple[float, float] | None = None
    for threshold in candidates:
        predictions = probabilities >= threshold
        tp = int(np.sum((predictions == 1) & (labels == 1)))
        fp = int(np.sum((predictions == 1) & (labels == 0)))
        fn = int(np.sum((predictions == 0) & (labels == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, threshold)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    quantized = Decimal(str(best[1])).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return min(Decimal("1.0000"), max(Decimal("0.0000"), quantized))


def validation_metrics(
    labels: np.ndarray[Any, np.dtype[np.integer[Any]]],
    probabilities: np.ndarray[Any, np.dtype[np.floating[Any]]],
    threshold: Decimal,
    *,
    training_seconds: float,
    inference_seconds: float,
) -> ValidationMetrics:
    predictions = probabilities >= float(threshold)
    tn, fp, fn, tp = (
        int(value) for value in confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ValidationMetrics(
        sample_count=len(labels),
        fraud_count=int(labels.sum()),
        fraud_prevalence=float(labels.mean()),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=float(average_precision_score(labels, probabilities)),
        threshold=threshold,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
    )


def lightgbm_pipeline(parameters: dict[str, Any], ratio: float) -> Pipeline:
    classifier = LGBMClassifier(
        **parameters,
        objective="binary",
        random_state=MODEL_RANDOM_SEED,
        deterministic=True,
        force_col_wise=True,
        n_jobs=4,
        verbosity=-1,
        scale_pos_weight=ratio,
    )
    imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
    return Pipeline([("imputer", imputer), ("model", classifier)])


def logistic_tabular_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    random_state=MODEL_RANDOM_SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _model_version_id(lineage: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(lineage)).hexdigest()
    return f"model:{lineage['candidate']}:{digest[:24]}"


def _evaluation_id(model_version_id: str, batch_id: str) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {"model_version_id": model_version_id, "batch_id": batch_id, "split": "validation"}
        )
    ).hexdigest()
    return f"evaluation:{digest}"


def _probabilities(model: Any, matrix: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
    if isinstance(model, dict):
        return np.full(len(matrix), float(model["prevalence"]), dtype=np.float64)
    values = model.predict_proba(matrix)
    return np.asarray(values[:, 1], dtype=np.float64)


def train_and_release_models(
    feature_artifact: Path,
    feature_metadata: Path,
    database_url: str,
    artifact_root: Path,
    *,
    expected_train_count: int = 120_000,
    expected_validation_count: int = 40_000,
) -> TrainingRunResult:
    started = time.perf_counter()
    frame, feature_result = read_and_validate_feature_artifact(
        feature_artifact,
        feature_metadata,
        expected_train_count=expected_train_count,
        expected_validation_count=expected_validation_count,
    )
    train = frame.filter(pl.col("data_split") == "train")
    validation = frame.filter(pl.col("data_split") == "validation")
    if train.height != expected_train_count or validation.height != expected_validation_count:
        raise ValueError("model development requires exact 120k/40k train/validation rows")
    train_labels = train["is_fraud"].to_numpy().astype(np.int8)
    validation_labels = validation["is_fraud"].to_numpy().astype(np.int8)
    train_steps = train["step"].to_numpy()
    fit_mask, tune_mask = chronological_tuning_masks(train_steps)
    internal_fit_ratio = class_weight_ratio(train_labels[fit_mask])
    final_ratio = class_weight_ratio(train_labels)
    batch_id = str(train["batch_id"][0])
    dependency_versions = {
        "python": platform.python_version(),
        "lightgbm": lightgbm.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "numpy": np.__version__,
    }
    candidates: list[CandidateResult] = []
    model_records: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []

    candidate_specs: list[tuple[str, str]] = [
        ("prevalence_baseline", "tabular"),
        ("logistic_tabular", "tabular"),
        ("lightgbm_tabular", "tabular"),
        ("lightgbm_tabular_graph", "tabular_graph"),
        ("lightgbm_tabular_vector", "tabular_vector"),
        ("lightgbm_combined", "combined"),
    ]
    trained: list[tuple[str, str, tuple[str, ...], Any, dict[str, Any], list[float], float]] = []
    for name, family in candidate_specs:
        features = FEATURE_FAMILIES[family]
        validate_predictor_allowlist(features)
        x_train = train.select(features).to_pandas()
        training_started = time.perf_counter()
        tuning_scores: list[float] = []
        if name == "prevalence_baseline":
            model: Any = {"prevalence": float(train_labels.mean())}
            selected: dict[str, Any] = {"strategy": "training_prevalence"}
        elif name == "logistic_tabular":
            model = logistic_tabular_pipeline()
            model.fit(x_train, train_labels)
            selected = {"class_weight": "balanced", "solver": "liblinear"}
        else:
            best_score = -1.0
            selected = {}
            for parameters in LIGHTGBM_SEARCH_SPACE:
                tuning_model = lightgbm_pipeline(parameters, internal_fit_ratio)
                tuning_model.fit(x_train.loc[fit_mask], train_labels[fit_mask])
                tune_probability = _probabilities(tuning_model, x_train.loc[tune_mask])
                score = float(average_precision_score(train_labels[tune_mask], tune_probability))
                tuning_scores.append(score)
                if score > best_score:
                    best_score = score
                    selected = dict(parameters)
            model = lightgbm_pipeline(selected, final_ratio)
            model.fit(x_train, train_labels)
        training_seconds = time.perf_counter() - training_started
        trained.append((name, family, features, model, selected, tuning_scores, training_seconds))

    combined_probabilities: np.ndarray[Any, np.dtype[np.float64]] | None = None
    cached_probabilities: dict[str, tuple[np.ndarray[Any, np.dtype[np.float64]], float]] = {}
    for name, _, features, model, _, _, _ in trained:
        matrix = validation.select(features).to_pandas()
        inference_started = time.perf_counter()
        probabilities = _probabilities(model, matrix)
        inference_seconds = time.perf_counter() - inference_started
        cached_probabilities[name] = (probabilities, inference_seconds)
        if name == "lightgbm_combined":
            combined_probabilities = probabilities
    assert combined_probabilities is not None
    locked_threshold = select_quantized_f1_threshold(validation_labels, combined_probabilities)

    for name, family, features, model, selected, tuning_scores, training_seconds in trained:
        probabilities, inference_seconds = cached_probabilities[name]
        threshold = locked_threshold if name == "lightgbm_combined" else Decimal("0.5000")
        metrics = validation_metrics(
            validation_labels,
            probabilities,
            threshold,
            training_seconds=training_seconds,
            inference_seconds=inference_seconds,
        )
        lineage = {
            "candidate": name,
            "feature_family": family,
            "ordered_features": features,
            "feature_logical_digest": feature_result.logical_digest,
            "feature_physical_sha256": feature_result.physical_sha256,
            "feature_versions": {
                "tabular": feature_result.tabular_version,
                "graph": feature_result.graph_version,
                "vector": feature_result.vector_version,
                "combined": feature_result.combined_version,
            },
            "embedding_logical_digest": PHASE6_LOGICAL_EMBEDDING_DIGEST,
            "embedding_model_id": EMBEDDING_MODEL_ID,
            "embedding_model_revision": EMBEDDING_MODEL_REVISION,
            "hyperparameters": selected,
            "random_seed": MODEL_RANDOM_SEED,
            "internal_fit_class_weight_ratio": internal_fit_ratio,
            "final_class_weight_ratio": final_ratio,
            "threshold": str(threshold),
            "training_steps": "1-281",
            "validation_steps": "282-354",
            "internal_fit_steps": f"1-{TUNING_CUTOFF_STEP}",
            "internal_tuning_steps": f"{TUNING_CUTOFF_STEP + 1}-281",
            "dependency_versions": dependency_versions,
            "schema_version": MODEL_RELEASE_SCHEMA_VERSION,
        }
        model_version_id = _model_version_id(lineage)
        metadata = {
            **lineage,
            "model_version_id": model_version_id,
            "metrics": metrics.model_dump(mode="json"),
            "tuning_scores": tuning_scores,
            "search_budget": len(LIGHTGBM_SEARCH_SPACE) if name.startswith("lightgbm") else 0,
            "internal_fit_class_weight_ratio": internal_fit_ratio,
            "final_class_weight_ratio": final_ratio,
            "feature_artifact": feature_result.model_dump(mode="json"),
            "source_sha256": feature_result.source_sha256,
            "sample_sha256": feature_result.sample_sha256,
            "test_data_used": False,
        }
        directory, model_sha, metadata_sha = write_trusted_model_artifact(
            artifact_root, model_version_id, model, metadata
        )
        candidate = CandidateResult(
            name=name,
            model_version_id=model_version_id,
            feature_family=family,
            ordered_features=features,
            hyperparameters=selected,
            tuning_scores=tuning_scores,
            search_budget=len(LIGHTGBM_SEARCH_SPACE) if name.startswith("lightgbm") else 0,
            internal_fit_class_weight_ratio=internal_fit_ratio,
            final_class_weight_ratio=final_ratio,
            metrics=metrics,
            artifact_directory=str(directory),
            model_artifact_sha256=model_sha,
            metadata_sha256=metadata_sha,
        )
        candidates.append(candidate)
        model_records.append(
            {
                "model_version_id": model_version_id,
                "feature_set": family,
                "training_cutoff_step": TRAIN_MAX_STEP,
                "threshold": threshold,
                "embedding_version": (
                    None
                    if family in {"tabular", "tabular_graph"}
                    else "minilm-l6-v2-1110a243-template-v1-normalized-float32"
                ),
                "artifact_uri": str(directory),
                "created_at": datetime.now(UTC),
            }
        )
        evaluation_records.append(
            {
                "evaluation_id": _evaluation_id(model_version_id, batch_id),
                "model_version_id": model_version_id,
                "batch_id": batch_id,
                "split_name": "validation",
                "tp": metrics.tp,
                "fp": metrics.fp,
                "tn": metrics.tn,
                "fn": metrics.fn,
                "precision": Decimal(str(metrics.precision)),
                "recall": Decimal(str(metrics.recall)),
                "f1": Decimal(str(metrics.f1)),
                "pr_auc": Decimal(str(metrics.pr_auc)),
            }
        )
    model_count, evaluation_count, score_count, flag_count, status = persist_model_release_records(
        database_url, batch_id, model_records, evaluation_records
    )
    if status != "pending":
        raise ValueError("Phase 7 batch must remain pending")
    return TrainingRunResult(
        batch_id=batch_id,
        batch_status="pending",
        feature_physical_sha256=feature_result.physical_sha256,
        feature_logical_digest=feature_result.logical_digest,
        internal_fit_steps=f"1-{TUNING_CUTOFF_STEP}",
        internal_tuning_steps=f"{TUNING_CUTOFF_STEP + 1}-281",
        internal_fit_class_weight_ratio=internal_fit_ratio,
        final_class_weight_ratio=final_ratio,
        candidates=candidates,
        locked_combined_threshold=locked_threshold,
        model_version_count=model_count,
        validation_evaluation_count=evaluation_count,
        transaction_score_count=score_count,
        fraud_flag_count=flag_count,
        total_seconds=time.perf_counter() - started,
    )
