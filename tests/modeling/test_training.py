"""Deterministic model training, threshold, metrics, and artifact tests."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import StandardScaler

from fraud_detection.ingestion.hashing import sha256_file
from fraud_detection.modeling.artifacts import (
    ModelArtifactConflictError,
    load_trusted_model_artifact,
    write_trusted_model_artifact,
)
from fraud_detection.modeling.contracts import (
    ALL_FEATURES,
    FEATURE_FAMILIES,
    TUNING_CUTOFF_STEP,
    FeatureArtifactResult,
)
from fraud_detection.modeling.features import feature_logical_digest
from fraud_detection.modeling.training import (
    LIGHTGBM_SEARCH_SPACE,
    chronological_tuning_masks,
    class_weight_ratio,
    lightgbm_pipeline,
    logistic_tabular_pipeline,
    select_quantized_f1_threshold,
    train_and_release_models,
    validation_metrics,
)


def test_preprocessing_fits_training_only() -> None:
    train = pd.DataFrame({"amount_log1p": [0.0, 2.0]})
    validation = pd.DataFrame({"amount_log1p": [1000.0]})
    labels = np.asarray([0, 1])
    pipeline = logistic_tabular_pipeline()
    pipeline.fit(train, labels)
    scaler = pipeline.named_steps["scaler"]
    assert isinstance(scaler, StandardScaler)
    assert scaler.mean_[0] == 1.0
    assert float(validation.mean().iloc[0]) not in scaler.mean_


def test_internal_tuning_split_uses_disjoint_whole_steps() -> None:
    steps = np.asarray([1, TUNING_CUTOFF_STEP, TUNING_CUTOFF_STEP + 1, 281], dtype=np.int64)
    fit, tune = chronological_tuning_masks(steps)
    assert steps[fit].tolist() == [1, TUNING_CUTOFF_STEP]
    assert steps[tune].tolist() == [TUNING_CUTOFF_STEP + 1, 281]
    assert not np.any(fit & tune)


def test_class_weight_uses_supplied_training_labels_only() -> None:
    assert class_weight_ratio(np.asarray([0, 0, 0, 1], dtype=np.int64)) == 3.0
    with pytest.raises(ValueError, match="no fraud"):
        class_weight_ratio(np.asarray([0, 0], dtype=np.int64))


def test_threshold_tie_break_quantization_and_metric_recomputation() -> None:
    labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
    probabilities = np.asarray([0.90004, 0.8, 0.7, 0.1])
    threshold = select_quantized_f1_threshold(labels, probabilities)
    assert threshold.as_tuple().exponent == -4
    metrics = validation_metrics(
        labels,
        probabilities,
        threshold,
        training_seconds=1.0,
        inference_seconds=0.1,
    )
    assert metrics.tp + metrics.fn == 2
    assert metrics.fp + metrics.tn == 2
    assert 0 <= metrics.pr_auc <= 1


def test_lightgbm_search_budget_and_training_are_deterministic() -> None:
    assert len(LIGHTGBM_SEARCH_SPACE) == 3
    assert all(
        FEATURE_FAMILIES[name][:8] == FEATURE_FAMILIES["tabular"] for name in FEATURE_FAMILIES
    )
    matrix = pd.DataFrame({"amount_log1p": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    first = lightgbm_pipeline(dict(LIGHTGBM_SEARCH_SPACE[0]), 1.0).fit(matrix, labels)
    second = lightgbm_pipeline(dict(LIGHTGBM_SEARCH_SPACE[0]), 1.0).fit(matrix, labels)
    assert np.allclose(first.predict_proba(matrix), second.predict_proba(matrix))


def test_trusted_artifact_identical_rerun_is_immutable_and_runtime_independent(
    tmp_path: Path,
) -> None:
    metadata = {
        "model_version_id": "model:test",
        "feature_logical_digest": "abc",
        "training_seconds": 1.0,
        "metrics": {"f1": 0.5, "inference_seconds": 0.1},
    }
    directory, first_model_sha, first_metadata_sha = write_trusted_model_artifact(
        tmp_path, "model:test", {"value": 1}, metadata
    )
    mtimes = {path.name: path.stat().st_mtime_ns for path in directory.iterdir()}
    rerun = dict(metadata)
    rerun["training_seconds"] = 999.0
    rerun["metrics"] = {"f1": 0.5, "inference_seconds": 888.0}
    _, second_model_sha, second_metadata_sha = write_trusted_model_artifact(
        tmp_path, "model:test", {"value": 1}, rerun
    )
    assert first_model_sha == second_model_sha
    assert first_metadata_sha == second_metadata_sha
    assert mtimes == {path.name: path.stat().st_mtime_ns for path in directory.iterdir()}
    model, loaded = load_trusted_model_artifact(
        tmp_path, "model:test", expected_feature_logical_digest="abc"
    )
    assert model == {"value": 1}
    assert loaded["metrics"] == {"f1": 0.5}
    assert "training_seconds" not in loaded
    with pytest.raises(ValueError, match="lineage"):
        load_trusted_model_artifact(tmp_path, "model:test", expected_feature_logical_digest="wrong")
    with pytest.raises(ValueError, match="artifact hash"):
        load_trusted_model_artifact(
            tmp_path,
            "model:test",
            expected_feature_logical_digest="abc",
            expected_model_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="metadata hash"):
        load_trusted_model_artifact(
            tmp_path,
            "model:test",
            expected_feature_logical_digest="abc",
            expected_metadata_sha256="0" * 64,
        )


def test_trusted_artifact_rejects_model_and_metadata_conflicts(tmp_path: Path) -> None:
    metadata = {"model_version_id": "model:test", "feature_logical_digest": "abc"}
    write_trusted_model_artifact(tmp_path, "model:test", {"value": 1}, metadata)
    with pytest.raises(ModelArtifactConflictError, match="model_artifact_sha256"):
        write_trusted_model_artifact(tmp_path, "model:test", {"value": 2}, metadata)
    conflict = dict(metadata)
    conflict["threshold"] = "0.9000"
    with pytest.raises(ModelArtifactConflictError, match="metadata_sha256"):
        write_trusted_model_artifact(tmp_path, "model:test", {"value": 1}, conflict)
    assert not list(tmp_path.glob(".model:test.tmp-*"))


@pytest.mark.parametrize(
    "tampered_file", ["model.joblib", "metadata.json", "artifact_manifest.json"]
)
def test_trusted_artifact_rejects_tampered_files(tmp_path: Path, tampered_file: str) -> None:
    version = f"model:{tampered_file}"
    metadata = {"model_version_id": version, "feature_logical_digest": "abc"}
    directory, _, _ = write_trusted_model_artifact(tmp_path, version, {"value": 1}, metadata)
    path = directory / tampered_file
    if tampered_file == "artifact_manifest.json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["unexpected"] = True
        path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match=r"manifest|hash"):
        load_trusted_model_artifact(tmp_path, version, expected_feature_logical_digest="abc")


def test_trusted_artifact_rejects_unsafe_manifest_paths(tmp_path: Path) -> None:
    version = "model:unsafe"
    metadata = {"model_version_id": version, "feature_logical_digest": "abc"}
    directory, _, _ = write_trusted_model_artifact(tmp_path, version, {"value": 1}, metadata)
    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_artifact"] = "../outside.joblib"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe path"):
        load_trusted_model_artifact(tmp_path, version, expected_feature_logical_digest="abc")


def test_small_end_to_end_training_uses_identical_ablation_rows_and_persists_no_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records: list[dict[str, Any]] = []
    steps = [1] * 10 + [224] * 10 + [225] * 10 + [281] * 10 + [282] * 10
    for index, step in enumerate(steps):
        record: dict[str, Any] = {
            "transaction_id": f"tx-{index:03d}",
            "batch_id": "batch-test",
            "step": step,
            "data_split": "train" if step <= 281 else "validation",
            "is_fraud": (index in {0, 10, 20, 21, 30, 31, 40, 45}),
        }
        record.update(
            {
                feature: float((index + position) % 7)
                for position, feature in enumerate(ALL_FEATURES)
            }
        )
        records.append(record)
    frame = pl.DataFrame(records).sort(["step", "transaction_id"])
    artifact = tmp_path / "features.parquet"
    frame.write_parquet(artifact)
    metadata = FeatureArtifactResult(
        artifact_path=str(artifact),
        metadata_path=str(artifact.with_suffix(".metadata.json")),
        physical_sha256=sha256_file(artifact),
        logical_digest=feature_logical_digest(frame),
        byte_size=artifact.stat().st_size,
        row_count=50,
        train_count=40,
        validation_count=10,
        test_count=0,
        ordered_features=ALL_FEATURES,
        feature_types={feature: "Float64" for feature in ALL_FEATURES},
        missing_value_rules={"all": "none"},
        tabular_version="tabular-test",
        graph_version="graph-test",
        vector_version="vector-test",
        combined_version="combined-test",
        embedding_logical_digest="embedding-test",
        source_sha256="a" * 64,
        sample_sha256="b" * 64,
        build_seconds=0.1,
    )
    metadata_path = artifact.with_suffix(".metadata.json")
    metadata_path.write_text(metadata.model_dump_json(), encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_persist(
        database_url: str,
        batch_id: str,
        model_records: list[dict[str, Any]],
        evaluation_records: list[dict[str, Any]],
    ) -> tuple[int, int, int, int, str]:
        captured.update(
            database_url=database_url,
            batch_id=batch_id,
            model_records=model_records,
            evaluation_records=evaluation_records,
        )
        return len(model_records), len(evaluation_records), 0, 0, "pending"

    monkeypatch.setattr(
        "fraud_detection.modeling.training.persist_model_release_records", fake_persist
    )
    result = train_and_release_models(
        artifact,
        metadata_path,
        "postgresql://unused",
        tmp_path / "models",
        expected_train_count=40,
        expected_validation_count=10,
    )

    assert len(result.candidates) == 6
    lightgbm_budgets = {
        candidate.search_budget
        for candidate in result.candidates
        if candidate.name.startswith("lightgbm")
    }
    assert lightgbm_budgets == {3}
    assert result.internal_fit_class_weight_ratio == 9.0
    assert result.final_class_weight_ratio == pytest.approx(34 / 6)
    assert result.transaction_score_count == result.fraud_flag_count == 0
    assert all(record["split_name"] == "validation" for record in captured["evaluation_records"])
    assert all(candidate.metrics.sample_count == 10 for candidate in result.candidates)
