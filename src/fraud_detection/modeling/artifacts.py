"""Atomic immutable trusted-local Phase 7 model artifacts."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import joblib

from fraud_detection.ingestion.hashing import sha256_file

VOLATILE_METADATA_KEYS = frozenset(
    {
        "training_seconds",
        "inference_seconds",
        "total_seconds",
        "build_seconds",
        "wall_clock_seconds",
        "created_at",
        "run_started_at",
        "run_finished_at",
    }
)
LEGACY_MANIFEST_KEYS = frozenset(
    {
        "model_version_id",
        "model_artifact",
        "model_artifact_sha256",
        "metadata",
        "metadata_sha256",
    }
)
RELEASE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "model_version_id",
        "model_artifact",
        "model_artifact_sha256",
        "preprocessing_artifact",
        "preprocessing_artifact_sha256",
        "estimator_artifact",
        "estimator_artifact_sha256",
        "metadata",
        "metadata_sha256",
    }
)


class ModelArtifactConflictError(ValueError):
    """An immutable model version already exists with different content."""


def _immutable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _immutable_value(item)
            for key, item in value.items()
            if key not in VOLATILE_METADATA_KEYS
        }
    if isinstance(value, list):
        return [_immutable_value(item) for item in value]
    return value


def _member(directory: Path, name: object) -> Path:
    text = str(name)
    if Path(text).name != text:
        raise ValueError("model artifact manifest contains an unsafe path")
    path = (directory / text).resolve(strict=True)
    if path.parent != directory.resolve(strict=True):
        raise ValueError("model artifact manifest contains an unsafe path")
    return path


def _validated_manifest(directory: Path, model_version_id: str) -> dict[str, Any]:
    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = frozenset(manifest)
    if keys not in {LEGACY_MANIFEST_KEYS, RELEASE_MANIFEST_KEYS}:
        raise ValueError("model artifact manifest contract is invalid")
    if manifest.get("model_version_id") != model_version_id:
        raise ValueError("model artifact version mismatch")
    pairs = [
        ("model_artifact", "model_artifact_sha256"),
        ("metadata", "metadata_sha256"),
    ]
    if keys == RELEASE_MANIFEST_KEYS:
        if manifest.get("schema_version") != "2.0.0":
            raise ValueError("model artifact manifest schema is invalid")
        pairs.extend(
            [
                ("preprocessing_artifact", "preprocessing_artifact_sha256"),
                ("estimator_artifact", "estimator_artifact_sha256"),
            ]
        )
    for path_field, hash_field in pairs:
        path = _member(directory, manifest[path_field])
        if sha256_file(path) != manifest[hash_field]:
            raise ValueError(f"model artifact hash mismatch for {path_field}")
    return cast(dict[str, Any], manifest)


def _components(model: Any) -> tuple[Any, Any]:
    steps = getattr(model, "steps", None)
    if isinstance(steps, list) and steps:
        return model[:-1], steps[-1][1]
    return None, model


def write_trusted_model_artifact(
    trusted_root: Path,
    model_version_id: str,
    model: Any,
    metadata: dict[str, Any],
) -> tuple[Path, str, str]:
    trusted_root.mkdir(parents=True, exist_ok=True)
    root = trusted_root.resolve(strict=True)
    directory = root / model_version_id
    temporary = Path(tempfile.mkdtemp(prefix=f".{model_version_id}.tmp-", dir=root))
    try:
        model_path = temporary / "model.joblib"
        preprocessing_path = temporary / "preprocessing.joblib"
        estimator_path = temporary / "estimator.joblib"
        metadata_path = temporary / "metadata.json"
        manifest_path = temporary / "artifact_manifest.json"
        preprocessing, estimator = _components(model)
        joblib.dump(model, model_path, compress=3)
        joblib.dump(preprocessing, preprocessing_path, compress=3)
        joblib.dump(estimator, estimator_path, compress=3)
        model_sha = sha256_file(model_path)
        preprocessing_sha = sha256_file(preprocessing_path)
        estimator_sha = sha256_file(estimator_path)
        immutable_metadata = _immutable_value(metadata)
        if not isinstance(immutable_metadata, dict):
            raise ValueError("model metadata must be an object")
        immutable_metadata["model_artifact_sha256"] = model_sha
        immutable_metadata["preprocessing_artifact_sha256"] = preprocessing_sha
        immutable_metadata["estimator_artifact_sha256"] = estimator_sha
        metadata_path.write_text(
            json.dumps(immutable_metadata, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        metadata_sha = sha256_file(metadata_path)
        manifest = {
            "schema_version": "2.0.0",
            "model_version_id": model_version_id,
            "model_artifact": model_path.name,
            "model_artifact_sha256": model_sha,
            "preprocessing_artifact": preprocessing_path.name,
            "preprocessing_artifact_sha256": preprocessing_sha,
            "estimator_artifact": estimator_path.name,
            "estimator_artifact_sha256": estimator_sha,
            "metadata": metadata_path.name,
            "metadata_sha256": metadata_sha,
        }
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        _validated_manifest(temporary, model_version_id)
        if directory.exists():
            existing = _validated_manifest(directory, model_version_id)
            for hash_field in (
                "model_artifact_sha256",
                "preprocessing_artifact_sha256",
                "estimator_artifact_sha256",
                "metadata_sha256",
            ):
                if existing.get(hash_field) != manifest[hash_field]:
                    raise ModelArtifactConflictError(
                        f"model version {model_version_id} conflicts on {hash_field}"
                    )
            return (
                directory,
                str(existing["model_artifact_sha256"]),
                str(existing["metadata_sha256"]),
            )
        temporary.rename(directory)
        _validated_manifest(directory, model_version_id)
        return directory, model_sha, metadata_sha
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_trusted_model_artifact(
    trusted_root: Path,
    model_version_id: str,
    *,
    expected_feature_logical_digest: str,
    expected_model_sha256: str | None = None,
    expected_metadata_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    root = trusted_root.resolve(strict=True)
    directory = (root / model_version_id).resolve(strict=True)
    if directory.parent != root or directory.name != model_version_id:
        raise ValueError("model artifact is outside the trusted local root")
    manifest = _validated_manifest(directory, model_version_id)
    if (
        expected_model_sha256 is not None
        and manifest["model_artifact_sha256"] != expected_model_sha256
    ):
        raise ValueError("locked model artifact hash mismatch")
    if (
        expected_metadata_sha256 is not None
        and manifest["metadata_sha256"] != expected_metadata_sha256
    ):
        raise ValueError("locked model metadata hash mismatch")
    model_path = _member(directory, manifest["model_artifact"])
    metadata_path = _member(directory, manifest["metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_version_id") != model_version_id:
        raise ValueError("model metadata version mismatch")
    if metadata.get("feature_logical_digest") != expected_feature_logical_digest:
        raise ValueError("model feature lineage mismatch")
    return joblib.load(model_path), metadata
