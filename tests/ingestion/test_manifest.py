"""Versioned manifest integrity tests."""

import json
from pathlib import Path

import pytest

from fraud_detection.ingestion.errors import ManifestValidationError
from fraud_detection.ingestion.manifest import read_manifest, validate_manifest_artifacts
from fraud_detection.ingestion.models import SampleManifest


def test_manifest_round_trips_and_validates(
    generated_artifacts: tuple[Path, Path, Path, SampleManifest],
) -> None:
    source, _, manifest_path, generated = generated_artifacts

    loaded = read_manifest(manifest_path)
    validated = validate_manifest_artifacts(source, manifest_path)

    assert loaded == generated
    assert validated.selected_row_digest == generated.selected_row_digest


def test_source_tampering_is_detected(
    generated_artifacts: tuple[Path, Path, Path, SampleManifest],
) -> None:
    source, _, manifest_path, _ = generated_artifacts
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match=r"source byte size|source SHA-256"):
        validate_manifest_artifacts(source, manifest_path)


def test_sample_tampering_is_detected(
    generated_artifacts: tuple[Path, Path, Path, SampleManifest],
) -> None:
    source, sample_path, manifest_path, _ = generated_artifacts
    sample_path.write_bytes(sample_path.read_bytes() + b"tamper")

    with pytest.raises(ManifestValidationError, match="sample SHA-256"):
        validate_manifest_artifacts(source, manifest_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("manifest_schema_version", "99", "schema validation"),
        ("selected_row_digest", "0" * 64, "selected source-row digest"),
        ("actual_sample_size", 31, "schema validation"),
    ],
)
def test_manifest_tampering_is_detected(
    generated_artifacts: tuple[Path, Path, Path, SampleManifest],
    field: str,
    value: object,
    message: str,
) -> None:
    source, _, manifest_path, _ = generated_artifacts
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload[field] = value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestValidationError, match=message):
        validate_manifest_artifacts(source, manifest_path)
