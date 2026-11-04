"""Manifest serialization and source/sample integrity validation."""

import json
from pathlib import Path

import polars as pl
from pydantic import ValidationError

from fraud_detection.ingestion.constants import REQUIRED_COLUMNS, SPLIT_NAMES
from fraud_detection.ingestion.errors import ManifestValidationError
from fraud_detection.ingestion.hashing import selected_rows_digest, sha256_file, transaction_id
from fraud_detection.ingestion.models import SampleManifest
from fraud_detection.ingestion.source import source_lazy_frame

GENERATED_COLUMNS = {"source_row", "transaction_id", "data_split"}


def read_manifest(path: Path) -> SampleManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SampleManifest.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ManifestValidationError("manifest parsing or schema validation failed") from exc


def sample_path_from_manifest(manifest: SampleManifest, manifest_path: Path) -> Path:
    configured = Path(manifest.sample_path)
    return configured if configured.is_absolute() else manifest_path.parent / configured


def _validate_sample_contents(sample: pl.DataFrame, manifest: SampleManifest) -> None:
    required = set(REQUIRED_COLUMNS) | GENERATED_COLUMNS
    missing = sorted(required - set(sample.columns))
    if missing:
        raise ManifestValidationError(f"sample is missing required columns: {', '.join(missing)}")
    if sample.height != manifest.actual_sample_size:
        raise ManifestValidationError("sample row count differs from manifest")
    if sample["source_row"].n_unique() != sample.height:
        raise ManifestValidationError("sample contains duplicate source rows")
    if sample["transaction_id"].n_unique() != sample.height:
        raise ManifestValidationError("sample contains duplicate transaction IDs")
    if not sample["source_row"].is_sorted():
        raise ManifestValidationError("sample rows are not ordered by source_row")

    rows = [int(value) for value in sample["source_row"].to_list()]
    if selected_rows_digest(rows) != manifest.selected_row_digest:
        raise ManifestValidationError("selected source-row digest differs from manifest")
    expected_ids = [transaction_id(manifest.source_sha256, source_row) for source_row in rows]
    if sample["transaction_id"].to_list() != expected_ids:
        raise ManifestValidationError("sample transaction IDs do not match canonical identities")

    for split_name in SPLIT_NAMES:
        split_manifest = manifest.split(split_name)
        split = sample.filter(pl.col("data_split") == split_name)
        fraud_count = int(split["isFraud"].sum()) if split.height else 0
        if split.height != split_manifest.actual_count or fraud_count != split_manifest.fraud_count:
            raise ManifestValidationError(f"sample counts differ for split {split_name}")
        if (
            split.height
            and not split["step"]
            .is_between(split_manifest.minimum_step, split_manifest.maximum_step, closed="both")
            .all()
        ):
            raise ManifestValidationError(f"sample step range differs for split {split_name}")
    if sample.filter(~pl.col("data_split").is_in(list(SPLIT_NAMES))).height:
        raise ManifestValidationError("sample contains an unsupported split name")


def validate_manifest_artifacts(source_path: Path, manifest_path: Path) -> SampleManifest:
    manifest = read_manifest(manifest_path)
    if source_path.name != manifest.source_filename:
        raise ManifestValidationError("source filename differs from manifest")
    if source_path.stat().st_size != manifest.source_byte_size:
        raise ManifestValidationError("source byte size differs from manifest")
    if sha256_file(source_path) != manifest.source_sha256:
        raise ManifestValidationError("source SHA-256 differs from manifest")

    sample_path = sample_path_from_manifest(manifest, manifest_path)
    if not sample_path.is_file():
        raise ManifestValidationError("generated sample file is missing")
    if sha256_file(sample_path) != manifest.sample_sha256:
        raise ManifestValidationError("generated sample SHA-256 differs from manifest")
    try:
        sample = pl.read_parquet(sample_path)
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise ManifestValidationError("unable to read generated sample") from exc
    _validate_sample_contents(sample, manifest)

    source_fraud_rows = (
        source_lazy_frame(source_path)
        .filter(pl.col("isFraud") == 1)
        .select("source_row")
        .collect(engine="streaming")["source_row"]
        .to_list()
    )
    selected_fraud_rows = sample.filter(pl.col("isFraud") == 1)["source_row"].to_list()
    if sorted(source_fraud_rows) != sorted(selected_fraud_rows):
        raise ManifestValidationError("sample does not preserve every source fraud row")
    return manifest
