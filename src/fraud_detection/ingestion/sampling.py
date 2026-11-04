"""Whole-step chronological splitting and deterministic fraud-preserving sampling."""

import os
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from fraud_detection import __version__
from fraud_detection.ingestion.constants import (
    DEFAULT_ENRICHMENT_SEED,
    DEFAULT_SAMPLING_SEED,
    DEFAULT_TARGET_SIZE,
    MANIFEST_SCHEMA_VERSION,
    SPLIT_NAMES,
    SPLIT_POLICY,
    SPLIT_WEIGHTS,
)
from fraud_detection.ingestion.errors import SourceValidationError
from fraud_detection.ingestion.hashing import (
    canonical_json_bytes,
    deterministic_rank_salt,
    selected_rows_digest,
    sha256_file,
)
from fraud_detection.ingestion.models import SampleManifest, SplitManifest
from fraud_detection.ingestion.source import source_lazy_frame, validate_source

RANK_MODULUS = 2_147_483_647


def _closest_index(cumulative: list[int], target: float, candidates: range) -> int:
    return min(candidates, key=lambda index: (abs(cumulative[index] - target), index))


def _source_split_ranges(frame: pl.LazyFrame) -> list[dict[str, int | str]]:
    per_step = (
        frame.group_by("step")
        .agg(
            pl.len().alias("source_count"),
            pl.col("isFraud").sum().alias("fraud_count"),
        )
        .sort("step")
        .collect(engine="streaming")
    )
    if per_step.height < 3:
        raise SourceValidationError("at least three distinct PaySim steps are required for splits")
    steps = [int(value) for value in per_step["step"].to_list()]
    counts = [int(value) for value in per_step["source_count"].to_list()]
    fraud_counts = [int(value) for value in per_step["fraud_count"].to_list()]
    cumulative: list[int] = []
    running = 0
    for count in counts:
        running += count
        cumulative.append(running)
    total = cumulative[-1]
    train_end_index = _closest_index(cumulative, total * 0.6, range(0, len(steps) - 2))
    validation_end_index = _closest_index(
        cumulative,
        total * 0.8,
        range(train_end_index + 1, len(steps) - 1),
    )
    slices = [
        ("train", 0, train_end_index),
        ("validation", train_end_index + 1, validation_end_index),
        ("test", validation_end_index + 1, len(steps) - 1),
    ]
    return [
        {
            "name": name,
            "minimum_step": steps[start],
            "maximum_step": steps[end],
            "source_count": sum(counts[start : end + 1]),
            "fraud_count": sum(fraud_counts[start : end + 1]),
        }
        for name, start, end in slices
    ]


def _target_budgets(target: int) -> dict[str, int]:
    train = round(target * SPLIT_WEIGHTS["train"])
    validation = round(target * SPLIT_WEIGHTS["validation"])
    return {"train": train, "validation": validation, "test": target - train - validation}


def _nonfraud_quotas(ranges: list[dict[str, int | str]], target: int) -> dict[str, int]:
    budgets = _target_budgets(target)
    available = {
        str(item["name"]): int(item["source_count"]) - int(item["fraud_count"]) for item in ranges
    }
    fraud = {str(item["name"]): int(item["fraud_count"]) for item in ranges}
    quotas = {
        name: min(available[name], max(0, budgets[name] - fraud[name])) for name in SPLIT_NAMES
    }
    desired_nonfraud = target - sum(fraud.values())
    selected_nonfraud = sum(quotas.values())
    if selected_nonfraud > desired_nonfraud:
        excess = selected_nonfraud - desired_nonfraud
        for name in sorted(SPLIT_NAMES, key=lambda item: (-quotas[item], item)):
            reduction = min(excess, quotas[name])
            quotas[name] -= reduction
            excess -= reduction
            if excess == 0:
                break
    remaining = desired_nonfraud - sum(quotas.values())
    while remaining > 0:
        candidates = [name for name in SPLIT_NAMES if quotas[name] < available[name]]
        if not candidates:
            break
        distributed = 0
        total_weight = sum(SPLIT_WEIGHTS[name] for name in candidates)
        for name in candidates:
            capacity = available[name] - quotas[name]
            addition = min(
                capacity,
                max(1, round(remaining * SPLIT_WEIGHTS[name] / total_weight)),
            )
            addition = min(addition, remaining - distributed)
            quotas[name] += addition
            distributed += addition
            if distributed == remaining:
                break
        if distributed == 0:
            break
        remaining -= distributed
    return quotas


def _split_condition(name: str, ranges: list[dict[str, int | str]]) -> pl.Expr:
    item = next(item for item in ranges if item["name"] == name)
    return pl.col("step").is_between(
        int(item["minimum_step"]), int(item["maximum_step"]), closed="both"
    )


def _selected_frame(
    frame: pl.LazyFrame,
    ranges: list[dict[str, int | str]],
    source_sha256: str,
    sampling_seed: int,
    quotas: dict[str, int],
) -> pl.DataFrame:
    selections: list[pl.LazyFrame] = []
    for name in SPLIT_NAMES:
        split_frame = frame.filter(_split_condition(name, ranges)).with_columns(
            pl.lit(name).alias("data_split")
        )
        fraud = split_frame.filter(pl.col("isFraud") == 1)
        salt = deterministic_rank_salt(source_sha256, sampling_seed, name)
        rank_multiplier = 1 + salt % (RANK_MODULUS - 1)
        nonfraud = (
            split_frame.filter(pl.col("isFraud") == 0)
            .with_columns(
                ((pl.col("source_row") * rank_multiplier + salt) % RANK_MODULUS).alias(
                    "_sample_rank"
                )
            )
            .sort(["_sample_rank", "source_row"])
            .limit(quotas[name])
            .drop("_sample_rank")
        )
        selections.extend([fraud, nonfraud])
    return (
        pl.concat(selections)
        .with_columns(
            pl.concat_str(
                [
                    pl.lit(f"paysim:{source_sha256}:"),
                    pl.col("source_row").cast(pl.String),
                ]
            ).alias("transaction_id")
        )
        .sort("source_row")
        .collect(engine="streaming")
    )


def _split_manifests(
    selected: pl.DataFrame,
    ranges: list[dict[str, int | str]],
    budgets: dict[str, int],
) -> list[SplitManifest]:
    results: list[SplitManifest] = []
    for name in SPLIT_NAMES:
        source_range = next(item for item in ranges if item["name"] == name)
        rows = selected.filter(pl.col("data_split") == name)
        actual = rows.height
        fraud = int(rows["isFraud"].sum()) if actual else 0
        results.append(
            SplitManifest(
                name=name,
                minimum_step=int(source_range["minimum_step"]),
                maximum_step=int(source_range["maximum_step"]),
                source_count=int(source_range["source_count"]),
                target_count=budgets[name],
                actual_count=actual,
                fraud_count=fraud,
                nonfraud_count=actual - fraud,
                fraud_prevalence=fraud / actual if actual else 0.0,
            )
        )
    return results


def generate_sample(
    source_path: Path,
    sample_path: Path,
    manifest_path: Path,
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
    sampling_seed: int = DEFAULT_SAMPLING_SEED,
    enrichment_seed: int = DEFAULT_ENRICHMENT_SEED,
) -> SampleManifest:
    if target_size <= 0:
        raise SourceValidationError("target sample size must be positive")
    report = validate_source(source_path)
    frame = source_lazy_frame(source_path)
    ranges = _source_split_ranges(frame)
    effective_target = min(target_size, report.source_row_count)
    if report.source_fraud_count > effective_target:
        effective_target = report.source_fraud_count
    quotas = _nonfraud_quotas(ranges, effective_target)
    selected = _selected_frame(frame, ranges, report.source_sha256, sampling_seed, quotas)
    if selected.height != effective_target:
        raise SourceValidationError(
            f"sample selection produced {selected.height} rows; expected {effective_target}"
        )
    fraud_selected = int(selected["isFraud"].sum())
    if fraud_selected != report.source_fraud_count:
        raise SourceValidationError("sample selection did not retain every source fraud row")

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_sample = sample_path.with_suffix(sample_path.suffix + ".tmp")
    selected.write_parquet(temporary_sample, compression="zstd", statistics=True)
    temporary_sample.replace(sample_path)
    row_digest = selected_rows_digest(int(value) for value in selected["source_row"].to_list())
    budgets = _target_budgets(effective_target)
    manifest = SampleManifest(
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        source_filename=report.source_filename,
        source_sha256=report.source_sha256,
        source_byte_size=report.source_byte_size,
        source_row_count=report.source_row_count,
        source_fraud_count=report.source_fraud_count,
        source_fraud_prevalence=report.source_fraud_prevalence,
        source_columns=report.detected_columns,
        source_types=report.detected_types,
        source_validation_timestamp=report.validation_timestamp,
        sampling_seed=sampling_seed,
        enrichment_seed=enrichment_seed,
        target_sample_size=target_size,
        actual_sample_size=selected.height,
        split_policy=SPLIT_POLICY,
        splits=_split_manifests(selected, ranges, budgets),
        selected_row_digest=row_digest,
        sample_path=os.path.relpath(sample_path.resolve(), manifest_path.parent.resolve()),
        sample_sha256=sha256_file(sample_path),
        created_at=datetime.now(UTC),
        code_version=__version__,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n")
    temporary_manifest.replace(manifest_path)
    return manifest
