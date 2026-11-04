"""Chronological deterministic sampling tests."""

from pathlib import Path

import polars as pl
from tests.paysim_factory import paysim_rows, write_paysim_csv

from fraud_detection.ingestion.hashing import transaction_id
from fraud_detection.ingestion.models import SampleManifest
from fraud_detection.ingestion.sampling import generate_sample


def test_source_row_identity_and_whole_step_splits(
    generated_artifacts: tuple[Path, Path, Path, SampleManifest],
) -> None:
    source, sample_path, _, manifest = generated_artifacts
    sample = pl.read_parquet(sample_path)

    assert sample["source_row"].is_sorted()
    split_counts = sample.group_by("step").agg(pl.col("data_split").n_unique())
    assert split_counts["data_split"].max() == 1
    train_max = int(
        sample.filter(pl.col("data_split") == "train").select(pl.col("step").max()).item()
    )
    validation_min = int(
        sample.filter(pl.col("data_split") == "validation").select(pl.col("step").min()).item()
    )
    validation_max = int(
        sample.filter(pl.col("data_split") == "validation").select(pl.col("step").max()).item()
    )
    test_min = int(
        sample.filter(pl.col("data_split") == "test").select(pl.col("step").min()).item()
    )
    assert train_max < validation_min
    assert validation_max < test_min
    for row in sample.select("source_row", "transaction_id").iter_rows(named=True):
        assert row["transaction_id"] == transaction_id(manifest.source_sha256, row["source_row"])
    assert source.is_file()


def test_every_fraud_row_is_retained(
    generated_artifacts: tuple[Path, Path, Path, SampleManifest],
) -> None:
    source, sample_path, _, manifest = generated_artifacts
    source_fraud = (
        pl.read_csv(source, row_index_name="source_row")
        .filter(pl.col("isFraud") == 1)["source_row"]
        .to_list()
    )
    sample_fraud = (
        pl.read_parquet(sample_path).filter(pl.col("isFraud") == 1)["source_row"].to_list()
    )

    assert sample_fraud == source_fraud
    assert len(sample_fraud) == manifest.source_fraud_count


def test_same_seed_repeats_and_different_seed_changes_only_nonfraud(
    paysim_source: Path, tmp_path: Path
) -> None:
    artifacts: list[tuple[pl.DataFrame, SampleManifest]] = []
    for label, seed in [("a", 123), ("b", 123), ("c", 999)]:
        sample_path = tmp_path / f"{label}.parquet"
        manifest = generate_sample(
            paysim_source,
            sample_path,
            tmp_path / f"{label}.json",
            target_size=24,
            sampling_seed=seed,
        )
        artifacts.append((pl.read_parquet(sample_path), manifest))

    first, second, changed = artifacts
    assert first[0]["source_row"].to_list() == second[0]["source_row"].to_list()
    assert first[1].selected_row_digest == second[1].selected_row_digest
    assert (
        first[0].filter(pl.col("isFraud") == 1)["source_row"].to_list()
        == changed[0].filter(pl.col("isFraud") == 1)["source_row"].to_list()
    )
    assert (
        first[0].filter(pl.col("isFraud") == 0)["source_row"].to_list()
        != changed[0].filter(pl.col("isFraud") == 0)["source_row"].to_list()
    )


def test_identical_business_values_have_distinct_row_identities(tmp_path: Path) -> None:
    rows = paysim_rows()
    rows[1] = dict(rows[0])
    duplicate_source = tmp_path / "duplicates.csv"
    write_paysim_csv(duplicate_source, rows=rows)
    sample_path = tmp_path / "duplicates.parquet"
    manifest = generate_sample(
        duplicate_source,
        sample_path,
        tmp_path / "duplicates.json",
        target_size=60,
    )
    sample = pl.read_parquet(sample_path)

    first_two = sample.head(2)
    assert first_two["source_row"].to_list() == [0, 1]
    assert first_two["transaction_id"].n_unique() == 2
    assert first_two["transaction_id"][0] == transaction_id(manifest.source_sha256, 0)
