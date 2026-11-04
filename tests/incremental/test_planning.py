"""Deterministic Phase 8 whole-step planning tests."""

import json
from itertools import pairwise
from pathlib import Path

import polars as pl
import pytest

from fraud_detection.incremental.planning import plan_test_chunks
from fraud_detection.ingestion.hashing import sha256_file
from tests.incremental.helpers import incremental_fixture


def test_default_plan_is_stable_gapless_and_never_splits_steps(tmp_path: Path) -> None:
    source, _, manifest = incremental_fixture(tmp_path, [355, 355, 378, 379, 743])
    first = plan_test_chunks(source, manifest, expected_test_count=5)
    second = plan_test_chunks(source, manifest, expected_test_count=5)

    assert first == second
    assert len(first.chunks) == 17
    assert (first.chunks[0].start_step, first.chunks[0].end_step) == (355, 378)
    assert (first.chunks[-1].start_step, first.chunks[-1].end_step) == (739, 743)
    assert sum(chunk.row_count for chunk in first.chunks) == 5
    assert first.chunks[0].row_count == 3
    assert first.chunks[1].row_count == 1
    assert all(
        previous.end_step + 1 == current.start_step for previous, current in pairwise(first.chunks)
    )
    assert len({chunk.batch_id for chunk in first.chunks}) == len(first.chunks)


def test_planner_rejects_altered_test_identity(tmp_path: Path) -> None:
    source, sample_path, manifest_path = incremental_fixture(tmp_path, [355])
    sample = pl.read_parquet(sample_path)
    sample = sample.with_columns(
        pl.when(pl.col("data_split") == "test")
        .then(pl.lit("altered"))
        .otherwise(pl.col("transaction_id"))
        .alias("transaction_id")
    )
    sample.write_parquet(sample_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sample_sha256"] = sha256_file(sample_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="identities"):
        plan_test_chunks(source, manifest_path, expected_test_count=1)
