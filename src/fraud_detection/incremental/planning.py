"""Deterministic whole-step planning for the reserved PaySim test period."""

import hashlib
from itertools import pairwise
from pathlib import Path

import polars as pl

from fraud_detection.ingestion.hashing import (
    canonical_json_bytes,
    selected_rows_digest,
    sha256_file,
    transaction_id,
)
from fraud_detection.ingestion.loader import initial_batch_id
from fraud_detection.ingestion.manifest import read_manifest, sample_path_from_manifest
from fraud_detection.ingestion.models import SampleManifest

from .contracts import ChunkPlan, TestChunk


def _lineage_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def reserved_test_rows(
    source_path: Path,
    manifest_path: Path,
    *,
    expected_test_count: int = 40_000,
) -> tuple[pl.DataFrame, SampleManifest]:
    manifest = read_manifest(manifest_path)
    if source_path.name != manifest.source_filename:
        raise ValueError("source filename differs from the approved manifest")
    if source_path.stat().st_size != manifest.source_byte_size:
        raise ValueError("source size differs from the approved manifest")
    if sha256_file(source_path) != manifest.source_sha256:
        raise ValueError("source hash differs from the approved manifest")
    sample_path = sample_path_from_manifest(manifest, manifest_path)
    if sha256_file(sample_path) != manifest.sample_sha256:
        raise ValueError("sample hash differs from the approved manifest")
    rows = (
        pl.scan_parquet(sample_path)
        .filter(pl.col("data_split") == "test")
        .sort(["step", "transaction_id"])
        .collect(engine="streaming")
    )
    expected = manifest.split("test")
    if rows.height != expected.actual_count or rows.height != expected_test_count:
        raise ValueError("reserved test row count differs from the approved manifest")
    if rows["source_row"].n_unique() != rows.height:
        raise ValueError("reserved test rows contain duplicate source rows")
    if rows["transaction_id"].n_unique() != rows.height:
        raise ValueError("reserved test rows contain duplicate transaction IDs")
    if not rows["step"].is_between(355, 743, closed="both").all():
        raise ValueError("reserved rows fall outside approved test steps 355-743")
    expected_ids = [
        transaction_id(manifest.source_sha256, int(source_row))
        for source_row in rows["source_row"].to_list()
    ]
    if rows["transaction_id"].to_list() != expected_ids:
        raise ValueError("reserved test transaction identities were altered")
    return rows, manifest


def plan_test_chunks(
    source_path: Path,
    manifest_path: Path,
    *,
    chunk_steps: int = 24,
    expected_test_count: int = 40_000,
) -> ChunkPlan:
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")
    rows, manifest = reserved_test_rows(
        source_path, manifest_path, expected_test_count=expected_test_count
    )
    chunks: list[TestChunk] = []
    for index, start_step in enumerate(range(355, 744, chunk_steps)):
        end_step = min(start_step + chunk_steps - 1, 743)
        selected = rows.filter(pl.col("step").is_between(start_step, end_step, closed="both"))
        source_rows = [int(value) for value in selected["source_row"].to_list()]
        row_digest = selected_rows_digest(source_rows)
        lineage: dict[str, object] = {
            "source_sha256": manifest.source_sha256,
            "sample_sha256": manifest.sample_sha256,
            "sampling_seed": manifest.sampling_seed,
            "enrichment_seed": manifest.enrichment_seed,
            "split_policy": manifest.split_policy,
            "start_step": start_step,
            "end_step": end_step,
            "row_digest": row_digest,
            "policy": f"whole-step-test-chunks-v1:{chunk_steps}",
        }
        digest = _lineage_digest(lineage)
        chunk_id = f"test-steps-{start_step:04d}-{end_step:04d}-{digest[:16]}"
        batch_id = f"paysim-incremental-{manifest.source_sha256[:16]}-{digest[:24]}"
        chunks.append(
            TestChunk(
                index=index,
                start_step=start_step,
                end_step=end_step,
                row_count=selected.height,
                chunk_id=chunk_id,
                batch_id=batch_id,
                row_digest=row_digest,
            )
        )
    if not chunks or chunks[0].start_step != 355 or chunks[-1].end_step != 743:
        raise ValueError("chunk plan does not cover the complete test range")
    for previous, current in pairwise(chunks):
        if previous.end_step + 1 != current.start_step:
            raise ValueError("chunk plan contains a gap or overlap")
    if sum(chunk.row_count for chunk in chunks) != rows.height:
        raise ValueError("chunk row counts do not reconcile with reserved test rows")
    return ChunkPlan(
        policy=f"whole-step-test-chunks-v1:{chunk_steps}",
        chunk_steps=chunk_steps,
        source_sha256=manifest.source_sha256,
        sample_sha256=manifest.sample_sha256,
        selected_test_row_digest=selected_rows_digest(
            int(value) for value in rows["source_row"].to_list()
        ),
        initial_batch_id=initial_batch_id(manifest),
        test_row_count=rows.height,
        chunks=tuple(chunks),
    )


def rows_for_chunk(rows: pl.DataFrame, chunk: TestChunk) -> pl.DataFrame:
    selected = rows.filter(
        pl.col("step").is_between(chunk.start_step, chunk.end_step, closed="both")
    ).sort(["step", "transaction_id"])
    if selected.height != chunk.row_count:
        raise ValueError("chunk rows differ from deterministic plan")
    if (
        selected_rows_digest(int(value) for value in selected["source_row"].to_list())
        != chunk.row_digest
    ):
        raise ValueError("chunk source-row digest differs from deterministic plan")
    return selected
