"""Small deterministic Phase 8 source/sample fixtures."""

import json
from pathlib import Path

import polars as pl

from fraud_detection.ingestion.hashing import sha256_file, transaction_id


def incremental_fixture(tmp_path: Path, test_steps: list[int]) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.csv"
    source.write_text("phase8-source", encoding="utf-8")
    source_sha = sha256_file(source)
    rows: list[dict[str, object]] = []
    specifications = [(1, "train"), (282, "validation"), *[(step, "test") for step in test_steps]]
    for source_row, (step, split) in enumerate(specifications):
        rows.append(
            {
                "step": step,
                "type": "TRANSFER" if source_row % 2 == 0 else "PAYMENT",
                "amount": float(source_row + 1),
                "nameOrig": f"C{source_row + 1}",
                "oldbalanceOrg": 100.0,
                "newbalanceOrig": 99.0,
                "nameDest": f"C{source_row + 2}",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 1.0,
                "isFraud": source_row % 2 == 0,
                "isFlaggedFraud": False,
                "source_row": source_row,
                "transaction_id": transaction_id(source_sha, source_row),
                "data_split": split,
            }
        )
    sample = pl.DataFrame(rows).sort("source_row")
    sample_path = tmp_path / "sample.parquet"
    sample.write_parquet(sample_path)
    split_specs = [
        ("train", 1, 281),
        ("validation", 282, 354),
        ("test", 355, 743),
    ]
    splits: list[dict[str, object]] = []
    for name, minimum, maximum in split_specs:
        selected = sample.filter(pl.col("data_split") == name)
        fraud = int(selected["isFraud"].sum()) if selected.height else 0
        splits.append(
            {
                "name": name,
                "minimum_step": minimum,
                "maximum_step": maximum,
                "source_count": selected.height,
                "target_count": selected.height,
                "actual_count": selected.height,
                "fraud_count": fraud,
                "nonfraud_count": selected.height - fraud,
                "fraud_prevalence": fraud / selected.height if selected.height else 0,
            }
        )
    fraud_count = int(sample["isFraud"].sum())
    manifest = {
        "manifest_schema_version": "1.0.0",
        "source_filename": source.name,
        "source_sha256": source_sha,
        "source_byte_size": source.stat().st_size,
        "source_row_count": sample.height,
        "source_fraud_count": fraud_count,
        "source_fraud_prevalence": fraud_count / sample.height,
        "source_columns": list(sample.columns),
        "source_types": {},
        "source_validation_timestamp": "2026-10-31T00:00:00Z",
        "sampling_seed": 73750,
        "enrichment_seed": 73751,
        "target_sample_size": sample.height,
        "actual_sample_size": sample.height,
        "split_policy": "whole-step-60-20-20-v1",
        "splits": splits,
        "selected_row_digest": "a" * 64,
        "sample_path": sample_path.name,
        "sample_sha256": sha256_file(sample_path),
        "created_at": "2026-10-31T00:00:00Z",
        "code_version": "test",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source, sample_path, manifest_path
