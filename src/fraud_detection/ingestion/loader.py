"""Transactional, idempotent PostgreSQL initial loader for train and validation rows."""

from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
from sqlalchemy import Connection, Engine, create_engine, func, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert

from fraud_detection.ingestion.constants import DEFAULT_BATCH_SIZE, PROVENANCE
from fraud_detection.ingestion.errors import LoadConflictError
from fraud_detection.ingestion.manifest import (
    sample_path_from_manifest,
    validate_manifest_artifacts,
)
from fraud_detection.ingestion.models import InitialLoadResult, SampleManifest
from fraud_detection.persistence.relational import Base

ingestion_batch = Base.metadata.tables["ingestion_batch"]
account = Base.metadata.tables["account"]
merchant = Base.metadata.tables["merchant"]
transaction = Base.metadata.tables["transaction"]
transaction_label = Base.metadata.tables["transaction_label"]


def initial_batch_id(manifest: SampleManifest) -> str:
    return (
        f"paysim-{manifest.source_sha256[:16]}-{manifest.sampling_seed}-"
        f"{manifest.selected_row_digest[:16]}-initial"
    )


def initial_chunk_id(manifest: SampleManifest) -> str:
    return f"initial-train-validation-{manifest.selected_row_digest[:16]}"


def _chunks(records: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def _assert_rows_match(
    connection: Connection,
    table: Any,
    key_column: Any,
    expected_records: list[dict[str, Any]],
    compared_fields: tuple[str, ...],
) -> None:
    expected = {record[str(key_column.name)]: record for record in expected_records}
    stored = connection.execute(select(table).where(key_column.in_(list(expected)))).mappings()
    actual = {row[str(key_column.name)]: row for row in stored}
    if set(actual) != set(expected):
        raise LoadConflictError(f"stored {table.name} identities do not match the load chunk")
    for key, expected_row in expected.items():
        actual_row = actual[key]
        for field in compared_fields:
            if actual_row[field] != expected_row[field]:
                raise LoadConflictError(
                    f"immutable {table.name} identity {key!s} conflicts on {field}"
                )


def _insert_and_verify(
    connection: Connection,
    table: Any,
    records: list[dict[str, Any]],
    key_column: Any,
    compared_fields: tuple[str, ...],
    *,
    batch_size: int,
) -> None:
    for chunk in _chunks(records, batch_size):
        connection.execute(
            postgres_insert(table)
            .values(chunk)
            .on_conflict_do_nothing(index_elements=[key_column.name])
        )
        _assert_rows_match(connection, table, key_column, chunk, compared_fields)


def _batch_record(manifest: SampleManifest) -> dict[str, Any]:
    return {
        "batch_id": initial_batch_id(manifest),
        "source_sha256": manifest.source_sha256,
        "sample_seed": manifest.sampling_seed,
        "chunk_id": initial_chunk_id(manifest),
        "enrichment_seed": manifest.enrichment_seed,
        "split_policy": manifest.split_policy,
        "status": "pending",
        "created_at": manifest.created_at,
    }


def _account_records(sample: pl.DataFrame) -> list[dict[str, Any]]:
    identities: dict[str, str] = {}
    for sender, receiver in sample.select("nameOrig", "nameDest").iter_rows():
        for identifier in (str(sender), str(receiver)):
            kind = "merchant" if identifier.startswith("M") else "customer"
            previous = identities.setdefault(identifier, kind)
            if previous != kind:
                raise LoadConflictError(f"account classification conflict for {identifier}")
    return [
        {"account_id": identifier, "account_kind": kind}
        for identifier, kind in sorted(identities.items())
    ]


def _transaction_records(sample: pl.DataFrame, batch_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in sample.iter_rows(named=True):
        records.append(
            {
                "transaction_id": str(row["transaction_id"]),
                "batch_id": batch_id,
                "source_row": int(row["source_row"]),
                "sender_id": str(row["nameOrig"]),
                "receiver_id": str(row["nameDest"]),
                "step": int(row["step"]),
                "transaction_type": str(row["type"]),
                "amount": Decimal(str(row["amount"])),
                "data_split": str(row["data_split"]),
            }
        )
    return records


def _label_records(sample: pl.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "transaction_id": str(row["transaction_id"]),
            "is_fraud": bool(row["isFraud"]),
            "source_flag": bool(row["isFlaggedFraud"]),
            "provenance": PROVENANCE,
        }
        for row in sample.iter_rows(named=True)
    ]


def _load_with_connection(
    connection: Connection,
    sample: pl.DataFrame,
    manifest: SampleManifest,
    *,
    batch_size: int,
) -> InitialLoadResult:
    batch = _batch_record(manifest)
    _insert_and_verify(
        connection,
        ingestion_batch,
        [batch],
        ingestion_batch.c.batch_id,
        (
            "source_sha256",
            "sample_seed",
            "chunk_id",
            "enrichment_seed",
            "split_policy",
            "status",
        ),
        batch_size=batch_size,
    )

    accounts = _account_records(sample)
    _insert_and_verify(
        connection,
        account,
        accounts,
        account.c.account_id,
        ("account_kind",),
        batch_size=batch_size,
    )
    merchants = [
        {"account_id": record["account_id"]}
        for record in accounts
        if record["account_kind"] == "merchant"
    ]
    if merchants:
        _insert_and_verify(
            connection,
            merchant,
            merchants,
            merchant.c.account_id,
            (),
            batch_size=batch_size,
        )

    transactions = _transaction_records(sample, str(batch["batch_id"]))
    _insert_and_verify(
        connection,
        transaction,
        transactions,
        transaction.c.transaction_id,
        (
            "batch_id",
            "source_row",
            "sender_id",
            "receiver_id",
            "step",
            "transaction_type",
            "amount",
            "data_split",
        ),
        batch_size=batch_size,
    )
    labels = _label_records(sample)
    _insert_and_verify(
        connection,
        transaction_label,
        labels,
        transaction_label.c.transaction_id,
        ("is_fraud", "source_flag", "provenance"),
        batch_size=batch_size,
    )

    batch_id = str(batch["batch_id"])
    transaction_count = connection.scalar(
        select(func.count()).select_from(transaction).where(transaction.c.batch_id == batch_id)
    )
    label_count = connection.scalar(
        select(func.count())
        .select_from(transaction_label.join(transaction))
        .where(transaction.c.batch_id == batch_id)
    )
    return InitialLoadResult(
        batch_id=batch_id,
        status="pending",
        expected_rows=sample.height,
        transaction_count=int(transaction_count or 0),
        label_count=int(label_count or 0),
        account_count=len(accounts),
        merchant_count=len(merchants),
    )


def load_initial_population(
    source_path: Path,
    manifest_path: Path,
    database_url: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> InitialLoadResult:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    manifest = validate_manifest_artifacts(source_path, manifest_path)
    sample_path = sample_path_from_manifest(manifest, manifest_path)
    sample = (
        pl.scan_parquet(sample_path)
        .filter(pl.col("data_split").is_in(["train", "validation"]))
        .sort("source_row")
        .collect(engine="streaming")
    )
    expected = manifest.split("train").actual_count + manifest.split("validation").actual_count
    if sample.height != expected or sample.filter(pl.col("data_split") == "test").height:
        raise LoadConflictError(
            "initial sample must contain only expected train and validation rows"
        )
    engine: Engine = create_engine(database_url, hide_parameters=True)
    try:
        with engine.begin() as connection:
            return _load_with_connection(
                connection,
                sample,
                manifest,
                batch_size=batch_size,
            )
    finally:
        engine.dispose()
