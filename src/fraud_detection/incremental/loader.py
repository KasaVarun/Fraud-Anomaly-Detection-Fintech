"""Idempotent PostgreSQL ingestion for one deterministic test chunk."""

from decimal import Decimal
from typing import Any

import polars as pl
from sqlalchemy import Connection, Engine, create_engine, func, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert

from fraud_detection.ingestion.constants import PROVENANCE
from fraud_detection.ingestion.errors import LoadConflictError
from fraud_detection.ingestion.models import SampleManifest
from fraud_detection.persistence.relational import Base

from .contracts import TestChunk

ingestion_batch = Base.metadata.tables["ingestion_batch"]
account = Base.metadata.tables["account"]
merchant = Base.metadata.tables["merchant"]
transaction = Base.metadata.tables["transaction"]
transaction_label = Base.metadata.tables["transaction_label"]
INSERT_BATCH_SIZE = 500


def _insert_verify(
    connection: Connection,
    table: Any,
    records: list[dict[str, Any]],
    key: str,
    immutable_fields: tuple[str, ...],
) -> None:
    if not records:
        return
    for start in range(0, len(records), INSERT_BATCH_SIZE):
        values = records[start : start + INSERT_BATCH_SIZE]
        connection.execute(
            postgres_insert(table).values(values).on_conflict_do_nothing(index_elements=[key])
        )
        keys = [record[key] for record in values]
        stored = {
            row[key]: row
            for row in connection.execute(select(table).where(table.c[key].in_(keys))).mappings()
        }
        if set(stored) != set(keys):
            raise LoadConflictError(f"stored {table.name} identities differ from chunk")
        for record in values:
            actual = stored[record[key]]
            for field in immutable_fields:
                if actual[field] != record[field]:
                    raise LoadConflictError(f"immutable {table.name} identity conflicts on {field}")


def _account_records(rows: pl.DataFrame) -> list[dict[str, Any]]:
    values: dict[str, str] = {}
    for sender, receiver in rows.select("nameOrig", "nameDest").iter_rows():
        for identifier in (str(sender), str(receiver)):
            kind = "merchant" if identifier.startswith("M") else "customer"
            previous = values.setdefault(identifier, kind)
            if previous != kind:
                raise LoadConflictError(f"account kind conflicts for {identifier}")
    return [
        {"account_id": identifier, "account_kind": kind}
        for identifier, kind in sorted(values.items())
    ]


def load_test_chunk(
    database_url: str,
    rows: pl.DataFrame,
    chunk: TestChunk,
    manifest: SampleManifest,
) -> tuple[int, int, str]:
    if rows.filter(pl.col("data_split") != "test").height:
        raise LoadConflictError("incremental chunk contains non-test rows")
    if not rows["step"].is_between(chunk.start_step, chunk.end_step, closed="both").all():
        raise LoadConflictError("incremental chunk contains rows outside its step range")
    batch_record = {
        "batch_id": chunk.batch_id,
        "source_sha256": manifest.source_sha256,
        "sample_seed": manifest.sampling_seed,
        "chunk_id": chunk.chunk_id,
        "enrichment_seed": manifest.enrichment_seed,
        "split_policy": manifest.split_policy,
        "status": "pending",
        "created_at": manifest.created_at,
    }
    accounts = _account_records(rows)
    merchants = [
        {"account_id": value["account_id"]}
        for value in accounts
        if value["account_kind"] == "merchant"
    ]
    transactions = [
        {
            "transaction_id": str(row["transaction_id"]),
            "batch_id": chunk.batch_id,
            "source_row": int(row["source_row"]),
            "sender_id": str(row["nameOrig"]),
            "receiver_id": str(row["nameDest"]),
            "step": int(row["step"]),
            "transaction_type": str(row["type"]),
            "amount": Decimal(str(row["amount"])),
            "data_split": "test",
        }
        for row in rows.iter_rows(named=True)
    ]
    labels = [
        {
            "transaction_id": str(row["transaction_id"]),
            "is_fraud": bool(row["isFraud"]),
            "source_flag": bool(row["isFlaggedFraud"]),
            "provenance": PROVENANCE,
        }
        for row in rows.iter_rows(named=True)
    ]
    engine: Engine = create_engine(database_url, hide_parameters=True)
    try:
        with engine.begin() as connection:
            _insert_verify(
                connection,
                ingestion_batch,
                [batch_record],
                "batch_id",
                (
                    "source_sha256",
                    "sample_seed",
                    "chunk_id",
                    "enrichment_seed",
                    "split_policy",
                ),
            )
            _insert_verify(connection, account, accounts, "account_id", ("account_kind",))
            _insert_verify(connection, merchant, merchants, "account_id", ())
            _insert_verify(
                connection,
                transaction,
                transactions,
                "transaction_id",
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
            )
            _insert_verify(
                connection,
                transaction_label,
                labels,
                "transaction_id",
                ("is_fraud", "source_flag", "provenance"),
            )
            transaction_count = int(
                connection.scalar(
                    select(func.count())
                    .select_from(transaction)
                    .where(transaction.c.batch_id == chunk.batch_id)
                )
                or 0
            )
            label_count = int(
                connection.scalar(
                    select(func.count())
                    .select_from(transaction_label)
                    .join(transaction)
                    .where(transaction.c.batch_id == chunk.batch_id)
                )
                or 0
            )
            status = str(
                connection.scalar(
                    select(ingestion_batch.c.status).where(
                        ingestion_batch.c.batch_id == chunk.batch_id
                    )
                )
            )
        if transaction_count != chunk.row_count or label_count != chunk.row_count:
            raise LoadConflictError("incremental chunk counts do not reconcile")
        return transaction_count, label_count, status
    finally:
        engine.dispose()
