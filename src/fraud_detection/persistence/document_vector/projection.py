"""Deterministic PostgreSQL-to-Cosmos raw-document projection."""

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol, cast

import polars as pl
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine, create_engine, select

from fraud_detection.ingestion.constants import PROVENANCE
from fraud_detection.ingestion.errors import LoadConflictError
from fraud_detection.ingestion.loader import initial_batch_id
from fraud_detection.ingestion.manifest import (
    sample_path_from_manifest,
    validate_manifest_artifacts,
)
from fraud_detection.persistence.document_vector.models import (
    FORBIDDEN_RAW_PREDICTIVE_FIELDS,
    RAW_DOCUMENT_SCHEMA_VERSION,
    RAW_PROJECTION_VERSION,
    RawTransactionDocument,
    content_digest,
)
from fraud_detection.persistence.relational import Base
from fraud_detection.ports.services import RawDocumentPersistencePort

transaction = Base.metadata.tables["transaction"]
transaction_label = Base.metadata.tables["transaction_label"]
ingestion_batch = Base.metadata.tables["ingestion_batch"]


class RawProjectionManifest(Protocol):
    @property
    def source_filename(self) -> str: ...

    @property
    def source_sha256(self) -> str: ...

    @property
    def sampling_seed(self) -> int: ...

    @property
    def enrichment_seed(self) -> int: ...


class RawProjectionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_id: str
    batch_status: str
    expected_documents: int
    processed_documents: int
    created_documents: int
    existing_documents: int


def _decimal_string(value: object) -> str | None:
    if value is None:
        return None
    return format(Decimal(str(value)), "f")


def build_raw_document(
    sample_row: Mapping[str, Any],
    postgres_row: Mapping[str, Any],
    manifest: RawProjectionManifest,
    batch_id: str,
    projected_at: datetime,
) -> RawTransactionDocument:
    comparisons = {
        "transaction_id": (sample_row["transaction_id"], postgres_row["transaction_id"]),
        "batch_id": (batch_id, postgres_row["batch_id"]),
        "source_row": (int(sample_row["source_row"]), postgres_row["source_row"]),
        "sender_id": (sample_row["nameOrig"], postgres_row["sender_id"]),
        "receiver_id": (sample_row["nameDest"], postgres_row["receiver_id"]),
        "step": (int(sample_row["step"]), postgres_row["step"]),
        "transaction_type": (sample_row["type"], postgres_row["transaction_type"]),
        "amount": (_decimal_string(sample_row["amount"]), _decimal_string(postgres_row["amount"])),
        "data_split": (sample_row["data_split"], postgres_row["data_split"]),
        "is_fraud": (bool(sample_row["isFraud"]), postgres_row["is_fraud"]),
        "source_flag": (bool(sample_row["isFlaggedFraud"]), postgres_row["source_flag"]),
    }
    mismatches = [name for name, (expected, actual) in comparisons.items() if expected != actual]
    if mismatches:
        raise LoadConflictError(
            "PostgreSQL and validated sample differ for " + ", ".join(sorted(mismatches))
        )
    payload: dict[str, Any] = {
        "id": str(sample_row["transaction_id"]),
        "sender_id": str(sample_row["nameOrig"]),
        "receiver_id": str(sample_row["nameDest"]),
        "batch_id": batch_id,
        "source_row": int(sample_row["source_row"]),
        "step": int(sample_row["step"]),
        "data_split": str(sample_row["data_split"]),
        "transaction_type": str(sample_row["type"]),
        "amount": _decimal_string(sample_row["amount"]),
        "forensic_balances": {
            "oldbalanceOrg": _decimal_string(sample_row["oldbalanceOrg"]),
            "newbalanceOrig": _decimal_string(sample_row["newbalanceOrig"]),
            "oldbalanceDest": _decimal_string(sample_row["oldbalanceDest"]),
            "newbalanceDest": _decimal_string(sample_row["newbalanceDest"]),
            "predictive_use": "forbidden",
        },
        "source_truth": {
            "is_fraud": bool(sample_row["isFraud"]),
            "source_flag": bool(sample_row["isFlaggedFraud"]),
            "provenance": PROVENANCE,
            "predictive_use": "forbidden",
        },
        "source_provenance": {
            "source_filename": manifest.source_filename,
            "source_sha256": manifest.source_sha256,
            "sampling_seed": manifest.sampling_seed,
            "enrichment_seed": manifest.enrichment_seed,
        },
        "forbidden_predictive_fields": FORBIDDEN_RAW_PREDICTIVE_FIELDS,
        "document_schema_version": RAW_DOCUMENT_SCHEMA_VERSION,
        "projection_version": RAW_PROJECTION_VERSION,
        "projected_at": projected_at,
    }
    payload["content_digest"] = content_digest(payload)
    return RawTransactionDocument.model_validate(payload)


def _postgres_rows(
    engine: Engine,
    transaction_ids: list[str],
    batch_id: str,
) -> dict[str, Mapping[str, Any]]:
    statement = (
        select(
            transaction.c.transaction_id,
            transaction.c.batch_id,
            transaction.c.source_row,
            transaction.c.sender_id,
            transaction.c.receiver_id,
            transaction.c.step,
            transaction.c.transaction_type,
            transaction.c.amount,
            transaction.c.data_split,
            transaction_label.c.is_fraud,
            transaction_label.c.source_flag,
        )
        .join(transaction_label)
        .where(transaction.c.batch_id == batch_id)
        .where(transaction.c.transaction_id.in_(transaction_ids))
    )
    with engine.connect() as connection:
        rows = connection.execute(statement).mappings()
        return {str(row["transaction_id"]): cast(Mapping[str, Any], row) for row in rows}


def initial_batch_status(engine: Engine, batch_id: str) -> str:
    with engine.connect() as connection:
        value = connection.scalar(
            select(ingestion_batch.c.status).where(ingestion_batch.c.batch_id == batch_id)
        )
    if value is None:
        raise LoadConflictError("initial PostgreSQL ingestion batch is missing")
    return str(value)


def iter_expected_raw_documents(
    source_path: Path,
    manifest_path: Path,
    database_url: str,
    *,
    batch_size: int,
    projected_at: datetime | None = None,
) -> Iterator[RawTransactionDocument]:
    manifest = validate_manifest_artifacts(source_path, manifest_path)
    batch_id = initial_batch_id(manifest)
    sample = (
        pl.scan_parquet(sample_path_from_manifest(manifest, manifest_path))
        .filter(pl.col("data_split").is_in(["train", "validation"]))
        .sort("source_row")
        .collect(engine="streaming")
    )
    expected_count = (
        manifest.split("train").actual_count + manifest.split("validation").actual_count
    )
    if sample.height != expected_count or sample.filter(pl.col("data_split") == "test").height:
        raise LoadConflictError("raw projection input must exclude reserved test rows")
    engine: Engine = create_engine(database_url, hide_parameters=True)
    timestamp = projected_at or datetime.now(UTC)
    try:
        if initial_batch_status(engine, batch_id) != "pending":
            raise LoadConflictError("raw projection requires a pending ingestion batch")
        for sample_slice in sample.iter_slices(n_rows=batch_size):
            sample_rows = sample_slice.to_dicts()
            ids = [str(row["transaction_id"]) for row in sample_rows]
            postgres = _postgres_rows(engine, ids, batch_id)
            if set(postgres) != set(ids):
                raise LoadConflictError("PostgreSQL is missing projected transaction identities")
            for sample_row in sample_rows:
                document_id = str(sample_row["transaction_id"])
                yield build_raw_document(
                    sample_row,
                    postgres[document_id],
                    manifest,
                    batch_id,
                    timestamp,
                )
    finally:
        engine.dispose()


def project_initial_raw_documents(
    source_path: Path,
    manifest_path: Path,
    database_url: str,
    store: RawDocumentPersistencePort,
    *,
    batch_size: int,
) -> RawProjectionResult:
    manifest = validate_manifest_artifacts(source_path, manifest_path)
    batch_id = initial_batch_id(manifest)
    store.validate_container()
    created = 0
    existing = 0
    processed = 0
    for document in iter_expected_raw_documents(
        source_path,
        manifest_path,
        database_url,
        batch_size=batch_size,
    ):
        outcome = store.upsert_document(document.cosmos_payload())
        created += outcome == "created"
        existing += outcome == "existing"
        processed += 1
    engine: Engine = create_engine(database_url, hide_parameters=True)
    try:
        status = initial_batch_status(engine, batch_id)
    finally:
        engine.dispose()
    return RawProjectionResult(
        batch_id=batch_id,
        batch_status=status,
        expected_documents=manifest.split("train").actual_count
        + manifest.split("validation").actual_count,
        processed_documents=processed,
        created_documents=created,
        existing_documents=existing,
    )
