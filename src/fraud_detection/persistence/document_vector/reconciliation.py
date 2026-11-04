"""Exact PostgreSQL-to-Cosmos raw-document reconciliation."""

from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine, create_engine

from fraud_detection.ingestion.errors import ReconciliationError
from fraud_detection.ingestion.loader import initial_batch_id
from fraud_detection.ingestion.manifest import (
    sample_path_from_manifest,
    validate_manifest_artifacts,
)
from fraud_detection.persistence.document_vector.models import RawTransactionDocument
from fraud_detection.persistence.document_vector.projection import (
    initial_batch_status,
    iter_expected_raw_documents,
)
from fraud_detection.ports.services import RawDocumentPersistencePort


class IssueSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    sample_ids: list[str]


class RawProjectionReconciliationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_id: str
    batch_status: str
    expected_document_count: int
    cosmos_document_count: int
    matched_document_count: int
    missing: IssueSummary
    orphaned: IssueSummary
    conflicting: IssueSummary
    invalid: IssueSummary
    duplicate: IssueSummary
    reserved_test: IssueSummary
    reconciled: bool


def _summary(identifiers: set[str], *, limit: int = 100) -> IssueSummary:
    return IssueSummary(count=len(identifiers), sample_ids=sorted(identifiers)[:limit])


def reconcile_raw_projection(
    source_path: Path,
    manifest_path: Path,
    database_url: str,
    store: RawDocumentPersistencePort,
    *,
    batch_size: int,
    page_size: int,
) -> RawProjectionReconciliationReport:
    manifest = validate_manifest_artifacts(source_path, manifest_path)
    batch_id = initial_batch_id(manifest)
    store.validate_container()
    expected: dict[str, tuple[str, str]] = {}
    for document in iter_expected_raw_documents(
        source_path,
        manifest_path,
        database_url,
        batch_size=batch_size,
        projected_at=manifest.created_at,
    ):
        expected[document.id] = (document.sender_id, document.content_digest)

    seen: set[str] = set()
    duplicate: set[str] = set()
    orphaned: set[str] = set()
    conflicting: set[str] = set()
    invalid: set[str] = set()
    matched: set[str] = set()
    cosmos_count = 0
    for payload in store.iter_batch_documents(batch_id, page_size=page_size):
        cosmos_count += 1
        identifier = str(payload.get("id", "<missing-id>"))
        if identifier in seen:
            duplicate.add(identifier)
        seen.add(identifier)
        if identifier not in expected:
            orphaned.add(identifier)
        try:
            document = RawTransactionDocument.model_validate(dict(payload))
        except Exception:
            invalid.add(identifier)
            continue
        if document.batch_id != batch_id or document.data_split not in {"train", "validation"}:
            invalid.add(identifier)
            continue
        expected_identity = expected.get(identifier)
        if expected_identity is None:
            continue
        if (document.sender_id, document.content_digest) != expected_identity:
            conflicting.add(identifier)
            continue
        matched.add(identifier)

    missing = set(expected) - seen
    sample = pl.read_parquet(sample_path_from_manifest(manifest, manifest_path))
    reserved_ids = set(
        str(value)
        for value in sample.filter(pl.col("data_split") == "test")["transaction_id"].to_list()
    )
    reserved_test: set[str] = set()
    for payload in store.iter_documents_by_ids(
        sorted(reserved_ids),
        chunk_size=batch_size,
        page_size=page_size,
    ):
        identifier = str(payload.get("id", "<missing-id>"))
        if identifier in reserved_ids:
            reserved_test.add(identifier)

    engine: Engine = create_engine(database_url, hide_parameters=True)
    try:
        batch_status = initial_batch_status(engine, batch_id)
    finally:
        engine.dispose()
    reconciled = all(
        [
            batch_status == "pending",
            cosmos_count == len(expected),
            len(matched) == len(expected),
            not missing,
            not orphaned,
            not conflicting,
            not invalid,
            not duplicate,
            not reserved_test,
        ]
    )
    report = RawProjectionReconciliationReport(
        batch_id=batch_id,
        batch_status=batch_status,
        expected_document_count=len(expected),
        cosmos_document_count=cosmos_count,
        matched_document_count=len(matched),
        missing=_summary(missing),
        orphaned=_summary(orphaned),
        conflicting=_summary(conflicting),
        invalid=_summary(invalid),
        duplicate=_summary(duplicate),
        reserved_test=_summary(reserved_test),
        reconciled=reconciled,
    )
    if not report.reconciled:
        raise ReconciliationError(report.model_dump_json())
    return report
