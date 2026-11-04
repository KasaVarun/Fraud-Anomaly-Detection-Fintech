"""Exact raw/PostgreSQL-to-embedding reconciliation."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import Engine, create_engine

from fraud_detection.features.behavioral import iter_behavioral_templates_sorted
from fraud_detection.ingestion.errors import ReconciliationError
from fraud_detection.ingestion.loader import initial_batch_id
from fraud_detection.ingestion.manifest import (
    sample_path_from_manifest,
    validate_manifest_artifacts,
)
from fraud_detection.persistence.document_vector.artifacts import (
    logical_embedding_artifact_digest,
)
from fraud_detection.persistence.document_vector.embedding_models import EmbeddingDocument
from fraud_detection.persistence.document_vector.projection import (
    initial_batch_status,
    iter_expected_raw_documents,
)
from fraud_detection.ports.services import EmbeddingPersistencePort


class EmbeddingIssueSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    sample_ids: list[str]


class EmbeddingReconciliationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_id: str
    batch_status: str
    expected_count: int
    embedding_count: int
    reference_count: int
    logical_artifact_digest: str
    missing: EmbeddingIssueSummary
    orphaned: EmbeddingIssueSummary
    duplicate: EmbeddingIssueSummary
    conflicting: EmbeddingIssueSummary
    invalid: EmbeddingIssueSummary
    reserved_test: EmbeddingIssueSummary
    reconciled: bool


def _summary(values: set[str]) -> EmbeddingIssueSummary:
    return EmbeddingIssueSummary(count=len(values), sample_ids=sorted(values)[:100])


def reconcile_embedding_documents(
    source_path: Path,
    manifest_path: Path,
    database_url: str,
    documents: Iterable[EmbeddingDocument | Mapping[str, Any]],
    *,
    expected_model_id: str,
    expected_model_revision: str,
    expected_embedding_version: str,
    expected_template_version: str,
    expected_logical_artifact_digest: str | None = None,
    forced_reserved_ids: set[str] | None = None,
) -> EmbeddingReconciliationReport:
    manifest = validate_manifest_artifacts(source_path, manifest_path)
    batch_id = initial_batch_id(manifest)
    raw_documents = list(
        iter_expected_raw_documents(
            source_path,
            manifest_path,
            database_url,
            batch_size=500,
            projected_at=manifest.created_at,
        )
    )
    expected_raw = {document.id: document for document in raw_documents}
    expected_templates = {
        template.transaction_id: template
        for template in iter_behavioral_templates_sorted(raw_documents)
    }
    seen: set[str] = set()
    duplicate: set[str] = set()
    orphaned: set[str] = set()
    conflicting: set[str] = set()
    invalid: set[str] = set()
    reserved_test: set[str] = set()
    reference_count = 0
    count = 0
    content_digests: list[str] = []
    for value in documents:
        count += 1
        payload = value.payload() if isinstance(value, EmbeddingDocument) else value
        digest_value = payload.get("content_digest")
        if isinstance(digest_value, str):
            content_digests.append(digest_value)
        transaction_id = str(payload.get("transaction_id", "<missing-id>"))
        if transaction_id in seen:
            duplicate.add(transaction_id)
        seen.add(transaction_id)
        raw = expected_raw.get(transaction_id)
        if raw is None:
            orphaned.add(transaction_id)
        try:
            document = EmbeddingDocument.model_validate(payload)
        except ValidationError:
            invalid.add(transaction_id)
            continue
        if raw is None:
            continue
        template = expected_templates[transaction_id]
        if any(
            [
                document.sender_id != raw.sender_id,
                document.raw_id != raw.id,
                document.raw_pk != raw.sender_id,
                document.batch_id != raw.batch_id,
                document.step != raw.step,
                document.data_split != raw.data_split,
                document.template_text != template.text,
                document.template_digest != template.template_digest,
                document.template_version != template.template_version,
                document.amount_band_version != template.amount_band_version,
                document.model_id != expected_model_id,
                document.model_revision != expected_model_revision,
                document.embedding_version != expected_embedding_version,
                document.template_version != expected_template_version,
                document.source_version != raw.projection_version,
                document.reference_eligible != template.reference_eligible,
                document.reference_eligibility_provenance != "protected-training-label-v1",
            ]
        ):
            conflicting.add(transaction_id)
        reference_count += document.reference_eligible
    missing = set(expected_raw) - seen
    logical_digest = logical_embedding_artifact_digest(content_digests)
    if (
        expected_logical_artifact_digest is not None
        and logical_digest != expected_logical_artifact_digest
    ):
        invalid.add("<logical-artifact-digest>")

    sample = pl.read_parquet(sample_path_from_manifest(manifest, manifest_path))
    reserved_ids = set(
        str(value)
        for value in sample.filter(pl.col("data_split") == "test")["transaction_id"].to_list()
    )
    reserved_test.update(seen & reserved_ids)
    reserved_test.update(forced_reserved_ids or set())
    engine: Engine = create_engine(database_url, hide_parameters=True)
    try:
        status = initial_batch_status(engine, batch_id)
    finally:
        engine.dispose()
    reconciled = all(
        [
            status == "pending",
            count == len(expected_raw),
            not missing,
            not orphaned,
            not duplicate,
            not conflicting,
            not invalid,
            not reserved_test,
        ]
    )
    report = EmbeddingReconciliationReport(
        batch_id=batch_id,
        batch_status=status,
        expected_count=len(expected_raw),
        embedding_count=count,
        reference_count=reference_count,
        logical_artifact_digest=logical_digest,
        missing=_summary(missing),
        orphaned=_summary(orphaned),
        duplicate=_summary(duplicate),
        conflicting=_summary(conflicting),
        invalid=_summary(invalid),
        reserved_test=_summary(reserved_test),
        reconciled=reconciled,
    )
    if not report.reconciled:
        raise ReconciliationError(report.model_dump_json())
    return report


def reconcile_embedding_store(
    source_path: Path,
    manifest_path: Path,
    database_url: str,
    store: EmbeddingPersistencePort,
    *,
    page_size: int,
    chunk_size: int,
    expected_model_id: str,
    expected_model_revision: str,
    expected_embedding_version: str,
    expected_template_version: str,
    expected_logical_artifact_digest: str | None = None,
) -> EmbeddingReconciliationReport:
    manifest = validate_manifest_artifacts(source_path, manifest_path)
    batch_id = initial_batch_id(manifest)
    sample = pl.read_parquet(sample_path_from_manifest(manifest, manifest_path))
    reserved_ids = set(
        str(value)
        for value in sample.filter(pl.col("data_split") == "test")["transaction_id"].to_list()
    )
    forced_reserved = {
        str(payload.get("transaction_id", "<missing-id>"))
        for payload in store.iter_embeddings_by_transaction_ids(
            sorted(reserved_ids), chunk_size=chunk_size, page_size=page_size
        )
        if str(payload.get("transaction_id", "<missing-id>")) in reserved_ids
    }
    return reconcile_embedding_documents(
        source_path,
        manifest_path,
        database_url,
        (dict(payload) for payload in store.iter_embedding_batch(batch_id, page_size=page_size)),
        expected_model_id=expected_model_id,
        expected_model_revision=expected_model_revision,
        expected_embedding_version=expected_embedding_version,
        expected_template_version=expected_template_version,
        expected_logical_artifact_digest=expected_logical_artifact_digest,
        forced_reserved_ids=forced_reserved,
    )
