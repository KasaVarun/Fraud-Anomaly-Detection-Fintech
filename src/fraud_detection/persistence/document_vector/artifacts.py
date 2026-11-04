"""Bounded ignored Parquet artifacts for templates and embeddings."""

import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from fraud_detection.features.behavioral import BehavioralTemplate
from fraud_detection.ingestion.hashing import canonical_json_bytes, sha256_file
from fraud_detection.persistence.document_vector.embedding_models import (
    EMBEDDING_DIMENSION,
    EmbeddingDocument,
)


def _chunks(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    chunk: list[Any] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def write_template_artifact(
    path: Path,
    templates: Iterable[BehavioralTemplate],
    *,
    batch_size: int,
) -> tuple[int, str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer: pq.ParquetWriter | None = None
    count = 0
    try:
        for chunk in _chunks(templates, batch_size):
            rows = [template.model_dump(mode="json") for template in chunk]
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            count += len(rows)
    finally:
        if writer is not None:
            writer.close()
    if count == 0:
        raise ValueError("template artifact cannot be empty")
    temporary.replace(path)
    return count, sha256_file(path), path.stat().st_size


def iter_template_artifact(path: Path) -> Iterator[BehavioralTemplate]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches():
        for row in batch.to_pylist():
            yield BehavioralTemplate.model_validate(row)


EMBEDDING_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("sender_id", pa.string()),
        ("transaction_id", pa.string()),
        ("raw_id", pa.string()),
        ("raw_pk", pa.string()),
        ("batch_id", pa.string()),
        ("step", pa.int64()),
        ("data_split", pa.string()),
        ("template_text", pa.string()),
        ("template_version", pa.string()),
        ("amount_band_version", pa.string()),
        ("template_digest", pa.string()),
        ("model_id", pa.string()),
        ("model_revision", pa.string()),
        ("sentence_transformers_version", pa.string()),
        ("embedding_version", pa.string()),
        ("vector", pa.list_(pa.float32(), EMBEDDING_DIMENSION)),
        ("dimension", pa.int16()),
        ("normalized", pa.bool_()),
        ("dtype", pa.string()),
        ("similarity_metric", pa.string()),
        ("source_version", pa.string()),
        ("reference_eligible", pa.bool_()),
        ("reference_eligibility_provenance", pa.string()),
        ("document_schema_version", pa.string()),
        ("content_digest", pa.string()),
        ("projected_at", pa.timestamp("us", tz="UTC")),
    ]
)


def write_embedding_artifact(
    path: Path,
    documents: Iterable[EmbeddingDocument],
    *,
    batch_size: int,
) -> tuple[int, int, str, int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = reference_count = 0
    content_digests: list[str] = []
    with pq.ParquetWriter(temporary, EMBEDDING_SCHEMA, compression="zstd") as writer:
        for chunk in _chunks(documents, batch_size):
            rows = [document.model_dump(mode="python") for document in chunk]
            writer.write_table(pa.Table.from_pylist(rows, schema=EMBEDDING_SCHEMA))
            count += len(rows)
            reference_count += sum(document.reference_eligible for document in chunk)
            content_digests.extend(document.content_digest for document in chunk)
    if count == 0:
        raise ValueError("embedding artifact cannot be empty")
    temporary.replace(path)
    return (
        count,
        reference_count,
        sha256_file(path),
        path.stat().st_size,
        logical_embedding_artifact_digest(content_digests),
    )


def logical_embedding_artifact_digest(content_digests: Iterable[str]) -> str:
    """Hash sorted immutable embedding digests, independent of Parquet layout."""
    return hashlib.sha256(canonical_json_bytes(sorted(content_digests))).hexdigest()


def embedding_artifact_logical_digest(path: Path) -> str:
    parquet = pq.ParquetFile(path)
    digests: list[str] = []
    for batch in parquet.iter_batches(columns=["content_digest"]):
        digests.extend(str(value) for value in batch.column(0).to_pylist())
    return logical_embedding_artifact_digest(digests)


def embedding_artifact_identity(path: Path) -> tuple[int, set[str], str]:
    """Read bounded projection metadata without validating or writing documents."""
    parquet = pq.ParquetFile(path)
    count = 0
    batch_ids: set[str] = set()
    digests: list[str] = []
    for batch in parquet.iter_batches(columns=["batch_id", "content_digest"]):
        count += batch.num_rows
        batch_ids.update(str(value) for value in batch.column(0).to_pylist())
        digests.extend(str(value) for value in batch.column(1).to_pylist())
    return count, batch_ids, logical_embedding_artifact_digest(digests)


def iter_embedding_artifact(path: Path) -> Iterator[EmbeddingDocument]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches():
        for row in batch.to_pylist():
            yield EmbeddingDocument.model_validate(row)
