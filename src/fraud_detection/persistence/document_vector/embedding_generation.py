"""Bounded real embedding generation from deterministic behavioral templates."""

import logging
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from fraud_detection.config import Settings
from fraud_detection.features.behavioral import BehavioralTemplate
from fraud_detection.persistence.document_vector.artifacts import (
    iter_template_artifact,
    write_embedding_artifact,
)
from fraud_detection.persistence.document_vector.embedding_models import (
    EMBEDDING_DIMENSION,
    EMBEDDING_DOCUMENT_SCHEMA_VERSION,
    EmbeddingDocument,
    embedding_content_digest,
)
from fraud_detection.persistence.document_vector.encoder import BehavioralEncoder

logger = logging.getLogger(__name__)


class EmbeddingArtifactResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_path: str
    artifact_sha256: str
    logical_artifact_digest: str
    artifact_byte_size: int
    vector_count: int
    vector_dimension: int
    training_fraud_reference_count: int
    embedding_version: str
    template_version: str
    model_id: str
    model_revision: str
    sentence_transformers_version: str


def build_embedding_document(
    template: BehavioralTemplate,
    vector: np.ndarray[Any, np.dtype[np.float32]],
    encoder: BehavioralEncoder,
    settings: Settings,
    projected_at: datetime,
) -> EmbeddingDocument:
    payload: dict[str, Any] = {
        "id": f"embedding:{settings.embedding_version}:{template.transaction_id}",
        "sender_id": template.sender_id,
        "transaction_id": template.transaction_id,
        "raw_id": template.raw_id,
        "raw_pk": template.raw_pk,
        "batch_id": template.batch_id,
        "step": template.step,
        "data_split": template.data_split,
        "template_text": template.text,
        "template_version": template.template_version,
        "amount_band_version": template.amount_band_version,
        "template_digest": template.template_digest,
        "model_id": encoder.model_id,
        "model_revision": encoder.model_revision,
        "sentence_transformers_version": encoder.sentence_transformers_version,
        "embedding_version": settings.embedding_version,
        "vector": vector.astype(np.float32).tolist(),
        "dimension": EMBEDDING_DIMENSION,
        "normalized": True,
        "dtype": "float32",
        "similarity_metric": settings.vector_similarity_metric,
        "source_version": template.source_version,
        "reference_eligible": template.reference_eligible,
        "reference_eligibility_provenance": "protected-training-label-v1",
        "document_schema_version": EMBEDDING_DOCUMENT_SCHEMA_VERSION,
        "projected_at": projected_at,
    }
    payload["content_digest"] = embedding_content_digest(payload)
    return EmbeddingDocument.model_validate(payload)


def iter_embedding_documents(
    templates: Iterable[BehavioralTemplate],
    encoder: BehavioralEncoder,
    settings: Settings,
    *,
    projected_at: datetime,
) -> Iterator[EmbeddingDocument]:
    batch: list[BehavioralTemplate] = []
    processed = 0
    next_progress = settings.embedding_progress_interval

    def encode_batch(values: list[BehavioralTemplate]) -> Iterator[EmbeddingDocument]:
        vectors = encoder.encode(
            [template.text for template in values],
            batch_size=settings.embedding_batch_size,
        )
        for template, vector in zip(values, vectors, strict=True):
            yield build_embedding_document(template, vector, encoder, settings, projected_at)

    for template in templates:
        batch.append(template)
        if len(batch) == settings.embedding_batch_size:
            yield from encode_batch(batch)
            processed += len(batch)
            if processed >= next_progress:
                logger.info("embedding_progress", extra={"processed": processed})
                while next_progress <= processed:
                    next_progress += settings.embedding_progress_interval
            batch = []
    if batch:
        yield from encode_batch(batch)
        processed += len(batch)
        logger.info("embedding_progress", extra={"processed": processed})


def generate_local_embedding_artifact(
    template_artifact: Path,
    output_path: Path,
    encoder: BehavioralEncoder,
    settings: Settings,
) -> EmbeddingArtifactResult:
    projected_at = datetime.now(UTC)
    documents = iter_embedding_documents(
        iter_template_artifact(template_artifact),
        encoder,
        settings,
        projected_at=projected_at,
    )
    count, references, digest, size, logical_digest = write_embedding_artifact(
        output_path,
        documents,
        batch_size=settings.embedding_batch_size,
    )
    return EmbeddingArtifactResult(
        artifact_path=str(output_path),
        artifact_sha256=digest,
        logical_artifact_digest=logical_digest,
        artifact_byte_size=size,
        vector_count=count,
        vector_dimension=EMBEDDING_DIMENSION,
        training_fraud_reference_count=references,
        embedding_version=settings.embedding_version,
        template_version=settings.behavioral_template_version,
        model_id=encoder.model_id,
        model_revision=encoder.model_revision,
        sentence_transformers_version=encoder.sentence_transformers_version,
    )
