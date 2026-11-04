"""Template artifacts, embedding projection, and leakage-safe similarity evidence."""

from collections.abc import Iterable, Iterator
from heapq import nsmallest
from pathlib import Path
from typing import cast

import numpy as np
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine, create_engine

from fraud_detection.config import Settings
from fraud_detection.features.behavioral import (
    BehavioralTemplate,
    iter_behavioral_templates_sorted,
)
from fraud_detection.ingestion.manifest import read_manifest
from fraud_detection.persistence.document_vector.artifacts import (
    embedding_artifact_identity,
    iter_embedding_artifact,
    write_template_artifact,
)
from fraud_detection.persistence.document_vector.embedding_generation import (
    EmbeddingArtifactResult,
    generate_local_embedding_artifact,
)
from fraud_detection.persistence.document_vector.embedding_models import (
    EmbeddingDocument,
    SimilarityEvidence,
    SimilarityMatch,
)
from fraud_detection.persistence.document_vector.encoder import BehavioralEncoder, MiniLMEncoder
from fraud_detection.persistence.document_vector.projection import (
    initial_batch_status,
    iter_expected_raw_documents,
)
from fraud_detection.ports.services import EmbeddingPersistencePort


class TemplateArtifactResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_path: str
    artifact_sha256: str
    artifact_byte_size: int
    template_count: int
    template_version: str
    training_fraud_reference_count: int


class EmbeddingProjectionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    expected_documents: int
    processed_documents: int
    created_documents: int
    existing_documents: int
    batch_status: str


def build_template_artifact(
    source_path: Path,
    manifest_path: Path,
    database_url: str,
    output_path: Path,
    settings: Settings,
) -> TemplateArtifactResult:
    manifest = read_manifest(manifest_path)
    documents = iter_expected_raw_documents(
        source_path,
        manifest_path,
        database_url,
        batch_size=settings.cosmos_batch_size,
        projected_at=manifest.created_at,
    )
    reference_count = 0

    def templates() -> Iterator[BehavioralTemplate]:
        nonlocal reference_count
        for template in iter_behavioral_templates_sorted(documents):
            reference_count += template.reference_eligible
            yield template

    count, digest, size = write_template_artifact(
        output_path,
        templates(),
        batch_size=settings.embedding_batch_size,
    )
    return TemplateArtifactResult(
        artifact_path=str(output_path),
        artifact_sha256=digest,
        artifact_byte_size=size,
        template_count=count,
        template_version=settings.behavioral_template_version,
        training_fraud_reference_count=reference_count,
    )


def generate_embeddings(
    template_artifact: Path,
    output_path: Path,
    settings: Settings,
    *,
    encoder: BehavioralEncoder | None = None,
) -> EmbeddingArtifactResult:
    resolved_encoder: BehavioralEncoder = encoder or cast(
        BehavioralEncoder,
        MiniLMEncoder(cache_directory=settings.embedding_model_cache),
    )
    return generate_local_embedding_artifact(
        template_artifact,
        output_path,
        resolved_encoder,
        settings,
    )


def project_embedding_artifact(
    artifact_path: Path,
    database_url: str,
    store: EmbeddingPersistencePort,
) -> EmbeddingProjectionResult:
    artifact_count, batch_ids, _ = embedding_artifact_identity(artifact_path)
    if artifact_count == 0:
        raise ValueError("embedding artifact is empty")
    if len(batch_ids) != 1:
        raise ValueError("embedding artifact must contain exactly one batch")
    batch_id = next(iter(batch_ids))
    engine: Engine = create_engine(database_url, hide_parameters=True)
    try:
        status = initial_batch_status(engine, batch_id)
        if status != "pending":
            raise ValueError("embedding projection requires a pending batch")
        store.validate_embedding_container()
        created = existing = processed = 0
        for document in iter_embedding_artifact(artifact_path):
            outcome = store.upsert_embedding(document.payload())
            created += outcome == "created"
            existing += outcome == "existing"
            processed += 1
        final_status = initial_batch_status(engine, batch_id)
    finally:
        engine.dispose()
    if final_status != "pending":
        raise ValueError("embedding projection requires the batch to remain pending")
    return EmbeddingProjectionResult(
        expected_documents=artifact_count,
        processed_documents=processed,
        created_documents=created,
        existing_documents=existing,
        batch_status=final_status,
    )


def _validate_top_k(top_k: int, max_top_k: int | None) -> None:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if max_top_k is not None and top_k > max_top_k:
        raise ValueError(f"top_k must not exceed configured maximum {max_top_k}")


def similar_behavior_from_documents(
    documents: Iterable[EmbeddingDocument],
    query_transaction_id: str,
    *,
    top_k: int,
    max_top_k: int | None = 100,
) -> SimilarityEvidence:
    _validate_top_k(top_k, max_top_k)
    values = list(documents)
    queries = [value for value in values if value.transaction_id == query_transaction_id]
    if len(queries) != 1:
        raise ValueError("query transaction must identify exactly one embedding")
    query = queries[0]
    query_vector = np.asarray(query.vector, dtype=np.float32)
    candidates: list[tuple[float, str, EmbeddingDocument]] = []
    for reference in values:
        if not reference.reference_eligible:
            continue
        if reference.data_split != "train" or reference.step >= query.step:
            continue
        if reference.transaction_id == query.transaction_id:
            continue
        if (
            reference.embedding_version != query.embedding_version
            or reference.template_version != query.template_version
            or reference.model_revision != query.model_revision
        ):
            continue
        distance = float(1.0 - np.dot(query_vector, np.asarray(reference.vector, dtype=np.float32)))
        candidates.append((max(0.0, distance), reference.transaction_id, reference))
    nearest = nsmallest(top_k, candidates, key=lambda value: (value[0], value[1]))
    matches = [
        SimilarityMatch(
            transaction_id=reference.transaction_id,
            cosine_distance=distance,
            step=reference.step,
            embedding_version=reference.embedding_version,
            template_version=reference.template_version,
            provenance=reference.source_version,
        )
        for distance, _, reference in nearest
    ]
    return SimilarityEvidence(
        query_transaction_id=query.transaction_id,
        nearest_fraud_distance=matches[0].cosine_distance if matches else None,
        nearest_fraud_missing=not matches,
        matches=matches,
    )


def similar_behavior_from_artifact(
    artifact_path: Path,
    query_transaction_id: str,
    *,
    top_k: int,
    max_top_k: int | None = 100,
) -> SimilarityEvidence:
    _validate_top_k(top_k, max_top_k)
    query: EmbeddingDocument | None = None
    for document in iter_embedding_artifact(artifact_path):
        if document.transaction_id == query_transaction_id:
            if query is not None:
                raise ValueError("query transaction has duplicate embeddings")
            query = document
    if query is None:
        raise ValueError("query transaction embedding is missing")
    query_vector = np.asarray(query.vector, dtype=np.float32)
    candidates: list[tuple[float, str, EmbeddingDocument]] = []
    for reference in iter_embedding_artifact(artifact_path):
        if not reference.reference_eligible or reference.data_split != "train":
            continue
        if reference.step >= query.step or reference.transaction_id == query.transaction_id:
            continue
        if (
            reference.embedding_version != query.embedding_version
            or reference.template_version != query.template_version
            or reference.model_revision != query.model_revision
        ):
            continue
        distance = max(
            0.0,
            float(
                1.0
                - np.dot(
                    query_vector,
                    np.asarray(reference.vector, dtype=np.float32),
                )
            ),
        )
        candidates.append((distance, reference.transaction_id, reference))
    nearest = nsmallest(top_k, candidates, key=lambda value: (value[0], value[1]))
    matches = [
        SimilarityMatch(
            transaction_id=reference.transaction_id,
            cosine_distance=distance,
            step=reference.step,
            embedding_version=reference.embedding_version,
            template_version=reference.template_version,
            provenance=reference.source_version,
        )
        for distance, _, reference in nearest
    ]
    return SimilarityEvidence(
        query_transaction_id=query.transaction_id,
        nearest_fraud_distance=matches[0].cosine_distance if matches else None,
        nearest_fraud_missing=not matches,
        matches=matches,
    )
