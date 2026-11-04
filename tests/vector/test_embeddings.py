"""Embedding contract, fake encoder, and similarity eligibility tests."""

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from fraud_detection.config import Settings
from fraud_detection.features.behavioral import build_behavioral_templates, template_digest
from fraud_detection.persistence.document_vector.artifacts import (
    EMBEDDING_SCHEMA,
    embedding_artifact_logical_digest,
    write_embedding_artifact,
)
from fraud_detection.persistence.document_vector.embedding_generation import (
    build_embedding_document,
)
from fraud_detection.persistence.document_vector.embedding_models import (
    EMBEDDING_DIMENSION,
    EmbeddingDocument,
    embedding_content_digest,
)
from fraud_detection.persistence.document_vector.embedding_service import (
    project_embedding_artifact,
    similar_behavior_from_artifact,
    similar_behavior_from_documents,
)
from tests.graph.helpers import raw_document
from tests.vector.fakes import FakeEmbeddingStore, FakeEncoder


def settings() -> Settings:
    return Settings(_env_file=None, embedding_version="test-embedding-v1")


def encoded_document(
    document: object,
    *,
    projected_at: datetime | None = None,
) -> EmbeddingDocument:
    template = build_behavioral_templates([document])[0]  # type: ignore[list-item]
    encoder = FakeEncoder()
    vector = encoder.encode([template.text], batch_size=1)[0]
    return build_embedding_document(
        template,
        vector,
        encoder,
        settings(),
        projected_at or datetime(2026, 10, 31, tzinfo=UTC),
    )


def test_embedding_is_384_finite_normalized_and_digest_stable() -> None:
    first = encoded_document(raw_document())
    later = encoded_document(
        raw_document(), projected_at=datetime(2026, 10, 31, tzinfo=UTC) + timedelta(seconds=1)
    )

    assert len(first.vector) == EMBEDDING_DIMENSION
    assert all(math.isfinite(value) for value in first.vector)
    assert math.isclose(float(np.linalg.norm(first.vector)), 1.0, abs_tol=1e-5)
    assert first.content_digest == later.content_digest
    assert first.immutable_payload() == later.immutable_payload()

    invalid = first.model_dump(mode="python")
    invalid["vector"] = first.vector[:-1]
    with pytest.raises(ValidationError, match="exactly 384"):
        EmbeddingDocument.model_validate(invalid)


def test_embedding_recomputes_versioned_template_digest() -> None:
    document = encoded_document(raw_document())
    altered = document.model_dump(mode="python")
    altered["template_text"] = f"{document.template_text};prior_sender_count=99"
    altered["content_digest"] = document.content_digest
    with pytest.raises(ValidationError, match="template digest"):
        EmbeddingDocument.model_validate(altered)

    fabricated = document.model_dump(mode="python")
    fabricated["template_digest"] = "0" * 64
    fabricated["content_digest"] = embedding_content_digest(fabricated)
    with pytest.raises(ValidationError, match="template digest"):
        EmbeddingDocument.model_validate(fabricated)
    assert document.template_digest == template_digest(
        document.template_text, document.template_version, document.amount_band_version
    )


def test_changing_labels_does_not_change_template_or_embedding() -> None:
    clean = encoded_document(raw_document(is_fraud=False, source_flag=False))
    fraud = encoded_document(raw_document(is_fraud=True, source_flag=True))

    assert clean.template_text == fraud.template_text
    assert clean.template_digest == fraud.template_digest
    assert clean.vector == fraud.vector
    assert clean.reference_eligible is False
    assert fraud.reference_eligible is True


def test_similarity_uses_only_earlier_training_fraud_and_excludes_self() -> None:
    documents = [
        raw_document(transaction_id="fraud-a", step=1, source_row=1, is_fraud=True),
        raw_document(transaction_id="query", step=3, source_row=2, is_fraud=False),
        raw_document(transaction_id="fraud-future", step=4, source_row=3, is_fraud=True),
    ]
    templates = build_behavioral_templates(documents)
    encoder = FakeEncoder()
    embeddings = [
        build_embedding_document(
            template,
            encoder.encode([template.text], batch_size=1)[0],
            encoder,
            settings(),
            datetime(2026, 10, 31, tzinfo=UTC),
        )
        for template in templates
    ]

    result = similar_behavior_from_documents(embeddings, "query", top_k=10)

    assert [match.transaction_id for match in result.matches] == ["fraud-a"]
    assert result.nearest_fraud_missing is False
    assert result.nearest_fraud_distance is not None
    assert all(match.transaction_id != "query" for match in result.matches)


def test_validation_fraud_is_not_a_reference_and_early_query_is_missing() -> None:
    training_fraud = raw_document(transaction_id="early-query", step=1, source_row=1, is_fraud=True)
    validation_fraud = raw_document(
        transaction_id="validation-fraud", step=2, source_row=2, is_fraud=True
    ).model_copy(update={"data_split": "validation"})
    values = [encoded_document(training_fraud), encoded_document(validation_fraud)]

    assert values[1].reference_eligible is False
    result = similar_behavior_from_documents(values, "early-query", top_k=5)
    assert result.matches == []
    assert result.nearest_fraud_distance is None
    assert result.nearest_fraud_missing is True


def test_top_k_ties_use_transaction_id_order() -> None:
    references = [
        raw_document(transaction_id="fraud-b", step=1, source_row=1, is_fraud=True),
        raw_document(transaction_id="fraud-a", step=1, source_row=2, is_fraud=True),
        raw_document(transaction_id="query", step=2, source_row=3),
    ]
    templates = build_behavioral_templates(references)
    encoder = FakeEncoder()
    shared = encoder.encode([templates[0].text], batch_size=1)[0]
    embeddings = [
        build_embedding_document(
            template,
            shared,
            encoder,
            settings(),
            datetime(2026, 10, 31, tzinfo=UTC),
        )
        for template in templates
    ]

    result = similar_behavior_from_documents(embeddings, "query", top_k=2)
    assert [match.transaction_id for match in result.matches] == ["fraud-a", "fraud-b"]

    with pytest.raises(ValueError, match="at least 1"):
        similar_behavior_from_documents(embeddings, "query", top_k=0)
    with pytest.raises(ValueError, match="configured maximum"):
        similar_behavior_from_documents(embeddings, "query", top_k=3, max_top_k=2)


def test_streaming_artifact_similarity_uses_same_eligibility_rules(tmp_path: Path) -> None:
    values = [
        raw_document(transaction_id="fraud", step=1, source_row=1, is_fraud=True),
        raw_document(transaction_id="query", step=2, source_row=2),
    ]
    templates = build_behavioral_templates(values)
    encoder = FakeEncoder()
    embeddings = [
        build_embedding_document(
            template,
            encoder.encode([template.text], batch_size=1)[0],
            encoder,
            settings(),
            datetime(2026, 10, 31, tzinfo=UTC),
        )
        for template in templates
    ]
    artifact = tmp_path / "embeddings.parquet"
    write_embedding_artifact(artifact, embeddings, batch_size=1)

    result = similar_behavior_from_artifact(artifact, "query", top_k=1)

    assert [match.transaction_id for match in result.matches] == ["fraud"]
    assert result.nearest_fraud_missing is False


def test_logical_artifact_digest_ignores_projection_timestamp_and_layout(
    tmp_path: Path,
) -> None:
    raw = raw_document()
    first = encoded_document(raw, projected_at=datetime(2026, 10, 31, tzinfo=UTC))
    later = encoded_document(raw, projected_at=datetime(2026, 11, 1, tzinfo=UTC))
    first_path = tmp_path / "first.parquet"
    later_path = tmp_path / "later.parquet"
    write_embedding_artifact(first_path, [first], batch_size=1)
    write_embedding_artifact(later_path, [later], batch_size=100)

    assert first.content_digest == later.content_digest
    assert embedding_artifact_logical_digest(first_path) == embedding_artifact_logical_digest(
        later_path
    )


def test_projection_rejects_empty_and_multiple_batch_artifacts_before_store_access(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.parquet"
    pq.write_table(pa.Table.from_pylist([], schema=EMBEDDING_SCHEMA), empty)
    store = FakeEmbeddingStore()
    with pytest.raises(ValueError, match="empty"):
        project_embedding_artifact(empty, "postgresql://unused", store)
    assert store.calls == 0

    first = encoded_document(raw_document(transaction_id="tx-a"))
    second = encoded_document(raw_document(transaction_id="tx-b", source_row=2))
    rows = [first.model_dump(mode="python"), second.model_dump(mode="python")]
    rows[1]["batch_id"] = "another-batch"
    multiple = tmp_path / "multiple.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=EMBEDDING_SCHEMA), multiple)
    with pytest.raises(ValueError, match="exactly one batch"):
        project_embedding_artifact(multiple, "postgresql://unused", store)
    assert store.calls == 0
