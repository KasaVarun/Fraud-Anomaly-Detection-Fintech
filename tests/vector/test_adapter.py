"""Controlled Cosmos embedding-container adapter tests."""

from datetime import UTC, datetime

import pytest

from fraud_detection.config import Settings
from fraud_detection.features.behavioral import build_behavioral_templates, template_digest
from fraud_detection.persistence.document_vector.adapter import CosmosRawDocumentAdapter
from fraud_detection.persistence.document_vector.embedding_adapter import CosmosEmbeddingAdapter
from fraud_detection.persistence.document_vector.embedding_generation import (
    build_embedding_document,
)
from fraud_detection.persistence.document_vector.embedding_models import embedding_content_digest
from fraud_detection.persistence.document_vector.errors import (
    CosmosConfigurationError,
    CosmosDocumentConflictError,
)
from tests.cosmos.fakes import FakeClient, FakeContainer, FakeDatabase
from tests.graph.helpers import raw_document
from tests.vector.fakes import FakeEncoder


def settings() -> Settings:
    return Settings(
        _env_file=None,
        cosmos_nosql_endpoint="https://secret.documents.azure.com",
        cosmos_nosql_key="secret-key",
        cosmos_nosql_database="fraud",
        cosmos_embedding_container="behavior_embeddings",
        embedding_version="test-embedding-v1",
    )


def embedding_payload(transaction_id: str = "tx-1", sender_id: str = "C1") -> dict[str, object]:
    document = raw_document(transaction_id=transaction_id, sender_id=sender_id)
    template = build_behavioral_templates([document])[0]
    encoder = FakeEncoder()
    vector = encoder.encode([template.text], batch_size=1)[0]
    return build_embedding_document(
        template,
        vector,
        encoder,
        settings(),
        datetime(2026, 10, 31, tzinfo=UTC),
    ).payload()


def adapter_fixture() -> tuple[CosmosEmbeddingAdapter, FakeContainer]:
    container = FakeContainer()
    container.properties = {
        "id": "behavior_embeddings",
        "partitionKey": {"paths": ["/sender_id"]},
        "vectorEmbeddingPolicy": {
            "vectorEmbeddings": [
                {
                    "path": "/vector",
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": 384,
                }
            ]
        },
    }
    database = FakeDatabase(container)
    return CosmosEmbeddingAdapter(settings(), client=FakeClient(database)), container


def test_vector_policy_validation_and_idempotent_conflict_detection() -> None:
    adapter, container = adapter_fixture()
    adapter.validate_embedding_container()
    payload = embedding_payload()

    assert adapter.upsert_embedding(payload) == "created"
    assert adapter.upsert_embedding(payload) == "existing"
    conflict = embedding_payload()
    conflict["template_text"] = "transaction_type=DEBIT"
    conflict["template_digest"] = template_digest(
        str(conflict["template_text"]),
        str(conflict["template_version"]),
        str(conflict["amount_band_version"]),
    )
    conflict["content_digest"] = embedding_content_digest(conflict)
    with pytest.raises(CosmosDocumentConflictError):
        adapter.upsert_embedding(conflict)

    container.properties["vectorEmbeddingPolicy"] = {"vectorEmbeddings": []}
    with pytest.raises(CosmosConfigurationError, match="384"):
        adapter.validate_embedding_container()


def test_embedding_pagination_id_lookup_and_parameterized_vector_query() -> None:
    adapter, container = adapter_fixture()
    for index in range(5):
        adapter.upsert_embedding(embedding_payload(f"tx-{index}", f"C{index}"))

    batch = list(adapter.iter_embedding_batch("batch-1", page_size=2))
    by_ids = list(
        adapter.iter_embeddings_by_transaction_ids(
            ["tx-0", "tx-3", "missing"], chunk_size=2, page_size=1
        )
    )
    container.vector_results = [
        {
            "transaction_id": "tx-3",
            "step": 1,
            "embedding_version": "test-embedding-v1",
            "template_version": "behavior-template-v1",
            "source_version": "raw-paysim-v1",
            "distance": 0.2,
        },
        {
            "transaction_id": "tx-0",
            "step": 1,
            "embedding_version": "test-embedding-v1",
            "template_version": "behavior-template-v1",
            "source_version": "raw-paysim-v1",
            "distance": 0.1,
        },
    ]
    matches = adapter.vector_search(
        query_vector=[0.0] * 384,
        query_step=5,
        query_transaction_id="query-id",
        embedding_version="test-embedding-v1",
        template_version="behavior-template-v1",
        model_revision="fake-test-revision",
        limit=2,
    )

    assert len(batch) == 5
    assert {item["transaction_id"] for item in by_ids} == {"tx-0", "tx-3"}
    assert [item["transaction_id"] for item in matches] == ["tx-0", "tx-3"]
    vector_query, parameters = container.queries[-1]
    assert "query-id" not in vector_query
    assert {parameter["name"] for parameter in parameters} >= {
        "@query_vector",
        "@query_step",
        "@query_id",
    }
    with pytest.raises(ValueError, match="at least 1"):
        adapter.vector_search(
            query_vector=[0.0] * 384,
            query_step=5,
            query_transaction_id="query-id",
            embedding_version="test-embedding-v1",
            template_version="behavior-template-v1",
            model_revision="fake-test-revision",
            limit=0,
        )
    with pytest.raises(ValueError, match="configured maximum"):
        adapter.vector_search(
            query_vector=[0.0] * 384,
            query_step=5,
            query_transaction_id="query-id",
            embedding_version="test-embedding-v1",
            template_version="behavior-template-v1",
            model_revision="fake-test-revision",
            limit=11,
        )


def test_interleaved_raw_and_embedding_queries_keep_container_isolation() -> None:
    raw_container = FakeContainer()
    embedding_container = FakeContainer()
    embedding_container.properties = {
        "id": "behavior_embeddings",
        "partitionKey": {"paths": ["/sender_id"]},
        "vectorEmbeddingPolicy": {"vectorEmbeddings": []},
    }
    for index in range(3):
        raw = raw_document(transaction_id=f"raw-{index}", source_row=index + 1)
        raw_container.documents[(raw.id, raw.sender_id)] = raw.cosmos_payload()
        payload = embedding_payload(f"embedding-{index}", f"C{index}")
        embedding_container.documents[(str(payload["id"]), str(payload["sender_id"]))] = payload
    database = FakeDatabase(
        raw_container,
        {
            "transactions_raw": raw_container,
            "behavior_embeddings": embedding_container,
        },
    )
    client = FakeClient(database)
    raw_adapter = CosmosRawDocumentAdapter(settings(), client=client)
    embedding_adapter = CosmosEmbeddingAdapter(settings(), client=client)

    raw_values = raw_adapter.iter_batch_documents("batch-1", page_size=1)
    embedding_values = embedding_adapter.iter_embedding_batch("batch-1", page_size=1)
    assert str(next(raw_values)["id"]).startswith("raw-")
    assert str(next(embedding_values)["id"]).startswith("embedding:")
    assert str(next(raw_values)["id"]).startswith("raw-")
    assert str(next(embedding_values)["id"]).startswith("embedding:")
    assert len(raw_container.queries) == 2
    assert len(embedding_container.queries) == 2
