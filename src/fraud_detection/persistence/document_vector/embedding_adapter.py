"""Production Cosmos NoSQL adapter for the separate embedding container."""

from collections.abc import Iterator, Mapping
from typing import Any, cast

from azure.cosmos.exceptions import CosmosResourceExistsError, CosmosResourceNotFoundError
from pydantic import ValidationError

from fraud_detection.config import Settings
from fraud_detection.persistence.document_vector.adapter import CosmosRawDocumentAdapter
from fraud_detection.persistence.document_vector.embedding_models import (
    EMBEDDING_DIMENSION,
    EmbeddingDocument,
)
from fraud_detection.persistence.document_vector.errors import (
    CosmosAdapterError,
    CosmosConfigurationError,
    CosmosDocumentConflictError,
)
from fraud_detection.ports.health import DependencyHealth, DependencyState


class CosmosEmbeddingAdapter(CosmosRawDocumentAdapter):
    """Conflict-detecting vector access isolated from ``transactions_raw``."""

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        if not settings.embedding_configured:
            raise CosmosConfigurationError("Cosmos embedding settings are incomplete")
        super().__init__(settings, client=client)
        self._embedding_container = self._database.get_container_client(
            settings.cosmos_embedding_container
        )

    def validate_embedding_container(self) -> None:
        self._call(
            lambda: self._database.read(timeout=self._settings.cosmos_request_timeout_seconds),
            "embedding database validation",
        )
        properties = self._call(
            lambda: self._embedding_container.read(
                timeout=self._settings.cosmos_request_timeout_seconds
            ),
            "embedding container validation",
        )
        if properties.get("id") != self._settings.cosmos_embedding_container:
            raise CosmosConfigurationError("Cosmos embedding container identity does not match")
        if properties.get("partitionKey", {}).get("paths", []) != ["/sender_id"]:
            raise CosmosConfigurationError(
                "Cosmos embedding container partition key must be /sender_id"
            )
        policies = properties.get("vectorEmbeddingPolicy", {}).get("vectorEmbeddings", [])
        expected = {
            "path": "/vector",
            "dataType": "float32",
            "distanceFunction": self._settings.vector_similarity_metric,
            "dimensions": EMBEDDING_DIMENSION,
        }
        if expected not in policies:
            raise CosmosConfigurationError(
                "Cosmos embedding vector policy must define /vector float32 cosine 384"
            )

    async def readiness(self) -> DependencyHealth:
        try:
            self.validate_embedding_container()
        except CosmosAdapterError:
            return DependencyHealth(
                name="cosmos_nosql_embeddings",
                state=DependencyState.UNAVAILABLE,
                required=True,
                detail="configured embedding container is unavailable or invalid",
            )
        return DependencyHealth(
            name="cosmos_nosql_embeddings",
            state=DependencyState.HEALTHY,
            required=True,
            detail="embedding container, /sender_id, and 384-d cosine policy validated",
        )

    def read_embedding(self, document_id: str, sender_id: str) -> Mapping[str, Any] | None:
        try:
            value = self._call(
                lambda: self._embedding_container.read_item(
                    item=document_id,
                    partition_key=sender_id,
                    timeout=self._settings.cosmos_request_timeout_seconds,
                ),
                "embedding point read",
            )
        except CosmosResourceNotFoundError:
            return None
        return cast(Mapping[str, Any], value)

    def upsert_embedding(self, document: Mapping[str, Any]) -> str:
        try:
            expected = EmbeddingDocument.model_validate(dict(document))
        except ValidationError as exc:
            raise CosmosAdapterError("embedding document validation failed") from exc
        existing = self.read_embedding(expected.id, expected.sender_id)
        if existing is not None:
            try:
                stored = EmbeddingDocument.model_validate(dict(existing))
            except ValidationError as exc:
                raise CosmosDocumentConflictError(
                    f"embedding {expected.id} has invalid existing content"
                ) from exc
            if stored.immutable_payload() == expected.immutable_payload():
                return "existing"
            raise CosmosDocumentConflictError(
                f"embedding {expected.id} has conflicting immutable content"
            )
        try:
            self._call(
                lambda: self._embedding_container.create_item(
                    body=expected.payload(),
                    retry_write=0,
                    timeout=self._settings.cosmos_request_timeout_seconds,
                ),
                "embedding create",
            )
        except CosmosResourceExistsError:
            concurrent = self.read_embedding(expected.id, expected.sender_id)
            if concurrent is not None:
                stored = EmbeddingDocument.model_validate(dict(concurrent))
                if stored.immutable_payload() == expected.immutable_payload():
                    return "existing"
            raise CosmosDocumentConflictError(
                f"embedding {expected.id} has conflicting immutable content"
            ) from None
        return "created"

    def _embedding_query(
        self, query: str, parameters: list[dict[str, Any]], page_size: int
    ) -> Iterator[Mapping[str, Any]]:
        return self._query_container(self._embedding_container, query, parameters, page_size)

    def iter_embedding_batch(self, batch_id: str, *, page_size: int) -> Iterator[Mapping[str, Any]]:
        return self._embedding_query(
            "SELECT * FROM c WHERE c.batch_id = @batch_id",
            [{"name": "@batch_id", "value": batch_id}],
            page_size,
        )

    def iter_embeddings_by_transaction_ids(
        self,
        transaction_ids: list[str],
        *,
        chunk_size: int,
        page_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        if chunk_size <= 0:
            raise ValueError("Cosmos embedding ID lookup chunk size must be positive")
        unique_ids = list(dict.fromkeys(transaction_ids))
        for start in range(0, len(unique_ids), chunk_size):
            chunk = unique_ids[start : start + chunk_size]
            yield from self._embedding_query(
                "SELECT c.id, c.transaction_id FROM c "
                "WHERE ARRAY_CONTAINS(@transaction_ids, c.transaction_id)",
                [{"name": "@transaction_ids", "value": chunk}],
                page_size,
            )

    def vector_search(
        self,
        *,
        query_vector: list[float],
        query_step: int,
        query_transaction_id: str,
        embedding_version: str,
        template_version: str,
        model_revision: str,
        limit: int,
    ) -> list[Mapping[str, Any]]:
        if limit < 1:
            raise ValueError("vector search limit must be at least 1")
        if limit > self._settings.vector_search_limit:
            raise ValueError("vector search limit exceeds configured maximum")
        query = (
            "SELECT TOP @limit c.transaction_id, c.step, c.embedding_version, "
            "c.template_version, c.source_version, "
            "c.model_revision, "
            "VectorDistance(c.vector, @query_vector) AS distance FROM c "
            "WHERE c.reference_eligible = true AND c.data_split = 'train' "
            "AND c.step < @query_step AND c.transaction_id != @query_id "
            "AND c.embedding_version = @embedding_version "
            "AND c.template_version = @template_version "
            "AND c.model_revision = @model_revision "
            "ORDER BY VectorDistance(c.vector, @query_vector)"
        )
        values = list(
            self._embedding_query(
                query,
                [
                    {"name": "@limit", "value": limit},
                    {"name": "@query_vector", "value": query_vector},
                    {"name": "@query_step", "value": query_step},
                    {"name": "@query_id", "value": query_transaction_id},
                    {"name": "@embedding_version", "value": embedding_version},
                    {"name": "@template_version", "value": template_version},
                    {"name": "@model_revision", "value": model_revision},
                ],
                page_size=limit,
            )
        )
        return sorted(
            values,
            key=lambda value: (float(value["distance"]), str(value["transaction_id"])),
        )[:limit]
