"""Production Azure Cosmos DB for NoSQL ``transactions_raw`` adapter."""

import time
from collections.abc import Callable, Iterator, Mapping
from functools import partial
from typing import Any, TypeVar, cast

from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import (
    CosmosHttpResponseError,
    CosmosResourceExistsError,
    CosmosResourceNotFoundError,
)
from azure.identity import DefaultAzureCredential
from pydantic import ValidationError

from fraud_detection.config import Settings
from fraud_detection.persistence.document_vector.errors import (
    CosmosAdapterError,
    CosmosConfigurationError,
    CosmosDocumentConflictError,
)
from fraud_detection.persistence.document_vector.models import (
    RAW_PARTITION_KEY_PATH,
    RawTransactionDocument,
)
from fraud_detection.ports.health import DependencyHealth, DependencyState

T = TypeVar("T")
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class CosmosRawDocumentAdapter:
    """Bounded, conflict-detecting access to the approved raw container."""

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        if not settings.document_vector_configured:
            raise CosmosConfigurationError("Cosmos NoSQL raw-document settings are incomplete")
        self._settings = settings
        if client is None:
            endpoint = cast(Any, settings.cosmos_nosql_endpoint).get_secret_value()
            credential: Any
            if settings.cosmos_use_managed_identity:
                credential = DefaultAzureCredential(
                    managed_identity_client_id=settings.azure_managed_identity_client_id,
                    exclude_interactive_browser_credential=True,
                )
            else:
                credential = cast(Any, settings.cosmos_nosql_key).get_secret_value()
            client = CosmosClient(
                endpoint,
                credential=credential,
                connection_timeout=settings.cosmos_request_timeout_seconds,
                retry_total=0,
                retry_write=0,
            )
        self._client = client
        self._database = client.get_database_client(cast(str, settings.cosmos_nosql_database))
        self._container = self._database.get_container_client(settings.cosmos_raw_container)

    def _call(self, operation: Callable[[], T], operation_name: str) -> T:
        attempts = self._settings.cosmos_max_retries + 1
        for attempt in range(attempts):
            try:
                return operation()
            except (CosmosResourceNotFoundError, CosmosResourceExistsError):
                raise
            except CosmosHttpResponseError as exc:
                if exc.status_code not in TRANSIENT_STATUS_CODES or attempt + 1 == attempts:
                    raise CosmosAdapterError(
                        f"Cosmos {operation_name} failed with status {exc.status_code}"
                    ) from exc
                time.sleep(min(0.25 * (2**attempt), 2.0))
            except (OSError, TimeoutError) as exc:
                if attempt + 1 == attempts:
                    raise CosmosAdapterError(f"Cosmos {operation_name} failed") from exc
                time.sleep(min(0.25 * (2**attempt), 2.0))
        raise CosmosAdapterError(f"Cosmos {operation_name} exhausted retries")

    def validate_container(self) -> None:
        self._call(
            lambda: self._database.read(timeout=self._settings.cosmos_request_timeout_seconds),
            "database validation",
        )
        properties = self._call(
            lambda: self._container.read(timeout=self._settings.cosmos_request_timeout_seconds),
            "container validation",
        )
        if properties.get("id") != self._settings.cosmos_raw_container:
            raise CosmosConfigurationError("Cosmos raw container identity does not match settings")
        paths = properties.get("partitionKey", {}).get("paths", [])
        if paths != [RAW_PARTITION_KEY_PATH]:
            raise CosmosConfigurationError(
                f"Cosmos raw container partition key must be {RAW_PARTITION_KEY_PATH}"
            )

    async def readiness(self) -> DependencyHealth:
        try:
            self.validate_container()
        except CosmosAdapterError:
            return DependencyHealth(
                name="cosmos_nosql_raw",
                state=DependencyState.UNAVAILABLE,
                required=True,
                detail="configured Cosmos raw container is unavailable or invalid",
            )
        return DependencyHealth(
            name="cosmos_nosql_raw",
            state=DependencyState.HEALTHY,
            required=True,
            detail="database, container, and /sender_id partition key validated",
        )

    def read_document(self, document_id: str, sender_id: str) -> Mapping[str, Any] | None:
        try:
            result = self._call(
                lambda: self._container.read_item(
                    item=document_id,
                    partition_key=sender_id,
                    timeout=self._settings.cosmos_request_timeout_seconds,
                ),
                "point read",
            )
        except CosmosResourceNotFoundError:
            return None
        return cast(Mapping[str, Any], result)

    @staticmethod
    def _same_immutable_content(
        expected: RawTransactionDocument,
        existing_payload: Mapping[str, Any],
    ) -> bool:
        try:
            existing = RawTransactionDocument.model_validate(dict(existing_payload))
        except ValidationError:
            return False
        return (
            existing.content_digest == expected.content_digest
            and existing.immutable_payload() == expected.immutable_payload()
        )

    def upsert_document(self, document: Mapping[str, Any]) -> str:
        try:
            expected = RawTransactionDocument.model_validate(dict(document))
        except ValidationError as exc:
            raise CosmosAdapterError("raw document contract validation failed") from exc
        existing = self.read_document(expected.id, expected.sender_id)
        if existing is not None:
            if self._same_immutable_content(expected, existing):
                return "existing"
            raise CosmosDocumentConflictError(
                f"raw document {expected.id} has conflicting immutable content"
            )
        try:
            self._call(
                lambda: self._container.create_item(
                    body=expected.cosmos_payload(),
                    retry_write=0,
                    timeout=self._settings.cosmos_request_timeout_seconds,
                ),
                "document create",
            )
        except CosmosResourceExistsError:
            concurrent = self.read_document(expected.id, expected.sender_id)
            if concurrent is not None and self._same_immutable_content(expected, concurrent):
                return "existing"
            raise CosmosDocumentConflictError(
                f"raw document {expected.id} has conflicting immutable content"
            ) from None
        return "created"

    def _query_container(
        self,
        container: Any,
        query: str,
        parameters: list[dict[str, Any]],
        page_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        continuation: str | None = None
        while True:
            paged = self._call(
                lambda: container.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                    max_item_count=page_size,
                    timeout=self._settings.cosmos_request_timeout_seconds,
                ),
                "document query",
            )
            pages = paged.by_page(continuation_token=continuation)
            try:
                page = self._call(partial(next, pages), "document query page")
            except StopIteration:
                return
            for document in page:
                yield cast(Mapping[str, Any], document)
            continuation = getattr(pages, "continuation_token", None)
            if not continuation:
                return

    def _query(
        self, query: str, parameters: list[dict[str, Any]], page_size: int
    ) -> Iterator[Mapping[str, Any]]:
        return self._query_container(self._container, query, parameters, page_size)

    def iter_batch_documents(self, batch_id: str, *, page_size: int) -> Iterator[Mapping[str, Any]]:
        return self._query(
            "SELECT * FROM c WHERE c.batch_id = @batch_id",
            [{"name": "@batch_id", "value": batch_id}],
            page_size,
        )

    def iter_split_documents(
        self, data_split: str, *, page_size: int
    ) -> Iterator[Mapping[str, Any]]:
        return self._query(
            "SELECT * FROM c WHERE c.data_split = @data_split",
            [{"name": "@data_split", "value": data_split}],
            page_size,
        )

    def iter_documents_by_ids(
        self,
        transaction_ids: list[str],
        *,
        chunk_size: int,
        page_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        if chunk_size <= 0:
            raise ValueError("Cosmos ID lookup chunk size must be positive")
        unique_ids = list(dict.fromkeys(transaction_ids))
        for start in range(0, len(unique_ids), chunk_size):
            chunk = unique_ids[start : start + chunk_size]
            yield from self._query(
                "SELECT c.id FROM c WHERE ARRAY_CONTAINS(@transaction_ids, c.id)",
                [{"name": "@transaction_ids", "value": chunk}],
                page_size,
            )
