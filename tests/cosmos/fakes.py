"""Test-only Cosmos and raw-store doubles."""

from collections.abc import Iterator, Mapping
from typing import Any

from azure.cosmos.exceptions import CosmosResourceExistsError, CosmosResourceNotFoundError

from fraud_detection.persistence.document_vector.models import RawTransactionDocument


class FakePageIterator:
    def __init__(self, pages: list[list[dict[str, Any]]], start: int) -> None:
        self._pages = pages
        self._index = start
        self.continuation_token: str | None = str(start) if start < len(pages) else None

    def __iter__(self) -> "FakePageIterator":
        return self

    def __next__(self) -> list[dict[str, Any]]:
        if self._index >= len(self._pages):
            self.continuation_token = None
            raise StopIteration
        page = self._pages[self._index]
        self._index += 1
        self.continuation_token = str(self._index) if self._index < len(self._pages) else None
        return page


class FakePaged:
    def __init__(self, pages: list[list[dict[str, Any]]]) -> None:
        self._pages = pages

    def by_page(self, continuation_token: str | None = None) -> FakePageIterator:
        return FakePageIterator(self._pages, int(continuation_token or 0))


class FakeContainer:
    def __init__(self) -> None:
        self.properties: dict[str, Any] = {
            "id": "transactions_raw",
            "partitionKey": {"paths": ["/sender_id"]},
        }
        self.documents: dict[tuple[str, str], dict[str, Any]] = {}
        self.read_failures: list[Exception] = []
        self.create_conflict = False
        self.query_calls = 0
        self.queries: list[tuple[str, list[dict[str, Any]]]] = []
        self.vector_results: list[dict[str, Any]] = []

    def read(self, **_: Any) -> dict[str, Any]:
        return self.properties

    def read_item(self, *, item: str, partition_key: str, **_: Any) -> dict[str, Any]:
        if self.read_failures:
            raise self.read_failures.pop(0)
        try:
            return self.documents[(item, partition_key)]
        except KeyError as exc:
            raise CosmosResourceNotFoundError(  # type: ignore[no-untyped-call]
                status_code=404, message="not found"
            ) from exc

    def create_item(self, *, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        key = (str(body["id"]), str(body["sender_id"]))
        if self.create_conflict or key in self.documents:
            raise CosmosResourceExistsError(  # type: ignore[no-untyped-call]
                status_code=409, message="exists"
            )
        self.documents[key] = dict(body)
        return body

    def query_items(
        self,
        *,
        query: str,
        parameters: list[dict[str, Any]],
        max_item_count: int,
        **_: Any,
    ) -> FakePaged:
        self.query_calls += 1
        self.queries.append((query, parameters))
        value = parameters[0]["value"]
        name = parameters[0]["name"]
        if name == "@limit":
            documents = self.vector_results[: int(value)]
        elif name == "@transaction_ids":
            identifiers = set(value)
            identity_field = "transaction_id" if "c.transaction_id" in query else "id"
            documents = [
                {
                    "id": document["id"],
                    "transaction_id": document.get("transaction_id"),
                }
                for document in self.documents.values()
                if document.get(identity_field) in identifiers
            ]
        else:
            field = "batch_id" if name == "@batch_id" else "data_split"
            documents = [
                document for document in self.documents.values() if document.get(field) == value
            ]
        pages = [
            documents[index : index + max_item_count]
            for index in range(0, len(documents), max_item_count)
        ]
        return FakePaged(pages)


class FakeDatabase:
    def __init__(
        self,
        container: FakeContainer,
        containers: dict[str, FakeContainer] | None = None,
    ) -> None:
        self.container = container
        self.containers = containers or {}
        self.failure: Exception | None = None

    def read(self, **_: Any) -> dict[str, Any]:
        if self.failure is not None:
            raise self.failure
        return {"id": "fraud-db"}

    def get_container_client(self, name: str) -> FakeContainer:
        return self.containers.get(name, self.container)


class FakeClient:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database

    def get_database_client(self, _: str) -> FakeDatabase:
        return self.database


class FakeRawStore:
    """Test-only store with restart and fault controls."""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.fail_after: int | None = None
        self.calls = 0

    async def readiness(self) -> Any:
        raise NotImplementedError

    def validate_container(self) -> None:
        return None

    def read_document(self, document_id: str, sender_id: str) -> Mapping[str, Any] | None:
        return next(
            (
                document
                for document in self.documents
                if document.get("id") == document_id and document.get("sender_id") == sender_id
            ),
            None,
        )

    def upsert_document(self, document: Mapping[str, Any]) -> str:
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("controlled projection failure")
        self.calls += 1
        existing = self.read_document(str(document["id"]), str(document["sender_id"]))
        if existing is None:
            self.documents.append(dict(document))
            return "created"
        existing_document = RawTransactionDocument.model_validate(existing)
        expected_document = RawTransactionDocument.model_validate(dict(document))
        if existing_document.immutable_payload() != expected_document.immutable_payload():
            raise RuntimeError("controlled immutable conflict")
        return "existing"

    def iter_batch_documents(self, batch_id: str, *, page_size: int) -> Iterator[Mapping[str, Any]]:
        del page_size
        yield from (document for document in self.documents if document.get("batch_id") == batch_id)

    def iter_split_documents(
        self, data_split: str, *, page_size: int
    ) -> Iterator[Mapping[str, Any]]:
        del page_size
        yield from (
            document for document in self.documents if document.get("data_split") == data_split
        )

    def iter_documents_by_ids(
        self,
        transaction_ids: list[str],
        *,
        chunk_size: int,
        page_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        del page_size
        identifiers = set(transaction_ids)
        for start in range(0, len(transaction_ids), chunk_size):
            chunk = set(transaction_ids[start : start + chunk_size])
            yield from (
                {"id": document.get("id")}
                for document in self.documents
                if document.get("id") in identifiers and document.get("id") in chunk
            )
