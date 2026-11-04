"""Test-only deterministic encoder and embedding store."""

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fraud_detection.persistence.document_vector.embedding_models import EmbeddingDocument


class FakeEncoder:
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model_revision = "fake-test-revision"
    sentence_transformers_version = "test"
    dimension = 384
    normalized = True
    dtype = "float32"

    def encode(self, texts: Sequence[str], *, batch_size: int) -> NDArray[np.float32]:
        del batch_size
        vectors: list[NDArray[np.float32]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = np.frombuffer((digest * 12)[:384], dtype=np.uint8).astype(np.float32)
            raw -= raw.mean()
            raw /= np.linalg.norm(raw)
            vectors.append(raw)
        return np.stack(vectors).astype(np.float32)


class FakeEmbeddingStore:
    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.fail_after: int | None = None
        self.calls = 0

    async def readiness(self) -> Any:
        raise NotImplementedError

    def validate_embedding_container(self) -> None:
        return None

    def read_embedding(self, document_id: str, sender_id: str) -> Mapping[str, Any] | None:
        return next(
            (
                document
                for document in self.documents
                if document.get("id") == document_id and document.get("sender_id") == sender_id
            ),
            None,
        )

    def upsert_embedding(self, document: Mapping[str, Any]) -> str:
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("controlled embedding failure")
        self.calls += 1
        expected = EmbeddingDocument.model_validate(dict(document))
        existing = self.read_embedding(expected.id, expected.sender_id)
        if existing is None:
            self.documents.append(expected.payload())
            return "created"
        stored = EmbeddingDocument.model_validate(dict(existing))
        if stored.immutable_payload() != expected.immutable_payload():
            raise RuntimeError("controlled embedding conflict")
        return "existing"

    def iter_embedding_batch(self, batch_id: str, *, page_size: int) -> Iterator[Mapping[str, Any]]:
        del page_size
        yield from (document for document in self.documents if document.get("batch_id") == batch_id)

    def iter_embeddings_by_transaction_ids(
        self,
        transaction_ids: list[str],
        *,
        chunk_size: int,
        page_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        del page_size
        for start in range(0, len(transaction_ids), chunk_size):
            chunk = set(transaction_ids[start : start + chunk_size])
            yield from (
                {"id": document.get("id"), "transaction_id": document.get("transaction_id")}
                for document in self.documents
                if document.get("transaction_id") in chunk
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
        del (
            query_vector,
            query_step,
            query_transaction_id,
            embedding_version,
            template_version,
            model_revision,
            limit,
        )
        raise NotImplementedError
