"""Ports for the approved persistence and reconciliation boundaries."""

from collections.abc import Iterator, Mapping
from typing import Any, Protocol, runtime_checkable

from fraud_detection.ports.health import DependencyHealth


@runtime_checkable
class RelationalPersistencePort(Protocol):
    async def readiness(self) -> DependencyHealth:
        """Report PostgreSQL availability without exposing credentials."""


@runtime_checkable
class DocumentVectorPersistencePort(Protocol):
    async def readiness(self) -> DependencyHealth:
        """Report Cosmos NoSQL/vector availability."""


@runtime_checkable
class RawDocumentPersistencePort(Protocol):
    async def readiness(self) -> DependencyHealth:
        """Report the real Cosmos raw-container readiness state."""

    def validate_container(self) -> None:
        """Validate the configured database, container, and partition key."""

    def read_document(self, document_id: str, sender_id: str) -> Mapping[str, Any] | None:
        """Point-read a raw document by ID and sender partition key."""

    def upsert_document(self, document: Mapping[str, Any]) -> str:
        """Create an absent document or accept identical immutable content."""

    def iter_batch_documents(self, batch_id: str, *, page_size: int) -> Iterator[Mapping[str, Any]]:
        """Enumerate one projected batch across partitions in bounded pages."""

    def iter_split_documents(
        self, data_split: str, *, page_size: int
    ) -> Iterator[Mapping[str, Any]]:
        """Enumerate documents for one split across partitions."""

    def iter_documents_by_ids(
        self,
        transaction_ids: list[str],
        *,
        chunk_size: int,
        page_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        """Find IDs anywhere in the container using bounded parameterized queries."""


@runtime_checkable
class EmbeddingPersistencePort(Protocol):
    async def readiness(self) -> DependencyHealth:
        """Report the real embedding-container readiness state."""

    def validate_embedding_container(self) -> None:
        """Validate partition key and 384-dimensional vector policy."""

    def read_embedding(self, document_id: str, sender_id: str) -> Mapping[str, Any] | None:
        """Point-read one embedding document."""

    def upsert_embedding(self, document: Mapping[str, Any]) -> str:
        """Create an absent embedding or accept identical immutable content."""

    def iter_embedding_batch(self, batch_id: str, *, page_size: int) -> Iterator[Mapping[str, Any]]:
        """Enumerate one embedding versioned batch."""

    def iter_embeddings_by_transaction_ids(
        self,
        transaction_ids: list[str],
        *,
        chunk_size: int,
        page_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        """Find embedding documents by transaction IDs across partitions."""

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
        """Return bounded earlier-training-fraud vector matches."""


@runtime_checkable
class GraphPersistencePort(Protocol):
    async def readiness(self) -> DependencyHealth:
        """Report Cosmos Gremlin availability."""


@runtime_checkable
class GraphEvidencePersistencePort(Protocol):
    async def readiness(self) -> DependencyHealth:
        """Report real Cosmos Gremlin graph readiness."""

    def validate_graph(self) -> None:
        """Run a bounded graph traversal to validate connectivity."""

    def upsert_vertex(self, vertex: Mapping[str, Any]) -> str:
        """Create a vertex or accept identical immutable evidence."""

    def upsert_edge(self, edge: Mapping[str, Any]) -> str:
        """Create an edge or accept identical immutable evidence."""

    def read_vertex(self, vertex_id: str, pk: str) -> Mapping[str, Any] | None:
        """Read one deterministic vertex."""

    def read_edge(self, edge_id: str) -> Mapping[str, Any] | None:
        """Read one deterministic evidence edge."""

    def iter_transaction_edges(
        self, batch_id: str, *, page_size: int
    ) -> Iterator[Mapping[str, Any]]:
        """Enumerate projected money-flow edges for one batch."""

    def iter_batch_edges(self, batch_id: str, *, page_size: int) -> Iterator[Mapping[str, Any]]:
        """Enumerate all evidence edges for one projected batch."""

    def iter_vertices_by_ids(
        self,
        vertex_ids: list[str],
        *,
        chunk_size: int,
        page_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        """Enumerate required vertices using bounded ID chunks."""

    def iter_elements_by_transaction_ids(
        self,
        transaction_ids: list[str],
        *,
        chunk_size: int,
        page_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        """Find any vertex/edge carrying one of the transaction IDs."""

    def iter_evidence_before(
        self, cutoff_step: int, *, page_size: int
    ) -> Iterator[Mapping[str, Any]]:
        """Enumerate evidence with strict step less than the cutoff."""


@runtime_checkable
class ModelArtifactPort(Protocol):
    async def readiness(self) -> DependencyHealth:
        """Report model-artifact storage availability."""


@runtime_checkable
class BatchReconciliationPort(Protocol):
    async def readiness(self) -> DependencyHealth:
        """Report batch reconciliation capability availability."""
