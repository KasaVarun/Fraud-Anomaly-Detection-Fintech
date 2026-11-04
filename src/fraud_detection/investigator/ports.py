"""Typed read-only Phase 9 persistence ports."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from fraud_detection.ports.health import DependencyHealth


class InvestigatorRelationalPort(Protocol):
    async def readiness(self) -> DependencyHealth: ...

    def transaction_evidence(
        self, transaction_id: str, model_id: str
    ) -> Mapping[str, Any] | None: ...

    def published_account_exists(self, account_id: str) -> bool: ...

    def snapshot_lineage(self, snapshot_id: str | None) -> Mapping[str, Any] | None: ...

    def dashboard_summary(self, filters: Mapping[str, Any], model_id: str) -> Mapping[str, Any]: ...

    def flag_page(
        self,
        filters: Mapping[str, Any],
        position: Mapping[str, Any] | None,
        limit: int,
        model_id: str,
    ) -> Sequence[Mapping[str, Any]]: ...

    def freshness(self, model_id: str) -> Mapping[str, Any]: ...


class RawEvidencePort(Protocol):
    def read_document(self, document_id: str, sender_id: str) -> Mapping[str, Any] | None: ...


class VectorEvidencePort(Protocol):
    def read_embedding(self, document_id: str, sender_id: str) -> Mapping[str, Any] | None: ...

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
    ) -> Sequence[Mapping[str, Any]]: ...


class GraphInvestigatorPort(Protocol):
    def transaction_evidence(
        self, transaction_id: str, *, limit: int
    ) -> Sequence[Mapping[str, Any]]: ...

    def related_accounts(
        self,
        account_id: str,
        *,
        relationship_type: str,
        cutoff_step: int,
        minimum_weight: float,
        offset: int,
        limit: int,
    ) -> Sequence[Mapping[str, Any]]: ...
