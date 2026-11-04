"""Controlled read-only investigator stores."""

from collections.abc import Mapping, Sequence
from typing import Any

from fraud_detection.ports.health import DependencyHealth, DependencyState


class FakeRelational:
    def __init__(self, row: Mapping[str, Any] | None) -> None:
        self.row = dict(row) if row else None
        self.query_count = 0
        self.raise_unavailable = False
        self.summary: dict[str, Any] = {
            "transaction_count": 3,
            "scored_transaction_count": 2,
            "fraud_flag_count": 1,
            "amount_total": "60.00",
            "score_minimum": 0.1,
            "score_maximum": 0.99,
            "score_mean": 0.545,
            "transaction_type_counts": {"PAYMENT": 2, "TRANSFER": 1},
        }
        self.flag_rows: list[dict[str, Any]] = []
        self.snapshot: dict[str, Any] | None = {
            "snapshot_id": "snapshot-1",
            "batch_id": "published-batch",
            "cutoff_step": 9,
            "algorithm_version": "graph-v1",
        }

    def _call(self) -> None:
        self.query_count += 1
        if self.raise_unavailable:
            raise RuntimeError("secret database endpoint")

    async def readiness(self) -> DependencyHealth:
        return DependencyHealth(
            name="postgresql",
            state=(
                DependencyState.UNAVAILABLE if self.raise_unavailable else DependencyState.HEALTHY
            ),
            required=True,
            detail="controlled relational state",
        )

    def transaction_evidence(self, transaction_id: str, model_id: str) -> Mapping[str, Any] | None:
        del model_id
        self._call()
        if self.row and self.row["transaction_id"] == transaction_id:
            return self.row
        return None

    def published_account_exists(self, account_id: str) -> bool:
        self._call()
        return bool(self.row and account_id in {self.row["sender_id"], self.row["receiver_id"]})

    def snapshot_lineage(self, snapshot_id: str | None) -> Mapping[str, Any] | None:
        self._call()
        if snapshot_id and self.snapshot and snapshot_id != self.snapshot["snapshot_id"]:
            return None
        return self.snapshot

    def dashboard_summary(self, filters: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
        del filters, model_id
        self._call()
        return self.summary

    def flag_page(
        self,
        filters: Mapping[str, Any],
        position: Mapping[str, Any] | None,
        limit: int,
        model_id: str,
    ) -> Sequence[Mapping[str, Any]]:
        del filters, model_id
        self._call()
        rows = self.flag_rows
        if position:
            key = (
                int(position["step"]),
                str(position["transaction_id"]),
                str(position["score_id"]),
            )
            rows = [
                row
                for row in rows
                if (int(row["step"]), str(row["transaction_id"]), str(row["score_id"])) > key
            ]
        return rows[: limit + 1]

    def freshness(self, model_id: str) -> Mapping[str, Any]:
        self._call()
        return {
            "latest_published_batch": "published-batch" if self.row else None,
            "published_cutoff_step": 10 if self.row else None,
            "pending_batch_count": 18,
            "model_version_id": model_id,
            "training_cutoff_step": 281,
            "feature_set": "combined",
            "embedding_version": "test-embedding-v1",
        }


class FakeRaw:
    def __init__(self, document: Mapping[str, Any] | None) -> None:
        self.document = dict(document) if document else None
        self.fail = False
        self.calls = 0

    def read_document(self, document_id: str, sender_id: str) -> Mapping[str, Any] | None:
        del document_id, sender_id
        self.calls += 1
        if self.fail:
            raise TimeoutError("secret raw endpoint")
        return self.document


class FakeGraph:
    def __init__(self) -> None:
        self.transaction_values: list[dict[str, Any]] = []
        self.related_values: list[dict[str, Any]] = []
        self.fail = False
        self.calls = 0
        self.last_cutoff: int | None = None

    def transaction_evidence(self, transaction_id: str, *, limit: int) -> list[dict[str, Any]]:
        del transaction_id
        self.calls += 1
        if self.fail:
            raise TimeoutError("secret graph endpoint")
        return self.transaction_values[:limit]

    def related_accounts(
        self,
        account_id: str,
        *,
        relationship_type: str,
        cutoff_step: int,
        minimum_weight: float,
        offset: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        del account_id, relationship_type
        self.calls += 1
        self.last_cutoff = cutoff_step
        if self.fail:
            raise TimeoutError("secret graph endpoint")
        values = [
            value
            for value in self.related_values
            if float(value["evidence_weight"]) >= minimum_weight
        ]
        return values[offset : offset + limit]


class FakeVectors:
    def __init__(self, document: Mapping[str, Any] | None) -> None:
        self.document = dict(document) if document else None
        self.results: list[dict[str, Any]] = []
        self.fail = False
        self.calls = 0

    def read_embedding(self, document_id: str, sender_id: str) -> Mapping[str, Any] | None:
        del document_id, sender_id
        self.calls += 1
        if self.fail:
            raise TimeoutError("secret vector endpoint")
        return self.document

    def vector_search(self, **_: Any) -> list[dict[str, Any]]:
        self.calls += 1
        if self.fail:
            raise TimeoutError("secret vector endpoint")
        return self.results
