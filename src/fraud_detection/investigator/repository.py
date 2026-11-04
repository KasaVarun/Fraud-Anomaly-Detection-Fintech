"""Bounded published-only PostgreSQL investigator repository."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import Engine, Select, and_, create_engine, func, or_, select
from sqlalchemy.exc import SQLAlchemyError

from fraud_detection.persistence.relational import Base
from fraud_detection.ports.health import DependencyHealth, DependencyState

ingestion_batch = Base.metadata.tables["ingestion_batch"]
account = Base.metadata.tables["account"]
merchant = Base.metadata.tables["merchant"]
transaction = Base.metadata.tables["transaction"]
model_version = Base.metadata.tables["model_version"]
transaction_score = Base.metadata.tables["transaction_score"]
fraud_flag = Base.metadata.tables["fraud_flag"]
graph_snapshot = Base.metadata.tables["graph_snapshot"]


class UnavailableInvestigatorRepository:
    async def readiness(self) -> DependencyHealth:
        return DependencyHealth(
            name="postgresql",
            state=DependencyState.NOT_CONFIGURED,
            required=False,
            detail="not configured for this process",
        )

    def _raise(self) -> None:
        raise RuntimeError("relational investigator repository is unavailable")

    def transaction_evidence(self, transaction_id: str, model_id: str) -> None:
        del transaction_id, model_id
        self._raise()

    def published_account_exists(self, account_id: str) -> bool:
        del account_id
        self._raise()
        return False

    def snapshot_lineage(self, snapshot_id: str | None) -> None:
        del snapshot_id
        self._raise()

    def dashboard_summary(self, filters: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
        del filters, model_id
        self._raise()
        return {}

    def flag_page(
        self,
        filters: Mapping[str, Any],
        position: Mapping[str, Any] | None,
        limit: int,
        model_id: str,
    ) -> Sequence[Mapping[str, Any]]:
        del filters, position, limit, model_id
        self._raise()
        return []

    def freshness(self, model_id: str) -> Mapping[str, Any]:
        del model_id
        self._raise()
        return {}


class PostgresInvestigatorRepository:
    """Read-only queries with published-batch predicates inside every evidence query."""

    def __init__(self, database_url: str) -> None:
        self._engine: Engine = create_engine(database_url, hide_parameters=True)
        self.query_count = 0

    def _mappings(self, statement: Select[Any]) -> list[Mapping[str, Any]]:
        self.query_count += 1
        with self._engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]

    async def readiness(self) -> DependencyHealth:
        try:
            self._mappings(select(ingestion_batch.c.batch_id).limit(1))
        except SQLAlchemyError:
            return DependencyHealth(
                name="postgresql",
                state=DependencyState.UNAVAILABLE,
                required=True,
                detail="configured PostgreSQL is unavailable",
            )
        return DependencyHealth(
            name="postgresql",
            state=DependencyState.HEALTHY,
            required=True,
            detail="published-data query boundary is available",
        )

    def transaction_evidence(self, transaction_id: str, model_id: str) -> Mapping[str, Any] | None:
        sender_merchant = merchant.alias("sender_merchant")
        receiver_merchant = merchant.alias("receiver_merchant")
        latest_score_id = (
            select(transaction_score.c.score_id)
            .where(
                transaction_score.c.transaction_id == transaction.c.transaction_id,
                transaction_score.c.model_version_id == model_id,
            )
            .order_by(transaction_score.c.scored_at.desc(), transaction_score.c.score_id.desc())
            .limit(1)
            .correlate(transaction)
            .scalar_subquery()
        )
        statement = (
            select(
                transaction.c.transaction_id,
                transaction.c.batch_id,
                transaction.c.source_row,
                transaction.c.sender_id,
                transaction.c.receiver_id,
                transaction.c.step,
                transaction.c.transaction_type,
                transaction.c.amount,
                transaction.c.data_split,
                ingestion_batch.c.source_sha256.label("batch_source_sha256"),
                ingestion_batch.c.sample_seed.label("batch_sample_seed"),
                ingestion_batch.c.enrichment_seed.label("batch_enrichment_seed"),
                ingestion_batch.c.split_policy.label("batch_split_policy"),
                sender_merchant.c.account_id.label("sender_merchant_id"),
                receiver_merchant.c.account_id.label("receiver_merchant_id"),
                transaction_score.c.score_id,
                transaction_score.c.fraud_probability,
                transaction_score.c.vector_distance,
                transaction_score.c.input_version,
                transaction_score.c.snapshot_id,
                transaction_score.c.scored_at,
                model_version.c.model_version_id,
                model_version.c.feature_set,
                model_version.c.training_cutoff_step,
                model_version.c.threshold,
                model_version.c.embedding_version,
                model_version.c.artifact_uri,
                fraud_flag.c.flag_id,
                fraud_flag.c.reason.label("flag_reason"),
                fraud_flag.c.review_status,
                graph_snapshot.c.cutoff_step.label("snapshot_cutoff_step"),
                graph_snapshot.c.algorithm_version.label("snapshot_algorithm_version"),
            )
            .join(ingestion_batch, ingestion_batch.c.batch_id == transaction.c.batch_id)
            .outerjoin(sender_merchant, sender_merchant.c.account_id == transaction.c.sender_id)
            .outerjoin(
                receiver_merchant, receiver_merchant.c.account_id == transaction.c.receiver_id
            )
            .outerjoin(
                transaction_score,
                transaction_score.c.score_id == latest_score_id,
            )
            .outerjoin(
                model_version,
                model_version.c.model_version_id == transaction_score.c.model_version_id,
            )
            .outerjoin(fraud_flag, fraud_flag.c.score_id == transaction_score.c.score_id)
            .outerjoin(
                graph_snapshot,
                graph_snapshot.c.snapshot_id == transaction_score.c.snapshot_id,
            )
            .where(
                transaction.c.transaction_id == transaction_id,
                ingestion_batch.c.status == "published",
            )
            .order_by(
                transaction_score.c.scored_at.desc().nullslast(),
                transaction_score.c.score_id.desc().nullslast(),
            )
            .limit(1)
        )
        values = self._mappings(statement)
        return values[0] if values else None

    def published_account_exists(self, account_id: str) -> bool:
        statement = (
            select(transaction.c.transaction_id)
            .join(ingestion_batch, ingestion_batch.c.batch_id == transaction.c.batch_id)
            .where(
                ingestion_batch.c.status == "published",
                or_(
                    transaction.c.sender_id == account_id,
                    transaction.c.receiver_id == account_id,
                ),
            )
            .limit(1)
        )
        return bool(self._mappings(statement))

    def snapshot_lineage(self, snapshot_id: str | None) -> Mapping[str, Any] | None:
        statement = (
            select(
                graph_snapshot.c.snapshot_id,
                graph_snapshot.c.batch_id,
                graph_snapshot.c.cutoff_step,
                graph_snapshot.c.algorithm_version,
                graph_snapshot.c.computed_at,
            )
            .join(ingestion_batch, ingestion_batch.c.batch_id == graph_snapshot.c.batch_id)
            .where(ingestion_batch.c.status == "published")
            .order_by(graph_snapshot.c.cutoff_step.desc(), graph_snapshot.c.snapshot_id.desc())
            .limit(1)
        )
        if snapshot_id is not None:
            statement = statement.where(graph_snapshot.c.snapshot_id == snapshot_id)
        values = self._mappings(statement)
        return values[0] if values else None

    def _filtered_base(self, filters: Mapping[str, Any], model_id: str) -> Select[Any]:
        latest_score_id = (
            select(transaction_score.c.score_id)
            .where(
                transaction_score.c.transaction_id == transaction.c.transaction_id,
                transaction_score.c.model_version_id == model_id,
            )
            .order_by(transaction_score.c.scored_at.desc(), transaction_score.c.score_id.desc())
            .limit(1)
            .correlate(transaction)
            .scalar_subquery()
        )
        statement = (
            select(
                transaction.c.transaction_id,
                transaction.c.batch_id,
                transaction.c.step,
                transaction.c.transaction_type,
                transaction.c.amount,
                transaction.c.sender_id,
                transaction.c.receiver_id,
                transaction_score.c.score_id,
                transaction_score.c.fraud_probability,
                transaction_score.c.snapshot_id,
                transaction_score.c.scored_at,
                model_version.c.threshold,
                model_version.c.model_version_id,
                fraud_flag.c.flag_id,
                fraud_flag.c.reason.label("flag_reason"),
                fraud_flag.c.review_status,
            )
            .join(ingestion_batch, ingestion_batch.c.batch_id == transaction.c.batch_id)
            .outerjoin(
                transaction_score,
                transaction_score.c.score_id == latest_score_id,
            )
            .outerjoin(
                model_version,
                model_version.c.model_version_id == transaction_score.c.model_version_id,
            )
            .outerjoin(fraud_flag, fraud_flag.c.score_id == transaction_score.c.score_id)
            .where(ingestion_batch.c.status == "published")
        )
        if filters.get("start_step") is not None:
            statement = statement.where(transaction.c.step >= int(filters["start_step"]))
        if filters.get("end_step") is not None:
            statement = statement.where(transaction.c.step <= int(filters["end_step"]))
        if filters.get("transaction_type"):
            statement = statement.where(
                transaction.c.transaction_type == str(filters["transaction_type"])
            )
        if filters.get("flagged") is True:
            statement = statement.where(fraud_flag.c.flag_id.is_not(None))
        if filters.get("flagged") is False:
            statement = statement.where(fraud_flag.c.flag_id.is_(None))
        return statement

    def dashboard_summary(self, filters: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
        base = self._filtered_base(filters, model_id).subquery()
        aggregate = select(
            func.count(base.c.transaction_id).label("transaction_count"),
            func.count(base.c.score_id).label("scored_transaction_count"),
            func.count(base.c.flag_id).label("fraud_flag_count"),
            func.coalesce(func.sum(base.c.amount), 0).label("amount_total"),
            func.min(base.c.fraud_probability).label("score_minimum"),
            func.max(base.c.fraud_probability).label("score_maximum"),
            func.avg(base.c.fraud_probability).label("score_mean"),
        )
        aggregate_rows = self._mappings(aggregate)
        type_rows = self._mappings(
            select(
                base.c.transaction_type,
                func.count(base.c.transaction_id).label("transaction_count"),
            )
            .group_by(base.c.transaction_type)
            .order_by(base.c.transaction_type)
        )
        value = dict(aggregate_rows[0])
        value["transaction_type_counts"] = {
            str(row["transaction_type"]): int(row["transaction_count"]) for row in type_rows
        }
        return value

    def flag_page(
        self,
        filters: Mapping[str, Any],
        position: Mapping[str, Any] | None,
        limit: int,
        model_id: str,
    ) -> Sequence[Mapping[str, Any]]:
        statement = self._filtered_base(filters, model_id).where(fraud_flag.c.flag_id.is_not(None))
        if filters.get("review_status"):
            statement = statement.where(fraud_flag.c.review_status == str(filters["review_status"]))
        if filters.get("minimum_score") is not None:
            statement = statement.where(
                transaction_score.c.fraud_probability >= float(filters["minimum_score"])
            )
        if position is not None:
            step = int(position["step"])
            identifier = str(position["transaction_id"])
            score_identifier = str(position["score_id"])
            statement = statement.where(
                or_(
                    transaction.c.step > step,
                    and_(
                        transaction.c.step == step,
                        transaction.c.transaction_id > identifier,
                    ),
                    and_(
                        transaction.c.step == step,
                        transaction.c.transaction_id == identifier,
                        transaction_score.c.score_id > score_identifier,
                    ),
                )
            )
        statement = statement.order_by(
            transaction.c.step,
            transaction.c.transaction_id,
            transaction_score.c.score_id,
        ).limit(limit + 1)
        return self._mappings(statement)

    def freshness(self, model_id: str) -> Mapping[str, Any]:
        pending = (
            select(func.count())
            .select_from(ingestion_batch)
            .where(ingestion_batch.c.status == "pending")
        )
        latest = (
            select(
                ingestion_batch.c.batch_id,
                func.max(transaction.c.step).label("published_cutoff_step"),
                pending.scalar_subquery().label("pending_batch_count"),
            )
            .outerjoin(transaction, transaction.c.batch_id == ingestion_batch.c.batch_id)
            .where(ingestion_batch.c.status == "published")
            .group_by(ingestion_batch.c.batch_id, ingestion_batch.c.created_at)
            .order_by(ingestion_batch.c.created_at.desc(), ingestion_batch.c.batch_id.desc())
            .limit(1)
        )
        rows = self._mappings(latest)
        model_rows = self._mappings(
            select(
                model_version.c.model_version_id,
                model_version.c.training_cutoff_step,
                model_version.c.feature_set,
                model_version.c.embedding_version,
            ).where(model_version.c.model_version_id == model_id)
        )
        value: dict[str, Any] = {
            "latest_published_batch": None,
            "published_cutoff_step": None,
            "pending_batch_count": 0,
            "model_version_id": model_id,
            "training_cutoff_step": None,
            "feature_set": None,
            "embedding_version": None,
        }
        if rows:
            value["latest_published_batch"] = rows[0]["batch_id"]
            value["published_cutoff_step"] = rows[0]["published_cutoff_step"]
            value["pending_batch_count"] = int(rows[0]["pending_batch_count"] or 0)
        else:
            pending_rows = self._mappings(pending)
            value["pending_batch_count"] = int(pending_rows[0]["count_1"])
        if model_rows:
            value.update(dict(model_rows[0]))
        return value
