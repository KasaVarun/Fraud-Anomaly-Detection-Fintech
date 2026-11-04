"""Idempotent immutable model-version and validation-evaluation persistence."""

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from sqlalchemy import Connection, Engine, create_engine, func, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert

from fraud_detection.persistence.relational import Base

model_version_table = Base.metadata.tables["model_version"]
evaluation_table = Base.metadata.tables["model_evaluation"]
ingestion_batch_table = Base.metadata.tables["ingestion_batch"]
score_table = Base.metadata.tables["transaction_score"]
flag_table = Base.metadata.tables["fraud_flag"]


def _insert_verify(
    connection: Connection,
    table: Any,
    record: dict[str, Any],
    key: str,
    immutable_fields: tuple[str, ...],
) -> None:
    connection.execute(
        postgres_insert(table).values(record).on_conflict_do_nothing(index_elements=[key])
    )
    stored = connection.execute(select(table).where(table.c[key] == record[key])).mappings().one()
    for field in immutable_fields:
        expected = record[field]
        actual = stored[field]
        if isinstance(expected, float | Decimal):
            if Decimal(str(actual)) != Decimal(str(expected)):
                raise ValueError(f"immutable {table.name} conflicts on {field}")
        elif actual != expected:
            raise ValueError(f"immutable {table.name} conflicts on {field}")


def persist_model_release_records(
    database_url: str,
    batch_id: str,
    model_records: list[dict[str, Any]],
    evaluation_records: list[dict[str, Any]],
) -> tuple[int, int, int, int, str]:
    engine: Engine = create_engine(database_url, hide_parameters=True)
    try:
        with engine.begin() as connection:
            status = connection.scalar(
                select(ingestion_batch_table.c.status).where(
                    ingestion_batch_table.c.batch_id == batch_id
                )
            )
            if status != "pending":
                raise ValueError("model release persistence requires a pending batch")
            for record in model_records:
                _insert_verify(
                    connection,
                    model_version_table,
                    record,
                    "model_version_id",
                    (
                        "feature_set",
                        "training_cutoff_step",
                        "threshold",
                        "embedding_version",
                        "artifact_uri",
                    ),
                )
            for record in evaluation_records:
                if record["split_name"] == "test":
                    raise ValueError("Phase 7 cannot persist test evaluations")
                _insert_verify(
                    connection,
                    evaluation_table,
                    record,
                    "evaluation_id",
                    (
                        "model_version_id",
                        "batch_id",
                        "split_name",
                        "tp",
                        "fp",
                        "tn",
                        "fn",
                        "precision",
                        "recall",
                        "f1",
                        "pr_auc",
                    ),
                )
            model_count = int(
                connection.scalar(
                    select(func.count())
                    .select_from(model_version_table)
                    .where(
                        model_version_table.c.model_version_id.in_(
                            [record["model_version_id"] for record in model_records]
                        )
                    )
                )
                or 0
            )
            evaluation_count = int(
                connection.scalar(
                    select(func.count())
                    .select_from(evaluation_table)
                    .where(
                        evaluation_table.c.evaluation_id.in_(
                            [record["evaluation_id"] for record in evaluation_records]
                        )
                    )
                )
                or 0
            )
            score_count = int(connection.scalar(select(func.count()).select_from(score_table)) or 0)
            flag_count = int(connection.scalar(select(func.count()).select_from(flag_table)) or 0)
            final_status = str(
                connection.scalar(
                    select(ingestion_batch_table.c.status).where(
                        ingestion_batch_table.c.batch_id == batch_id
                    )
                )
            )
        return model_count, evaluation_count, score_count, flag_count, final_status
    finally:
        engine.dispose()


def evaluation_record_matches(record: Mapping[str, Any], stored: Mapping[str, Any]) -> bool:
    fields = ("tp", "fp", "tn", "fn", "precision", "recall", "f1", "pr_auc")
    return all(Decimal(str(record[field])) == Decimal(str(stored[field])) for field in fields)
