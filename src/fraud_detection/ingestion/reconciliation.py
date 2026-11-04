"""PostgreSQL-only verification for the Phase 3 initial load."""

from pathlib import Path
from typing import Any, cast

import polars as pl
from sqlalchemy import Connection, Engine, create_engine, text

from fraud_detection.ingestion.errors import ReconciliationError
from fraud_detection.ingestion.loader import initial_batch_id
from fraud_detection.ingestion.manifest import (
    sample_path_from_manifest,
    validate_manifest_artifacts,
)
from fraud_detection.ingestion.models import ReconciliationReport


def _scalar(connection: Connection, sql: str, parameters: dict[str, object]) -> int:
    value: Any = connection.execute(text(sql), parameters).scalar_one()
    return int(value)


def reconcile_initial_load(
    source_path: Path,
    manifest_path: Path,
    database_url: str,
) -> ReconciliationReport:
    manifest = validate_manifest_artifacts(source_path, manifest_path)
    sample = (
        pl.scan_parquet(sample_path_from_manifest(manifest, manifest_path))
        .filter(pl.col("data_split").is_in(["train", "validation"]))
        .collect(engine="streaming")
    )
    batch_id = initial_batch_id(manifest)
    expected_rows = sample.height
    expected_fraud = {
        split: int(sample.filter(pl.col("data_split") == split)["isFraud"].sum())
        for split in ("train", "validation")
    }
    expected_source_min = (
        int(sample.select(pl.col("source_row").min()).item()) if sample.height else None
    )
    expected_source_max = (
        int(sample.select(pl.col("source_row").max()).item()) if sample.height else None
    )

    engine: Engine = create_engine(database_url, hide_parameters=True)
    try:
        with engine.connect() as connection:
            batch = (
                connection.execute(
                    text("SELECT status FROM ingestion_batch WHERE batch_id = :batch_id"),
                    {"batch_id": batch_id},
                )
                .mappings()
                .one_or_none()
            )
            if batch is None:
                raise ReconciliationError("expected initial ingestion batch is missing")
            parameters: dict[str, object] = {"batch_id": batch_id}
            transaction_count = _scalar(
                connection,
                'SELECT count(*) FROM "transaction" WHERE batch_id = :batch_id',
                parameters,
            )
            label_count = _scalar(
                connection,
                """SELECT count(*) FROM transaction_label label
                JOIN "transaction" tx USING (transaction_id)
                WHERE tx.batch_id = :batch_id""",
                parameters,
            )
            sender_count = _scalar(
                connection,
                'SELECT count(DISTINCT sender_id) FROM "transaction" WHERE batch_id = :batch_id',
                parameters,
            )
            receiver_count = _scalar(
                connection,
                'SELECT count(DISTINCT receiver_id) FROM "transaction" WHERE batch_id = :batch_id',
                parameters,
            )
            participant_cte = """WITH participants AS (
                SELECT sender_id AS account_id FROM "transaction" WHERE batch_id = :batch_id
                UNION
                SELECT receiver_id AS account_id FROM "transaction" WHERE batch_id = :batch_id
            )"""
            merchant_account_count = _scalar(
                connection,
                participant_cte
                + " SELECT count(*) FROM account JOIN participants USING (account_id)"
                + " WHERE account_kind = 'merchant'",
                parameters,
            )
            merchant_subtype_count = _scalar(
                connection,
                participant_cte
                + " SELECT count(*) FROM merchant JOIN participants USING (account_id)",
                parameters,
            )
            fraud_rows = connection.execute(
                text(
                    """SELECT tx.data_split, count(*)
                    FROM "transaction" tx
                    JOIN transaction_label label USING (transaction_id)
                    WHERE tx.batch_id = :batch_id AND label.is_fraud
                    GROUP BY tx.data_split"""
                ),
                parameters,
            ).all()
            fraud_by_split = {"train": 0, "validation": 0}
            fraud_by_split.update(
                {str(split): int(cast(Any, count)) for split, count in fraud_rows}
            )
            source_range_row = connection.execute(
                text(
                    'SELECT min(source_row), max(source_row) FROM "transaction" '
                    "WHERE batch_id = :batch_id"
                ),
                parameters,
            ).one()
            source_range = (
                int(cast(Any, source_range_row[0])) if source_range_row[0] is not None else None,
                int(cast(Any, source_range_row[1])) if source_range_row[1] is not None else None,
            )
            duplicate_transaction_ids = _scalar(
                connection,
                """SELECT count(*) FROM (
                    SELECT transaction_id FROM "transaction"
                    WHERE batch_id = :batch_id GROUP BY transaction_id HAVING count(*) > 1
                ) duplicates""",
                parameters,
            )
            duplicate_batch_rows = _scalar(
                connection,
                """SELECT count(*) FROM (
                    SELECT batch_id, source_row FROM "transaction"
                    WHERE batch_id = :batch_id GROUP BY batch_id, source_row HAVING count(*) > 1
                ) duplicates""",
                parameters,
            )
            test_count = _scalar(
                connection,
                """SELECT count(*) FROM "transaction"
                WHERE batch_id = :batch_id AND data_split = 'test'""",
                parameters,
            )
    finally:
        engine.dispose()

    checks = [
        str(batch["status"]) == "pending",
        transaction_count == expected_rows,
        label_count == expected_rows,
        fraud_by_split.get("train", 0) == expected_fraud["train"],
        fraud_by_split.get("validation", 0) == expected_fraud["validation"],
        source_range[0] == expected_source_min,
        source_range[1] == expected_source_max,
        duplicate_transaction_ids == 0,
        duplicate_batch_rows == 0,
        test_count == 0,
        merchant_account_count == merchant_subtype_count,
    ]
    report = ReconciliationReport(
        batch_id=batch_id,
        status=str(batch["status"]),
        expected_train_validation_rows=expected_rows,
        loaded_transaction_count=transaction_count,
        loaded_label_count=label_count,
        distinct_sender_accounts=sender_count,
        distinct_receiver_accounts=receiver_count,
        merchant_account_count=merchant_account_count,
        merchant_subtype_count=merchant_subtype_count,
        fraud_count_by_split=fraud_by_split,
        source_row_minimum=source_range[0],
        source_row_maximum=source_range[1],
        duplicate_transaction_id_count=duplicate_transaction_ids,
        duplicate_batch_source_row_count=duplicate_batch_rows,
        test_row_count=test_count,
        reconciled=all(checks),
    )
    if not report.reconciled:
        raise ReconciliationError(report.model_dump_json())
    return report
