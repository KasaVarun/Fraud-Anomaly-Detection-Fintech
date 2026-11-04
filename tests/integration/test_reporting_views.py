"""PostgreSQL-backed Phase 10 reporting-view tests."""

from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import Engine, inspect, text

from fraud_detection.reporting.contracts import APPROVED_LINEAGE, REPORTING_VIEW_COLUMNS
from tests.integration.conftest import run_alembic, run_alembic_downgrade

pytestmark = pytest.mark.integration

APPLICATION_TABLES = {
    "ingestion_batch",
    "account",
    "merchant",
    "transaction",
    "transaction_label",
    "model_version",
    "transaction_score",
    "fraud_flag",
    "graph_snapshot",
    "community",
    "account_risk_profile",
    "model_evaluation",
}


@pytest.fixture
def reporting_database(postgres_engine: Engine) -> Iterator[Engine]:
    now = datetime(2026, 10, 31, tzinfo=UTC)
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))
        connection.execute(text("DELETE FROM model_version"))
        connection.execute(
            text(
                """INSERT INTO ingestion_batch
                (batch_id, source_sha256, sample_seed, chunk_id, enrichment_seed, split_policy,
                 status, created_at)
                VALUES
                ('published-batch', :published_sha, 73750, 'published', 73751,
                 'whole-step-v1', 'published', :now),
                ('pending-batch', :pending_sha, 73750, 'pending', 73751,
                 'whole-step-v1', 'pending', :now)"""
            ),
            {"published_sha": "a" * 64, "pending_sha": "b" * 64, "now": now},
        )
        connection.execute(
            text(
                """INSERT INTO account (account_id, account_kind) VALUES
                ('C1', 'customer'), ('C2', 'customer'), ('C3', 'customer')"""
            )
        )
        connection.execute(
            text(
                """INSERT INTO "transaction"
                (transaction_id, batch_id, source_row, sender_id, receiver_id, step,
                 transaction_type, amount, data_split)
                VALUES
                ('tx-published', 'published-batch', 1, 'C1', 'C2', 10, 'TRANSFER', 25.00,
                 'test'),
                ('tx-pending', 'pending-batch', 2, 'C2', 'C3', 11, 'PAYMENT', 50.00, 'test')"""
            )
        )
        connection.execute(
            text(
                """INSERT INTO model_version
                (model_version_id, feature_set, training_cutoff_step, threshold,
                 embedding_version, artifact_uri, created_at)
                VALUES ('model-reporting', 'combined-v2', 281, 0.8000, 'embedding-v1',
                        'local://ignored', :now)"""
            ),
            {"now": now},
        )
        connection.execute(
            text(
                """INSERT INTO transaction_score
                (score_id, transaction_id, model_version_id, snapshot_id, input_version,
                 fraud_probability, vector_distance, scored_at)
                VALUES
                ('score-published', 'tx-published', 'model-reporting', NULL, 'input-v1',
                 0.9000000, 0.2, :now),
                ('score-pending', 'tx-pending', 'model-reporting', NULL, 'input-v1',
                 0.9500000, 0.1, :now)"""
            ),
            {"now": now},
        )
        connection.execute(
            text(
                """INSERT INTO fraud_flag (flag_id, score_id, reason, review_status) VALUES
                ('flag-published', 'score-published', 'locked threshold', 'open'),
                ('flag-pending', 'score-pending', 'locked threshold', 'open')"""
            )
        )
        connection.execute(
            text(
                """INSERT INTO model_evaluation
                (evaluation_id, model_version_id, batch_id, split_name, tp, fp, tn, fn,
                 precision, recall, f1, pr_auc)
                VALUES
                ('eval-published', 'model-reporting', 'published-batch', 'test', 1, 2, 3, 4,
                 0.333, 0.2, 0.25, 0.4),
                ('eval-pending', 'model-reporting', 'pending-batch', 'test', 5, 6, 7, 8,
                 0.455, 0.385, 0.417, 0.5)"""
            )
        )
    yield postgres_engine
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))
        connection.execute(text("DELETE FROM model_version"))


def test_reporting_revision_downgrade_removes_only_views(postgres_engine: Engine) -> None:
    run_alembic_downgrade("20261031_0001")
    assert not set(REPORTING_VIEW_COLUMNS) & set(inspect(postgres_engine).get_view_names())
    assert (
        set(inspect(postgres_engine).get_table_names()) - {"alembic_version"} == APPLICATION_TABLES
    )
    run_alembic("head")
    assert set(REPORTING_VIEW_COLUMNS) <= set(inspect(postgres_engine).get_view_names())
    assert len(set(REPORTING_VIEW_COLUMNS)) == 5


def test_reporting_view_columns_match_power_bi_contract(postgres_engine: Engine) -> None:
    inspector = inspect(postgres_engine)
    for view_name, expected_columns in REPORTING_VIEW_COLUMNS.items():
        actual = tuple(column["name"] for column in inspector.get_columns(view_name))
        assert actual == expected_columns
    protected = {
        "is_fraud",
        "source_flag",
        "oldbalanceorg",
        "newbalanceorig",
        "oldbalancedest",
        "newbalancedest",
        "artifact_uri",
        "vector",
    }
    assert not protected & {
        column for columns in REPORTING_VIEW_COLUMNS.values() for column in columns
    }


def test_views_expose_published_facts_and_hide_pending_facts(
    reporting_database: Engine,
) -> None:
    table_counts: dict[str, int] = {}
    with reporting_database.connect() as connection:
        for table in APPLICATION_TABLES:
            table_counts[table] = connection.scalar(text(f'SELECT COUNT(*) FROM "{table}"'))

        transactions = (
            connection.execute(text("SELECT * FROM reporting_published_transaction"))
            .mappings()
            .all()
        )
        trends = (
            connection.execute(text("SELECT * FROM reporting_published_transaction_trend"))
            .mappings()
            .all()
        )
        evaluations = (
            connection.execute(text("SELECT * FROM reporting_published_model_evaluation"))
            .mappings()
            .all()
        )
        freshness = (
            connection.execute(text("SELECT * FROM reporting_batch_freshness ORDER BY batch_id"))
            .mappings()
            .all()
        )
        lineage = (
            connection.execute(text("SELECT * FROM reporting_system_lineage")).mappings().one()
        )

        assert [row["transaction_id"] for row in transactions] == ["tx-published"]
        assert transactions[0]["is_flagged"] is True
        assert transactions[0]["amount"] == Decimal("25.00")
        assert len(trends) == 1
        assert trends[0]["transaction_count"] == 1
        assert trends[0]["flagged_transaction_count"] == 1
        assert trends[0]["flagged_exposure"] == Decimal("25.00")
        assert [row["evaluation_id"] for row in evaluations] == ["eval-published"]
        assert evaluations[0]["evaluated_row_count"] == 10
        assert [row["batch_status"] for row in freshness] == ["pending", "published"]
        pending = freshness[0]
        assert pending["published_transaction_count"] == 0
        assert pending["published_min_step"] is None
        assert pending["published_max_step"] is None
        assert pending["published_score_count"] == 0
        assert pending["published_flag_count"] == 0
        assert dict(lineage) == APPROVED_LINEAGE

        for table, before in table_counts.items():
            assert connection.scalar(text(f'SELECT COUNT(*) FROM "{table}"')) == before


def test_reporting_views_are_empty_without_batches(postgres_engine: Engine) -> None:
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))
        connection.execute(text("DELETE FROM model_version"))
    with postgres_engine.connect() as connection:
        assert connection.scalar(text("SELECT COUNT(*) FROM reporting_published_transaction")) == 0
        assert (
            connection.scalar(text("SELECT COUNT(*) FROM reporting_published_transaction_trend"))
            == 0
        )
        assert (
            connection.scalar(text("SELECT COUNT(*) FROM reporting_published_model_evaluation"))
            == 0
        )
        assert connection.scalar(text("SELECT COUNT(*) FROM reporting_batch_freshness")) == 0
        assert connection.scalar(text("SELECT COUNT(*) FROM reporting_system_lineage")) == 1
