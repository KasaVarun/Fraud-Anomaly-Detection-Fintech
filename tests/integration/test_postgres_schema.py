"""PostgreSQL-specific migration, constraint, and trigger tests."""

from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError

from tests.integration.conftest import run_alembic

pytestmark = pytest.mark.integration

APPLICATION_TABLES = {
    "ingestion_batch",
    "account",
    "merchant",
    "transaction",
    "transaction_label",
    "model_version",
    "graph_snapshot",
    "community",
    "account_risk_profile",
    "transaction_score",
    "fraud_flag",
    "model_evaluation",
}
NOW = datetime(2026, 10, 31, tzinfo=UTC)
Statement = tuple[str, dict[str, object]]


def execute_all(connection: Any, statements: Iterable[Statement]) -> None:
    for sql, parameters in statements:
        connection.execute(text(sql), parameters)


def expect_integrity_error(
    engine: Engine,
    setup: Iterable[Statement],
    sql: str,
    parameters: dict[str, object],
) -> None:
    with pytest.raises(IntegrityError), engine.begin() as connection:
        execute_all(connection, setup)
        connection.execute(text(sql), parameters)


def base_batch(batch_id: str = "batch-1") -> Statement:
    return (
        """INSERT INTO ingestion_batch
        (batch_id, source_sha256, sample_seed, chunk_id, enrichment_seed, split_policy, status,
         created_at)
        VALUES (:batch_id, :source_sha256, 7, :chunk_id, 11, 'whole-step-v1', 'pending', :now)""",
        {
            "batch_id": batch_id,
            "source_sha256": f"sha-{batch_id}",
            "chunk_id": f"chunk-{batch_id}",
            "now": NOW,
        },
    )


def base_accounts() -> list[Statement]:
    return [
        (
            "INSERT INTO account (account_id, account_kind) VALUES (:id, :kind)",
            {"id": account_id, "kind": kind},
        )
        for account_id, kind in [("sender", "customer"), ("receiver", "customer")]
    ]


def base_transaction(transaction_id: str = "tx-1") -> Statement:
    return (
        """INSERT INTO "transaction"
        (transaction_id, batch_id, source_row, sender_id, receiver_id, step, transaction_type,
         amount, data_split)
        VALUES (:transaction_id, 'batch-1', 1, 'sender', 'receiver', 1, 'TRANSFER', 10.00,
                'train')""",
        {"transaction_id": transaction_id},
    )


def base_model(model_id: str = "model-1", threshold: str = "0.7000") -> Statement:
    return (
        """INSERT INTO model_version
        (model_version_id, feature_set, training_cutoff_step, threshold, embedding_version,
         artifact_uri, created_at)
        VALUES (:model_id, 'tabular', 0, :threshold, NULL, 'local://model', :now)""",
        {"model_id": model_id, "threshold": Decimal(threshold), "now": NOW},
    )


def base_score(score_id: str = "score-1", probability: str = "0.8000000") -> Statement:
    return (
        """INSERT INTO transaction_score
        (score_id, transaction_id, model_version_id, snapshot_id, input_version,
         fraud_probability, vector_distance, scored_at)
        VALUES (:score_id, 'tx-1', 'model-1', NULL, 'input-v1', :probability, NULL, :now)""",
        {"score_id": score_id, "probability": Decimal(probability), "now": NOW},
    )


def core_setup() -> list[Statement]:
    return [base_batch(), *base_accounts(), base_transaction(), base_model()]


def test_upgrade_downgrade_and_reupgrade(postgres_engine: Engine) -> None:
    assert set(inspect(postgres_engine).get_table_names()) >= APPLICATION_TABLES
    run_alembic("base")
    assert not (APPLICATION_TABLES & set(inspect(postgres_engine).get_table_names()))
    run_alembic("head")
    assert set(inspect(postgres_engine).get_table_names()) >= APPLICATION_TABLES


def test_exactly_twelve_application_tables_exist(postgres_engine: Engine) -> None:
    names = set(inspect(postgres_engine).get_table_names()) - {"alembic_version"}
    assert names == APPLICATION_TABLES
    assert len(names) == 12


def test_duplicate_batch_manifest_is_rejected(postgres_engine: Engine) -> None:
    first = base_batch()
    duplicate: Statement = (
        first[0],
        {**first[1], "batch_id": "batch-2"},
    )
    expect_integrity_error(postgres_engine, [first], *duplicate)


def test_duplicate_batch_source_row_is_rejected(postgres_engine: Engine) -> None:
    setup: list[Statement] = [base_batch(), *base_accounts(), base_transaction()]
    duplicate: Statement = (
        """INSERT INTO "transaction"
        (transaction_id, batch_id, source_row, sender_id, receiver_id, step, transaction_type,
         amount, data_split)
        VALUES ('tx-2', 'batch-1', 1, 'sender', 'receiver', 2, 'TRANSFER', 5.00, 'train')""",
        {},
    )
    expect_integrity_error(postgres_engine, setup, *duplicate)


@pytest.mark.parametrize(
    ("sql", "parameters"),
    [
        ("INSERT INTO account VALUES ('bad', 'unknown')", {}),
        (
            """INSERT INTO ingestion_batch VALUES
            ('bad', 'sha', 1, 'chunk', 1, 'policy', 'unknown', :now)""",
            {"now": NOW},
        ),
        (
            """INSERT INTO "transaction" VALUES
            ('bad', 'batch-1', 2, 'sender', 'receiver', 1, 'TRANSFER', 1.00, 'unknown')""",
            {},
        ),
        (
            "INSERT INTO fraud_flag VALUES ('bad', 'score-1', 'reason', 'unknown')",
            {},
        ),
        (
            """INSERT INTO model_version VALUES
            ('bad', 'tabular', 0, 1.1000, NULL, 'local://bad', :now)""",
            {"now": NOW},
        ),
        (
            """INSERT INTO transaction_score VALUES
            ('bad', 'tx-1', 'model-1', NULL, 'input-bad', 1.1000000, NULL, :now)""",
            {"now": NOW},
        ),
        (
            """INSERT INTO transaction_score VALUES
            ('bad', 'tx-1', 'model-1', NULL, 'input-bad', 0.5000000, -0.1, :now)""",
            {"now": NOW},
        ),
    ],
)
def test_invalid_enumerations_and_probabilities_are_rejected(
    postgres_engine: Engine, sql: str, parameters: dict[str, object]
) -> None:
    expect_integrity_error(postgres_engine, [*core_setup(), base_score()], sql, parameters)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("degree_centrality", "1.1"),
        ("weighted_degree", "-0.1"),
    ],
)
def test_invalid_risk_values_are_rejected(postgres_engine: Engine, column: str, value: str) -> None:
    setup: list[Statement] = [
        base_batch(),
        *base_accounts(),
        (
            """INSERT INTO graph_snapshot
            VALUES ('snapshot-1', 'batch-1', 0, 'algo-v1', :now)""",
            {"now": NOW},
        ),
    ]
    sql = f"""INSERT INTO account_risk_profile
        (account_id, snapshot_id, community_id, degree_centrality, weighted_degree)
        VALUES ('sender', 'snapshot-1', NULL,
                {"CAST(:value AS DOUBLE PRECISION)" if column == "degree_centrality" else "0.5"},
                {"CAST(:value AS NUMERIC)" if column == "weighted_degree" else "1.0"})"""
    expect_integrity_error(postgres_engine, setup, sql, {"value": value})


@pytest.mark.parametrize(
    ("overrides",),
    [
        ({"tp": -1},),
        ({"precision": Decimal("1.1")},),
        ({"recall": Decimal("-0.1")},),
        ({"f1": Decimal("1.1")},),
        ({"pr_auc": Decimal("-0.1")},),
    ],
)
def test_invalid_evaluation_counts_and_metrics_are_rejected(
    postgres_engine: Engine, overrides: dict[str, object]
) -> None:
    values: dict[str, object] = {
        "tp": 1,
        "fp": 1,
        "tn": 1,
        "fn": 1,
        "precision": Decimal("0.5"),
        "recall": Decimal("0.5"),
        "f1": Decimal("0.5"),
        "pr_auc": Decimal("0.5"),
    }
    values.update(overrides)
    sql = """INSERT INTO model_evaluation
        (evaluation_id, model_version_id, batch_id, split_name, tp, fp, tn, fn,
         precision, recall, f1, pr_auc)
        VALUES ('eval-1', 'model-1', 'batch-1', 'test', :tp, :fp, :tn, :fn,
                :precision, :recall, :f1, :pr_auc)"""
    expect_integrity_error(postgres_engine, [base_batch(), base_model()], sql, values)


def test_invalid_evaluation_split_is_rejected(postgres_engine: Engine) -> None:
    sql = """INSERT INTO model_evaluation
        (evaluation_id, model_version_id, batch_id, split_name, tp, fp, tn, fn)
        VALUES ('eval-1', 'model-1', 'batch-1', 'unknown', 0, 0, 0, 0)"""
    expect_integrity_error(postgres_engine, [base_batch(), base_model()], sql, {})


@pytest.mark.parametrize(
    ("source_row", "step", "amount"),
    [(-1, 0, "1.00"), (1, -1, "1.00"), (1, 0, "-0.01")],
)
def test_negative_transaction_values_are_rejected(
    postgres_engine: Engine, source_row: int, step: int, amount: str
) -> None:
    sql = """INSERT INTO "transaction"
        (transaction_id, batch_id, source_row, sender_id, receiver_id, step, transaction_type,
         amount, data_split)
        VALUES ('bad', 'batch-1', :source_row, 'sender', 'receiver', :step, 'TRANSFER',
                :amount, 'train')"""
    expect_integrity_error(
        postgres_engine,
        [base_batch(), *base_accounts()],
        sql,
        {"source_row": source_row, "step": step, "amount": Decimal(amount)},
    )


@pytest.mark.parametrize(
    ("sql", "setup"),
    [
        (
            """INSERT INTO "transaction" VALUES
            ('tx', 'missing', 1, 'sender', 'receiver', 1, 'TRANSFER', 1.00, 'train')""",
            base_accounts(),
        ),
        (
            """INSERT INTO "transaction" VALUES
            ('tx', 'batch-1', 1, 'missing', 'receiver', 1, 'TRANSFER', 1.00, 'train')""",
            [base_batch(), *base_accounts()],
        ),
        (
            """INSERT INTO "transaction" VALUES
            ('tx', 'batch-1', 1, 'sender', 'missing', 1, 'TRANSFER', 1.00, 'train')""",
            [base_batch(), *base_accounts()],
        ),
        (
            "INSERT INTO transaction_label VALUES ('missing', true, false, 'PaySim')",
            [],
        ),
        (
            """INSERT INTO transaction_score VALUES
            ('score', 'missing', 'model-1', NULL, 'input', 0.5, NULL, :now)""",
            [base_model()],
        ),
        (
            """INSERT INTO transaction_score VALUES
            ('score', 'tx-1', 'missing', NULL, 'input', 0.5, NULL, :now)""",
            [base_batch(), *base_accounts(), base_transaction()],
        ),
        (
            """INSERT INTO transaction_score VALUES
            ('score', 'tx-1', 'model-1', 'missing', 'input', 0.5, NULL, :now)""",
            core_setup(),
        ),
    ],
)
def test_missing_references_are_rejected(
    postgres_engine: Engine,
    sql: str,
    setup: list[Statement],
) -> None:
    expect_integrity_error(postgres_engine, setup, sql, {"now": NOW})


def test_merchant_subtype_requires_merchant_account(postgres_engine: Engine) -> None:
    setup: list[Statement] = [
        (
            "INSERT INTO account (account_id, account_kind) VALUES ('merchant-1', 'customer')",
            {},
        )
    ]
    expect_integrity_error(
        postgres_engine,
        setup,
        "INSERT INTO merchant (account_id) VALUES ('merchant-1')",
        {},
    )


def test_cross_snapshot_community_membership_is_rejected(postgres_engine: Engine) -> None:
    setup: list[Statement] = [
        base_batch(),
        *base_accounts(),
        (
            "INSERT INTO graph_snapshot VALUES ('s1', 'batch-1', 0, 'algo', :now)",
            {"now": NOW},
        ),
        (
            "INSERT INTO graph_snapshot VALUES ('s2', 'batch-1', 0, 'algo', :now)",
            {"now": NOW},
        ),
        ("INSERT INTO community VALUES ('s1', 1)", {}),
    ]
    expect_integrity_error(
        postgres_engine,
        setup,
        "INSERT INTO account_risk_profile VALUES ('sender', 's2', 1, 0.5, 1.0)",
        {},
    )


def test_duplicate_score_version_is_rejected(postgres_engine: Engine) -> None:
    setup: list[Statement] = [*core_setup(), base_score()]
    duplicate: Statement = (
        """INSERT INTO transaction_score VALUES
        ('score-2', 'tx-1', 'model-1', NULL, 'input-v1', 0.9, NULL, :now)""",
        {"now": NOW},
    )
    expect_integrity_error(postgres_engine, setup, *duplicate)


def test_only_one_flag_per_score_is_allowed(postgres_engine: Engine) -> None:
    setup: list[Statement] = [
        *core_setup(),
        base_score(),
        ("INSERT INTO fraud_flag VALUES ('flag-1', 'score-1', 'threshold', 'open')", {}),
    ]
    expect_integrity_error(
        postgres_engine,
        setup,
        "INSERT INTO fraud_flag VALUES ('flag-2', 'score-1', 'duplicate', 'open')",
        {},
    )


def test_flag_below_model_threshold_is_rejected(postgres_engine: Engine) -> None:
    setup: list[Statement] = [*core_setup(), base_score(probability="0.6000000")]
    expect_integrity_error(
        postgres_engine,
        setup,
        "INSERT INTO fraud_flag VALUES ('flag-1', 'score-1', 'below threshold', 'open')",
        {},
    )
