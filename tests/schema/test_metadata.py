"""Fast contract tests for the complete relational metadata."""

from collections.abc import Collection
from typing import Any, cast

import pytest
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Integer,
    Numeric,
    Text,
)
from sqlalchemy.sql.schema import Table

from fraud_detection.persistence.relational import Base

pytestmark = pytest.mark.schema

EXPECTED_TABLES = {
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


def table(name: str) -> Table:
    return Base.metadata.tables[name]


def names(items: Collection[object]) -> set[str | None]:
    return {getattr(item, "name", None) for item in items}


def test_exactly_the_approved_tables_are_registered() -> None:
    assert set(Base.metadata.tables) == EXPECTED_TABLES
    assert len(Base.metadata.tables) == 12


@pytest.mark.parametrize(
    ("table_name", "expected_columns"),
    [
        (
            "ingestion_batch",
            {
                "batch_id",
                "source_sha256",
                "sample_seed",
                "chunk_id",
                "enrichment_seed",
                "split_policy",
                "status",
                "created_at",
            },
        ),
        ("account", {"account_id", "account_kind"}),
        ("merchant", {"account_id"}),
        (
            "transaction",
            {
                "transaction_id",
                "batch_id",
                "source_row",
                "sender_id",
                "receiver_id",
                "step",
                "transaction_type",
                "amount",
                "data_split",
            },
        ),
        ("transaction_label", {"transaction_id", "is_fraud", "source_flag", "provenance"}),
        (
            "model_version",
            {
                "model_version_id",
                "feature_set",
                "training_cutoff_step",
                "threshold",
                "embedding_version",
                "artifact_uri",
                "created_at",
            },
        ),
        (
            "graph_snapshot",
            {"snapshot_id", "batch_id", "cutoff_step", "algorithm_version", "computed_at"},
        ),
        ("community", {"snapshot_id", "community_id"}),
        (
            "account_risk_profile",
            {
                "account_id",
                "snapshot_id",
                "community_id",
                "degree_centrality",
                "weighted_degree",
            },
        ),
        (
            "transaction_score",
            {
                "score_id",
                "transaction_id",
                "model_version_id",
                "snapshot_id",
                "input_version",
                "fraud_probability",
                "vector_distance",
                "scored_at",
            },
        ),
        ("fraud_flag", {"flag_id", "score_id", "reason", "review_status"}),
        (
            "model_evaluation",
            {
                "evaluation_id",
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
            },
        ),
    ],
)
def test_columns_match_contract(table_name: str, expected_columns: set[str]) -> None:
    assert set(table(table_name).columns.keys()) == expected_columns


def test_approved_types_and_timezone_awareness() -> None:
    assert isinstance(table("ingestion_batch").c.batch_id.type, Text)
    assert isinstance(table("ingestion_batch").c.sample_seed.type, Integer)
    assert isinstance(table("transaction").c.source_row.type, BigInteger)
    amount_type = cast(Numeric[Any], table("transaction").c.amount.type)
    assert isinstance(amount_type, Numeric)
    assert (
        amount_type.precision,
        amount_type.scale,
    ) == (
        18,
        2,
    )
    assert isinstance(table("transaction_label").c.is_fraud.type, Boolean)
    assert isinstance(table("account_risk_profile").c.degree_centrality.type, Float)
    assert isinstance(table("account_risk_profile").c.weighted_degree.type, Numeric)
    probability_type = cast(Numeric[Any], table("transaction_score").c.fraud_probability.type)
    assert isinstance(probability_type, Numeric)
    assert (
        probability_type.precision,
        probability_type.scale,
    ) == (8, 7)
    threshold_type = cast(Numeric[Any], table("model_version").c.threshold.type)
    assert isinstance(threshold_type, Numeric)
    assert (
        threshold_type.precision,
        threshold_type.scale,
    ) == (
        5,
        4,
    )
    for table_name, column_name in [
        ("ingestion_batch", "created_at"),
        ("model_version", "created_at"),
        ("graph_snapshot", "computed_at"),
        ("transaction_score", "scored_at"),
    ]:
        column_type = table(table_name).c[column_name].type
        assert isinstance(column_type, DateTime)
        assert column_type.timezone is True


def test_nullable_columns_are_only_the_approved_optional_fields() -> None:
    nullable = {
        (table_name, column.name)
        for table_name, model_table in Base.metadata.tables.items()
        for column in model_table.columns
        if column.nullable
    }
    assert nullable == {
        ("model_version", "embedding_version"),
        ("account_risk_profile", "community_id"),
        ("transaction_score", "snapshot_id"),
        ("transaction_score", "vector_distance"),
        ("model_evaluation", "precision"),
        ("model_evaluation", "recall"),
        ("model_evaluation", "f1"),
        ("model_evaluation", "pr_auc"),
    }


def test_primary_keys_match_the_approved_grains() -> None:
    assert list(table("community").primary_key.columns.keys()) == ["snapshot_id", "community_id"]
    assert list(table("account_risk_profile").primary_key.columns.keys()) == [
        "account_id",
        "snapshot_id",
    ]
    for table_name, key in {
        "ingestion_batch": "batch_id",
        "account": "account_id",
        "merchant": "account_id",
        "transaction": "transaction_id",
        "transaction_label": "transaction_id",
        "model_version": "model_version_id",
        "graph_snapshot": "snapshot_id",
        "transaction_score": "score_id",
        "fraud_flag": "flag_id",
        "model_evaluation": "evaluation_id",
    }.items():
        assert list(table(table_name).primary_key.columns.keys()) == [key]


def test_foreign_keys_include_cross_snapshot_membership() -> None:
    targets = {
        foreign_key.target_fullname
        for model_table in Base.metadata.tables.values()
        for foreign_key in model_table.foreign_keys
    }
    assert {
        "account.account_id",
        "ingestion_batch.batch_id",
        "transaction.transaction_id",
        "model_version.model_version_id",
        "graph_snapshot.snapshot_id",
        "community.snapshot_id",
        "community.community_id",
        "transaction_score.score_id",
    } <= targets
    composite = next(
        constraint
        for constraint in table("account_risk_profile").foreign_key_constraints
        if constraint.name == "fk_account_risk_profile_snapshot_community"
    )
    assert list(composite.column_keys) == ["snapshot_id", "community_id"]
    assert [element.target_fullname for element in composite.elements] == [
        "community.snapshot_id",
        "community.community_id",
    ]


def test_constraints_and_indexes_have_stable_names() -> None:
    constraint_names = {
        constraint.name
        for model_table in Base.metadata.tables.values()
        for constraint in model_table.constraints
    }
    assert {
        "uq_ingestion_batch_source_seed_chunk",
        "uq_transaction_batch_source_row",
        "uq_transaction_score_transaction_model_input",
        "uq_fraud_flag_score_id",
        "uq_model_evaluation_model_batch_split",
        "ck_account_kind_allowed",
        "ck_transaction_data_split_allowed",
        "ck_model_version_threshold_unit_interval",
        "ck_transaction_score_fraud_probability_unit_interval",
        "ck_fraud_flag_review_status_allowed",
    } <= constraint_names

    index_names = {
        index.name for model_table in Base.metadata.tables.values() for index in model_table.indexes
    }
    assert index_names == {
        "ix_transaction_sender_step",
        "ix_transaction_receiver_step",
        "ix_transaction_score_transaction_scored_at",
        "ix_transaction_score_model_version",
        "ix_fraud_flag_review_status",
        "ix_account_risk_profile_snapshot_community",
    }
