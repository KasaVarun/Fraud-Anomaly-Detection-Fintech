"""SQLAlchemy 2 models for the approved PostgreSQL relational design."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    UniqueConstraint,
)
from sqlalchemy import (
    Text as String,
)
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.elements import quoted_name

from fraud_detection.persistence.relational.base import Base


class IngestionBatch(Base):
    __tablename__ = "ingestion_batch"
    __table_args__ = (
        UniqueConstraint(
            "source_sha256",
            "sample_seed",
            "chunk_id",
            name="uq_ingestion_batch_source_seed_chunk",
        ),
        CheckConstraint(
            "status IN ('pending', 'projected', 'published', 'failed')",
            name="status_allowed",
        ),
    )

    batch_id: Mapped[str] = mapped_column(String, primary_key=True)
    source_sha256: Mapped[str] = mapped_column(String, nullable=False)
    sample_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_id: Mapped[str] = mapped_column(String, nullable=False)
    enrichment_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    split_policy: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Account(Base):
    __tablename__ = "account"
    __table_args__ = (
        CheckConstraint("account_kind IN ('customer', 'merchant')", name="kind_allowed"),
    )

    account_id: Mapped[str] = mapped_column(String, primary_key=True)
    account_kind: Mapped[str] = mapped_column(String, nullable=False)


class Merchant(Base):
    __tablename__ = "merchant"

    account_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "account.account_id",
            name="fk_merchant_account_id_account",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )


class Transaction(Base):
    __tablename__ = quoted_name("transaction", quote=True)
    __table_args__ = (
        UniqueConstraint("batch_id", "source_row", name="uq_transaction_batch_source_row"),
        CheckConstraint("source_row >= 0", name="source_row_nonnegative"),
        CheckConstraint("step >= 0", name="step_nonnegative"),
        CheckConstraint("amount >= 0", name="amount_nonnegative"),
        CheckConstraint(
            "data_split IN ('train', 'validation', 'test')",
            name="data_split_allowed",
        ),
        Index("ix_transaction_sender_step", "sender_id", "step"),
        Index("ix_transaction_receiver_step", "receiver_id", "step"),
    )

    transaction_id: Mapped[str] = mapped_column(String, primary_key=True)
    batch_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "ingestion_batch.batch_id",
            name="fk_transaction_batch_id_ingestion_batch",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    source_row: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sender_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "account.account_id",
            name="fk_transaction_sender_id_account",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    receiver_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "account.account_id",
            name="fk_transaction_receiver_id_account",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    transaction_type: Mapped[str] = mapped_column(String, nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    data_split: Mapped[str] = mapped_column(String, nullable=False)


class TransactionLabel(Base):
    __tablename__ = "transaction_label"

    transaction_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "transaction.transaction_id",
            name="fk_transaction_label_transaction_id_transaction",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    is_fraud: Mapped[bool] = mapped_column(Boolean, nullable=False)
    source_flag: Mapped[bool] = mapped_column(Boolean, nullable=False)
    provenance: Mapped[str] = mapped_column(String, nullable=False)


class ModelVersion(Base):
    __tablename__ = "model_version"
    __table_args__ = (
        CheckConstraint("training_cutoff_step >= 0", name="training_cutoff_nonnegative"),
        CheckConstraint("threshold >= 0 AND threshold <= 1", name="threshold_unit_interval"),
    )

    model_version_id: Mapped[str] = mapped_column(String, primary_key=True)
    feature_set: Mapped[str] = mapped_column(String, nullable=False)
    training_cutoff_step: Mapped[int] = mapped_column(Integer, nullable=False)
    threshold: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    embedding_version: Mapped[str | None] = mapped_column(String, nullable=True)
    artifact_uri: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class GraphSnapshot(Base):
    __tablename__ = "graph_snapshot"
    __table_args__ = (CheckConstraint("cutoff_step >= 0", name="cutoff_step_nonnegative"),)

    snapshot_id: Mapped[str] = mapped_column(String, primary_key=True)
    batch_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "ingestion_batch.batch_id",
            name="fk_graph_snapshot_batch_id_ingestion_batch",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    cutoff_step: Mapped[int] = mapped_column(Integer, nullable=False)
    algorithm_version: Mapped[str] = mapped_column(String, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Community(Base):
    __tablename__ = "community"
    __table_args__ = (PrimaryKeyConstraint("snapshot_id", "community_id", name="pk_community"),)

    snapshot_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "graph_snapshot.snapshot_id",
            name="fk_community_snapshot_id_graph_snapshot",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    community_id: Mapped[int] = mapped_column(Integer, nullable=False)


class AccountRiskProfile(Base):
    __tablename__ = "account_risk_profile"
    __table_args__ = (
        PrimaryKeyConstraint("account_id", "snapshot_id", name="pk_account_risk_profile"),
        ForeignKeyConstraint(
            ["snapshot_id", "community_id"],
            ["community.snapshot_id", "community.community_id"],
            name="fk_account_risk_profile_snapshot_community",
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "degree_centrality >= 0 AND degree_centrality <= 1",
            name="degree_centrality_unit_interval",
        ),
        CheckConstraint("weighted_degree >= 0", name="weighted_degree_nonnegative"),
        Index("ix_account_risk_profile_snapshot_community", "snapshot_id", "community_id"),
    )

    account_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "account.account_id",
            name="fk_account_risk_profile_account_id_account",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    snapshot_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "graph_snapshot.snapshot_id",
            name="fk_account_risk_profile_snapshot_id_graph_snapshot",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    community_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    degree_centrality: Mapped[float] = mapped_column(DOUBLE_PRECISION, nullable=False)
    weighted_degree: Mapped[Decimal] = mapped_column(Numeric, nullable=False)


class TransactionScore(Base):
    __tablename__ = "transaction_score"
    __table_args__ = (
        UniqueConstraint(
            "transaction_id",
            "model_version_id",
            "input_version",
            name="uq_transaction_score_transaction_model_input",
        ),
        CheckConstraint(
            "fraud_probability >= 0 AND fraud_probability <= 1",
            name="fraud_probability_unit_interval",
        ),
        CheckConstraint(
            "vector_distance IS NULL OR vector_distance >= 0",
            name="vector_distance_nonnegative",
        ),
        Index("ix_transaction_score_transaction_scored_at", "transaction_id", "scored_at"),
        Index("ix_transaction_score_model_version", "model_version_id"),
    )

    score_id: Mapped[str] = mapped_column(String, primary_key=True)
    transaction_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "transaction.transaction_id",
            name="fk_transaction_score_transaction_id_transaction",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    model_version_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "model_version.model_version_id",
            name="fk_transaction_score_model_version_id_model_version",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    snapshot_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey(
            "graph_snapshot.snapshot_id",
            name="fk_transaction_score_snapshot_id_graph_snapshot",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )
    input_version: Mapped[str] = mapped_column(String, nullable=False)
    fraud_probability: Mapped[Decimal] = mapped_column(Numeric(8, 7), nullable=False)
    vector_distance: Mapped[float | None] = mapped_column(DOUBLE_PRECISION, nullable=True)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class FraudFlag(Base):
    __tablename__ = "fraud_flag"
    __table_args__ = (
        UniqueConstraint("score_id", name="uq_fraud_flag_score_id"),
        CheckConstraint("review_status IN ('open', 'reviewed')", name="review_status_allowed"),
        Index("ix_fraud_flag_review_status", "review_status"),
    )

    flag_id: Mapped[str] = mapped_column(String, primary_key=True)
    score_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "transaction_score.score_id",
            name="fk_fraud_flag_score_id_transaction_score",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    reason: Mapped[str] = mapped_column(String, nullable=False)
    review_status: Mapped[str] = mapped_column(String, nullable=False)


class ModelEvaluation(Base):
    __tablename__ = "model_evaluation"
    __table_args__ = (
        UniqueConstraint(
            "model_version_id",
            "batch_id",
            "split_name",
            name="uq_model_evaluation_model_batch_split",
        ),
        CheckConstraint(
            "split_name IN ('train', 'validation', 'test')",
            name="split_name_allowed",
        ),
        CheckConstraint("tp >= 0 AND fp >= 0 AND tn >= 0 AND fn >= 0", name="counts_nonnegative"),
        CheckConstraint(
            "precision IS NULL OR (precision >= 0 AND precision <= 1)",
            name="precision_unit_interval",
        ),
        CheckConstraint(
            "recall IS NULL OR (recall >= 0 AND recall <= 1)",
            name="recall_unit_interval",
        ),
        CheckConstraint("f1 IS NULL OR (f1 >= 0 AND f1 <= 1)", name="f1_unit_interval"),
        CheckConstraint(
            "pr_auc IS NULL OR (pr_auc >= 0 AND pr_auc <= 1)",
            name="pr_auc_unit_interval",
        ),
    )

    evaluation_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_version_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "model_version.model_version_id",
            name="fk_model_evaluation_model_version_id_model_version",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    batch_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "ingestion_batch.batch_id",
            name="fk_model_evaluation_batch_id_ingestion_batch",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    split_name: Mapped[str] = mapped_column(String, nullable=False)
    tp: Mapped[int] = mapped_column(BigInteger, nullable=False)
    fp: Mapped[int] = mapped_column(BigInteger, nullable=False)
    tn: Mapped[int] = mapped_column(BigInteger, nullable=False)
    fn: Mapped[int] = mapped_column(BigInteger, nullable=False)
    precision: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    recall: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    f1: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    pr_auc: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
