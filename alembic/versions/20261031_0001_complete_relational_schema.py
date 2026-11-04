"""Create the complete approved relational schema.

Revision ID: 20261031_0001
Revises:
Create Date: 2026-10-31
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.elements import quoted_name

revision: str = "20261031_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingestion_batch",
        sa.Column("batch_id", sa.Text(), nullable=False),
        sa.Column("source_sha256", sa.Text(), nullable=False),
        sa.Column("sample_seed", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.Text(), nullable=False),
        sa.Column("enrichment_seed", sa.Integer(), nullable=False),
        sa.Column("split_policy", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("batch_id", name="pk_ingestion_batch"),
        sa.UniqueConstraint(
            "source_sha256",
            "sample_seed",
            "chunk_id",
            name="uq_ingestion_batch_source_seed_chunk",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'projected', 'published', 'failed')",
            name="status_allowed",
        ),
    )
    op.create_table(
        "account",
        sa.Column("account_id", sa.Text(), nullable=False),
        sa.Column("account_kind", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("account_id", name="pk_account"),
        sa.CheckConstraint(
            "account_kind IN ('customer', 'merchant')",
            name="kind_allowed",
        ),
    )
    op.create_table(
        "merchant",
        sa.Column("account_id", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["account_id"],
            ["account.account_id"],
            name="fk_merchant_account_id_account",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("account_id", name="pk_merchant"),
    )
    op.create_table(
        "transaction",
        sa.Column("transaction_id", sa.Text(), nullable=False),
        sa.Column("batch_id", sa.Text(), nullable=False),
        sa.Column("source_row", sa.BigInteger(), nullable=False),
        sa.Column("sender_id", sa.Text(), nullable=False),
        sa.Column("receiver_id", sa.Text(), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("transaction_type", sa.Text(), nullable=False),
        sa.Column("amount", sa.Numeric(18, 2), nullable=False),
        sa.Column("data_split", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["ingestion_batch.batch_id"],
            name="fk_transaction_batch_id_ingestion_batch",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["sender_id"],
            ["account.account_id"],
            name="fk_transaction_sender_id_account",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["receiver_id"],
            ["account.account_id"],
            name="fk_transaction_receiver_id_account",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("transaction_id", name="pk_transaction"),
        sa.UniqueConstraint("batch_id", "source_row", name="uq_transaction_batch_source_row"),
        sa.CheckConstraint("source_row >= 0", name="source_row_nonnegative"),
        sa.CheckConstraint("step >= 0", name="step_nonnegative"),
        sa.CheckConstraint("amount >= 0", name="amount_nonnegative"),
        sa.CheckConstraint(
            "data_split IN ('train', 'validation', 'test')",
            name="data_split_allowed",
        ),
        quote=True,
    )
    op.create_index(
        "ix_transaction_sender_step",
        quoted_name("transaction", True),
        ["sender_id", "step"],
        unique=False,
    )
    op.create_index(
        "ix_transaction_receiver_step",
        quoted_name("transaction", True),
        ["receiver_id", "step"],
        unique=False,
    )
    op.create_table(
        "transaction_label",
        sa.Column("transaction_id", sa.Text(), nullable=False),
        sa.Column("is_fraud", sa.Boolean(), nullable=False),
        sa.Column("source_flag", sa.Boolean(), nullable=False),
        sa.Column("provenance", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("transaction_id", name="pk_transaction_label"),
    )
    op.create_foreign_key(
        "fk_transaction_label_transaction_id_transaction",
        "transaction_label",
        quoted_name("transaction", True),
        ["transaction_id"],
        ["transaction_id"],
        ondelete="CASCADE",
    )
    op.create_table(
        "model_version",
        sa.Column("model_version_id", sa.Text(), nullable=False),
        sa.Column("feature_set", sa.Text(), nullable=False),
        sa.Column("training_cutoff_step", sa.Integer(), nullable=False),
        sa.Column("threshold", sa.Numeric(5, 4), nullable=False),
        sa.Column("embedding_version", sa.Text(), nullable=True),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("model_version_id", name="pk_model_version"),
        sa.CheckConstraint(
            "training_cutoff_step >= 0",
            name="training_cutoff_nonnegative",
        ),
        sa.CheckConstraint(
            "threshold >= 0 AND threshold <= 1",
            name="threshold_unit_interval",
        ),
    )
    op.create_table(
        "graph_snapshot",
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("batch_id", sa.Text(), nullable=False),
        sa.Column("cutoff_step", sa.Integer(), nullable=False),
        sa.Column("algorithm_version", sa.Text(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["ingestion_batch.batch_id"],
            name="fk_graph_snapshot_batch_id_ingestion_batch",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("snapshot_id", name="pk_graph_snapshot"),
        sa.CheckConstraint("cutoff_step >= 0", name="cutoff_step_nonnegative"),
    )
    op.create_table(
        "community",
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("community_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["graph_snapshot.snapshot_id"],
            name="fk_community_snapshot_id_graph_snapshot",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("snapshot_id", "community_id", name="pk_community"),
    )
    op.create_table(
        "account_risk_profile",
        sa.Column("account_id", sa.Text(), nullable=False),
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("community_id", sa.Integer(), nullable=True),
        sa.Column("degree_centrality", postgresql.DOUBLE_PRECISION(), nullable=False),
        sa.Column("weighted_degree", sa.Numeric(), nullable=False),
        sa.ForeignKeyConstraint(
            ["account_id"],
            ["account.account_id"],
            name="fk_account_risk_profile_account_id_account",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["graph_snapshot.snapshot_id"],
            name="fk_account_risk_profile_snapshot_id_graph_snapshot",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id", "community_id"],
            ["community.snapshot_id", "community.community_id"],
            name="fk_account_risk_profile_snapshot_community",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("account_id", "snapshot_id", name="pk_account_risk_profile"),
        sa.CheckConstraint(
            "degree_centrality >= 0 AND degree_centrality <= 1",
            name="degree_centrality_unit_interval",
        ),
        sa.CheckConstraint(
            "weighted_degree >= 0",
            name="weighted_degree_nonnegative",
        ),
    )
    op.create_index(
        "ix_account_risk_profile_snapshot_community",
        "account_risk_profile",
        ["snapshot_id", "community_id"],
        unique=False,
    )
    op.create_table(
        "transaction_score",
        sa.Column("score_id", sa.Text(), nullable=False),
        sa.Column("transaction_id", sa.Text(), nullable=False),
        sa.Column("model_version_id", sa.Text(), nullable=False),
        sa.Column("snapshot_id", sa.Text(), nullable=True),
        sa.Column("input_version", sa.Text(), nullable=False),
        sa.Column("fraud_probability", sa.Numeric(8, 7), nullable=False),
        sa.Column("vector_distance", postgresql.DOUBLE_PRECISION(), nullable=True),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_version.model_version_id"],
            name="fk_transaction_score_model_version_id_model_version",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["graph_snapshot.snapshot_id"],
            name="fk_transaction_score_snapshot_id_graph_snapshot",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("score_id", name="pk_transaction_score"),
        sa.UniqueConstraint(
            "transaction_id",
            "model_version_id",
            "input_version",
            name="uq_transaction_score_transaction_model_input",
        ),
        sa.CheckConstraint(
            "fraud_probability >= 0 AND fraud_probability <= 1",
            name="fraud_probability_unit_interval",
        ),
        sa.CheckConstraint(
            "vector_distance IS NULL OR vector_distance >= 0",
            name="vector_distance_nonnegative",
        ),
    )
    op.create_foreign_key(
        "fk_transaction_score_transaction_id_transaction",
        "transaction_score",
        quoted_name("transaction", True),
        ["transaction_id"],
        ["transaction_id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "ix_transaction_score_transaction_scored_at",
        "transaction_score",
        ["transaction_id", "scored_at"],
        unique=False,
    )
    op.create_index(
        "ix_transaction_score_model_version",
        "transaction_score",
        ["model_version_id"],
        unique=False,
    )
    op.create_table(
        "fraud_flag",
        sa.Column("flag_id", sa.Text(), nullable=False),
        sa.Column("score_id", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("review_status", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["score_id"],
            ["transaction_score.score_id"],
            name="fk_fraud_flag_score_id_transaction_score",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("flag_id", name="pk_fraud_flag"),
        sa.UniqueConstraint("score_id", name="uq_fraud_flag_score_id"),
        sa.CheckConstraint(
            "review_status IN ('open', 'reviewed')",
            name="review_status_allowed",
        ),
    )
    op.create_index("ix_fraud_flag_review_status", "fraud_flag", ["review_status"], unique=False)
    op.create_table(
        "model_evaluation",
        sa.Column("evaluation_id", sa.Text(), nullable=False),
        sa.Column("model_version_id", sa.Text(), nullable=False),
        sa.Column("batch_id", sa.Text(), nullable=False),
        sa.Column("split_name", sa.Text(), nullable=False),
        sa.Column("tp", sa.BigInteger(), nullable=False),
        sa.Column("fp", sa.BigInteger(), nullable=False),
        sa.Column("tn", sa.BigInteger(), nullable=False),
        sa.Column("fn", sa.BigInteger(), nullable=False),
        sa.Column("precision", sa.Numeric(), nullable=True),
        sa.Column("recall", sa.Numeric(), nullable=True),
        sa.Column("f1", sa.Numeric(), nullable=True),
        sa.Column("pr_auc", sa.Numeric(), nullable=True),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_version.model_version_id"],
            name="fk_model_evaluation_model_version_id_model_version",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["ingestion_batch.batch_id"],
            name="fk_model_evaluation_batch_id_ingestion_batch",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("evaluation_id", name="pk_model_evaluation"),
        sa.UniqueConstraint(
            "model_version_id",
            "batch_id",
            "split_name",
            name="uq_model_evaluation_model_batch_split",
        ),
        sa.CheckConstraint(
            "split_name IN ('train', 'validation', 'test')",
            name="split_name_allowed",
        ),
        sa.CheckConstraint(
            "tp >= 0 AND fp >= 0 AND tn >= 0 AND fn >= 0",
            name="counts_nonnegative",
        ),
        sa.CheckConstraint(
            "precision IS NULL OR (precision >= 0 AND precision <= 1)",
            name="precision_unit_interval",
        ),
        sa.CheckConstraint(
            "recall IS NULL OR (recall >= 0 AND recall <= 1)",
            name="recall_unit_interval",
        ),
        sa.CheckConstraint(
            "f1 IS NULL OR (f1 >= 0 AND f1 <= 1)",
            name="f1_unit_interval",
        ),
        sa.CheckConstraint(
            "pr_auc IS NULL OR (pr_auc >= 0 AND pr_auc <= 1)",
            name="pr_auc_unit_interval",
        ),
    )

    op.execute(
        """
        CREATE FUNCTION enforce_merchant_account_kind() RETURNS trigger AS $$
        BEGIN
            IF TG_TABLE_NAME = 'merchant' THEN
                IF NOT EXISTS (
                    SELECT 1 FROM account
                    WHERE account_id = NEW.account_id AND account_kind = 'merchant'
                ) THEN
                    RAISE EXCEPTION 'merchant subtype requires account_kind merchant'
                        USING ERRCODE = '23514';
                END IF;
            ELSIF TG_TABLE_NAME = 'account' AND NEW.account_kind <> 'merchant' THEN
                IF EXISTS (SELECT 1 FROM merchant WHERE account_id = NEW.account_id) THEN
                    RAISE EXCEPTION 'merchant account kind cannot be changed while subtype exists'
                        USING ERRCODE = '23514';
                END IF;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_merchant_account_kind
        BEFORE INSERT OR UPDATE OF account_id ON merchant
        FOR EACH ROW EXECUTE FUNCTION enforce_merchant_account_kind();
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_account_merchant_kind
        BEFORE UPDATE OF account_kind ON account
        FOR EACH ROW EXECUTE FUNCTION enforce_merchant_account_kind();
        """
    )
    op.execute(
        """
        CREATE FUNCTION enforce_fraud_flag_threshold() RETURNS trigger AS $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM transaction_score score
                JOIN model_version model
                  ON model.model_version_id = score.model_version_id
                WHERE score.score_id = NEW.score_id
                  AND score.fraud_probability >= model.threshold
            ) THEN
                RAISE EXCEPTION 'fraud flag requires score at or above model threshold'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_fraud_flag_threshold
        BEFORE INSERT OR UPDATE OF score_id ON fraud_flag
        FOR EACH ROW EXECUTE FUNCTION enforce_fraud_flag_threshold();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_fraud_flag_threshold ON fraud_flag")
    op.execute("DROP FUNCTION IF EXISTS enforce_fraud_flag_threshold()")
    op.execute("DROP TRIGGER IF EXISTS trg_account_merchant_kind ON account")
    op.execute("DROP TRIGGER IF EXISTS trg_merchant_account_kind ON merchant")
    op.execute("DROP FUNCTION IF EXISTS enforce_merchant_account_kind()")
    op.drop_table("model_evaluation")
    op.drop_table("fraud_flag")
    op.drop_table("transaction_score")
    op.drop_table("account_risk_profile")
    op.drop_table("community")
    op.drop_table("graph_snapshot")
    op.drop_table("model_version")
    op.drop_table("transaction_label")
    op.drop_table("transaction", quote=True)
    op.drop_table("merchant")
    op.drop_table("account")
    op.drop_table("ingestion_batch")
