"""Create published-only Power BI reporting views.

Revision ID: 20261031_0002
Revises: 20261031_0001
Create Date: 2026-10-31
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20261031_0002"
down_revision: str | None = "20261031_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE VIEW reporting_published_transaction AS
        WITH ranked_scores AS (
            SELECT
                score.*,
                ROW_NUMBER() OVER (
                    PARTITION BY score.transaction_id, score.model_version_id
                    ORDER BY score.scored_at DESC, score.input_version DESC, score.score_id DESC
                ) AS score_rank
            FROM transaction_score AS score
        )
        SELECT
            txn.transaction_id,
            txn.batch_id,
            txn.source_row,
            txn.sender_id,
            txn.receiver_id,
            txn.step,
            txn.transaction_type,
            txn.amount,
            txn.data_split,
            batch.created_at AS batch_created_at,
            score.score_id,
            score.model_version_id,
            model.feature_set,
            model.training_cutoff_step,
            model.threshold,
            model.embedding_version,
            score.input_version,
            score.snapshot_id,
            snapshot.cutoff_step AS snapshot_cutoff_step,
            snapshot.algorithm_version AS snapshot_algorithm_version,
            score.fraud_probability,
            score.vector_distance,
            score.scored_at,
            flag.flag_id,
            flag.reason AS flag_reason,
            flag.review_status,
            (flag.flag_id IS NOT NULL) AS is_flagged
        FROM "transaction" AS txn
        JOIN ingestion_batch AS batch
          ON batch.batch_id = txn.batch_id
         AND batch.status = 'published'
        LEFT JOIN ranked_scores AS score
          ON score.transaction_id = txn.transaction_id
         AND score.score_rank = 1
        LEFT JOIN model_version AS model
          ON model.model_version_id = score.model_version_id
        LEFT JOIN graph_snapshot AS snapshot
          ON snapshot.snapshot_id = score.snapshot_id
        LEFT JOIN fraud_flag AS flag
          ON flag.score_id = score.score_id
        """
    )
    op.execute(
        """
        CREATE VIEW reporting_published_transaction_trend AS
        SELECT
            batch_id,
            model_version_id,
            step,
            transaction_type,
            COUNT(DISTINCT transaction_id)::BIGINT AS transaction_count,
            COUNT(score_id)::BIGINT AS scored_transaction_count,
            COUNT(flag_id)::BIGINT AS flagged_transaction_count,
            COALESCE(SUM(amount), 0)::NUMERIC(24, 2) AS transaction_amount,
            COALESCE(SUM(amount) FILTER (WHERE is_flagged), 0)::NUMERIC(24, 2)
                AS flagged_exposure,
            MIN(fraud_probability) AS score_minimum,
            MAX(fraud_probability) AS score_maximum,
            AVG(fraud_probability) AS score_mean,
            CASE
                WHEN COUNT(score_id) = 0 THEN 0::NUMERIC
                ELSE COUNT(flag_id)::NUMERIC / COUNT(score_id)::NUMERIC
            END AS flag_rate
        FROM reporting_published_transaction
        GROUP BY batch_id, model_version_id, step, transaction_type
        """
    )
    op.execute(
        """
        CREATE VIEW reporting_published_model_evaluation AS
        SELECT
            evaluation.evaluation_id,
            evaluation.model_version_id,
            model.feature_set,
            model.training_cutoff_step,
            model.threshold,
            model.embedding_version,
            evaluation.batch_id,
            evaluation.split_name,
            evaluation.tp,
            evaluation.fp,
            evaluation.tn,
            evaluation.fn,
            (evaluation.tp + evaluation.fp + evaluation.tn + evaluation.fn)::BIGINT
                AS evaluated_row_count,
            evaluation.precision,
            evaluation.recall,
            evaluation.f1,
            evaluation.pr_auc,
            batch.created_at AS batch_created_at
        FROM model_evaluation AS evaluation
        JOIN ingestion_batch AS batch
          ON batch.batch_id = evaluation.batch_id
         AND batch.status = 'published'
        JOIN model_version AS model
          ON model.model_version_id = evaluation.model_version_id
        """
    )
    op.execute(
        """
        CREATE VIEW reporting_batch_freshness AS
        SELECT
            batch.batch_id,
            batch.status AS batch_status,
            batch.created_at AS batch_created_at,
            (batch.status = 'published') AS is_published,
            COUNT(DISTINCT txn.transaction_id)::BIGINT AS published_transaction_count,
            MIN(txn.step) AS published_min_step,
            MAX(txn.step) AS published_max_step,
            COUNT(DISTINCT score.score_id)::BIGINT AS published_score_count,
            COUNT(DISTINCT flag.flag_id)::BIGINT AS published_flag_count
        FROM ingestion_batch AS batch
        LEFT JOIN "transaction" AS txn
          ON txn.batch_id = batch.batch_id
         AND batch.status = 'published'
        LEFT JOIN transaction_score AS score
          ON score.transaction_id = txn.transaction_id
        LEFT JOIN fraud_flag AS flag
          ON flag.score_id = score.score_id
        GROUP BY batch.batch_id, batch.status, batch.created_at
        """
    )
    op.execute(
        """
        CREATE VIEW reporting_system_lineage AS
        SELECT
            'power-bi-dashboard-v1'::TEXT AS reporting_contract_version,
            'model:lightgbm_combined:1e9a3fc0a52dbac541bf4f92'::TEXT
                AS active_model_version,
            'tabular-graph-vector-v2'::TEXT AS feature_version,
            'minilm-l6-v2-1110a243-template-v1-normalized-float32'::TEXT
                AS embedding_version,
            'raw-paysim-v1'::TEXT AS raw_projection_version,
            'gremlin-evidence-v1'::TEXT AS graph_projection_version
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS reporting_system_lineage")
    op.execute("DROP VIEW IF EXISTS reporting_batch_freshness")
    op.execute("DROP VIEW IF EXISTS reporting_published_model_evaluation")
    op.execute("DROP VIEW IF EXISTS reporting_published_transaction_trend")
    op.execute("DROP VIEW IF EXISTS reporting_published_transaction")
