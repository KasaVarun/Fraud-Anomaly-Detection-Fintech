"""PostgreSQL-backed published-only investigator repository and API tests."""

from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, insert, select, text, update

from fraud_detection.api.app import create_app
from fraud_detection.api.container import build_container
from fraud_detection.config import Settings
from fraud_detection.incremental.contracts import LOCKED_MODEL_ID
from fraud_detection.investigator.repository import PostgresInvestigatorRepository
from fraud_detection.persistence.relational import Base

pytestmark = pytest.mark.integration


@pytest.fixture
def published_investigator_database(postgres_engine: Engine) -> Iterator[Engine]:
    tables = Base.metadata.tables
    now = datetime(2026, 10, 31, tzinfo=UTC)
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))
        connection.execute(
            tables["model_version"]
            .delete()
            .where(tables["model_version"].c.model_version_id == LOCKED_MODEL_ID)
        )
        connection.execute(
            insert(tables["ingestion_batch"]),
            [
                {
                    "batch_id": "published-batch",
                    "source_sha256": "a" * 64,
                    "sample_seed": 1,
                    "chunk_id": "published",
                    "enrichment_seed": 2,
                    "split_policy": "test",
                    "status": "published",
                    "created_at": now,
                },
                {
                    "batch_id": "pending-batch",
                    "source_sha256": "b" * 64,
                    "sample_seed": 1,
                    "chunk_id": "pending",
                    "enrichment_seed": 2,
                    "split_policy": "test",
                    "status": "pending",
                    "created_at": now,
                },
            ],
        )
        connection.execute(
            insert(tables["account"]),
            [
                {"account_id": "C1", "account_kind": "customer"},
                {"account_id": "C2", "account_kind": "customer"},
                {"account_id": "C9", "account_kind": "customer"},
                {"account_id": "M1", "account_kind": "merchant"},
            ],
        )
        connection.execute(insert(tables["merchant"]).values(account_id="M1"))
        connection.execute(
            insert(tables["transaction"]),
            [
                {
                    "transaction_id": "tx-1",
                    "batch_id": "published-batch",
                    "source_row": 1,
                    "sender_id": "C1",
                    "receiver_id": "M1",
                    "step": 10,
                    "transaction_type": "PAYMENT",
                    "amount": Decimal("25.00"),
                    "data_split": "test",
                },
                {
                    "transaction_id": "tx-2",
                    "batch_id": "published-batch",
                    "source_row": 2,
                    "sender_id": "C2",
                    "receiver_id": "C1",
                    "step": 11,
                    "transaction_type": "TRANSFER",
                    "amount": Decimal("35.00"),
                    "data_split": "test",
                },
                {
                    "transaction_id": "tx-pending",
                    "batch_id": "pending-batch",
                    "source_row": 3,
                    "sender_id": "C9",
                    "receiver_id": "C1",
                    "step": 12,
                    "transaction_type": "TRANSFER",
                    "amount": Decimal("50.00"),
                    "data_split": "test",
                },
            ],
        )
        connection.execute(
            insert(tables["model_version"]).values(
                model_version_id=LOCKED_MODEL_ID,
                feature_set="combined",
                training_cutoff_step=281,
                threshold=Decimal("0.9496"),
                embedding_version="test-embedding-v1",
                artifact_uri="trusted",
                created_at=now,
            )
        )
        connection.execute(
            insert(tables["graph_snapshot"]).values(
                snapshot_id="snapshot-1",
                batch_id="published-batch",
                cutoff_step=9,
                algorithm_version="graph-v1",
                computed_at=now,
            )
        )
        connection.execute(
            insert(tables["transaction_score"]),
            [
                {
                    "score_id": "score-1",
                    "transaction_id": "tx-1",
                    "model_version_id": LOCKED_MODEL_ID,
                    "snapshot_id": "snapshot-1",
                    "input_version": "input-1",
                    "fraud_probability": Decimal("0.9900000"),
                    "vector_distance": 0.2,
                    "scored_at": now,
                },
                {
                    "score_id": "score-2",
                    "transaction_id": "tx-2",
                    "model_version_id": LOCKED_MODEL_ID,
                    "snapshot_id": "snapshot-1",
                    "input_version": "input-2",
                    "fraud_probability": Decimal("0.1000000"),
                    "vector_distance": 0.8,
                    "scored_at": now,
                },
            ],
        )
        connection.execute(
            insert(tables["fraud_flag"]).values(
                flag_id="flag-1",
                score_id="score-1",
                reason="locked threshold",
                review_status="open",
            )
        )
    yield postgres_engine
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))
        connection.execute(
            tables["model_version"]
            .delete()
            .where(tables["model_version"].c.model_version_id == LOCKED_MODEL_ID)
        )


def test_repository_and_api_enforce_published_only_with_bounded_queries(
    published_investigator_database: Engine,
) -> None:
    url = published_investigator_database.url.render_as_string(hide_password=False)
    repository = PostgresInvestigatorRepository(url)
    assert repository.transaction_evidence("tx-1", LOCKED_MODEL_ID) is not None
    assert repository.transaction_evidence("tx-pending", LOCKED_MODEL_ID) is None
    assert repository.transaction_evidence("unknown", LOCKED_MODEL_ID) is None

    repository.query_count = 0
    summary = repository.dashboard_summary(
        {
            "start_step": None,
            "end_step": None,
            "transaction_type": None,
            "flagged": None,
        },
        LOCKED_MODEL_ID,
    )
    assert summary["transaction_count"] == 2
    assert summary["fraud_flag_count"] == 1
    assert summary["amount_total"] == Decimal("60.00")
    assert summary["transaction_type_counts"] == {"PAYMENT": 1, "TRANSFER": 1}
    assert repository.query_count == 2
    flagged_summary = repository.dashboard_summary(
        {
            "start_step": 10,
            "end_step": 10,
            "transaction_type": "PAYMENT",
            "flagged": True,
        },
        LOCKED_MODEL_ID,
    )
    assert flagged_summary["transaction_count"] == 1
    unflagged_summary = repository.dashboard_summary(
        {
            "start_step": None,
            "end_step": None,
            "transaction_type": "TRANSFER",
            "flagged": False,
        },
        LOCKED_MODEL_ID,
    )
    assert unflagged_summary["transaction_count"] == 1

    repository.query_count = 0
    flags = repository.flag_page({}, None, 25, LOCKED_MODEL_ID)
    assert [row["transaction_id"] for row in flags] == ["tx-1"]
    assert repository.query_count == 1
    assert (
        repository.flag_page(
            {
                "review_status": "reviewed",
                "minimum_score": 0.5,
                "transaction_type": "PAYMENT",
            },
            None,
            25,
            LOCKED_MODEL_ID,
        )
        == []
    )

    settings = Settings(_env_file=None, postgres_dsn=url)
    container = build_container(settings)
    with TestClient(create_app(settings, container=container)) as client:
        published = client.get("/v1/investigations/transactions/tx-1")
        pending = client.get("/v1/investigations/transactions/tx-pending")
        unknown = client.get("/v1/investigations/transactions/unknown")
        api_summary = client.get("/v1/dashboard/summary")
        flag_page = client.get("/v1/dashboard/flags")
        freshness = client.get("/v1/meta/freshness")
    assert published.status_code == 200
    assert published.json()["partial_evidence"] is True
    assert pending.status_code == unknown.status_code == 404
    assert pending.json()["error"]["message"] == unknown.json()["error"]["message"]
    assert api_summary.json()["data"]["transaction_count"] == 2
    assert len(flag_page.json()["data"]["items"]) == 1
    assert freshness.json()["data"]["latest_published_batch"] == "published-batch"
    assert freshness.json()["data"]["pending_batch_count"] == 1

    tables = Base.metadata.tables
    with published_investigator_database.connect() as connection:
        before = {
            "scores": connection.scalar(
                select(func.count()).select_from(tables["transaction_score"])
            ),
            "flags": connection.scalar(select(func.count()).select_from(tables["fraud_flag"])),
            "published": connection.scalar(
                select(func.count())
                .select_from(tables["ingestion_batch"])
                .where(tables["ingestion_batch"].c.status == "published")
            ),
        }
    assert before == {"scores": 2, "flags": 1, "published": 1}


def test_zero_published_data_returns_empty_aggregates_and_safe_404(
    published_investigator_database: Engine,
) -> None:
    tables = Base.metadata.tables
    with published_investigator_database.begin() as connection:
        connection.execute(
            update(tables["ingestion_batch"])
            .where(tables["ingestion_batch"].c.batch_id == "published-batch")
            .values(status="pending")
        )
    url = published_investigator_database.url.render_as_string(hide_password=False)
    settings = Settings(_env_file=None, postgres_dsn=url)
    with TestClient(create_app(settings)) as client:
        transaction_response = client.get("/v1/investigations/transactions/tx-1")
        summary = client.get("/v1/dashboard/summary")
        flags = client.get("/v1/dashboard/flags")
        freshness = client.get("/v1/meta/freshness")
    assert transaction_response.status_code == 404
    assert summary.json()["data"]["transaction_count"] == 0
    assert summary.json()["data"]["fraud_flag_count"] == 0
    assert flags.json()["data"]["items"] == []
    assert freshness.json()["data"]["latest_published_batch"] is None
    assert freshness.json()["data"]["pending_batch_count"] == 2
