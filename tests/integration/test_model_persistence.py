"""PostgreSQL-backed Phase 7 immutable release persistence tests."""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select, text

from fraud_detection.ingestion.loader import initial_batch_id, load_initial_population
from fraud_detection.ingestion.manifest import read_manifest
from fraud_detection.ingestion.sampling import generate_sample
from fraud_detection.modeling.persistence import persist_model_release_records
from fraud_detection.persistence.relational import Base
from tests.paysim_factory import write_paysim_csv

pytestmark = pytest.mark.integration


@pytest.fixture
def clean_model_database(postgres_engine: Engine) -> Engine:
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))
    return postgres_engine


def test_model_and_validation_persistence_is_idempotent_conflict_detecting_and_score_free(
    clean_model_database: Engine, tmp_path: Path
) -> None:
    source = write_paysim_csv(tmp_path / "paysim.csv")
    sample = tmp_path / "sample.parquet"
    manifest_path = tmp_path / "manifest.json"
    generate_sample(source, sample, manifest_path, target_size=30, sampling_seed=123)
    url = clean_model_database.url.render_as_string(hide_password=False)
    load_initial_population(source, manifest_path, url)
    batch_id = initial_batch_id(read_manifest(manifest_path))
    model = {
        "model_version_id": "model:test",
        "feature_set": "combined",
        "training_cutoff_step": 281,
        "threshold": Decimal("0.4321"),
        "embedding_version": "embedding-v1",
        "artifact_uri": str(tmp_path / "trusted"),
        "created_at": datetime(2026, 10, 31, tzinfo=UTC),
    }
    evaluation = {
        "evaluation_id": "evaluation:test",
        "model_version_id": "model:test",
        "batch_id": batch_id,
        "split_name": "validation",
        "tp": 2,
        "fp": 3,
        "tn": 10,
        "fn": 1,
        "precision": Decimal("0.4"),
        "recall": Decimal("0.666666"),
        "f1": Decimal("0.5"),
        "pr_auc": Decimal("0.6"),
    }
    first = persist_model_release_records(url, batch_id, [model], [evaluation])
    second = persist_model_release_records(url, batch_id, [model], [evaluation])
    assert first == second == (1, 1, 0, 0, "pending")

    conflicting = dict(model)
    conflicting["threshold"] = Decimal("0.9999")
    with pytest.raises(ValueError, match="threshold"):
        persist_model_release_records(url, batch_id, [conflicting], [evaluation])

    with clean_model_database.connect() as connection:
        assert (
            connection.scalar(
                select(func.count()).select_from(Base.metadata.tables["transaction_score"])
            )
            == 0
        )
        assert (
            connection.scalar(select(func.count()).select_from(Base.metadata.tables["fraud_flag"]))
            == 0
        )
        assert (
            connection.scalar(
                select(Base.metadata.tables["ingestion_batch"].c.status).where(
                    Base.metadata.tables["ingestion_batch"].c.batch_id == batch_id
                )
            )
            == "pending"
        )
