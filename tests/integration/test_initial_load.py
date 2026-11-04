"""PostgreSQL 16 tests for the deterministic Phase 3 initial load."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, func, select, text
from sqlalchemy.exc import DataError

from fraud_detection.ingestion.errors import LoadConflictError, ReconciliationError
from fraud_detection.ingestion.loader import load_initial_population
from fraud_detection.ingestion.reconciliation import reconcile_initial_load
from fraud_detection.ingestion.sampling import generate_sample
from fraud_detection.persistence.relational import Base
from tests.paysim_factory import paysim_rows, write_paysim_csv

pytestmark = pytest.mark.integration


def database_url(engine: Engine) -> str:
    return engine.url.render_as_string(hide_password=False)


@pytest.fixture
def clean_loader_database(postgres_engine: Engine) -> Iterator[Engine]:
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))
    yield postgres_engine
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))


def artifacts(tmp_path: Path, *, rows: list[dict[str, object]] | None = None) -> tuple[Path, Path]:
    source = write_paysim_csv(tmp_path / "paysim.csv", rows=rows)
    manifest = tmp_path / "manifest.json"
    generate_sample(
        source,
        tmp_path / "sample.parquet",
        manifest,
        target_size=len(rows) if rows is not None else 30,
        sampling_seed=123,
        enrichment_seed=456,
    )
    return source, manifest


def test_initial_load_maps_rows_is_idempotent_and_reconciles(
    clean_loader_database: Engine, tmp_path: Path
) -> None:
    source, manifest_path = artifacts(tmp_path)
    url = database_url(clean_loader_database)

    first = load_initial_population(source, manifest_path, url, batch_size=7)
    second = load_initial_population(source, manifest_path, url, batch_size=7)
    report = reconcile_initial_load(source, manifest_path, url)

    assert first == second
    assert first.status == "pending"
    assert first.transaction_count == first.expected_rows
    assert first.label_count == first.expected_rows
    assert report.reconciled is True
    assert report.test_row_count == 0
    assert report.loaded_transaction_count == first.expected_rows
    with clean_loader_database.connect() as connection:
        raw_splits: list[Any] = list(
            connection.execute(
                text('SELECT DISTINCT data_split FROM "transaction" ORDER BY data_split')
            ).scalars()
        )
        splits = [str(value) for value in raw_splits]
        assert splits == ["train", "validation"]
        assert connection.scalar(
            select(func.count()).select_from(Base.metadata.tables["merchant"])
        ) == (
            connection.scalar(
                select(func.count())
                .select_from(Base.metadata.tables["account"])
                .where(Base.metadata.tables["account"].c.account_kind == "merchant")
            )
        )
        for later_table in [
            "model_version",
            "transaction_score",
            "fraud_flag",
            "graph_snapshot",
            "community",
            "account_risk_profile",
            "model_evaluation",
        ]:
            assert (
                connection.scalar(
                    select(func.count()).select_from(Base.metadata.tables[later_table])
                )
                == 0
            )


def test_conflicting_stable_transaction_is_rejected(
    clean_loader_database: Engine, tmp_path: Path
) -> None:
    source, manifest_path = artifacts(tmp_path)
    url = database_url(clean_loader_database)
    result = load_initial_population(source, manifest_path, url)
    with clean_loader_database.begin() as connection:
        transaction_id = str(
            connection.execute(
                text(
                    'SELECT transaction_id FROM "transaction" '
                    "WHERE batch_id = :batch_id ORDER BY source_row LIMIT 1"
                ),
                {"batch_id": result.batch_id},
            ).scalar_one()
        )
        connection.execute(
            text('UPDATE "transaction" SET amount = amount + 1 WHERE transaction_id = :id'),
            {"id": transaction_id},
        )

    with pytest.raises(LoadConflictError, match="conflicts on amount"):
        load_initial_population(source, manifest_path, url)


def test_failed_chunk_rolls_back_completely(clean_loader_database: Engine, tmp_path: Path) -> None:
    rows = paysim_rows()
    rows[0]["amount"] = "100000000000000000000.00"
    source, manifest_path = artifacts(tmp_path, rows=rows)

    with pytest.raises(DataError):
        load_initial_population(
            source,
            manifest_path,
            database_url(clean_loader_database),
            batch_size=5,
        )

    with clean_loader_database.connect() as connection:
        assert (
            connection.scalar(
                select(func.count()).select_from(Base.metadata.tables["ingestion_batch"])
            )
            == 0
        )
        assert (
            connection.scalar(select(func.count()).select_from(Base.metadata.tables["account"]))
            == 0
        )
        assert (
            connection.scalar(select(func.count()).select_from(Base.metadata.tables["transaction"]))
            == 0
        )


def test_reconciliation_fails_when_loaded_counts_differ(
    clean_loader_database: Engine, tmp_path: Path
) -> None:
    source, manifest_path = artifacts(tmp_path)
    url = database_url(clean_loader_database)
    load_initial_population(source, manifest_path, url)
    with clean_loader_database.begin() as connection:
        connection.execute(
            text(
                """DELETE FROM transaction_label WHERE transaction_id = (
                SELECT transaction_id FROM transaction_label ORDER BY transaction_id LIMIT 1
                )"""
            )
        )

    with pytest.raises(ReconciliationError, match='"reconciled":false'):
        reconcile_initial_load(source, manifest_path, url)
