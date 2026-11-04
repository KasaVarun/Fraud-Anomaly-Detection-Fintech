"""PostgreSQL-backed raw projection and cross-store reconciliation tests."""

import json
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import Engine, text

from fraud_detection.ingestion.errors import ReconciliationError
from fraud_detection.ingestion.loader import load_initial_population
from fraud_detection.ingestion.sampling import generate_sample
from fraud_detection.persistence.document_vector.models import content_digest
from fraud_detection.persistence.document_vector.projection import project_initial_raw_documents
from fraud_detection.persistence.document_vector.reconciliation import reconcile_raw_projection
from tests.cosmos.fakes import FakeRawStore
from tests.paysim_factory import write_paysim_csv

pytestmark = pytest.mark.integration


@pytest.fixture
def clean_projection_database(postgres_engine: Engine) -> Iterator[Engine]:
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))
    yield postgres_engine
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))


def projection_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = write_paysim_csv(tmp_path / "paysim.csv")
    sample = tmp_path / "sample.parquet"
    manifest = tmp_path / "manifest.json"
    generate_sample(source, sample, manifest, target_size=30, sampling_seed=123)
    return source, sample, manifest


def database_url(engine: Engine) -> str:
    return engine.url.render_as_string(hide_password=False)


def load_projection_fixture(engine: Engine, tmp_path: Path) -> tuple[Path, Path, Path, str]:
    source, sample, manifest = projection_artifacts(tmp_path)
    url = database_url(engine)
    load_initial_population(source, manifest, url)
    return source, sample, manifest, url


def test_projection_is_train_validation_only_and_idempotent(
    clean_projection_database: Engine, tmp_path: Path
) -> None:
    source, sample, manifest, url = load_projection_fixture(clean_projection_database, tmp_path)
    store = FakeRawStore()

    first = project_initial_raw_documents(source, manifest, url, store, batch_size=7)
    second = project_initial_raw_documents(source, manifest, url, store, batch_size=7)
    report = reconcile_raw_projection(source, manifest, url, store, batch_size=7, page_size=5)

    assert first.created_documents == 24
    assert first.existing_documents == 0
    assert second.created_documents == 0
    assert second.existing_documents == 24
    assert len(store.documents) == 24
    assert {document["data_split"] for document in store.documents} == {"train", "validation"}
    test_ids = set(
        pl.read_parquet(sample).filter(pl.col("data_split") == "test")["transaction_id"].to_list()
    )
    assert not (test_ids & {document["id"] for document in store.documents})
    assert first.batch_status == second.batch_status == report.batch_status == "pending"
    assert report.reconciled is True
    assert report.reserved_test.count == 0


def test_partial_failure_recovers_without_publishing(
    clean_projection_database: Engine, tmp_path: Path
) -> None:
    source, _, manifest, url = load_projection_fixture(clean_projection_database, tmp_path)
    store = FakeRawStore()
    store.fail_after = 5

    with pytest.raises(RuntimeError, match="controlled projection failure"):
        project_initial_raw_documents(source, manifest, url, store, batch_size=5)
    assert len(store.documents) == 5
    with clean_projection_database.connect() as connection:
        assert connection.scalar(text("SELECT status FROM ingestion_batch")) == "pending"

    store.fail_after = None
    recovered = project_initial_raw_documents(source, manifest, url, store, batch_size=5)
    assert recovered.created_documents == 19
    assert recovered.existing_documents == 5
    assert recovered.batch_status == "pending"


def test_immutable_conflict_fails_loudly(clean_projection_database: Engine, tmp_path: Path) -> None:
    source, _, manifest, url = load_projection_fixture(clean_projection_database, tmp_path)
    store = FakeRawStore()
    project_initial_raw_documents(source, manifest, url, store, batch_size=10)
    store.documents[0]["amount"] = "999.00"
    store.documents[0]["content_digest"] = content_digest(store.documents[0])

    with pytest.raises(RuntimeError, match="controlled immutable conflict"):
        project_initial_raw_documents(source, manifest, url, store, batch_size=10)


def test_reconciliation_reports_issue_categories_separately(
    clean_projection_database: Engine, tmp_path: Path
) -> None:
    source, sample_path, manifest, url = load_projection_fixture(
        clean_projection_database, tmp_path
    )
    store = FakeRawStore()
    project_initial_raw_documents(source, manifest, url, store, batch_size=10)

    store.documents.pop(0)
    store.documents[0]["amount"] = "88.00"
    store.documents[0]["content_digest"] = content_digest(store.documents[0])
    store.documents[1]["content_digest"] = "invalid"
    store.documents.append(dict(store.documents[2]))
    orphan = dict(store.documents[3])
    orphan["id"] = "orphan-id"
    orphan["content_digest"] = content_digest(orphan)
    store.documents.append(orphan)
    test_row = (
        pl.read_parquet(sample_path).filter(pl.col("data_split") == "test").row(0, named=True)
    )
    reserved = dict(store.documents[4])
    reserved["id"] = str(test_row["transaction_id"])
    reserved["sender_id"] = str(test_row["nameOrig"])
    reserved["data_split"] = "test"
    reserved["content_digest"] = content_digest(reserved)
    store.documents.append(reserved)

    with pytest.raises(ReconciliationError) as captured:
        reconcile_raw_projection(source, manifest, url, store, batch_size=10, page_size=5)
    report = json.loads(str(captured.value))
    assert report["missing"]["count"] == 1
    assert report["orphaned"]["count"] >= 2
    assert report["conflicting"]["count"] == 1
    assert report["invalid"]["count"] >= 2
    assert report["duplicate"]["count"] == 1
    assert report["reserved_test"]["count"] == 1


@pytest.mark.parametrize(
    ("stored_split", "batch_mode", "sender_mode"),
    [
        ("test", "projected", "expected"),
        ("train", "projected", "expected"),
        ("validation", "different", "expected"),
        ("train", "different", "unexpected"),
    ],
)
def test_reserved_id_is_detected_regardless_of_document_claims(
    clean_projection_database: Engine,
    tmp_path: Path,
    stored_split: str,
    batch_mode: str,
    sender_mode: str,
) -> None:
    source, sample_path, manifest, url = load_projection_fixture(
        clean_projection_database, tmp_path
    )
    store = FakeRawStore()
    projected = project_initial_raw_documents(source, manifest, url, store, batch_size=10)
    test_row = (
        pl.read_parquet(sample_path).filter(pl.col("data_split") == "test").row(0, named=True)
    )
    store.documents.append(
        {
            "id": str(test_row["transaction_id"]),
            "sender_id": (
                str(test_row["nameOrig"]) if sender_mode == "expected" else "unexpected-partition"
            ),
            "batch_id": projected.batch_id if batch_mode == "projected" else "other-batch",
            "data_split": stored_split,
        }
    )

    with pytest.raises(ReconciliationError) as captured:
        reconcile_raw_projection(source, manifest, url, store, batch_size=2, page_size=1)
    report = json.loads(str(captured.value))
    assert report["reserved_test"]["count"] == 1
    assert report["reserved_test"]["sample_ids"] == [str(test_row["transaction_id"])]
