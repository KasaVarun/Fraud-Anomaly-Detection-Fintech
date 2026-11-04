"""PostgreSQL-backed template, embedding, projection, and reconciliation tests."""

import json
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import Engine, text

from fraud_detection.config import Settings
from fraud_detection.features.behavioral import template_digest
from fraud_detection.ingestion.errors import ReconciliationError
from fraud_detection.ingestion.loader import load_initial_population
from fraud_detection.ingestion.sampling import generate_sample
from fraud_detection.persistence.document_vector.artifacts import (
    embedding_artifact_logical_digest,
    iter_embedding_artifact,
)
from fraud_detection.persistence.document_vector.embedding_models import embedding_content_digest
from fraud_detection.persistence.document_vector.embedding_reconciliation import (
    reconcile_embedding_documents,
    reconcile_embedding_store,
)
from fraud_detection.persistence.document_vector.embedding_service import (
    build_template_artifact,
    generate_embeddings,
    project_embedding_artifact,
)
from tests.paysim_factory import write_paysim_csv
from tests.vector.fakes import FakeEmbeddingStore, FakeEncoder

pytestmark = pytest.mark.integration


@pytest.fixture
def clean_embedding_database(postgres_engine: Engine) -> Iterator[Engine]:
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))
    yield postgres_engine
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))


def embedding_fixture(engine: Engine, tmp_path: Path) -> tuple[Path, Path, Path, str, Settings]:
    source = write_paysim_csv(tmp_path / "paysim.csv")
    sample = tmp_path / "sample.parquet"
    manifest = tmp_path / "manifest.json"
    generate_sample(source, sample, manifest, target_size=30, sampling_seed=123)
    url = engine.url.render_as_string(hide_password=False)
    load_initial_population(source, manifest, url)
    settings = Settings(
        _env_file=None,
        behavioral_template_artifact=tmp_path / "templates.parquet",
        embedding_local_artifact=tmp_path / "embeddings.parquet",
        embedding_batch_size=5,
        embedding_progress_interval=5,
        embedding_version="test-embedding-v1",
    )
    build_template_artifact(
        source,
        manifest,
        url,
        settings.behavioral_template_artifact,
        settings,
    )
    generate_embeddings(
        settings.behavioral_template_artifact,
        settings.embedding_local_artifact,
        settings,
        encoder=FakeEncoder(),
    )
    return source, sample, manifest, url, settings


def test_local_artifact_reconciles_and_excludes_test_rows(
    clean_embedding_database: Engine, tmp_path: Path
) -> None:
    source, sample, manifest, url, settings = embedding_fixture(clean_embedding_database, tmp_path)
    documents = list(iter_embedding_artifact(settings.embedding_local_artifact))
    report = reconcile_embedding_documents(
        source,
        manifest,
        url,
        documents,
        expected_model_id=FakeEncoder.model_id,
        expected_model_revision=FakeEncoder.model_revision,
        expected_embedding_version=settings.embedding_version,
        expected_template_version=settings.behavioral_template_version,
        expected_logical_artifact_digest=embedding_artifact_logical_digest(
            settings.embedding_local_artifact
        ),
    )

    assert report.reconciled is True
    assert report.embedding_count == 24
    assert report.reference_count == 2
    assert report.batch_status == "pending"
    assert report.logical_artifact_digest == embedding_artifact_logical_digest(
        settings.embedding_local_artifact
    )
    test_ids = set(
        pl.read_parquet(sample).filter(pl.col("data_split") == "test")["transaction_id"].to_list()
    )
    assert not (test_ids & {document.transaction_id for document in documents})


def test_projection_is_idempotent_and_restart_safe(
    clean_embedding_database: Engine, tmp_path: Path
) -> None:
    _, _, _, url, settings = embedding_fixture(clean_embedding_database, tmp_path)
    store = FakeEmbeddingStore()

    first = project_embedding_artifact(settings.embedding_local_artifact, url, store)
    second = project_embedding_artifact(settings.embedding_local_artifact, url, store)
    assert first.created_documents == 24
    assert second.created_documents == 0
    assert second.existing_documents == 24
    assert second.batch_status == "pending"

    store = FakeEmbeddingStore()
    store.fail_after = 5
    with pytest.raises(RuntimeError, match="controlled embedding failure"):
        project_embedding_artifact(settings.embedding_local_artifact, url, store)
    store.fail_after = None
    recovered = project_embedding_artifact(settings.embedding_local_artifact, url, store)
    assert recovered.created_documents == 19
    assert recovered.existing_documents == 5


def test_non_pending_batch_causes_zero_embedding_writes(
    clean_embedding_database: Engine, tmp_path: Path
) -> None:
    _, _, _, url, settings = embedding_fixture(clean_embedding_database, tmp_path)
    with clean_embedding_database.begin() as connection:
        connection.execute(text("UPDATE ingestion_batch SET status = 'projected'"))
    store = FakeEmbeddingStore()

    with pytest.raises(ValueError, match="pending batch"):
        project_embedding_artifact(settings.embedding_local_artifact, url, store)

    assert store.calls == 0
    assert store.documents == []


def test_projection_conflict_and_reconciliation_issue_categories(
    clean_embedding_database: Engine, tmp_path: Path
) -> None:
    source, sample, manifest, url, settings = embedding_fixture(clean_embedding_database, tmp_path)
    store = FakeEmbeddingStore()
    project_embedding_artifact(settings.embedding_local_artifact, url, store)
    store.documents[0]["template_text"] = "transaction_type=DEBIT"
    with pytest.raises(
        Exception, match=r"controlled embedding conflict|content digest|template digest"
    ):
        project_embedding_artifact(settings.embedding_local_artifact, url, store)

    documents = [
        document.payload()
        for document in iter_embedding_artifact(settings.embedding_local_artifact)
    ]
    documents.pop(0)
    documents.append(dict(documents[0]))
    documents[1]["content_digest"] = "invalid"
    orphan = dict(documents[2])
    orphan["transaction_id"] = "orphan"
    documents.append(orphan)
    test_id = str(
        pl.read_parquet(sample).filter(pl.col("data_split") == "test")["transaction_id"].item(0)
    )
    hidden_test = dict(documents[3])
    hidden_test["transaction_id"] = test_id
    hidden_test["data_split"] = "train"
    documents.append(hidden_test)

    with pytest.raises(ReconciliationError) as captured:
        reconcile_embedding_documents(
            source,
            manifest,
            url,
            documents,
            expected_model_id=FakeEncoder.model_id,
            expected_model_revision=FakeEncoder.model_revision,
            expected_embedding_version=settings.embedding_version,
            expected_template_version=settings.behavioral_template_version,
        )
    report = json.loads(str(captured.value))
    assert report["missing"]["count"] >= 1
    assert report["duplicate"]["count"] >= 1
    assert report["orphaned"]["count"] >= 1
    assert report["invalid"]["count"] >= 1
    assert report["reserved_test"]["sample_ids"] == [test_id]


def test_store_reconciliation_finds_hidden_reserved_id(
    clean_embedding_database: Engine, tmp_path: Path
) -> None:
    source, sample, manifest, url, settings = embedding_fixture(clean_embedding_database, tmp_path)
    store = FakeEmbeddingStore()
    project_embedding_artifact(settings.embedding_local_artifact, url, store)
    test_id = str(
        pl.read_parquet(sample).filter(pl.col("data_split") == "test")["transaction_id"].item(0)
    )
    hidden = dict(store.documents[0])
    hidden["transaction_id"] = test_id
    hidden["batch_id"] = "wrong-batch"
    hidden["data_split"] = "train"
    store.documents.append(hidden)

    with pytest.raises(ReconciliationError) as captured:
        reconcile_embedding_store(
            source,
            manifest,
            url,
            store,
            page_size=2,
            chunk_size=2,
            expected_model_id=FakeEncoder.model_id,
            expected_model_revision=FakeEncoder.model_revision,
            expected_embedding_version=settings.embedding_version,
            expected_template_version=settings.behavioral_template_version,
        )
    report = json.loads(str(captured.value))
    assert report["reserved_test"]["sample_ids"] == [test_id]


def _replace_template_field(text_value: str, field: str, value: str) -> str:
    parts = text_value.split(";")
    return ";".join(f"{field}={value}" if part.startswith(f"{field}=") else part for part in parts)


def _resign_embedding(payload: dict[str, object], *, resign_template: bool = True) -> None:
    if resign_template:
        payload["template_digest"] = template_digest(
            str(payload["template_text"]),
            str(payload["template_version"]),
            str(payload["amount_band_version"]),
        )
    payload["content_digest"] = embedding_content_digest(payload)


@pytest.mark.parametrize(
    ("case", "expected_category"),
    [
        ("altered-template", "conflicting"),
        ("fabricated-template-digest", "invalid"),
        ("incorrect-amount-band", "conflicting"),
        ("future-same-step-count", "conflicting"),
        ("nonfraud-eligible", "conflicting"),
        ("fraud-ineligible", "conflicting"),
        ("validation-eligible", "invalid"),
        ("incorrect-provenance", "conflicting"),
    ],
)
def test_independent_reconciliation_rejects_semantic_embedding_tampering(
    clean_embedding_database: Engine,
    tmp_path: Path,
    case: str,
    expected_category: str,
) -> None:
    source, _, manifest, url, settings = embedding_fixture(clean_embedding_database, tmp_path)
    documents = [
        document.payload()
        for document in iter_embedding_artifact(settings.embedding_local_artifact)
    ]
    train_nonfraud = next(
        value
        for value in documents
        if value["data_split"] == "train" and not value["reference_eligible"]
    )
    training_fraud = next(value for value in documents if value["reference_eligible"])
    validation = next(value for value in documents if value["data_split"] == "validation")
    target = train_nonfraud
    if case == "altered-template":
        target["template_text"] = _replace_template_field(
            str(target["template_text"]), "prior_receiver_count", "999"
        )
        _resign_embedding(target)
    elif case == "fabricated-template-digest":
        target["template_digest"] = "f" * 64
        _resign_embedding(target, resign_template=False)
    elif case == "incorrect-amount-band":
        target["template_text"] = _replace_template_field(
            str(target["template_text"]), "amount_band", "gt10000"
        )
        _resign_embedding(target)
    elif case == "future-same-step-count":
        target["template_text"] = _replace_template_field(
            str(target["template_text"]), "prior_sender_count", "1"
        )
        _resign_embedding(target)
    elif case == "nonfraud-eligible":
        target["reference_eligible"] = True
        _resign_embedding(target)
    elif case == "fraud-ineligible":
        target = training_fraud
        target["reference_eligible"] = False
        _resign_embedding(target)
    elif case == "validation-eligible":
        target = validation
        target["reference_eligible"] = True
        _resign_embedding(target)
    else:
        target["reference_eligibility_provenance"] = "untrusted-artifact-claim"
        _resign_embedding(target)

    with pytest.raises(ReconciliationError) as captured:
        reconcile_embedding_documents(
            source,
            manifest,
            url,
            documents,
            expected_model_id=FakeEncoder.model_id,
            expected_model_revision=FakeEncoder.model_revision,
            expected_embedding_version=settings.embedding_version,
            expected_template_version=settings.behavioral_template_version,
        )
    report = json.loads(str(captured.value))
    assert report[expected_category]["count"] >= 1
