"""PostgreSQL-backed graph projection, reconciliation, and snapshot tests."""

import json
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import Engine, text

from fraud_detection.ingestion.errors import ReconciliationError
from fraud_detection.ingestion.loader import load_initial_population
from fraud_detection.ingestion.sampling import generate_sample
from fraud_detection.persistence.document_vector.projection import iter_expected_raw_documents
from fraud_detection.persistence.graph.analysis import (
    analyze_historical_evidence,
    persist_historical_analysis,
)
from fraud_detection.persistence.graph.errors import GremlinEvidenceConflictError
from fraud_detection.persistence.graph.models import GraphEdge, graph_digest
from fraud_detection.persistence.graph.projection import project_initial_graph
from fraud_detection.persistence.graph.reconciliation import reconcile_graph_projection
from tests.cosmos.fakes import FakeRawStore
from tests.graph.fakes import FakeGraphStore
from tests.paysim_factory import write_paysim_csv

pytestmark = pytest.mark.integration


@pytest.fixture
def clean_graph_database(postgres_engine: Engine) -> Iterator[Engine]:
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))
    yield postgres_engine
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))


def setup_graph_fixture(
    engine: Engine, tmp_path: Path
) -> tuple[Path, Path, Path, str, FakeRawStore]:
    source = write_paysim_csv(tmp_path / "paysim.csv")
    sample = tmp_path / "sample.parquet"
    manifest = tmp_path / "manifest.json"
    generated = generate_sample(source, sample, manifest, target_size=30, sampling_seed=123)
    url = engine.url.render_as_string(hide_password=False)
    load_initial_population(source, manifest, url)
    raw_store = FakeRawStore()
    raw_store.documents.extend(
        document.cosmos_payload()
        for document in iter_expected_raw_documents(
            source, manifest, url, batch_size=7, projected_at=generated.created_at
        )
    )
    return source, sample, manifest, url, raw_store


def test_graph_projection_is_idempotent_label_safe_and_reconciles(
    clean_graph_database: Engine, tmp_path: Path
) -> None:
    source, sample, manifest, url, raw_store = setup_graph_fixture(clean_graph_database, tmp_path)
    graph_store = FakeGraphStore()

    first = project_initial_graph(
        source,
        manifest,
        url,
        raw_store,
        graph_store,
        shard_count=64,
        page_size=5,
    )
    second = project_initial_graph(
        source,
        manifest,
        url,
        raw_store,
        graph_store,
        shard_count=64,
        page_size=5,
    )
    report = reconcile_graph_projection(
        source,
        manifest,
        url,
        raw_store,
        graph_store,
        shard_count=64,
        chunk_size=5,
        page_size=5,
    )

    assert first.processed_transactions == second.processed_transactions == 24
    assert first.created_edges == 72
    assert second.existing_edges == 72
    assert len([edge for edge in graph_store.edges if edge["kind"] == "money_flow"]) == 24
    assert report.reconciled is True
    assert report.batch_status == first.batch_status == second.batch_status == "pending"
    forbidden = {"isFraud", "isFlaggedFraud", "is_fraud", "source_flag"}
    assert not any(
        forbidden & set(element) for element in [*graph_store.vertices, *graph_store.edges]
    )
    test_ids = set(
        pl.read_parquet(sample).filter(pl.col("data_split") == "test")["transaction_id"].to_list()
    )
    assert not (test_ids & {edge["transaction_id"] for edge in graph_store.edges})


def test_partial_graph_failure_recovers_and_batch_stays_pending(
    clean_graph_database: Engine, tmp_path: Path
) -> None:
    source, _, manifest, url, raw_store = setup_graph_fixture(clean_graph_database, tmp_path)
    graph_store = FakeGraphStore()
    graph_store.fail_after = 6

    with pytest.raises(RuntimeError, match="controlled graph failure"):
        project_initial_graph(
            source,
            manifest,
            url,
            raw_store,
            graph_store,
            shard_count=64,
            page_size=5,
        )
    graph_store.fail_after = None
    recovered = project_initial_graph(
        source,
        manifest,
        url,
        raw_store,
        graph_store,
        shard_count=64,
        page_size=5,
    )

    assert recovered.processed_transactions == 24
    assert recovered.batch_status == "pending"


def test_graph_immutable_conflict_is_rejected(clean_graph_database: Engine, tmp_path: Path) -> None:
    source, _, manifest, url, raw_store = setup_graph_fixture(clean_graph_database, tmp_path)
    graph_store = FakeGraphStore()
    project_initial_graph(
        source, manifest, url, raw_store, graph_store, shard_count=64, page_size=5
    )
    graph_store.edges[0]["amount"] = "999.00"
    graph_store.edges[0]["content_digest"] = graph_digest(graph_store.edges[0])

    with pytest.raises(RuntimeError, match="controlled edge conflict"):
        project_initial_graph(
            source, manifest, url, raw_store, graph_store, shard_count=64, page_size=5
        )


def test_graph_reconciliation_reports_failures_and_hidden_reserved_test_id(
    clean_graph_database: Engine, tmp_path: Path
) -> None:
    source, sample, manifest, url, raw_store = setup_graph_fixture(clean_graph_database, tmp_path)
    graph_store = FakeGraphStore()
    project_initial_graph(
        source, manifest, url, raw_store, graph_store, shard_count=64, page_size=5
    )
    graph_store.edges.pop(0)
    graph_store.edges[0]["amount"] = "88.00"
    graph_store.edges[0]["content_digest"] = graph_digest(graph_store.edges[0])
    graph_store.edges[1]["content_digest"] = "invalid"
    graph_store.edges.append(dict(graph_store.edges[2]))
    orphan = dict(graph_store.edges[3])
    orphan["id"] = "orphan-edge"
    orphan["content_digest"] = graph_digest(orphan)
    graph_store.edges.append(orphan)
    test_id = str(
        pl.read_parquet(sample).filter(pl.col("data_split") == "test")["transaction_id"].item(0)
    )
    graph_store.edges.append(
        {
            "id": "hidden-test-edge",
            "transaction_id": test_id,
            "batch_id": "wrong-batch",
            "data_split": "train",
            "pk": "wrong-partition",
        }
    )

    with pytest.raises(ReconciliationError) as captured:
        reconcile_graph_projection(
            source,
            manifest,
            url,
            raw_store,
            graph_store,
            shard_count=64,
            chunk_size=2,
            page_size=2,
        )
    report = json.loads(str(captured.value))
    assert report["missing"]["count"] >= 1
    assert report["orphaned"]["count"] >= 1
    assert report["duplicate"]["count"] >= 1
    assert report["conflicting"]["count"] >= 1
    assert report["invalid"]["count"] >= 1
    assert report["reserved_test"]["sample_ids"] == [test_id]


def test_snapshot_persistence_is_idempotent_and_immutable(
    clean_graph_database: Engine, tmp_path: Path
) -> None:
    source, _, manifest, url, raw_store = setup_graph_fixture(clean_graph_database, tmp_path)
    graph_store = FakeGraphStore()
    projected = project_initial_graph(
        source, manifest, url, raw_store, graph_store, shard_count=64, page_size=5
    )
    evidence = [GraphEdge.model_validate(edge) for edge in graph_store.edges]
    result = analyze_historical_evidence(
        evidence,
        batch_id=projected.batch_id,
        cutoff_step=5,
        louvain_seed=73751,
        resolution=1.0,
    )

    persist_historical_analysis(url, result)
    persist_historical_analysis(url, result)
    with clean_graph_database.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM graph_snapshot")) == 1
        assert connection.scalar(text("SELECT count(*) FROM community")) == result.community_count
        assert (
            connection.scalar(text("SELECT count(*) FROM account_risk_profile"))
            == result.node_count
        )
        assert connection.scalar(text("SELECT status FROM ingestion_batch")) == "pending"
    with clean_graph_database.begin() as connection:
        connection.execute(text("UPDATE graph_snapshot SET algorithm_version='conflict'"))
    with pytest.raises(GremlinEvidenceConflictError, match="conflicts on algorithm_version"):
        persist_historical_analysis(url, result)
