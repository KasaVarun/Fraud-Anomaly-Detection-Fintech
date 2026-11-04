"""PostgreSQL-backed Phase 8 recovery, scoring, reconciliation, and publication tests."""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from sqlalchemy import Engine, func, insert, select, text, update

from fraud_detection.config import Settings
from fraud_detection.features.behavioral import build_behavioral_templates
from fraud_detection.incremental import loader as incremental_loader
from fraud_detection.incremental.contracts import (
    LOCKED_MODEL_ID,
    LOCKED_THRESHOLD,
)
from fraud_detection.incremental.evidence import _raw_document
from fraud_detection.incremental.planning import plan_test_chunks
from fraud_detection.incremental.reconciliation import (
    ExpectedLiveEvidence,
    LiveStores,
    attempt_publication,
    evaluate_pooled_locked_test,
    reconcile_chunk,
)
from fraud_detection.incremental.scoring import run_offline_chunk
from fraud_detection.ingestion.hashing import sha256_file
from fraud_detection.ingestion.manifest import read_manifest
from fraud_detection.modeling.contracts import ALL_FEATURES
from fraud_detection.persistence.document_vector.embedding_generation import (
    build_embedding_document,
)
from fraud_detection.persistence.document_vector.models import content_digest
from fraud_detection.persistence.graph.projection import build_graph_evidence
from fraud_detection.persistence.relational import Base
from tests.cosmos.fakes import FakeRawStore
from tests.graph.fakes import FakeGraphStore
from tests.incremental.helpers import incremental_fixture
from tests.vector.fakes import FakeEmbeddingStore, FakeEncoder

pytestmark = pytest.mark.integration


class FakeLockedModel:
    def predict_proba(self, matrix: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
        probabilities = np.asarray(matrix["amount_log1p"], dtype=np.float64)
        return np.column_stack((1.0 - probabilities, probabilities))


@pytest.fixture
def clean_incremental_database(postgres_engine: Engine) -> Iterator[Engine]:
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))
        connection.execute(
            Base.metadata.tables["model_version"]
            .delete()
            .where(Base.metadata.tables["model_version"].c.model_version_id == LOCKED_MODEL_ID)
        )
        connection.execute(
            insert(Base.metadata.tables["model_version"]).values(
                model_version_id=LOCKED_MODEL_ID,
                feature_set="combined",
                training_cutoff_step=281,
                threshold=LOCKED_THRESHOLD,
                embedding_version="minilm-l6-v2-1110a243-template-v1-normalized-float32",
                artifact_uri="trusted-test-artifact",
                created_at=datetime(2026, 10, 31, tzinfo=UTC),
            )
        )
    yield postgres_engine
    with postgres_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ingestion_batch, account CASCADE"))
        connection.execute(
            Base.metadata.tables["model_version"]
            .delete()
            .where(Base.metadata.tables["model_version"].c.model_version_id == LOCKED_MODEL_ID)
        )


def _feature_artifact(
    tmp_path: Path, transaction_ids: list[str], batch_id: str
) -> tuple[Path, Path]:
    probabilities = [0.9496, 0.9495999, 0.1]
    records: list[dict[str, object]] = []
    for index, transaction_id in enumerate(transaction_ids):
        record: dict[str, object] = {
            "transaction_id": transaction_id,
            "batch_id": batch_id,
            "step": 355 if index < 2 else 356,
            "data_split": "test",
        }
        record.update({feature: 0.0 for feature in ALL_FEATURES})
        record["amount_log1p"] = probabilities[index]
        record["nearest_earlier_training_fraud_cosine_distance"] = 0.5
        records.append(record)
    artifact = tmp_path / "features.parquet"
    pl.DataFrame(records).write_parquet(artifact)
    metadata = tmp_path / "features.metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "feature_sha256": sha256_file(artifact),
                "feature_version": "tabular-graph-vector-v2",
                "feature_logical_digest": "f" * 64,
            }
        ),
        encoding="utf-8",
    )
    return artifact, metadata


def test_interrupted_run_recovers_idempotently_and_publication_stays_blocked(
    clean_incremental_database: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, sample_path, manifest = incremental_fixture(tmp_path, [355, 355, 356])
    full_plan = plan_test_chunks(source, manifest, expected_test_count=3)
    chunk = full_plan.chunks[0]
    plan = full_plan.model_copy(update={"chunks": (chunk,)})
    test_ids = (
        pl.read_parquet(sample_path)
        .filter(pl.col("data_split") == "test")
        .sort(["step", "transaction_id"])["transaction_id"]
        .to_list()
    )
    feature_artifact, feature_metadata = _feature_artifact(
        tmp_path, [str(value) for value in test_ids], chunk.batch_id
    )
    monkeypatch.setattr(
        "fraud_detection.incremental.scoring._verify_locked_model",
        lambda *args: (FakeLockedModel(), {}),
    )
    url = clean_incremental_database.url.render_as_string(hide_password=False)

    original_insert = incremental_loader._insert_verify

    def fail_relational(
        connection: Any,
        table: Any,
        records: list[dict[str, Any]],
        key: str,
        immutable_fields: tuple[str, ...],
    ) -> None:
        if table.name == "transaction":
            raise RuntimeError("controlled relational ingestion failure")
        original_insert(connection, table, records, key, immutable_fields)

    monkeypatch.setattr(incremental_loader, "_insert_verify", fail_relational)
    with pytest.raises(RuntimeError, match="relational ingestion"):
        run_offline_chunk(
            source,
            manifest,
            feature_artifact,
            feature_metadata,
            url,
            tmp_path,
            plan,
            chunk,
            expected_test_count=3,
            verify_authoritative_metadata=False,
        )
    monkeypatch.setattr(incremental_loader, "_insert_verify", original_insert)
    with clean_incremental_database.connect() as connection:
        assert (
            connection.scalar(
                select(func.count()).select_from(Base.metadata.tables["ingestion_batch"])
            )
            == 0
        )

    def interrupt(stage: str) -> None:
        if stage == "scores":
            raise RuntimeError("controlled interruption")

    with pytest.raises(RuntimeError, match="controlled interruption"):
        run_offline_chunk(
            source,
            manifest,
            feature_artifact,
            feature_metadata,
            url,
            tmp_path,
            plan,
            chunk,
            stage_hook=interrupt,
            expected_test_count=3,
            verify_authoritative_metadata=False,
        )
    with clean_incremental_database.connect() as connection:
        assert (
            connection.scalar(
                select(func.count()).select_from(Base.metadata.tables["transaction_score"])
            )
            == 0
        )
        assert (
            connection.scalar(
                select(func.count()).select_from(Base.metadata.tables["graph_snapshot"])
            )
            == 0
        )

    first = run_offline_chunk(
        source,
        manifest,
        feature_artifact,
        feature_metadata,
        url,
        tmp_path,
        plan,
        chunk,
        expected_test_count=3,
        verify_authoritative_metadata=False,
    )
    second = run_offline_chunk(
        source,
        manifest,
        feature_artifact,
        feature_metadata,
        url,
        tmp_path,
        plan,
        chunk,
        expected_test_count=3,
        verify_authoritative_metadata=False,
    )
    assert first.metrics == second.metrics
    assert first.score_count == second.score_count == 3
    assert first.flag_count == second.flag_count == 1
    assert first.snapshot_count == 2
    assert first.evaluation_count == 1
    assert (first.metrics.tp, first.metrics.fp, first.metrics.tn, first.metrics.fn) == (
        1,
        0,
        1,
        1,
    )
    for interrupted_stage in ("ingestion", "snapshots", "flags", "evaluation"):

        def interrupt_later(stage: str, target: str = interrupted_stage) -> None:
            if stage == target:
                raise RuntimeError(f"controlled {target} interruption")

        with pytest.raises(RuntimeError, match=interrupted_stage):
            run_offline_chunk(
                source,
                manifest,
                feature_artifact,
                feature_metadata,
                url,
                tmp_path,
                plan,
                chunk,
                stage_hook=interrupt_later,
                expected_test_count=3,
                verify_authoritative_metadata=False,
            )
        resumed = run_offline_chunk(
            source,
            manifest,
            feature_artifact,
            feature_metadata,
            url,
            tmp_path,
            plan,
            chunk,
            expected_test_count=3,
            verify_authoritative_metadata=False,
        )
        assert (resumed.score_count, resumed.flag_count, resumed.snapshot_count) == (3, 1, 2)

    settings = Settings(_env_file=None)
    reconciliation = reconcile_chunk(
        url, feature_artifact, chunk, settings, verify_offline_artifacts=False
    )
    assert reconciliation.offline_reconciled is True
    assert reconciliation.publication_eligible is False
    assert set(reconciliation.missing_live_stores) == {
        "cosmos_nosql_raw",
        "cosmos_nosql_embeddings",
        "cosmos_gremlin",
    }
    publication = attempt_publication(
        url,
        feature_artifact,
        plan,
        chunk,
        settings,
        verify_offline_artifacts=False,
    )
    assert publication.published is False
    assert publication.status == "pending"
    assert publication.reason == "initial_batch_not_published"

    manifest_value = read_manifest(manifest)
    with clean_incremental_database.begin() as connection:
        connection.execute(
            insert(Base.metadata.tables["ingestion_batch"]).values(
                batch_id=plan.initial_batch_id,
                source_sha256=manifest_value.source_sha256,
                sample_seed=manifest_value.sampling_seed,
                chunk_id="initial-dependency",
                enrichment_seed=manifest_value.enrichment_seed,
                split_policy=manifest_value.split_policy,
                status="pending",
                created_at=manifest_value.created_at,
            )
        )
    blocked_initial = attempt_publication(
        url,
        feature_artifact,
        plan,
        chunk,
        settings,
        verify_offline_artifacts=False,
    )
    assert blocked_initial.reason == "initial_batch_not_published"
    with clean_incremental_database.begin() as connection:
        connection.execute(
            update(Base.metadata.tables["ingestion_batch"])
            .where(Base.metadata.tables["ingestion_batch"].c.batch_id == plan.initial_batch_id)
            .values(status="published")
        )
    blocked_later = attempt_publication(
        url,
        feature_artifact,
        full_plan,
        full_plan.chunks[1],
        settings,
        verify_offline_artifacts=False,
    )
    assert blocked_later.reason == "earlier_chunk_not_published"

    configured = Settings(
        _env_file=None,
        cosmos_nosql_endpoint="https://example.documents.azure.com",
        cosmos_nosql_key="secret",
        cosmos_nosql_database="db",
        gremlin_endpoint="wss://example.gremlin.cosmos.azure.com",
        gremlin_username="/dbs/db/colls/graph",
        gremlin_database="db",
        gremlin_graph="graph",
        gremlin_key="secret",
        gremlin_partition_shards=8,
        embedding_version="test-embedding-v1",
    )
    raw_documents = tuple(
        _raw_document(
            row,
            batch_id=chunk.batch_id,
            manifest=manifest_value,
            projected_at=manifest_value.created_at,
        )
        for row in (
            pl.read_parquet(sample_path)
            .filter(pl.col("data_split") == "test")
            .sort(["step", "transaction_id"])
            .iter_rows(named=True)
        )
    )
    templates = build_behavioral_templates(raw_documents)
    encoder = FakeEncoder()
    embeddings = tuple(
        build_embedding_document(
            template,
            encoder.encode([template.text], batch_size=1)[0],
            encoder,
            configured,
            manifest_value.created_at,
        )
        for template in templates
    )
    vertices: dict[str, Any] = {}
    edges: list[Any] = []
    for document in raw_documents:
        bundle = build_graph_evidence(document, shard_count=8)
        vertices.update({vertex.id: vertex for vertex in bundle.vertices})
        edges.extend(bundle.edges)
    expected = ExpectedLiveEvidence(
        raw_documents=raw_documents,
        embeddings=embeddings,
        vertices=tuple(vertices.values()),
        edges=tuple(edges),
    )
    raw_store = FakeRawStore()
    embedding_store = FakeEmbeddingStore()
    graph_store = FakeGraphStore()
    for document in raw_documents:
        raw_store.upsert_document(document.cosmos_payload())
    for embedding in embeddings:
        embedding_store.upsert_embedding(embedding.payload())
    for vertex in expected.vertices:
        graph_store.upsert_vertex(vertex.payload())
    for edge in expected.edges:
        graph_store.upsert_edge(edge.payload())
    stores = LiveStores(raw_store, embedding_store, graph_store)

    corrupted = dict(raw_store.documents[0])
    corrupted["amount"] = "999.00"
    corrupted["content_digest"] = content_digest(corrupted)
    raw_store.documents[0] = corrupted
    with pytest.raises(ValueError, match="raw document"):
        attempt_publication(
            url,
            feature_artifact,
            plan,
            chunk,
            configured,
            verify_offline_artifacts=False,
            live_stores=stores,
            expected_evidence=expected,
        )
    with clean_incremental_database.connect() as connection:
        assert (
            connection.scalar(
                select(Base.metadata.tables["ingestion_batch"].c.status).where(
                    Base.metadata.tables["ingestion_batch"].c.batch_id == chunk.batch_id
                )
            )
            == "pending"
        )
    raw_store.documents[0] = raw_documents[0].cosmos_payload()

    def transition_failure() -> None:
        raise RuntimeError("controlled publication transition failure")

    with pytest.raises(RuntimeError, match="publication transition"):
        attempt_publication(
            url,
            feature_artifact,
            plan,
            chunk,
            configured,
            verify_offline_artifacts=False,
            live_stores=stores,
            expected_evidence=expected,
            transition_hook=transition_failure,
        )
    with clean_incremental_database.connect() as connection:
        assert (
            connection.scalar(
                select(Base.metadata.tables["ingestion_batch"].c.status).where(
                    Base.metadata.tables["ingestion_batch"].c.batch_id == chunk.batch_id
                )
            )
            == "pending"
        )
    published = attempt_publication(
        url,
        feature_artifact,
        plan,
        chunk,
        configured,
        verify_offline_artifacts=False,
        live_stores=stores,
        expected_evidence=expected,
    )
    assert published.published is True
    published_rerun = run_offline_chunk(
        source,
        manifest,
        feature_artifact,
        feature_metadata,
        url,
        tmp_path,
        plan,
        chunk,
        expected_test_count=3,
        verify_authoritative_metadata=False,
    )
    assert published_rerun.batch_status == "published"
    assert published_rerun.score_count == 3
    assert published_rerun.flag_count == 1
    assert attempt_publication(
        url,
        feature_artifact,
        plan,
        chunk,
        configured,
        verify_offline_artifacts=False,
        live_stores=stores,
        expected_evidence=expected,
    ).published

    pooled = evaluate_pooled_locked_test(url, plan, tmp_path / "pooled.json")
    rerun = evaluate_pooled_locked_test(url, plan, tmp_path / "pooled.json")
    assert pooled == rerun
    assert pooled.metrics == first.metrics

    with clean_incremental_database.begin() as connection:
        connection.execute(
            insert(Base.metadata.tables["ingestion_batch"]).values(
                batch_id="prior-published",
                source_sha256="b" * 64,
                sample_seed=1,
                chunk_id="prior",
                enrichment_seed=2,
                split_policy="prior",
                status="published",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
    attempt_publication(
        url,
        feature_artifact,
        plan,
        chunk,
        settings,
        verify_offline_artifacts=False,
    )
    with clean_incremental_database.connect() as connection:
        assert (
            connection.scalar(
                select(Base.metadata.tables["ingestion_batch"].c.status).where(
                    Base.metadata.tables["ingestion_batch"].c.batch_id == "prior-published"
                )
            )
            == "published"
        )

    with clean_incremental_database.begin() as connection:
        score = Base.metadata.tables["transaction_score"]
        score_identity = connection.scalar(select(score.c.score_id).limit(1))
        connection.execute(
            update(score)
            .where(score.c.score_id == score_identity)
            .values(fraud_probability=Decimal("0.1234567"))
        )
    with pytest.raises(ValueError, match="fraud_probability"):
        run_offline_chunk(
            source,
            manifest,
            feature_artifact,
            feature_metadata,
            url,
            tmp_path,
            plan,
            chunk,
            expected_test_count=3,
            verify_authoritative_metadata=False,
        )
