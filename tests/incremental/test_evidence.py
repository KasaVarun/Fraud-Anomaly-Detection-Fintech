"""Phase 8 local evidence and leakage-boundary tests."""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fraud_detection.config import Settings
from fraud_detection.features.behavioral import build_behavioral_templates
from fraud_detection.incremental.evidence import prepare_offline_evidence
from fraud_detection.incremental.planning import plan_test_chunks, reserved_test_rows
from fraud_detection.incremental.reconciliation import _offline_evidence_counts
from fraud_detection.incremental.scoring import _validated_scoring_metadata
from fraud_detection.ingestion.hashing import sha256_file
from fraud_detection.modeling.contracts import ALL_FEATURES, FORBIDDEN_MODEL_FIELDS
from fraud_detection.persistence.document_vector.artifacts import (
    embedding_artifact_logical_digest,
    write_embedding_artifact,
)
from fraud_detection.persistence.document_vector.embedding_generation import (
    build_embedding_document,
)
from tests.graph.helpers import raw_document
from tests.incremental.helpers import incremental_fixture
from tests.vector.fakes import FakeEncoder


def _prepared_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Settings, Path]:
    source, _, manifest = incremental_fixture(tmp_path, [355, 355, 356])
    settings = Settings(
        _env_file=None,
        incremental_chunk_steps=24,
        incremental_data_root=tmp_path / "phase8",
        embedding_batch_size=2,
        embedding_version="test-embedding-v1",
        gremlin_partition_shards=8,
    )
    training_raw = raw_document(transaction_id="training-fraud", is_fraud=True)
    template = build_behavioral_templates([training_raw])[0]
    encoder = FakeEncoder()
    training_embedding = build_embedding_document(
        template,
        encoder.encode([template.text], batch_size=1)[0],
        encoder,
        settings,
        datetime(2026, 10, 31, tzinfo=UTC),
    )
    training_path = tmp_path / "training-embeddings.parquet"
    write_embedding_artifact(training_path, [training_embedding], batch_size=1)
    settings = settings.model_copy(update={"embedding_local_artifact": training_path})
    monkeypatch.setattr(
        "fraud_detection.incremental.evidence.PHASE6_LOGICAL_EMBEDDING_DIGEST",
        embedding_artifact_logical_digest(training_path),
    )
    prepare_offline_evidence(
        source,
        manifest,
        training_path,
        settings.incremental_data_root,
        settings,
        encoder=encoder,
        expected_sample_count=5,
        expected_test_count=3,
    )
    return source, manifest, training_path, settings, settings.incremental_data_root


def test_offline_evidence_uses_test_rows_without_label_predictors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _, manifest = incremental_fixture(tmp_path, [355, 355, 356])
    settings = Settings(
        _env_file=None,
        incremental_chunk_steps=24,
        incremental_data_root=tmp_path / "phase8",
        embedding_batch_size=2,
        embedding_version="test-embedding-v1",
        gremlin_partition_shards=8,
    )
    training_raw = raw_document(transaction_id="training-fraud", is_fraud=True)
    template = build_behavioral_templates([training_raw])[0]
    encoder = FakeEncoder()
    training_embedding = build_embedding_document(
        template,
        encoder.encode([template.text], batch_size=1)[0],
        encoder,
        settings,
        datetime(2026, 10, 31, tzinfo=UTC),
    )
    training_path = tmp_path / "training-embeddings.parquet"
    write_embedding_artifact(training_path, [training_embedding], batch_size=1)
    settings = settings.model_copy(update={"embedding_local_artifact": training_path})
    monkeypatch.setattr(
        "fraud_detection.incremental.evidence.PHASE6_LOGICAL_EMBEDDING_DIGEST",
        embedding_artifact_logical_digest(training_path),
    )

    result = prepare_offline_evidence(
        source,
        manifest,
        training_path,
        settings.incremental_data_root,
        settings,
        encoder=encoder,
        expected_sample_count=5,
        expected_test_count=3,
    )

    features = pl.read_parquet(result.feature_artifact)
    assert result.test_count == 3
    assert features.height == 3
    assert set(features["data_split"].to_list()) == {"test"}
    assert tuple(features.select(ALL_FEATURES).columns) == ALL_FEATURES
    nonpredictive_lineage = {"transaction_id", "batch_id", "data_split"}
    assert set(features.columns).isdisjoint(FORBIDDEN_MODEL_FIELDS - nonpredictive_lineage)
    assert "is_fraud" not in features.columns
    plan = plan_test_chunks(source, manifest, expected_test_count=3)
    test_rows, _ = reserved_test_rows(source, manifest, expected_test_count=3)
    expected_ids = {str(value) for value in test_rows["transaction_id"].to_list()}
    counts = _offline_evidence_counts(plan.chunks[0], expected_ids, settings)
    assert counts == {
        "raw_documents": 3,
        "graph_vertices_required": 10,
        "graph_edges": 9,
        "templates": 3,
        "embeddings": 3,
        "features": 3,
    }


@pytest.mark.parametrize(
    "artifact_name",
    [
        "test_raw_documents_v1.parquet",
        "test_graph_vertices_v1.parquet",
        "test_graph_edges_v1.parquet",
        "test_behavior_templates_v1.parquet",
        "test_behavior_embeddings_v1.parquet",
        "locked_test_features_v2.parquet",
        "locked_test_features_v2.metadata.json",
        "locked_test_features_v2.metadata.v2.json",
    ],
)
def test_tampered_offline_artifact_is_rejected_without_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact_name: str
) -> None:
    source, manifest, training, settings, root = _prepared_release(tmp_path, monkeypatch)
    path = root / artifact_name
    path.write_bytes(b"tampered-offline-evidence")
    tampered = path.read_bytes()

    with pytest.raises((OSError, ValueError, pl.exceptions.PolarsError)):
        prepare_offline_evidence(
            source,
            manifest,
            training,
            root,
            settings,
            encoder=FakeEncoder(),
            expected_sample_count=5,
            expected_test_count=3,
        )
    assert path.read_bytes() == tampered


def test_manifest_tampering_and_missing_artifacts_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, manifest, training, settings, root = _prepared_release(tmp_path, monkeypatch)
    authoritative = root / "offline_evidence_manifest_v2.json"
    payload = json.loads(authoritative.read_text(encoding="utf-8"))
    payload["feature_version"] = "tampered"
    authoritative.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        prepare_offline_evidence(
            source,
            manifest,
            training,
            root,
            settings,
            expected_sample_count=5,
            expected_test_count=3,
        )

    source2, manifest2, training2, settings2, root2 = _prepared_release(
        tmp_path / "missing", monkeypatch
    )
    missing = root2 / "test_raw_documents_v1.parquet"
    missing.unlink()
    with pytest.raises((OSError, ValueError)):
        prepare_offline_evidence(
            source2,
            manifest2,
            training2,
            root2,
            settings2,
            expected_sample_count=5,
            expected_test_count=3,
        )
    assert not missing.exists()


def test_feature_metadata_tampering_is_rejected_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, manifest, _, _, root = _prepared_release(tmp_path, monkeypatch)
    metadata_path = root / "locked_test_features_v2.metadata.v2.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_logical_digest"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    plan = plan_test_chunks(source, manifest, expected_test_count=3)
    with pytest.raises(ValueError, match=r"anchored|metadata"):
        _validated_scoring_metadata(root / "locked_test_features_v2.parquet", metadata_path, plan)


def test_identical_rerun_reuses_artifacts_without_rewriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, manifest, training, settings, root = _prepared_release(tmp_path, monkeypatch)
    before = {
        path.name: (sha256_file(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.iterdir()
    }
    result = prepare_offline_evidence(
        source,
        manifest,
        training,
        root,
        settings,
        expected_sample_count=5,
        expected_test_count=3,
    )
    after = {
        path.name: (sha256_file(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.iterdir()
    }
    assert before == after
    assert result.build_seconds == 0.0


def test_failed_staged_build_leaves_no_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _, manifest = incremental_fixture(tmp_path, [355])
    training = tmp_path / "training.parquet"
    training.write_bytes(b"placeholder")
    settings = Settings(_env_file=None, incremental_data_root=tmp_path / "fresh")

    def fail_build(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("controlled build failure")

    monkeypatch.setattr(
        "fraud_detection.incremental.evidence._build_offline_evidence_unpublished",
        fail_build,
    )
    with pytest.raises(RuntimeError, match="controlled build failure"):
        prepare_offline_evidence(
            source,
            manifest,
            training,
            settings.incremental_data_root,
            settings,
            expected_sample_count=3,
            expected_test_count=1,
        )
    assert not list(tmp_path.glob(".fresh.stage-*"))
    assert not settings.incremental_data_root.exists()


def test_unexpected_final_artifact_without_manifest_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _, manifest = incremental_fixture(tmp_path, [355])
    root = tmp_path / "unexpected"
    root.mkdir()
    artifact = root / "test_raw_documents_v1.parquet"
    artifact.write_bytes(b"unexpected-existing-content")
    settings = Settings(_env_file=None, incremental_data_root=root)
    training = tmp_path / "training.parquet"
    training.write_bytes(b"unused")
    with pytest.raises(ValueError, match="unexpected offline artifact"):
        prepare_offline_evidence(
            source,
            manifest,
            training,
            root,
            settings,
            expected_sample_count=3,
            expected_test_count=1,
        )
    assert artifact.read_bytes() == b"unexpected-existing-content"
