"""Leakage-safe Phase 7 feature contract tests."""

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pytest

from fraud_detection.config import Settings
from fraud_detection.features.behavioral import build_behavioral_templates
from fraud_detection.ingestion.hashing import sha256_file
from fraud_detection.modeling.contracts import (
    ALL_FEATURES,
    FEATURE_FAMILIES,
    FORBIDDEN_MODEL_FIELDS,
    GRAPH_FEATURES,
    TABULAR_FEATURES,
    VECTOR_FEATURES,
    validate_predictor_allowlist,
)
from fraud_detection.modeling.features import (
    build_causal_graph_feature_rows,
    build_feature_artifact,
    build_vector_feature_rows,
    feature_logical_digest,
)
from fraud_detection.persistence.document_vector.artifacts import (
    embedding_artifact_logical_digest,
    write_embedding_artifact,
)
from fraud_detection.persistence.document_vector.embedding_generation import (
    build_embedding_document,
)
from fraud_detection.persistence.graph.analysis import build_historical_account_graph
from fraud_detection.persistence.graph.models import GraphEdge, graph_digest
from tests.graph.helpers import raw_document
from tests.vector.fakes import FakeEncoder


def _row(
    identifier: str, step: int, sender: str, receiver: str, source_row: int
) -> dict[str, object]:
    return {
        "transaction_id": identifier,
        "step": step,
        "nameOrig": sender,
        "nameDest": receiver,
        "source_row": source_row,
    }


def test_exact_feature_allowlists_exclude_every_forbidden_field() -> None:
    assert ALL_FEATURES == TABULAR_FEATURES + GRAPH_FEATURES + VECTOR_FEATURES
    assert set(ALL_FEATURES).isdisjoint(FORBIDDEN_MODEL_FIELDS)
    assert FEATURE_FAMILIES["combined"] == ALL_FEATURES
    for features in FEATURE_FAMILIES.values():
        validate_predictor_allowlist(features)
    with pytest.raises(ValueError, match="forbidden"):
        validate_predictor_allowlist(("is_fraud",))
    with pytest.raises(ValueError, match="unsupported"):
        validate_predictor_allowlist(("future_graph_degree",))


def test_causal_graph_features_are_input_order_independent_and_same_step_isolated() -> None:
    rows = [
        _row("a", 1, "C1", "C2", 1),
        _row("b", 1, "C1", "C3", 2),
        _row("c", 2, "C1", "C2", 3),
    ]
    first = build_causal_graph_feature_rows(rows, enrichment_seed=73_751)
    reversed_result = build_causal_graph_feature_rows(reversed(rows), enrichment_seed=73_751)

    assert first == reversed_result
    assert first["a"]["sender_prior_weighted_degree"] == 0
    assert first["b"]["sender_prior_weighted_degree"] == 0
    assert first["c"]["sender_prior_weighted_degree"] == 2
    assert first["c"]["prior_sender_receiver_transaction_count"] == 1


def test_future_graph_mutation_cannot_change_earlier_features() -> None:
    earlier = [_row("a", 1, "C1", "C2", 1), _row("b", 2, "C1", "C3", 2)]
    baseline = build_causal_graph_feature_rows(earlier, enrichment_seed=73_751)
    with_future = build_causal_graph_feature_rows(
        [*earlier, _row("future", 99, "C1", "C2", 3)], enrichment_seed=73_751
    )
    assert baseline["a"] == with_future["a"]
    assert baseline["b"] == with_future["b"]


def _graph_edge(
    identifier: str,
    *,
    kind: str,
    sender: str,
    receiver: str,
    step: int,
) -> GraphEdge:
    if kind == "money_flow":
        label = "TRANSFER_TO"
        amount: str | None = "10.00"
        synthetic = False
    elif kind == "device_observation":
        label = "USES_DEVICE"
        amount = None
        synthetic = True
    else:
        label = "USES_IP"
        amount = None
        synthetic = True
    payload: dict[str, object] = {
        "id": identifier,
        "label": label,
        "out_id": sender,
        "out_pk": "shard-0001",
        "in_id": receiver,
        "in_pk": "shard-0002",
        "transaction_id": identifier.split(":")[-1],
        "step": step,
        "batch_id": "batch-1",
        "data_split": "train",
        "transaction_type": "TRANSFER",
        "amount": amount,
        "kind": kind,
        "synthetic": synthetic,
        "source_version": "raw-paysim-v1",
        "provenance": "test",
        "enrichment_seed": 73751 if synthetic else None,
        "enrichment_algorithm_version": "synthetic-artifact-v1" if synthetic else None,
        "graph_schema_version": "1.0.0",
        "projection_version": "gremlin-evidence-v1",
    }
    payload["content_digest"] = graph_digest(payload)
    return GraphEdge.model_validate(payload)


def test_causal_graph_features_match_phase5_historical_graph_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _row("t1", 1, "C1", "C2", 1),
        _row("t2", 2, "C1", "C2", 2),
        _row("t3", 2, "C3", "C3", 3),
        _row("t4", 3, "C3", "C4", 4),
        _row("t5", 4, "C2", "C4", 5),
        _row("future", 99, "C9", "C1", 6),
    ]
    devices = {value: "device:A" for value in ("t1", "t2", "t3", "t4", "t5")}
    devices["future"] = "device:future"
    ips = {"t1": "ip:A", "t2": "ip:B", "t3": "ip:A", "t4": "ip:A", "t5": "ip:A"}
    ips["future"] = "ip:future"

    def artifact_id(
        kind: str, *, transaction_id: str, source_row: int, step: int, seed: int
    ) -> str:
        del source_row, step, seed
        return devices[transaction_id] if kind == "device" else ips[transaction_id]

    monkeypatch.setattr("fraud_detection.modeling.features.deterministic_artifact_id", artifact_id)
    features = build_causal_graph_feature_rows(rows, enrichment_seed=73751)
    evidence: list[GraphEdge] = []
    for row in rows:
        transaction_id = str(row["transaction_id"])
        step = int(cast(Any, row["step"]))
        sender = str(row["nameOrig"])
        receiver = str(row["nameDest"])
        evidence.extend(
            [
                _graph_edge(
                    f"money:{transaction_id}",
                    kind="money_flow",
                    sender=sender,
                    receiver=receiver,
                    step=step,
                ),
                _graph_edge(
                    f"device:{transaction_id}",
                    kind="device_observation",
                    sender=sender,
                    receiver=devices[transaction_id],
                    step=step,
                ),
                _graph_edge(
                    f"ip:{transaction_id}",
                    kind="ip_observation",
                    sender=sender,
                    receiver=ips[transaction_id],
                    step=step,
                ),
            ]
        )
    evidence.extend([evidence[1], evidence[2]])  # duplicate observations must be idempotent

    for row in rows[:-1]:
        cutoff = int(cast(Any, row["step"]))
        sender = str(row["nameOrig"])
        receiver = str(row["nameDest"])
        graph = build_historical_account_graph(evidence, cutoff_step=cutoff)
        node_count = graph.number_of_nodes()
        denominator = node_count - 1 if node_count >= 2 else 1
        expected_sender_centrality = (
            graph.degree(sender) / denominator if sender in graph and node_count >= 2 else 0.0
        )
        expected_receiver_centrality = (
            graph.degree(receiver) / denominator if receiver in graph and node_count >= 2 else 0.0
        )
        expected_sender_weight = (
            float(graph.degree(sender, weight="weight")) if sender in graph else 0.0
        )
        expected_receiver_weight = (
            float(graph.degree(receiver, weight="weight")) if receiver in graph else 0.0
        )
        actual = features[str(row["transaction_id"])]
        assert actual["sender_prior_degree_centrality"] == expected_sender_centrality
        assert actual["receiver_prior_degree_centrality"] == expected_receiver_centrality
        assert actual["sender_prior_weighted_degree"] == expected_sender_weight
        assert actual["receiver_prior_weighted_degree"] == expected_receiver_weight

    without_future = build_causal_graph_feature_rows(rows[:-1], enrichment_seed=73751)
    assert all(features[key] == without_future[key] for key in without_future)
    assert features["t2"] == build_causal_graph_feature_rows(rows[:2], enrichment_seed=73751)["t2"]
    assert features["t3"]["sender_prior_weighted_degree"] == 0.0  # self-loop not yet visible


def test_feature_logical_digest_is_stable_under_input_order() -> None:
    frame = pl.DataFrame(
        {
            "transaction_id": ["a", "b"],
            "step": [1, 2],
            "value": [1.0, 2.0],
        }
    )
    first = feature_logical_digest(frame.sort(["step", "transaction_id"]))
    second = feature_logical_digest(frame.reverse().sort(["step", "transaction_id"]))
    assert first == second


def test_vector_features_use_strictly_earlier_training_fraud_only(tmp_path: Path) -> None:
    raw = [
        raw_document(transaction_id="fraud", step=1, source_row=1, is_fraud=True),
        raw_document(transaction_id="same-step", step=1, source_row=2),
        raw_document(transaction_id="query", step=2, source_row=3),
        raw_document(
            transaction_id="validation-fraud", step=2, source_row=4, is_fraud=True
        ).model_copy(update={"data_split": "validation"}),
    ]
    templates = build_behavioral_templates(raw)
    encoder = FakeEncoder()
    settings = Settings(_env_file=None, embedding_version="test-embedding-v1")
    documents = [
        build_embedding_document(
            template,
            encoder.encode([template.text], batch_size=1)[0],
            encoder,
            settings,
            datetime(2026, 10, 31, tzinfo=UTC),
        )
        for template in templates
    ]
    artifact = tmp_path / "embeddings.parquet"
    write_embedding_artifact(artifact, documents, batch_size=2)
    rows = [
        {
            "transaction_id": document.id,
            "step": document.step,
            "data_split": document.data_split,
            "isFraud": document.source_truth.is_fraud,
        }
        for document in raw
    ]
    result = build_vector_feature_rows(
        rows,
        artifact,
        expected_logical_digest=embedding_artifact_logical_digest(artifact),
        query_batch_size=1,
    )

    assert result["fraud"]["missing_earlier_training_fraud_reference"] == 1
    assert result["same-step"]["missing_earlier_training_fraud_reference"] == 1
    assert result["query"]["missing_earlier_training_fraud_reference"] == 0
    assert result["validation-fraud"]["missing_earlier_training_fraud_reference"] == 0
    assert np.isfinite(result["query"]["nearest_earlier_training_fraud_cosine_distance"])


def test_feature_artifact_build_excludes_reserved_test_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    source.write_text("approved-source", encoding="utf-8")
    sample = pl.DataFrame(
        {
            "transaction_id": ["train-a", "train-b", "validation-a", "validation-b", "test-a"],
            "source_row": [1, 2, 3, 4, 5],
            "step": [1, 281, 282, 354, 355],
            "data_split": ["train", "train", "validation", "validation", "test"],
            "type": ["TRANSFER", "PAYMENT", "DEBIT", "CASH_OUT", "CASH_IN"],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
            "nameOrig": ["C1", "C2", "C3", "C4", "C5"],
            "nameDest": ["C2", "M1", "C4", "C5", "C6"],
            "isFraud": [True, False, True, False, True],
        }
    )
    sample_path = tmp_path / "sample.parquet"
    sample.write_parquet(sample_path)
    manifest = {
        "manifest_schema_version": "1.0.0",
        "source_filename": source.name,
        "source_sha256": sha256_file(source),
        "source_byte_size": source.stat().st_size,
        "source_row_count": 5,
        "source_fraud_count": 3,
        "source_fraud_prevalence": 0.6,
        "source_columns": list(sample.columns),
        "source_types": {},
        "source_validation_timestamp": "2026-10-31T00:00:00Z",
        "sampling_seed": 73750,
        "enrichment_seed": 73751,
        "target_sample_size": 5,
        "actual_sample_size": 5,
        "split_policy": "whole-step-60-20-20-v1",
        "splits": [
            {
                "name": "train",
                "minimum_step": 1,
                "maximum_step": 281,
                "source_count": 2,
                "target_count": 2,
                "actual_count": 2,
                "fraud_count": 1,
                "nonfraud_count": 1,
                "fraud_prevalence": 0.5,
            },
            {
                "name": "validation",
                "minimum_step": 282,
                "maximum_step": 354,
                "source_count": 2,
                "target_count": 2,
                "actual_count": 2,
                "fraud_count": 1,
                "nonfraud_count": 1,
                "fraud_prevalence": 0.5,
            },
            {
                "name": "test",
                "minimum_step": 355,
                "maximum_step": 743,
                "source_count": 1,
                "target_count": 1,
                "actual_count": 1,
                "fraud_count": 1,
                "nonfraud_count": 0,
                "fraud_prevalence": 1.0,
            },
        ],
        "selected_row_digest": "c" * 64,
        "sample_path": sample_path.name,
        "sample_sha256": sha256_file(sample_path),
        "created_at": "2026-10-31T00:00:00Z",
        "code_version": "test",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("fraud_detection.modeling.features._validate_postgres", lambda *args: None)

    def vector_rows(
        rows: Iterable[Mapping[str, object]], path: Path
    ) -> dict[str, dict[str, float]]:
        del path
        return {
            str(row["transaction_id"]): {
                "nearest_earlier_training_fraud_cosine_distance": 1.0,
                "missing_earlier_training_fraud_reference": 1.0,
            }
            for row in rows
        }

    monkeypatch.setattr("fraud_detection.modeling.features.build_vector_feature_rows", vector_rows)
    output = tmp_path / "features.parquet"
    result = build_feature_artifact(
        source,
        manifest_path,
        tmp_path / "unused-embeddings.parquet",
        "postgresql://unused",
        output,
        expected_train_count=2,
        expected_validation_count=2,
    )

    built = pl.read_parquet(output)
    assert result.row_count == 4
    assert result.test_count == 0
    assert "test-a" not in built["transaction_id"].to_list()
