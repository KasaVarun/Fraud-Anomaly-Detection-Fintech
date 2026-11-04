"""Phase 8 workflow orchestration tests."""

from pathlib import Path

import pytest

from fraud_detection.config import Settings
from fraud_detection.incremental import workflow
from fraud_detection.incremental.contracts import (
    ChunkPlan,
    ChunkRunResult,
    LockedMetrics,
)
from fraud_detection.incremental.contracts import (
    TestChunk as ChunkContract,
)


def _values() -> tuple[ChunkPlan, ChunkContract, ChunkRunResult]:
    chunk = ChunkContract(
        index=0,
        start_step=355,
        end_step=378,
        row_count=1,
        chunk_id="chunk",
        batch_id="batch",
        row_digest="a" * 64,
    )
    plan = ChunkPlan(
        policy="policy",
        chunk_steps=24,
        source_sha256="b" * 64,
        sample_sha256="c" * 64,
        selected_test_row_digest="d" * 64,
        initial_batch_id="initial",
        test_row_count=1,
        chunks=(chunk,),
    )
    metrics = LockedMetrics(
        row_count=1,
        fraud_count=0,
        nonfraud_count=1,
        tp=0,
        fp=0,
        tn=1,
        fn=0,
        precision=0,
        recall=0,
        f1=0,
        pr_auc=0,
        predicted_fraud_count=0,
        flag_rate=0,
        score_minimum=0.1,
        score_maximum=0.1,
        score_mean=0.1,
    )
    result = ChunkRunResult(
        chunk=chunk,
        batch_status="pending",
        transaction_count=1,
        label_count=1,
        score_count=1,
        flag_count=0,
        snapshot_count=1,
        evaluation_count=1,
        metrics=metrics,
        live_projection=False,
    )
    return plan, chunk, result


def test_next_chunk_and_offline_run_routing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, chunk, result = _values()
    monkeypatch.setattr(workflow, "list_chunk_status", lambda *args: [{"complete": False}])
    assert workflow.next_eligible_chunk("postgresql://unused", plan) == chunk
    monkeypatch.setattr(workflow, "list_chunk_status", lambda *args: [{"complete": True}])
    assert workflow.next_eligible_chunk("postgresql://unused", plan) is None

    monkeypatch.setattr(workflow, "run_offline_chunk", lambda *args: result)
    settings = Settings(
        _env_file=None,
        incremental_data_root=tmp_path,
        model_artifact_root=tmp_path,
    )
    actual = workflow.run_chunk(
        tmp_path / "source",
        tmp_path / "manifest",
        "postgresql://unused",
        settings,
        plan,
        chunk,
    )
    assert actual == result


def test_default_plan_passes_configured_chunk_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, _, _ = _values()
    captured: dict[str, object] = {}

    def fake_plan(*args: object, **kwargs: object) -> ChunkPlan:
        captured.update(kwargs)
        return plan

    monkeypatch.setattr(workflow, "plan_test_chunks", fake_plan)
    settings = Settings(_env_file=None, incremental_chunk_steps=48)
    assert workflow.default_plan(tmp_path / "s", tmp_path / "m", settings) == plan
    assert captured["chunk_steps"] == 48
