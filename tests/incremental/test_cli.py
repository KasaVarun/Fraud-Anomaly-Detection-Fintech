"""Phase 8 operational CLI routing tests."""

from argparse import Namespace
from pathlib import Path

import pytest

from fraud_detection.config import Settings
from fraud_detection.incremental import cli
from fraud_detection.incremental.contracts import ChunkPlan
from fraud_detection.incremental.contracts import TestChunk as ChunkContract


def _plan() -> ChunkPlan:
    chunk = ChunkContract(
        index=0,
        start_step=355,
        end_step=378,
        row_count=1,
        chunk_id="chunk",
        batch_id="batch",
        row_digest="a" * 64,
    )
    return ChunkPlan(
        policy="policy",
        chunk_steps=24,
        source_sha256="b" * 64,
        sample_sha256="c" * 64,
        selected_test_row_digest="d" * 64,
        initial_batch_id="initial",
        test_row_count=1,
        chunks=(chunk,),
    )


def test_incremental_cli_routes_every_operation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(
        _env_file=None,
        postgres_dsn="postgresql://local/test",
        incremental_data_root=tmp_path / "data",
        incremental_artifact_root=tmp_path / "artifacts",
    )
    plan = _plan()
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "default_plan", lambda *args: plan)
    monkeypatch.setattr(cli, "list_chunk_status", lambda *args: [{"complete": False}])
    monkeypatch.setattr(cli, "prepare_offline_evidence", lambda *args: {"prepared": True})
    monkeypatch.setattr(cli, "run_chunk", lambda *args: {"ran": True})
    monkeypatch.setattr(cli, "next_eligible_chunk", lambda *args: plan.chunks[0])
    monkeypatch.setattr(cli, "reconcile_chunk", lambda *args: {"reconciled": True})
    monkeypatch.setattr(cli, "evaluate_pooled_locked_test", lambda *args: {"evaluated": True})
    monkeypatch.setattr(cli, "attempt_publication", lambda *args: {"published": False})
    common = {"source": tmp_path / "source", "manifest": tmp_path / "manifest"}

    assert cli._run(Namespace(command="plan", **common)) == plan
    assert cli._run(Namespace(command="list", **common)) == [{"complete": False}]
    assert cli._run(Namespace(command="prepare-offline", **common)) == {"prepared": True}
    assert cli._run(Namespace(command="run-chunk", chunk_index=0, **common)) == {"ran": True}
    assert cli._run(Namespace(command="resume", chunk_index=0, **common)) == {"ran": True}
    assert cli._run(Namespace(command="run-next", **common)) == {"ran": True}
    assert cli._run(Namespace(command="reconcile", chunk_index=0, **common)) == {"reconciled": True}
    assert cli._run(Namespace(command="evaluate", **common)) == {"evaluated": True}
    assert cli._run(Namespace(command="publish", chunk_index=0, **common)) == {"published": False}


def test_run_next_reports_complete_and_main_sanitizes_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = Settings(_env_file=None, postgres_dsn="postgresql://local/test")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "default_plan", lambda *args: _plan())
    monkeypatch.setattr(cli, "next_eligible_chunk", lambda *args: None)
    result = cli._run(Namespace(command="run-next", source=tmp_path / "s", manifest=tmp_path / "m"))
    assert result["status"] == "complete"

    def fail(*_: object) -> object:
        raise ValueError("safe failure")

    monkeypatch.setattr(cli, "_run", fail)
    assert cli.main(["plan", "--source", "s", "--manifest", "m"]) == 2
    assert "safe failure" in capsys.readouterr().err
