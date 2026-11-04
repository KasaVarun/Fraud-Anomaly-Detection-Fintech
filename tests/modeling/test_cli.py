"""Phase 7 CLI routing tests."""

from pathlib import Path

import pytest

from fraud_detection.config import Settings
from fraud_detection.modeling import cli


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        postgres_dsn="postgresql://local/test",
        embedding_local_artifact=tmp_path / "embeddings.parquet",
        model_feature_artifact=tmp_path / "features.parquet",
        model_artifact_root=tmp_path / "models",
    )


def test_feature_and_training_cli_routes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(cli, "build_feature_artifact", lambda *args: {"feature": "ok"})
    assert (
        cli.main(
            [
                "features-build",
                "--source",
                str(tmp_path / "source.csv"),
                "--manifest",
                str(tmp_path / "manifest.json"),
            ]
        )
        == 0
    )
    assert '"feature": "ok"' in capsys.readouterr().out

    monkeypatch.setattr(cli, "train_and_release_models", lambda *args: {"training": "ok"})
    assert cli.main(["train-release"]) == 0
    assert '"training": "ok"' in capsys.readouterr().out


def test_cli_sanitizes_value_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: _settings(tmp_path))

    def fail(*_: object) -> object:
        raise ValueError("bounded failure")

    monkeypatch.setattr(cli, "train_and_release_models", fail)
    assert cli.main(["train-release"]) == 2
    assert "bounded failure" in capsys.readouterr().err
