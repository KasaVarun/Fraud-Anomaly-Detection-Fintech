"""CLI exit-code behavior tests."""

from pathlib import Path

from fraud_detection.config import get_settings
from fraud_detection.ingestion.cli import main


def test_validation_command_returns_nonzero_for_missing_source(tmp_path: Path) -> None:
    result = main(["validate-source", "--source", str(tmp_path / "missing.csv")])

    assert result == 2


def test_cosmos_validation_returns_nonzero_when_not_configured() -> None:
    get_settings.cache_clear()
    try:
        result = main(["cosmos-validate"])
    finally:
        get_settings.cache_clear()

    assert result == 4
