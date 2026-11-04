"""PaySim source validation tests."""

from pathlib import Path

import pytest
from tests.paysim_factory import paysim_rows, write_paysim_csv

from fraud_detection.ingestion.constants import REQUIRED_COLUMNS
from fraud_detection.ingestion.errors import SourceValidationError
from fraud_detection.ingestion.source import validate_source


def test_required_source_columns_and_statistics_are_validated(paysim_source: Path) -> None:
    report = validate_source(paysim_source)

    assert report.source_row_count == 60
    assert report.source_fraud_count == 4
    assert report.minimum_step == 1
    assert report.maximum_step == 6
    assert set(REQUIRED_COLUMNS) <= set(report.detected_columns)
    assert len(report.source_sha256) == 64


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    columns = tuple(column for column in REQUIRED_COLUMNS if column != "nameDest")
    source = write_paysim_csv(tmp_path / "missing.csv", columns=columns)

    with pytest.raises(SourceValidationError, match="missing required PaySim columns: nameDest"):
        validate_source(source)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("step", "not-an-int", "row parsing failed"),
        ("amount", "not-a-decimal", "row parsing failed"),
        ("step", -1, "negative step"),
        ("amount", "-0.01", "negative amount"),
        ("isFraud", 2, "non-binary isFraud"),
        ("isFlaggedFraud", 2, "non-binary isFlaggedFraud"),
        ("nameOrig", "", "missing sender identifier"),
        ("nameDest", "", "missing receiver identifier"),
    ],
)
def test_invalid_source_values_are_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    rows = paysim_rows()
    rows[0][field] = value
    source = write_paysim_csv(tmp_path / f"invalid-{field}.csv", rows=rows)

    with pytest.raises(SourceValidationError, match=message):
        validate_source(source)


def test_empty_source_is_rejected(tmp_path: Path) -> None:
    source = write_paysim_csv(tmp_path / "empty.csv", rows=[])

    with pytest.raises(SourceValidationError, match=r"row parsing failed|no data rows"):
        validate_source(source)
