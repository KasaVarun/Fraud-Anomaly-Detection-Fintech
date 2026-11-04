"""PaySim source-file schema and content validation."""

import csv
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from fraud_detection import __version__
from fraud_detection.ingestion.constants import REQUIRED_COLUMNS
from fraud_detection.ingestion.errors import SourceValidationError
from fraud_detection.ingestion.hashing import sha256_file
from fraud_detection.ingestion.models import SourceValidationReport

SOURCE_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "step": pl.Int64,
    "type": pl.String,
    "amount": pl.Decimal(38, 2),
    "nameOrig": pl.String,
    "oldbalanceOrg": pl.Decimal(38, 2),
    "newbalanceOrig": pl.Decimal(38, 2),
    "nameDest": pl.String,
    "oldbalanceDest": pl.Decimal(38, 2),
    "newbalanceDest": pl.Decimal(38, 2),
    "isFraud": pl.Int8,
    "isFlaggedFraud": pl.Int8,
}


def source_lazy_frame(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        schema_overrides=SOURCE_SCHEMA,
        row_index_name="source_row",
        row_index_offset=0,
        null_values=[""],
        infer_schema_length=10_000,
        raise_if_empty=True,
    )


def validate_source(path: Path) -> SourceValidationReport:
    if not path.is_file():
        raise SourceValidationError(f"PaySim source file not found: {path.name}")
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            header = next(csv.reader(stream))
    except (OSError, csv.Error, StopIteration) as exc:
        raise SourceValidationError("unable to read PaySim CSV header") from exc
    missing = sorted(set(REQUIRED_COLUMNS) - set(header))
    if missing:
        raise SourceValidationError(f"missing required PaySim columns: {', '.join(missing)}")

    try:
        frame = source_lazy_frame(path)
        invalid = (
            frame.select(
                pl.len().alias("row_count"),
                (pl.col("step").is_null() | (pl.col("step") < 0)).sum().alias("negative_steps"),
                (pl.col("amount").is_null() | (pl.col("amount") < 0))
                .sum()
                .alias("negative_amounts"),
                (pl.col("isFraud").is_null() | ~pl.col("isFraud").is_in([0, 1]))
                .sum()
                .alias("invalid_fraud"),
                (pl.col("isFlaggedFraud").is_null() | ~pl.col("isFlaggedFraud").is_in([0, 1]))
                .sum()
                .alias("invalid_source_flag"),
                (pl.col("nameOrig").is_null() | (pl.col("nameOrig").str.strip_chars() == ""))
                .sum()
                .alias("missing_senders"),
                (pl.col("nameDest").is_null() | (pl.col("nameDest").str.strip_chars() == ""))
                .sum()
                .alias("missing_receivers"),
                pl.col("isFraud").sum().alias("fraud_count"),
                pl.col("step").min().alias("minimum_step"),
                pl.col("step").max().alias("maximum_step"),
            )
            .collect(engine="streaming")
            .row(0, named=True)
        )
        schema = frame.collect_schema()
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise SourceValidationError("PaySim row parsing failed") from exc

    row_count = int(invalid["row_count"])
    if row_count == 0:
        raise SourceValidationError("PaySim source contains no data rows")
    checks = {
        "negative step": invalid["negative_steps"],
        "negative amount": invalid["negative_amounts"],
        "non-binary isFraud": invalid["invalid_fraud"],
        "non-binary isFlaggedFraud": invalid["invalid_source_flag"],
        "missing sender identifier": invalid["missing_senders"],
        "missing receiver identifier": invalid["missing_receivers"],
    }
    failures = [f"{label} rows={count}" for label, count in checks.items() if int(count) > 0]
    if failures:
        raise SourceValidationError("; ".join(failures))

    fraud_count = int(invalid["fraud_count"])
    return SourceValidationReport(
        source_filename=path.name,
        source_sha256=sha256_file(path),
        source_byte_size=path.stat().st_size,
        source_row_count=row_count,
        source_fraud_count=fraud_count,
        source_fraud_prevalence=fraud_count / row_count,
        minimum_step=int(invalid["minimum_step"]),
        maximum_step=int(invalid["maximum_step"]),
        detected_columns=header,
        detected_types={name: str(schema[name]) for name in header},
        validation_timestamp=datetime.now(UTC),
        code_version=__version__,
    )
