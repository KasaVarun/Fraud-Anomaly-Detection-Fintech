"""Small deterministic PaySim-like CSV factory for tests."""

import csv
from pathlib import Path
from typing import Any

from fraud_detection.ingestion.constants import REQUIRED_COLUMNS


def paysim_rows(count: int = 60) -> list[dict[str, Any]]:
    fraud_rows = {3, 17, 44, 58}
    rows: list[dict[str, Any]] = []
    for index in range(count):
        step = index // 10 + 1
        rows.append(
            {
                "step": step,
                "type": "TRANSFER" if index % 3 else "PAYMENT",
                "amount": f"{10 + index / 10:.2f}",
                "nameOrig": f"C{index % 13:04d}",
                "oldbalanceOrg": "1000.00",
                "newbalanceOrig": "990.00",
                "nameDest": f"M{index % 5:04d}" if index % 4 == 0 else f"C{100 + index % 11:04d}",
                "oldbalanceDest": "100.00",
                "newbalanceDest": "110.00",
                "isFraud": 1 if index in fraud_rows else 0,
                "isFlaggedFraud": 1 if index == 44 else 0,
            }
        )
    return rows


def write_paysim_csv(
    path: Path,
    *,
    rows: list[dict[str, Any]] | None = None,
    columns: tuple[str, ...] = REQUIRED_COLUMNS,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows if rows is not None else paysim_rows())
    return path
