"""Optional Kaggle CLI acquisition without credential handling."""

import shutil
import subprocess
from pathlib import Path

from fraud_detection.ingestion.constants import KAGGLE_DATASET, PAYSIM_FILENAME
from fraud_detection.ingestion.errors import IngestionError


def acquire_paysim(output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    expected = output_directory / PAYSIM_FILENAME
    if expected.is_file():
        return expected
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise IngestionError(
            "Kaggle CLI is not installed; install and authenticate it without committing tokens"
        )
    try:
        subprocess.run(
            [
                kaggle,
                "datasets",
                "download",
                "--dataset",
                KAGGLE_DATASET,
                "--path",
                str(output_directory),
                "--unzip",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise IngestionError("Kaggle PaySim download failed; check local authentication") from exc
    if not expected.is_file():
        raise IngestionError(f"Kaggle archive did not contain expected file {PAYSIM_FILENAME}")
    return expected
