"""Phase 3 ingestion fixtures."""

from pathlib import Path

import pytest
from tests.paysim_factory import write_paysim_csv

from fraud_detection.ingestion.models import SampleManifest
from fraud_detection.ingestion.sampling import generate_sample


@pytest.fixture
def paysim_source(tmp_path: Path) -> Path:
    return write_paysim_csv(tmp_path / "PS_20174392719_1491204439457_log.csv")


@pytest.fixture
def generated_artifacts(
    paysim_source: Path, tmp_path: Path
) -> tuple[Path, Path, Path, SampleManifest]:
    sample_path = tmp_path / "sample.parquet"
    manifest_path = tmp_path / "manifest.json"
    manifest = generate_sample(
        paysim_source,
        sample_path,
        manifest_path,
        target_size=30,
        sampling_seed=123,
        enrichment_seed=456,
    )
    return paysim_source, sample_path, manifest_path, manifest
