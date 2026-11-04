"""PaySim ingestion constants and versioned contracts."""

from typing import Final

PAYSIM_FILENAME: Final = "PS_20174392719_1491204439457_log.csv"
KAGGLE_DATASET: Final = "ealaxi/paysim1"
MANIFEST_SCHEMA_VERSION: Final = "1.0.0"
SPLIT_POLICY: Final = "whole-step-60-20-20-v1"
PROVENANCE: Final = "PaySim:ealaxi/paysim1"
DEFAULT_TARGET_SIZE: Final = 200_000
DEFAULT_SAMPLING_SEED: Final = 73_750
DEFAULT_ENRICHMENT_SEED: Final = 73_751
DEFAULT_BATCH_SIZE: Final = 5_000

REQUIRED_COLUMNS: Final = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)

SPLIT_NAMES: Final = ("train", "validation", "test")
SPLIT_WEIGHTS: Final = {"train": 0.6, "validation": 0.2, "test": 0.2}
