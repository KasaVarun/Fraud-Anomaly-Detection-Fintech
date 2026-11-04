"""Locked threshold, snapshot, and metric contract tests."""

from decimal import Decimal

import numpy as np

from fraud_detection.incremental.contracts import LOCKED_THRESHOLD
from fraud_detection.incremental.contracts import TestChunk as ChunkContract
from fraud_detection.incremental.scoring import (
    flag_id,
    input_version,
    locked_metrics,
    score_id,
    snapshot_id,
)


def test_threshold_equality_flags_and_metrics_are_exact() -> None:
    labels = np.asarray([1, 0, 1], dtype=np.int8)
    scores = np.asarray([float(LOCKED_THRESHOLD), 0.9495999, 0.1], dtype=np.float64)
    metrics = locked_metrics(labels, scores)
    assert metrics.predicted_fraud_count == 1
    assert (metrics.tp, metrics.fp, metrics.tn, metrics.fn) == (1, 0, 1, 1)
    assert metrics.flag_rate == 1 / 3


def test_score_flag_input_and_snapshot_identities_are_stable_and_causal() -> None:
    chunk = ChunkContract(
        index=0,
        start_step=355,
        end_step=378,
        row_count=1,
        chunk_id="chunk",
        batch_id="batch",
        row_digest="a" * 64,
    )
    version = input_version("b" * 64, 355)
    score = score_id("transaction", version)
    assert score == score_id("transaction", version)
    assert flag_id(score) == flag_id(score)
    assert snapshot_id(chunk, 355, "c" * 64) == snapshot_id(chunk, 355, "c" * 64)
    assert Decimal(str(LOCKED_THRESHOLD)) == Decimal("0.9496")
