"""Controlled contract tests for the production MiniLM encoder wrapper."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fraud_detection.persistence.document_vector.encoder import MiniLMEncoder


class FakeSentenceTransformer:
    dimension = 384
    initialization: tuple[str, dict[str, object]] | None = None

    def __init__(self, model_id: str, **kwargs: object) -> None:
        type(self).initialization = (model_id, kwargs)

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        del kwargs
        vector = np.ones(384, dtype=np.float32)
        vector /= np.linalg.norm(vector)
        return np.stack([vector for _ in texts])


def test_production_encoder_uses_pinned_model_cpu_and_normalized_float32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    encoder = MiniLMEncoder(cache_directory=tmp_path / "cache")
    vectors = encoder.encode(["one", "two"], batch_size=2)

    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert FakeSentenceTransformer.initialization is not None
    model_id, options = FakeSentenceTransformer.initialization
    assert model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert options["revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert options["device"] == "cpu"


def test_production_encoder_rejects_wrong_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class WrongDimension(FakeSentenceTransformer):
        dimension = 128

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=WrongDimension),
    )
    with pytest.raises(ValueError, match="does not match required 384"):
        MiniLMEncoder(cache_directory=tmp_path / "cache")
