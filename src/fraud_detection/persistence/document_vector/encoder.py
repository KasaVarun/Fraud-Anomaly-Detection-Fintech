"""Real MiniLM encoder and testable encoder protocol."""

from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from fraud_detection.persistence.document_vector.embedding_models import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
)


class BehavioralEncoder(Protocol):
    model_id: str
    model_revision: str
    sentence_transformers_version: str
    dimension: int
    normalized: bool
    dtype: str

    def encode(self, texts: Sequence[str], *, batch_size: int) -> NDArray[np.float32]:
        """Encode a bounded batch into normalized float32 vectors."""


class MiniLMEncoder:
    """CPU-compatible all-MiniLM-L6-v2 encoder pinned to one model revision."""

    model_id = EMBEDDING_MODEL_ID
    model_revision = EMBEDDING_MODEL_REVISION
    sentence_transformers_version = version("sentence-transformers")
    dimension = EMBEDDING_DIMENSION
    normalized = True
    dtype = "float32"

    def __init__(self, *, cache_directory: Path) -> None:
        from sentence_transformers import SentenceTransformer

        cache_directory.mkdir(parents=True, exist_ok=True)
        self._model = SentenceTransformer(
            self.model_id,
            revision=self.model_revision,
            cache_folder=str(cache_directory),
            device="cpu",
        )
        actual_dimension = self._model.get_embedding_dimension()
        if actual_dimension != self.dimension:
            raise ValueError(
                f"MiniLM dimension {actual_dimension} does not match required {self.dimension}"
            )

    def encode(self, texts: Sequence[str], *, batch_size: int) -> NDArray[np.float32]:
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            precision="float32",
        )
        array = np.asarray(vectors, dtype=np.float32)
        if array.shape != (len(texts), self.dimension):
            raise ValueError(f"MiniLM returned unexpected shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("MiniLM returned non-finite values")
        return array
