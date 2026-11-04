"""Versioned behavioral embedding document and similarity contracts."""

import hashlib
import math
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fraud_detection.features.behavioral import FORBIDDEN_TEMPLATE_TOKENS, template_digest
from fraud_detection.ingestion.hashing import canonical_json_bytes

EMBEDDING_DOCUMENT_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
EMBEDDING_DIMENSION: Literal[384] = 384
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def embedding_content_digest(payload: dict[str, Any]) -> str:
    immutable = {
        key: value
        for key, value in payload.items()
        if key not in {"content_digest", "projected_at"}
    }
    return hashlib.sha256(canonical_json_bytes(immutable)).hexdigest()


class EmbeddingDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    sender_id: str
    transaction_id: str
    raw_id: str
    raw_pk: str
    batch_id: str
    step: int = Field(ge=0)
    data_split: Literal["train", "validation", "test"]
    template_text: str
    template_version: str
    amount_band_version: str
    template_digest: str
    model_id: str
    model_revision: str
    sentence_transformers_version: str
    embedding_version: str
    vector: list[float]
    dimension: Literal[384] = EMBEDDING_DIMENSION
    normalized: Literal[True] = True
    dtype: Literal["float32"] = "float32"
    similarity_metric: Literal["cosine"] = "cosine"
    source_version: str
    reference_eligible: bool
    reference_eligibility_provenance: str = "protected-training-label-v1"
    document_schema_version: Literal["1.0.0"] = EMBEDDING_DOCUMENT_SCHEMA_VERSION
    content_digest: str
    projected_at: datetime

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if len(self.vector) != EMBEDDING_DIMENSION:
            raise ValueError("embedding vector must contain exactly 384 values")
        if not all(math.isfinite(value) for value in self.vector):
            raise ValueError("embedding vector contains non-finite values")
        norm = math.sqrt(sum(value * value for value in self.vector))
        if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError("normalized embedding vector must have unit length")
        lowered = self.template_text.lower()
        if any(token in lowered for token in FORBIDDEN_TEMPLATE_TOKENS):
            raise ValueError("embedding template contains forbidden fields")
        if (
            self.sender_id in self.template_text
            or self.transaction_id in self.template_text
            or self.raw_id in self.template_text
        ):
            raise ValueError("embedding template contains a protected identifier")
        if self.reference_eligible and self.data_split != "train":
            raise ValueError("only training documents may be similarity references")
        if self.template_digest != template_digest(
            self.template_text,
            self.template_version,
            self.amount_band_version,
        ):
            raise ValueError("embedding template digest is invalid")
        if self.content_digest != embedding_content_digest(self.model_dump(mode="json")):
            raise ValueError("embedding content digest is invalid")
        return self

    def immutable_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"content_digest", "projected_at"})

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SimilarityMatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    transaction_id: str
    cosine_distance: float = Field(ge=0)
    step: int
    embedding_version: str
    template_version: str
    provenance: str


class SimilarityEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_transaction_id: str
    nearest_fraud_distance: float | None
    nearest_fraud_missing: bool
    matches: list[SimilarityMatch]
    metric: Literal["cosine"] = "cosine"
