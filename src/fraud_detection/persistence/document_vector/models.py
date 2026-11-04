"""Versioned ``transactions_raw`` document contract."""

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fraud_detection.ingestion.hashing import canonical_json_bytes

RAW_DOCUMENT_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
RAW_PROJECTION_VERSION: Literal["raw-paysim-v1"] = "raw-paysim-v1"
RAW_PARTITION_KEY_PATH = "/sender_id"

FORBIDDEN_RAW_PREDICTIVE_FIELDS = (
    "id",
    "sender_id",
    "receiver_id",
    "forensic_balances.oldbalanceOrg",
    "forensic_balances.newbalanceOrig",
    "forensic_balances.oldbalanceDest",
    "forensic_balances.newbalanceDest",
    "source_truth.is_fraud",
    "source_truth.source_flag",
)


class ForensicBalances(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    oldbalanceOrg: str | None
    newbalanceOrig: str | None
    oldbalanceDest: str | None
    newbalanceDest: str | None
    predictive_use: Literal["forbidden"] = "forbidden"


class ProtectedSourceTruth(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    is_fraud: bool
    source_flag: bool
    provenance: str
    predictive_use: Literal["forbidden"] = "forbidden"


class RawSourceProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_filename: str
    source_sha256: str
    sampling_seed: int
    enrichment_seed: int


class RawTransactionDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    sender_id: str
    receiver_id: str
    batch_id: str
    source_row: int = Field(ge=0)
    step: int = Field(ge=0)
    data_split: Literal["train", "validation", "test"]
    transaction_type: str
    amount: str
    forensic_balances: ForensicBalances
    source_truth: ProtectedSourceTruth
    source_provenance: RawSourceProvenance
    forbidden_predictive_fields: tuple[str, ...] = FORBIDDEN_RAW_PREDICTIVE_FIELDS
    document_schema_version: Literal["1.0.0"] = RAW_DOCUMENT_SCHEMA_VERSION
    projection_version: Literal["raw-paysim-v1"] = RAW_PROJECTION_VERSION
    content_digest: str
    projected_at: datetime

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        if self.content_digest != self.compute_content_digest():
            raise ValueError("raw document content digest is invalid")
        return self

    def immutable_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            exclude={"content_digest", "projected_at"},
        )

    def compute_content_digest(self) -> str:
        import hashlib

        return hashlib.sha256(canonical_json_bytes(self.immutable_payload())).hexdigest()

    def cosmos_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def content_digest(payload: dict[str, Any]) -> str:
    import hashlib

    immutable = {
        key: value
        for key, value in payload.items()
        if key not in {"content_digest", "projected_at"}
    }
    return hashlib.sha256(canonical_json_bytes(immutable)).hexdigest()
