"""Typed source, manifest, load, and reconciliation contracts."""

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fraud_detection.ingestion.constants import MANIFEST_SCHEMA_VERSION, SPLIT_NAMES

SplitName = Literal["train", "validation", "test"]


class SourceValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_filename: str
    source_sha256: str
    source_byte_size: int = Field(ge=1)
    source_row_count: int = Field(gt=0)
    source_fraud_count: int = Field(ge=0)
    source_fraud_prevalence: float = Field(ge=0, le=1)
    minimum_step: int = Field(ge=0)
    maximum_step: int = Field(ge=0)
    detected_columns: list[str]
    detected_types: dict[str, str]
    validation_timestamp: datetime
    code_version: str


class SplitManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: SplitName
    minimum_step: int = Field(ge=0)
    maximum_step: int = Field(ge=0)
    source_count: int = Field(ge=0)
    target_count: int = Field(ge=0)
    actual_count: int = Field(ge=0)
    fraud_count: int = Field(ge=0)
    nonfraud_count: int = Field(ge=0)
    fraud_prevalence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def counts_reconcile(self) -> Self:
        if self.minimum_step > self.maximum_step:
            raise ValueError("split minimum_step exceeds maximum_step")
        if self.fraud_count + self.nonfraud_count != self.actual_count:
            raise ValueError("split fraud and nonfraud counts do not reconcile")
        return self


class SampleManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest_schema_version: str
    source_filename: str
    source_sha256: str
    source_byte_size: int = Field(ge=1)
    source_row_count: int = Field(gt=0)
    source_fraud_count: int = Field(ge=0)
    source_fraud_prevalence: float = Field(ge=0, le=1)
    source_columns: list[str]
    source_types: dict[str, str]
    source_validation_timestamp: datetime
    sampling_seed: int
    enrichment_seed: int
    target_sample_size: int = Field(gt=0)
    actual_sample_size: int = Field(gt=0)
    split_policy: str
    splits: list[SplitManifest]
    selected_row_digest: str
    sample_path: str
    sample_sha256: str
    created_at: datetime
    code_version: str

    @model_validator(mode="after")
    def manifest_contract(self) -> Self:
        if self.manifest_schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema version: {self.manifest_schema_version}")
        if [split.name for split in self.splits] != list(SPLIT_NAMES):
            raise ValueError("manifest must contain train, validation, and test splits in order")
        if sum(split.actual_count for split in self.splits) != self.actual_sample_size:
            raise ValueError("split counts do not equal actual sample size")
        if sum(split.fraud_count for split in self.splits) != self.source_fraud_count:
            raise ValueError("manifest does not retain every source fraud row")
        for previous, current in zip(self.splits, self.splits[1:], strict=False):
            if previous.maximum_step >= current.minimum_step:
                raise ValueError("manifest split step ranges overlap or are unordered")
        return self

    def split(self, name: SplitName) -> SplitManifest:
        return next(split for split in self.splits if split.name == name)


class InitialLoadResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_id: str
    status: Literal["pending"]
    expected_rows: int
    transaction_count: int
    label_count: int
    account_count: int
    merchant_count: int


class ReconciliationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_id: str
    status: str
    expected_train_validation_rows: int
    loaded_transaction_count: int
    loaded_label_count: int
    distinct_sender_accounts: int
    distinct_receiver_accounts: int
    merchant_account_count: int
    merchant_subtype_count: int
    fraud_count_by_split: dict[str, int]
    source_row_minimum: int | None
    source_row_maximum: int | None
    duplicate_transaction_id_count: int
    duplicate_batch_source_row_count: int
    test_row_count: int
    reconciled: bool
