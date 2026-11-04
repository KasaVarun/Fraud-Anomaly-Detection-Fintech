"""Typed Phase 9 investigator and dashboard response contracts."""

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EvidenceStatus = Literal[
    "available",
    "not_configured",
    "unavailable",
    "not_found",
    "version_mismatch",
]


class GraphTransactionEdgeData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    edge_id: str
    label: Literal["PAYS", "TRANSFER_TO", "USES_DEVICE", "USES_IP"]
    kind: Literal["money_flow", "device_observation", "ip_observation"]
    out_id: str
    in_id: str
    transaction_id: str
    step: int
    amount: str | None
    synthetic: bool
    source_version: str
    graph_schema_version: str
    projection_version: str


class RelatedAccountEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    related_account_id: str = Field(min_length=1, max_length=256)
    relationship_type: Literal["all", "money_flow", "shared_device", "shared_ip"]
    transaction_count: int = Field(ge=0)
    shared_device_count: int = Field(ge=0)
    shared_ip_count: int = Field(ge=0)
    evidence_weight: float = Field(ge=0)
    snapshot_id: str = Field(min_length=1)
    cutoff_step: int = Field(ge=0)
    algorithm_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_weight(self) -> "RelatedAccountEvidence":
        expected = (
            self.transaction_count + 0.5 * self.shared_device_count + 0.25 * self.shared_ip_count
        )
        if not math.isfinite(self.evidence_weight) or not math.isclose(
            self.evidence_weight, expected, abs_tol=1e-12
        ):
            raise ValueError("related-account evidence weight is invalid")
        return self


class SimilarityMatchData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    transaction_id: str = Field(min_length=1, max_length=256)
    step: int = Field(ge=0)
    cosine_distance: float = Field(ge=0, le=2)
    embedding_version: str
    template_version: str
    model_revision: str
    source_version: str | None = None

    @model_validator(mode="after")
    def validate_distance(self) -> "SimilarityMatchData":
        if not math.isfinite(self.cosine_distance):
            raise ValueError("cosine distance is not finite")
        return self


class EvidenceSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: EvidenceStatus
    data: dict[str, Any] | list[dict[str, Any]] | None = None
    version: str | None = None
    warning: str | None = None


class Provenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_id: str | None = None
    model_version_id: str | None = None
    input_version: str | None = None
    snapshot_id: str | None = None
    source_version: str | None = None


class Freshness(BaseModel):
    model_config = ConfigDict(frozen=True)

    latest_published_batch: str | None = None
    published_cutoff_step: int | None = None
    pending_batch_count: int = 0
    complete: bool = False


class TransactionInvestigation(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["transaction"] = "transaction"
    transaction: dict[str, Any]
    participants: dict[str, Any]
    prediction: dict[str, Any] | None
    fraud_flag: dict[str, Any] | None
    graph_snapshot: dict[str, Any] | None
    raw_evidence: EvidenceSection
    graph_evidence: EvidenceSection
    vector_evidence: EvidenceSection
    disclaimer: str = (
        "A fraud flag and graph evidence are investigation signals, not proof of fraud."
    )


class RelatedAccountsData(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["related_accounts"] = "related_accounts"
    account_id: str
    relationship_type: str
    snapshot: dict[str, Any] | None
    relationships: EvidenceSection
    disclaimer: str = (
        "Shared infrastructure and communities are evidence, not proof of coordination."
    )


class SimilarBehaviorData(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["similar_behavior"] = "similar_behavior"
    transaction_id: str
    matches: EvidenceSection
    metric: Literal["cosine_distance"] = "cosine_distance"
    disclaimer: str = "Cosine distance is similarity evidence, not fraud probability."


class DashboardSummaryData(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["dashboard_summary"] = "dashboard_summary"
    transaction_count: int
    scored_transaction_count: int
    fraud_flag_count: int
    flag_rate: float
    amount_total: str
    score_minimum: float | None
    score_maximum: float | None
    score_mean: float | None
    transaction_type_counts: dict[str, int]
    active_model_id: str


class FlagItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    transaction_id: str
    step: int
    transaction_type: str
    amount: str
    sender_id: str
    receiver_id: str
    score: float
    threshold: float
    model_version_id: str
    flag_reason: str
    review_status: str
    snapshot_id: str | None
    scored_at: datetime
    batch_id: str


class FlagPageData(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["fraud_flags"] = "fraud_flags"
    items: list[FlagItem]


class FreshnessData(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["freshness"] = "freshness"
    latest_published_batch: str | None
    published_cutoff_step: int | None
    pending_batch_count: int
    active_model_version: str
    model_training_cutoff: int
    versions: dict[str, str]
    dependencies: dict[str, EvidenceStatus]
    evidence_complete: bool


ResponseData = (
    TransactionInvestigation
    | RelatedAccountsData
    | SimilarBehaviorData
    | DashboardSummaryData
    | FlagPageData
    | FreshnessData
)


class EvidenceEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: ResponseData | None
    provenance: Provenance
    freshness: Freshness
    partial_evidence: bool
    evidence_status: EvidenceStatus
    warnings: list[str] = Field(default_factory=list)
    next_cursor: str | None = None
