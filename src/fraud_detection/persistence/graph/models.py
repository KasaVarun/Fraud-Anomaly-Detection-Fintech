"""Versioned graph evidence contracts and deterministic identifiers."""

import hashlib
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fraud_detection.ingestion.hashing import canonical_json_bytes

GRAPH_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
GRAPH_PROJECTION_VERSION: Literal["gremlin-evidence-v1"] = "gremlin-evidence-v1"
PARTITION_ALGORITHM_VERSION = "sha256-shard-v1"
ENRICHMENT_ALGORITHM_VERSION = "synthetic-artifact-v1"
DEVICE_POOL_SIZE = 32_768
IP_POOL_SIZE = 16_384

MONEY_EDGE_BY_TRANSACTION_TYPE: dict[str, Literal["PAYS", "TRANSFER_TO"]] = {
    "CASH_IN": "TRANSFER_TO",
    "CASH_OUT": "TRANSFER_TO",
    "DEBIT": "TRANSFER_TO",
    "PAYMENT": "PAYS",
    "TRANSFER": "TRANSFER_TO",
}

FORBIDDEN_GRAPH_PROPERTIES = frozenset(
    {
        "isFraud",
        "isFlaggedFraud",
        "is_fraud",
        "source_flag",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "fraud_probability",
        "model_version_id",
    }
)


def graph_digest(payload: dict[str, Any]) -> str:
    immutable = {key: value for key, value in payload.items() if key != "content_digest"}
    return hashlib.sha256(canonical_json_bytes(immutable)).hexdigest()


def partition_key(vertex_id: str, shard_count: int) -> str:
    material = f"{PARTITION_ALGORITHM_VERSION}:{vertex_id}".encode()
    shard = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % shard_count
    return f"shard-{shard:04d}"


def deterministic_artifact_id(
    kind: Literal["device", "ip"],
    *,
    transaction_id: str,
    source_row: int,
    step: int,
    seed: int,
) -> str:
    pool_size = DEVICE_POOL_SIZE if kind == "device" else IP_POOL_SIZE
    material = (
        f"{ENRICHMENT_ALGORITHM_VERSION}:{kind}:{seed}:{transaction_id}:{source_row}:{step}"
    ).encode()
    bucket = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % pool_size
    return f"{kind}:{ENRICHMENT_ALGORITHM_VERSION}:{bucket:05d}"


def money_edge_label(transaction_type: str) -> Literal["PAYS", "TRANSFER_TO"]:
    try:
        return MONEY_EDGE_BY_TRANSACTION_TYPE[transaction_type]
    except KeyError as exc:
        raise ValueError(f"unsupported PaySim transaction type: {transaction_type}") from exc


class GraphVertex(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    pk: str
    label: Literal["account", "merchant", "device", "ip"]
    kind: Literal["customer", "merchant", "device", "ip"]
    account_id: str | None = None
    synthetic: bool
    enrichment_seed: int | None = None
    enrichment_algorithm_version: str | None = None
    graph_schema_version: Literal["1.0.0"] = GRAPH_SCHEMA_VERSION
    projection_version: Literal["gremlin-evidence-v1"] = GRAPH_PROJECTION_VERSION
    partition_algorithm_version: str = PARTITION_ALGORITHM_VERSION
    content_digest: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.content_digest != graph_digest(self.model_dump(mode="json")):
            raise ValueError("graph vertex digest is invalid")
        if self.label in {"account", "merchant"} and self.account_id != self.id:
            raise ValueError("account vertex must preserve the PostgreSQL account ID")
        return self

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class GraphEdge(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    label: Literal["PAYS", "TRANSFER_TO", "USES_DEVICE", "USES_IP"]
    out_id: str
    out_pk: str
    in_id: str
    in_pk: str
    transaction_id: str
    step: int = Field(ge=0)
    batch_id: str
    data_split: Literal["train", "validation", "test"]
    transaction_type: str
    amount: str | None = None
    kind: Literal["money_flow", "device_observation", "ip_observation"]
    synthetic: bool
    source_version: str
    provenance: str
    enrichment_seed: int | None = None
    enrichment_algorithm_version: str | None = None
    graph_schema_version: Literal["1.0.0"] = GRAPH_SCHEMA_VERSION
    projection_version: Literal["gremlin-evidence-v1"] = GRAPH_PROJECTION_VERSION
    content_digest: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.content_digest != graph_digest(self.model_dump(mode="json")):
            raise ValueError("graph edge digest is invalid")
        forbidden = FORBIDDEN_GRAPH_PROPERTIES & set(self.model_fields_set)
        if forbidden:
            raise ValueError("graph edge contains forbidden predictive properties")
        return self

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class GraphEvidenceBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    vertices: tuple[GraphVertex, ...]
    edges: tuple[GraphEdge, ...]
