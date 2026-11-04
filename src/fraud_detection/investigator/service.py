"""Published-only investigator orchestration with explicit partial evidence."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from fraud_detection.incremental.contracts import (
    LOCKED_FEATURE_VERSION,
    LOCKED_MODEL_ID,
)
from fraud_detection.persistence.document_vector.embedding_models import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    EmbeddingDocument,
)
from fraud_detection.persistence.document_vector.models import (
    RAW_DOCUMENT_SCHEMA_VERSION,
    RAW_PROJECTION_VERSION,
    RawTransactionDocument,
)
from fraud_detection.persistence.graph.models import (
    ENRICHMENT_ALGORITHM_VERSION,
    GRAPH_PROJECTION_VERSION,
    GRAPH_SCHEMA_VERSION,
    GraphEdge,
    deterministic_artifact_id,
    money_edge_label,
    partition_key,
)
from fraud_detection.persistence.graph.projection import build_graph_evidence
from fraud_detection.reconciliation.service import ReadinessService

from .contracts import (
    DashboardSummaryData,
    EvidenceEnvelope,
    EvidenceSection,
    FlagItem,
    FlagPageData,
    Freshness,
    FreshnessData,
    GraphTransactionEdgeData,
    Provenance,
    RelatedAccountEvidence,
    RelatedAccountsData,
    SimilarBehaviorData,
    SimilarityMatchData,
    TransactionInvestigation,
)
from .cursor import decode_cursor, encode_cursor
from .ports import (
    GraphInvestigatorPort,
    InvestigatorRelationalPort,
    RawEvidencePort,
    VectorEvidencePort,
)


class PublishedEvidenceNotFound(Exception):
    """Unknown and unpublished records share this safe outcome."""


class RelationalEvidenceUnavailable(Exception):
    """The authoritative relational entry point is unavailable."""


PROTECTED_EVIDENCE_FIELDS = frozenset(
    {
        "is_fraud",
        "isFraud",
        "source_flag",
        "isFlaggedFraud",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "fraud_probability",
        "model_version_id",
    }
)


def _contains_protected_fields(value: Any) -> bool:
    if isinstance(value, Mapping):
        if PROTECTED_EVIDENCE_FIELDS & set(value):
            return True
        return any(_contains_protected_fields(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_protected_fields(item) for item in value)
    return False


def _contract_payload(value: Mapping[str, Any], fields: set[str]) -> dict[str, Any]:
    unexpected = {key for key in value if key not in fields and not str(key).startswith("_")}
    if unexpected:
        raise ValueError("evidence contains unexpected fields")
    return {key: item for key, item in value.items() if key in fields}


def _flatten_graph_edge(value: Mapping[str, Any]) -> dict[str, Any]:
    if _contains_protected_fields(value):
        raise ValueError("graph evidence contains protected fields")
    if "properties" not in value:
        return dict(value)
    allowed_top = {"id", "label", "out_id", "out_pk", "in_id", "in_pk", "properties"}
    if set(value) - allowed_top:
        raise ValueError("graph evidence contains unexpected fields")
    properties = value["properties"]
    if not isinstance(properties, Mapping):
        raise ValueError("graph properties are malformed")
    flattened: dict[str, Any] = {
        key: value[key] for key in ("id", "label", "out_id", "out_pk", "in_id", "in_pk")
    }
    for key, item in properties.items():
        if isinstance(item, list) and len(item) == 1:
            flattened[str(key)] = item[0]
        else:
            flattened[str(key)] = item
    return flattened


def _public_row(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    return value


class InvestigatorService:
    def __init__(
        self,
        relational: InvestigatorRelationalPort,
        readiness: ReadinessService,
        *,
        raw: RawEvidencePort | None = None,
        graph: GraphInvestigatorPort | None = None,
        vectors: VectorEvidencePort | None = None,
        graph_shard_count: int = 64,
    ) -> None:
        self._relational = relational
        self._readiness = readiness
        self._raw = raw
        self._graph = graph
        self._vectors = vectors
        self._graph_shard_count = graph_shard_count

    def _relational_call(self, operation: Any) -> Any:
        try:
            return operation()
        except (SQLAlchemyError, RuntimeError) as exc:
            raise RelationalEvidenceUnavailable from exc

    def _freshness(self) -> Freshness:
        value = self._relational_call(lambda: self._relational.freshness(LOCKED_MODEL_ID))
        return Freshness(
            latest_published_batch=value.get("latest_published_batch"),
            published_cutoff_step=value.get("published_cutoff_step"),
            pending_batch_count=int(value.get("pending_batch_count") or 0),
            complete=value.get("latest_published_batch") is not None,
        )

    def _transaction_row(self, transaction_id: str) -> dict[str, Any]:
        value = self._relational_call(
            lambda: self._relational.transaction_evidence(transaction_id, LOCKED_MODEL_ID)
        )
        if value is None:
            raise PublishedEvidenceNotFound
        return dict(value)

    def _raw_evidence(
        self, row: Mapping[str, Any]
    ) -> tuple[EvidenceSection, RawTransactionDocument | None]:
        if self._raw is None:
            return (
                EvidenceSection(
                    status="not_configured",
                    warning="Raw document evidence is not configured.",
                ),
                None,
            )
        try:
            value = self._raw.read_document(str(row["transaction_id"]), str(row["sender_id"]))
            if value is None:
                return (
                    EvidenceSection(
                        status="not_found", warning="Raw document evidence is missing."
                    ),
                    None,
                )
            payload = _contract_payload(dict(value), set(RawTransactionDocument.model_fields))
            document = RawTransactionDocument.model_validate(payload)
        except Exception:
            return (
                EvidenceSection(
                    status="unavailable", warning="Raw document evidence is unavailable."
                ),
                None,
            )
        matches = all(
            [
                document.id == row["transaction_id"],
                document.batch_id == row["batch_id"],
                document.source_row == int(row["source_row"]),
                document.sender_id == row["sender_id"],
                document.receiver_id == row["receiver_id"],
                document.step == int(row["step"]),
                document.data_split == row["data_split"],
                document.transaction_type == row["transaction_type"],
                Decimal(document.amount) == Decimal(str(row["amount"])),
                document.source_provenance.source_sha256 == row["batch_source_sha256"],
                document.source_provenance.sampling_seed == int(row["batch_sample_seed"]),
                document.source_provenance.enrichment_seed == int(row["batch_enrichment_seed"]),
                document.document_schema_version == RAW_DOCUMENT_SCHEMA_VERSION,
                document.projection_version == RAW_PROJECTION_VERSION,
            ]
        )
        if not matches:
            return (
                EvidenceSection(
                    status="version_mismatch",
                    warning="Raw document lineage conflicts with the published transaction.",
                ),
                None,
            )
        return (
            EvidenceSection(
                status="available",
                version=document.projection_version,
                data={
                    "transaction_id": document.id,
                    "batch_id": document.batch_id,
                    "source_row": document.source_row,
                    "sender_id": document.sender_id,
                    "receiver_id": document.receiver_id,
                    "step": document.step,
                    "data_split": document.data_split,
                    "transaction_type": document.transaction_type,
                    "amount": document.amount,
                    "source_filename": document.source_provenance.source_filename,
                    "source_sha256": document.source_provenance.source_sha256,
                    "document_schema_version": document.document_schema_version,
                    "projection_version": document.projection_version,
                },
            ),
            document,
        )

    def _embedding_evidence(
        self, row: Mapping[str, Any]
    ) -> tuple[EvidenceSection, EmbeddingDocument | None]:
        if self._vectors is None:
            return (
                EvidenceSection(
                    status="not_configured", warning="Embedding evidence is not configured."
                ),
                None,
            )
        expected_version = row.get("embedding_version")
        if not expected_version:
            return (
                EvidenceSection(
                    status="not_found",
                    warning="No active embedding lineage exists for this transaction.",
                ),
                None,
            )
        expected_id = f"embedding:{expected_version}:{row['transaction_id']}"
        try:
            value = self._vectors.read_embedding(expected_id, str(row["sender_id"]))
            if value is None:
                return (
                    EvidenceSection(status="not_found", warning="Embedding evidence is missing."),
                    None,
                )
            payload = _contract_payload(dict(value), set(EmbeddingDocument.model_fields))
            document = EmbeddingDocument.model_validate(payload)
        except Exception:
            return (
                EvidenceSection(status="unavailable", warning="Embedding evidence is unavailable."),
                None,
            )
        matches = all(
            [
                document.id == expected_id,
                document.transaction_id == row["transaction_id"],
                document.raw_id == row["transaction_id"],
                document.raw_pk == row["sender_id"],
                document.sender_id == row["sender_id"],
                document.batch_id == row["batch_id"],
                document.step == int(row["step"]),
                document.data_split == row["data_split"],
                document.embedding_version == expected_version,
                document.model_id == EMBEDDING_MODEL_ID,
                document.model_revision == EMBEDDING_MODEL_REVISION,
                document.template_version == "behavior-template-v1",
                document.source_version == RAW_PROJECTION_VERSION,
                not (document.data_split in {"validation", "test"} and document.reference_eligible),
            ]
        )
        if not matches:
            return (
                EvidenceSection(
                    status="version_mismatch",
                    warning="Embedding lineage conflicts with the published transaction.",
                ),
                None,
            )
        return (
            EvidenceSection(
                status="available",
                version=document.embedding_version,
                data={
                    "embedding_version": document.embedding_version,
                    "template_version": document.template_version,
                    "model_id": document.model_id,
                    "model_revision": document.model_revision,
                    "dimension": document.dimension,
                    "source_version": document.source_version,
                },
            ),
            document,
        )

    def _graph_evidence(
        self,
        row: Mapping[str, Any],
        raw_document: RawTransactionDocument | None = None,
    ) -> EvidenceSection:
        if self._graph is None:
            return EvidenceSection(
                status="not_configured", warning="Graph evidence is not configured."
            )
        try:
            values = list(self._graph.transaction_evidence(str(row["transaction_id"]), limit=4))
        except Exception:
            return EvidenceSection(status="unavailable", warning="Graph evidence is unavailable.")
        if not values:
            return EvidenceSection(
                status="not_found", warning="Graph transaction evidence is missing."
            )
        try:
            edges = [GraphEdge.model_validate(_flatten_graph_edge(value)) for value in values]
        except Exception:
            return EvidenceSection(
                status="unavailable", warning="Graph evidence failed contract validation."
            )
        if len(edges) != 3 or len({edge.id for edge in edges}) != 3:
            return EvidenceSection(
                status="version_mismatch",
                warning="Graph evidence is partial, duplicated, or contains orphan records.",
            )
        sender = str(row["sender_id"])
        receiver = str(row["receiver_id"])
        transaction_id = str(row["transaction_id"])
        enrichment_seed = int(row["batch_enrichment_seed"])
        shard_count = self._graph_shard_count
        device_id = deterministic_artifact_id(
            "device",
            transaction_id=transaction_id,
            source_row=int(row["source_row"]),
            step=int(row["step"]),
            seed=enrichment_seed,
        )
        ip_id = deterministic_artifact_id(
            "ip",
            transaction_id=transaction_id,
            source_row=int(row["source_row"]),
            step=int(row["step"]),
            seed=enrichment_seed,
        )
        money_label = money_edge_label(str(row["transaction_type"]))
        expected: dict[str, dict[str, Any]] = {
            f"edge:{money_label}:{transaction_id}": {
                "label": money_label,
                "kind": "money_flow",
                "out_id": sender,
                "out_pk": partition_key(sender, shard_count),
                "in_id": receiver,
                "in_pk": partition_key(receiver, shard_count),
                "amount": Decimal(str(row["amount"])),
                "synthetic": False,
                "enrichment_seed": None,
                "enrichment_algorithm_version": None,
            },
            f"edge:USES_DEVICE:{transaction_id}": {
                "label": "USES_DEVICE",
                "kind": "device_observation",
                "out_id": sender,
                "out_pk": partition_key(sender, shard_count),
                "in_id": device_id,
                "in_pk": partition_key(device_id, shard_count),
                "amount": None,
                "synthetic": True,
                "enrichment_seed": enrichment_seed,
                "enrichment_algorithm_version": ENRICHMENT_ALGORITHM_VERSION,
            },
            f"edge:USES_IP:{transaction_id}": {
                "label": "USES_IP",
                "kind": "ip_observation",
                "out_id": sender,
                "out_pk": partition_key(sender, shard_count),
                "in_id": ip_id,
                "in_pk": partition_key(ip_id, shard_count),
                "amount": None,
                "synthetic": True,
                "enrichment_seed": enrichment_seed,
                "enrichment_algorithm_version": ENRICHMENT_ALGORITHM_VERSION,
            },
        }
        by_id = {edge.id: edge for edge in edges}
        if set(by_id) != set(expected):
            return EvidenceSection(
                status="version_mismatch", warning="Graph edge identities do not reconcile."
            )
        for identifier, values_expected in expected.items():
            edge = by_id[identifier]
            common_matches = all(
                [
                    edge.transaction_id == transaction_id,
                    edge.batch_id == row["batch_id"],
                    edge.step == int(row["step"]),
                    edge.data_split == row["data_split"],
                    edge.transaction_type == row["transaction_type"],
                    edge.source_version == RAW_PROJECTION_VERSION,
                    edge.graph_schema_version == GRAPH_SCHEMA_VERSION,
                    edge.projection_version == GRAPH_PROJECTION_VERSION,
                ]
            )
            field_matches = all(
                [
                    edge.label == values_expected["label"],
                    edge.kind == values_expected["kind"],
                    edge.out_id == values_expected["out_id"],
                    edge.out_pk == values_expected["out_pk"],
                    edge.in_id == values_expected["in_id"],
                    edge.in_pk == values_expected["in_pk"],
                    (
                        Decimal(str(edge.amount)) == values_expected["amount"]
                        if edge.amount is not None
                        else values_expected["amount"] is None
                    ),
                    edge.synthetic == values_expected["synthetic"],
                    edge.enrichment_seed == values_expected["enrichment_seed"],
                    edge.enrichment_algorithm_version
                    == values_expected["enrichment_algorithm_version"],
                ]
            )
            if not common_matches or not field_matches:
                return EvidenceSection(
                    status="version_mismatch",
                    warning="Graph evidence conflicts with the published transaction.",
                )
        if raw_document is not None:
            exact = {
                edge.id: edge.payload()
                for edge in build_graph_evidence(
                    raw_document, shard_count=self._graph_shard_count
                ).edges
            }
            if {edge.id: edge.payload() for edge in edges} != exact:
                return EvidenceSection(
                    status="version_mismatch",
                    warning="Graph immutable content differs from deterministic projection.",
                )
        response = [
            GraphTransactionEdgeData(
                edge_id=edge.id,
                label=edge.label,
                kind=edge.kind,
                out_id=edge.out_id,
                in_id=edge.in_id,
                transaction_id=edge.transaction_id,
                step=edge.step,
                amount=edge.amount,
                synthetic=edge.synthetic,
                source_version=edge.source_version,
                graph_schema_version=edge.graph_schema_version,
                projection_version=edge.projection_version,
            ).model_dump(mode="json")
            for edge in sorted(edges, key=lambda value: value.id)
        ]
        return EvidenceSection(status="available", data=response, version=GRAPH_PROJECTION_VERSION)

    def transaction(self, transaction_id: str) -> EvidenceEnvelope:
        row = self._transaction_row(transaction_id)
        warnings: list[str] = []
        raw_section, raw_document = self._raw_evidence(row)
        graph_section = self._graph_evidence(row, raw_document)
        vector_section, _ = self._embedding_evidence(row)
        sections = [raw_section, graph_section, vector_section]
        for section in sections:
            if section.warning:
                warnings.append(section.warning)
        probability = row.get("fraud_probability")
        threshold = row.get("threshold")
        prediction = None
        if probability is not None and threshold is not None:
            prediction = {
                "score_id": row.get("score_id"),
                "fraud_probability": float(probability),
                "threshold": float(threshold),
                "at_or_above_threshold": Decimal(str(probability)) >= Decimal(str(threshold)),
                "model_version_id": row.get("model_version_id"),
                "feature_set": row.get("feature_set"),
                "training_cutoff_step": row.get("training_cutoff_step"),
                "input_version": row.get("input_version"),
                "scored_at": row.get("scored_at"),
            }
        flag = None
        if row.get("flag_id") is not None:
            flag = {
                "flag_id": row["flag_id"],
                "reason": row["flag_reason"],
                "review_status": row["review_status"],
                "proof_of_fraud": False,
            }
        snapshot = None
        if row.get("snapshot_id") is not None:
            snapshot = {
                "snapshot_id": row["snapshot_id"],
                "cutoff_step": row.get("snapshot_cutoff_step"),
                "algorithm_version": row.get("snapshot_algorithm_version"),
            }
        data = TransactionInvestigation(
            transaction={
                "transaction_id": row["transaction_id"],
                "source_row": row["source_row"],
                "step": row["step"],
                "transaction_type": row["transaction_type"],
                "amount": _public_row(row["amount"]),
                "data_split": row["data_split"],
            },
            participants={
                "sender_id": row["sender_id"],
                "sender_is_merchant": row.get("sender_merchant_id") is not None,
                "receiver_id": row["receiver_id"],
                "receiver_is_merchant": row.get("receiver_merchant_id") is not None,
            },
            prediction=prediction,
            fraud_flag=flag,
            graph_snapshot=snapshot,
            raw_evidence=raw_section,
            graph_evidence=graph_section,
            vector_evidence=vector_section,
        )
        partial = any(section.status != "available" for section in sections)
        return EvidenceEnvelope(
            data=data,
            provenance=Provenance(
                batch_id=str(row["batch_id"]),
                model_version_id=row.get("model_version_id"),
                input_version=row.get("input_version"),
                snapshot_id=row.get("snapshot_id"),
                source_version=raw_section.version,
            ),
            freshness=self._freshness(),
            partial_evidence=partial,
            evidence_status="available",
            warnings=warnings,
        )

    def related_accounts(
        self,
        account_id: str,
        *,
        relationship_type: str,
        limit: int,
        minimum_weight: float,
        snapshot_id: str | None,
        cursor: str | None,
    ) -> EvidenceEnvelope:
        exists = self._relational_call(
            lambda: self._relational.published_account_exists(account_id)
        )
        if not exists:
            raise PublishedEvidenceNotFound
        snapshot = self._relational_call(lambda: self._relational.snapshot_lineage(snapshot_id))
        filters = {
            "account_id": account_id,
            "relationship_type": relationship_type,
            "minimum_weight": minimum_weight,
            "snapshot_id": snapshot_id,
        }
        offset = 0
        if cursor:
            position = decode_cursor(cursor, filters, expected_position_fields={"offset"})
            offset = int(position["offset"])
            if offset < 0 or offset > 1_000_000:
                raise ValueError("cursor is invalid")
        warning = "Shared infrastructure and community evidence are not proof of coordination."
        if snapshot is None:
            section = EvidenceSection(
                status="not_found", warning="No published graph snapshot is available."
            )
            values: list[Mapping[str, Any]] = []
        elif self._graph is None:
            section = EvidenceSection(
                status="not_configured", warning="Graph evidence is not configured."
            )
            values = []
        else:
            try:
                values = list(
                    self._graph.related_accounts(
                        account_id,
                        relationship_type=relationship_type,
                        cutoff_step=int(snapshot["cutoff_step"]) + 1,
                        minimum_weight=minimum_weight,
                        offset=offset,
                        limit=limit + 1,
                    )
                )
                validated: list[RelatedAccountEvidence] = []
                for value in values:
                    if _contains_protected_fields(value):
                        raise ValueError("related evidence contains protected fields")
                    candidate = RelatedAccountEvidence.model_validate(
                        {
                            **dict(value),
                            "snapshot_id": snapshot["snapshot_id"],
                            "cutoff_step": snapshot["cutoff_step"],
                            "algorithm_version": snapshot["algorithm_version"],
                        }
                    )
                    if (
                        candidate.related_account_id == account_id
                        or candidate.relationship_type != relationship_type
                        or candidate.snapshot_id != snapshot["snapshot_id"]
                        or candidate.cutoff_step != snapshot["cutoff_step"]
                        or candidate.algorithm_version != snapshot["algorithm_version"]
                    ):
                        raise ValueError("related evidence lineage is invalid")
                    validated.append(candidate)
                if len({value.related_account_id for value in validated}) != len(validated):
                    raise ValueError("related evidence contains duplicate accounts")
                values = [
                    value.model_dump(mode="json")
                    for value in sorted(validated, key=lambda value: value.related_account_id)
                ]
                section = EvidenceSection(
                    status="available", data=values[:limit], version=snapshot["algorithm_version"]
                )
            except Exception:
                values = []
                section = EvidenceSection(
                    status="unavailable", warning="Graph evidence is unavailable."
                )
        next_cursor = None
        if len(values) > limit:
            next_cursor = encode_cursor({"offset": offset + limit}, filters)
        warnings = [warning]
        if section.warning:
            warnings.append(section.warning)
        return EvidenceEnvelope(
            data=RelatedAccountsData(
                account_id=account_id,
                relationship_type=relationship_type,
                snapshot=dict(snapshot) if snapshot else None,
                relationships=section,
            ),
            provenance=Provenance(snapshot_id=snapshot.get("snapshot_id") if snapshot else None),
            freshness=self._freshness(),
            partial_evidence=section.status != "available",
            evidence_status="available",
            warnings=warnings,
            next_cursor=next_cursor,
        )

    def similar(
        self,
        transaction_id: str,
        *,
        limit: int,
        maximum_distance: float | None,
    ) -> EvidenceEnvelope:
        row = self._transaction_row(transaction_id)
        warning = "Cosine distance is similarity evidence, not fraud probability."
        section, embedding = self._embedding_evidence(row)
        if section.status == "available" and embedding is not None:
            try:
                assert self._vectors is not None
                results = self._vectors.vector_search(
                    query_vector=embedding.vector,
                    query_step=embedding.step,
                    query_transaction_id=transaction_id,
                    embedding_version=embedding.embedding_version,
                    template_version=embedding.template_version,
                    model_revision=embedding.model_revision,
                    limit=limit,
                )
                matches: list[SimilarityMatchData] = []
                mismatch = False
                allowed_fields = {
                    "transaction_id",
                    "step",
                    "distance",
                    "embedding_version",
                    "template_version",
                    "model_revision",
                    "source_version",
                }
                for result in results:
                    if _contains_protected_fields(result) or set(result) - allowed_fields:
                        raise ValueError("similarity evidence contains protected fields")
                    candidate = SimilarityMatchData.model_validate(
                        {
                            "transaction_id": result.get("transaction_id"),
                            "step": result.get("step"),
                            "cosine_distance": result.get("distance"),
                            "embedding_version": result.get("embedding_version"),
                            "template_version": result.get("template_version"),
                            "model_revision": result.get("model_revision"),
                            "source_version": result.get("source_version"),
                        }
                    )
                    if (
                        candidate.transaction_id == transaction_id
                        or candidate.step >= embedding.step
                    ):
                        raise ValueError(
                            "similarity result is not strictly earlier or excludes self"
                        )
                    if (
                        candidate.embedding_version != embedding.embedding_version
                        or candidate.template_version != embedding.template_version
                        or candidate.model_revision != embedding.model_revision
                    ):
                        mismatch = True
                        continue
                    if (
                        maximum_distance is not None
                        and candidate.cosine_distance > maximum_distance
                    ):
                        continue
                    matches.append(candidate)
                if mismatch:
                    section = EvidenceSection(
                        status="version_mismatch",
                        warning="Similar evidence versions are incompatible.",
                    )
                else:
                    section = EvidenceSection(
                        status="available",
                        data=[value.model_dump(mode="json") for value in matches],
                        version=embedding.embedding_version,
                    )
            except Exception:
                section = EvidenceSection(
                    status="unavailable", warning="Vector evidence is unavailable."
                )
        warnings = [warning]
        if section.warning:
            warnings.append(section.warning)
        return EvidenceEnvelope(
            data=SimilarBehaviorData(transaction_id=transaction_id, matches=section),
            provenance=Provenance(
                batch_id=str(row["batch_id"]),
                model_version_id=row.get("model_version_id"),
                input_version=row.get("input_version"),
            ),
            freshness=self._freshness(),
            partial_evidence=section.status != "available",
            evidence_status="available",
            warnings=warnings,
        )

    def dashboard_summary(self, filters: dict[str, Any]) -> EvidenceEnvelope:
        value = self._relational_call(
            lambda: self._relational.dashboard_summary(filters, LOCKED_MODEL_ID)
        )
        transaction_count = int(value["transaction_count"] or 0)
        flag_count = int(value["fraud_flag_count"] or 0)
        data = DashboardSummaryData(
            transaction_count=transaction_count,
            scored_transaction_count=int(value["scored_transaction_count"] or 0),
            fraud_flag_count=flag_count,
            flag_rate=flag_count / transaction_count if transaction_count else 0.0,
            amount_total=format(Decimal(str(value["amount_total"] or 0)), "f"),
            score_minimum=(
                float(value["score_minimum"]) if value["score_minimum"] is not None else None
            ),
            score_maximum=(
                float(value["score_maximum"]) if value["score_maximum"] is not None else None
            ),
            score_mean=(float(value["score_mean"]) if value["score_mean"] is not None else None),
            transaction_type_counts=dict(value["transaction_type_counts"]),
            active_model_id=LOCKED_MODEL_ID,
        )
        return EvidenceEnvelope(
            data=data,
            provenance=Provenance(model_version_id=LOCKED_MODEL_ID),
            freshness=self._freshness(),
            partial_evidence=False,
            evidence_status="available",
            warnings=[],
        )

    def flags(
        self,
        filters: dict[str, Any],
        *,
        limit: int,
        cursor: str | None,
    ) -> EvidenceEnvelope:
        cursor_filters = dict(filters)
        position = None
        if cursor:
            position = decode_cursor(
                cursor,
                cursor_filters,
                expected_position_fields={"step", "transaction_id", "score_id"},
            )
        rows = list(
            self._relational_call(
                lambda: self._relational.flag_page(filters, position, limit, LOCKED_MODEL_ID)
            )
        )
        items = [
            FlagItem(
                transaction_id=str(row["transaction_id"]),
                step=int(row["step"]),
                transaction_type=str(row["transaction_type"]),
                amount=format(Decimal(str(row["amount"])), "f"),
                sender_id=str(row["sender_id"]),
                receiver_id=str(row["receiver_id"]),
                score=float(row["fraud_probability"]),
                threshold=float(row["threshold"]),
                model_version_id=str(row["model_version_id"]),
                flag_reason=str(row["flag_reason"]),
                review_status=str(row["review_status"]),
                snapshot_id=row.get("snapshot_id"),
                scored_at=row["scored_at"],
                batch_id=str(row["batch_id"]),
            )
            for row in rows[:limit]
        ]
        next_cursor = None
        if len(rows) > limit:
            last = rows[limit - 1]
            next_cursor = encode_cursor(
                {
                    "step": int(last["step"]),
                    "transaction_id": str(last["transaction_id"]),
                    "score_id": str(last["score_id"]),
                },
                cursor_filters,
            )
        return EvidenceEnvelope(
            data=FlagPageData(items=items),
            provenance=Provenance(model_version_id=LOCKED_MODEL_ID),
            freshness=self._freshness(),
            partial_evidence=False,
            evidence_status="available",
            warnings=[],
            next_cursor=next_cursor,
        )

    async def freshness(self) -> EvidenceEnvelope:
        value = self._relational_call(lambda: self._relational.freshness(LOCKED_MODEL_ID))
        readiness = await self._readiness.check()
        dependencies = {
            item.name: ("available" if item.state.value == "healthy" else item.state.value)
            for item in readiness.dependencies
        }
        latest = value.get("latest_published_batch")
        complete = latest is not None and all(
            state == "available"
            for name, state in dependencies.items()
            if name
            in {"postgresql", "cosmos_nosql_raw", "cosmos_nosql_embeddings", "cosmos_gremlin"}
        )
        data = FreshnessData(
            latest_published_batch=latest,
            published_cutoff_step=value.get("published_cutoff_step"),
            pending_batch_count=int(value.get("pending_batch_count") or 0),
            active_model_version=LOCKED_MODEL_ID,
            model_training_cutoff=int(value.get("training_cutoff_step") or 281),
            versions={
                "raw": "raw-paysim-v1",
                "graph": "gremlin-evidence-v1",
                "embedding": str(value.get("embedding_version") or "not_available"),
                "feature": LOCKED_FEATURE_VERSION,
                "score": "phase8-locked-test-input-v1",
            },
            dependencies=dependencies,
            evidence_complete=complete,
        )
        warnings = [] if complete else ["Evidence is partial or no published batch is available."]
        return EvidenceEnvelope(
            data=data,
            provenance=Provenance(model_version_id=LOCKED_MODEL_ID),
            freshness=Freshness(
                latest_published_batch=latest,
                published_cutoff_step=value.get("published_cutoff_step"),
                pending_batch_count=int(value.get("pending_batch_count") or 0),
                complete=complete,
            ),
            partial_evidence=not complete,
            evidence_status="available",
            warnings=warnings,
        )
