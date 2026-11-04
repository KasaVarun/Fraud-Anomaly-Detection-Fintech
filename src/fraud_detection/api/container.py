"""Application dependency container."""

from dataclasses import dataclass
from typing import cast

from fraud_detection.config import Settings
from fraud_detection.investigator.ports import (
    GraphInvestigatorPort,
    InvestigatorRelationalPort,
    RawEvidencePort,
    VectorEvidencePort,
)
from fraud_detection.investigator.repository import (
    PostgresInvestigatorRepository,
    UnavailableInvestigatorRepository,
)
from fraud_detection.investigator.service import InvestigatorService
from fraud_detection.persistence.document_vector.adapter import CosmosRawDocumentAdapter
from fraud_detection.persistence.document_vector.embedding_adapter import CosmosEmbeddingAdapter
from fraud_detection.persistence.graph.adapter import CosmosGremlinAdapter
from fraud_detection.ports.services import (
    BatchReconciliationPort,
    DocumentVectorPersistencePort,
    GraphPersistencePort,
    ModelArtifactPort,
    RelationalPersistencePort,
)
from fraud_detection.ports.unavailable import UnavailableAdapter
from fraud_detection.reconciliation.service import ReadinessService


@dataclass(frozen=True, slots=True)
class AppContainer:
    """Dependencies available to API services and future read-only routes."""

    settings: Settings
    relational: RelationalPersistencePort
    document_vector: DocumentVectorPersistencePort
    embeddings: DocumentVectorPersistencePort
    graph: GraphPersistencePort
    model_artifacts: ModelArtifactPort
    reconciliation: BatchReconciliationPort
    readiness: ReadinessService
    investigator: InvestigatorService


def build_container(settings: Settings) -> AppContainer:
    """Wire Phase 1 unavailable adapters without simulating production behavior."""

    investigator_relational: InvestigatorRelationalPort
    if settings.postgres_configured:
        assert settings.postgres_dsn is not None
        investigator_relational = PostgresInvestigatorRepository(
            settings.postgres_dsn.get_secret_value()
        )
        relational: RelationalPersistencePort = investigator_relational
    else:
        investigator_relational = UnavailableInvestigatorRepository()
        relational = investigator_relational
    document_vector: DocumentVectorPersistencePort
    if settings.document_vector_configured:
        document_vector = CosmosRawDocumentAdapter(settings)
    else:
        document_vector = UnavailableAdapter("cosmos_nosql_raw", False, "Phase 4")
    embeddings: DocumentVectorPersistencePort
    if settings.embedding_configured:
        embeddings = CosmosEmbeddingAdapter(settings)
    else:
        embeddings = UnavailableAdapter("cosmos_nosql_embeddings", False, "Phase 6")
    graph: GraphPersistencePort
    if settings.graph_configured:
        graph = CosmosGremlinAdapter(settings)
    else:
        graph = UnavailableAdapter("cosmos_gremlin", False, "Phase 5")
    model_artifacts = UnavailableAdapter(
        "model_artifacts", settings.model_artifact_configured, "Phase 7"
    )
    reconciliation = UnavailableAdapter("batch_reconciliation", False, "Phase 8")
    readiness = ReadinessService(
        [relational, document_vector, embeddings, graph, model_artifacts, reconciliation]
    )
    investigator = InvestigatorService(
        investigator_relational,
        readiness,
        raw=(
            cast(RawEvidencePort, document_vector) if settings.document_vector_configured else None
        ),
        graph=(cast(GraphInvestigatorPort, graph) if settings.graph_configured else None),
        vectors=(cast(VectorEvidencePort, embeddings) if settings.embedding_configured else None),
        graph_shard_count=settings.gremlin_partition_shards,
    )
    return AppContainer(
        settings=settings,
        relational=relational,
        document_vector=document_vector,
        embeddings=embeddings,
        graph=graph,
        model_artifacts=model_artifacts,
        reconciliation=reconciliation,
        readiness=readiness,
        investigator=investigator,
    )
