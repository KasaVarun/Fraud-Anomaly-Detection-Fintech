"""Operational Phase 8 chunk workflow and optional live projections."""

from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, func, select

from fraud_detection.config import Settings
from fraud_detection.persistence.document_vector.adapter import CosmosRawDocumentAdapter
from fraud_detection.persistence.document_vector.embedding_adapter import CosmosEmbeddingAdapter
from fraud_detection.persistence.graph.adapter import CosmosGremlinAdapter
from fraud_detection.persistence.relational import Base

from .contracts import ChunkPlan, ChunkRunResult, TestChunk
from .planning import plan_test_chunks
from .reconciliation import ExpectedLiveEvidence, LiveStores, expected_live_evidence
from .scoring import run_offline_chunk

transaction = Base.metadata.tables["transaction"]
transaction_score = Base.metadata.tables["transaction_score"]
model_evaluation = Base.metadata.tables["model_evaluation"]


def project_chunk_live(
    chunk: TestChunk,
    settings: Settings,
) -> dict[str, int]:
    if not (
        settings.document_vector_configured
        and settings.embedding_configured
        and settings.graph_configured
    ):
        raise ValueError("all Cosmos raw, embedding, and Gremlin settings are required")
    raw_adapter = CosmosRawDocumentAdapter(settings)
    embedding_adapter = CosmosEmbeddingAdapter(settings)
    graph_adapter = CosmosGremlinAdapter(settings)
    try:
        return project_expected_live_evidence(
            chunk,
            expected_live_evidence(settings, chunk),
            LiveStores(raw_adapter, embedding_adapter, graph_adapter),
        )
    finally:
        graph_adapter.close()


def project_expected_live_evidence(
    chunk: TestChunk,
    expected: ExpectedLiveEvidence,
    stores: LiveStores,
    *,
    stage_hook: Any | None = None,
) -> dict[str, int]:
    stores.raw.validate_container()
    stores.embeddings.validate_embedding_container()
    stores.graph.validate_graph()
    raw_created = graph_vertices = graph_edges = embedding_created = 0
    for document in expected.raw_documents:
        raw_created += stores.raw.upsert_document(document.cosmos_payload()) == "created"
    if stage_hook:
        stage_hook("live_raw")
    for vertex in expected.vertices:
        graph_vertices += stores.graph.upsert_vertex(vertex.payload()) == "created"
    for edge in expected.edges:
        graph_edges += stores.graph.upsert_edge(edge.payload()) == "created"
    if stage_hook:
        stage_hook("live_graph")
    for embedding in expected.embeddings:
        embedding_created += stores.embeddings.upsert_embedding(embedding.payload()) == "created"
    if stage_hook:
        stage_hook("live_embedding")
    if (
        len(expected.raw_documents) != chunk.row_count
        or len(expected.embeddings) != chunk.row_count
        or len(expected.edges) != 3 * chunk.row_count
    ):
        raise ValueError("live projection evidence counts differ from chunk plan")
    return {
        "raw_created": raw_created,
        "graph_vertices_created": graph_vertices,
        "graph_edges_created": graph_edges,
        "embeddings_created": embedding_created,
    }


def list_chunk_status(database_url: str, plan: ChunkPlan) -> list[dict[str, object]]:
    engine: Engine = create_engine(database_url, hide_parameters=True)
    try:
        with engine.connect() as connection:
            values: list[dict[str, object]] = []
            for chunk in plan.chunks:
                transaction_count = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(transaction)
                        .where(transaction.c.batch_id == chunk.batch_id)
                    )
                    or 0
                )
                score_count = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(transaction_score)
                        .join(transaction)
                        .where(transaction.c.batch_id == chunk.batch_id)
                    )
                    or 0
                )
                evaluation_count = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(model_evaluation)
                        .where(model_evaluation.c.batch_id == chunk.batch_id)
                    )
                    or 0
                )
                values.append(
                    {
                        **chunk.model_dump(mode="json"),
                        "transaction_count": transaction_count,
                        "score_count": score_count,
                        "evaluation_count": evaluation_count,
                        "complete": score_count == chunk.row_count and evaluation_count == 1,
                    }
                )
        return values
    finally:
        engine.dispose()


def run_chunk(
    source_path: Path,
    manifest_path: Path,
    database_url: str,
    settings: Settings,
    plan: ChunkPlan,
    chunk: TestChunk,
) -> ChunkRunResult:
    live = False
    if (
        settings.document_vector_configured
        and settings.embedding_configured
        and settings.graph_configured
    ):
        project_chunk_live(chunk, settings)
        live = True
    result = run_offline_chunk(
        source_path,
        manifest_path,
        settings.incremental_data_root / "locked_test_features_v2.parquet",
        settings.incremental_data_root / "locked_test_features_v2.metadata.v2.json",
        database_url,
        settings.model_artifact_root,
        plan,
        chunk,
    )
    return result.model_copy(update={"live_projection": live})


def next_eligible_chunk(database_url: str, plan: ChunkPlan) -> TestChunk | None:
    statuses = list_chunk_status(database_url, plan)
    for chunk, status in zip(plan.chunks, statuses, strict=True):
        if not bool(status["complete"]):
            return chunk
    return None


def default_plan(source_path: Path, manifest_path: Path, settings: Settings) -> ChunkPlan:
    return plan_test_chunks(
        source_path, manifest_path, chunk_steps=settings.incremental_chunk_steps
    )
