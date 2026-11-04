"""Small typed command-line interface for Phase 3 ingestion workflows."""

import argparse
import sys
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from fraud_detection.config import get_settings
from fraud_detection.ingestion.acquisition import acquire_paysim
from fraud_detection.ingestion.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENRICHMENT_SEED,
    DEFAULT_SAMPLING_SEED,
    DEFAULT_TARGET_SIZE,
)
from fraud_detection.ingestion.errors import IngestionError
from fraud_detection.ingestion.loader import initial_batch_id, load_initial_population
from fraud_detection.ingestion.manifest import validate_manifest_artifacts
from fraud_detection.ingestion.reconciliation import reconcile_initial_load
from fraud_detection.ingestion.sampling import generate_sample
from fraud_detection.ingestion.source import validate_source
from fraud_detection.logging import configure_logging
from fraud_detection.persistence.document_vector.adapter import CosmosRawDocumentAdapter
from fraud_detection.persistence.document_vector.artifacts import (
    embedding_artifact_logical_digest,
    iter_embedding_artifact,
)
from fraud_detection.persistence.document_vector.embedding_adapter import CosmosEmbeddingAdapter
from fraud_detection.persistence.document_vector.embedding_models import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
)
from fraud_detection.persistence.document_vector.embedding_reconciliation import (
    reconcile_embedding_documents,
    reconcile_embedding_store,
)
from fraud_detection.persistence.document_vector.embedding_service import (
    build_template_artifact,
    generate_embeddings,
    project_embedding_artifact,
    similar_behavior_from_artifact,
)
from fraud_detection.persistence.document_vector.errors import CosmosAdapterError
from fraud_detection.persistence.document_vector.projection import project_initial_raw_documents
from fraud_detection.persistence.document_vector.reconciliation import reconcile_raw_projection
from fraud_detection.persistence.graph.adapter import CosmosGremlinAdapter
from fraud_detection.persistence.graph.analysis import (
    analyze_historical_evidence,
    persist_historical_analysis,
)
from fraud_detection.persistence.graph.errors import GremlinAdapterError
from fraud_detection.persistence.graph.models import GraphEdge
from fraud_detection.persistence.graph.projection import project_initial_graph
from fraud_detection.persistence.graph.reconciliation import reconcile_graph_projection


def _path(value: str) -> Path:
    return Path(value)


def _database_url() -> str:
    dsn = get_settings().postgres_dsn
    if dsn is None:
        raise IngestionError("FRAUD_POSTGRES_DSN is required for database commands")
    return dsn.get_secret_value()


def _cosmos_adapter() -> CosmosRawDocumentAdapter:
    return CosmosRawDocumentAdapter(get_settings())


def _graph_adapter() -> CosmosGremlinAdapter:
    return CosmosGremlinAdapter(get_settings())


def _embedding_adapter() -> CosmosEmbeddingAdapter:
    return CosmosEmbeddingAdapter(get_settings())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud-ingest")
    subcommands = parser.add_subparsers(dest="command", required=True)

    acquire = subcommands.add_parser(
        "acquire", help="download PaySim with an authenticated Kaggle CLI"
    )
    acquire.add_argument("--output-directory", type=_path, default=Path("data/raw"))

    validate = subcommands.add_parser("validate-source", help="validate a PaySim source CSV")
    validate.add_argument("--source", type=_path, required=True)

    sample = subcommands.add_parser("sample", help="generate deterministic Parquet and manifest")
    sample.add_argument("--source", type=_path, required=True)
    sample.add_argument("--sample-out", type=_path, required=True)
    sample.add_argument("--manifest-out", type=_path, required=True)
    sample.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    sample.add_argument("--sampling-seed", type=int, default=DEFAULT_SAMPLING_SEED)
    sample.add_argument("--enrichment-seed", type=int, default=DEFAULT_ENRICHMENT_SEED)

    validate_manifest = subcommands.add_parser(
        "validate-manifest", help="validate source, sample, and manifest integrity"
    )
    validate_manifest.add_argument("--source", type=_path, required=True)
    validate_manifest.add_argument("--manifest", type=_path, required=True)

    load = subcommands.add_parser("load-initial", help="load train and validation rows")
    load.add_argument("--source", type=_path, required=True)
    load.add_argument("--manifest", type=_path, required=True)
    load.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    verify = subcommands.add_parser("verify-load", help="reconcile the initial PostgreSQL load")
    verify.add_argument("--source", type=_path, required=True)
    verify.add_argument("--manifest", type=_path, required=True)

    subcommands.add_parser(
        "cosmos-validate", help="validate the Cosmos database, raw container, and partition key"
    )

    cosmos_project = subcommands.add_parser(
        "cosmos-project-initial", help="project train/validation raw documents"
    )
    cosmos_project.add_argument("--source", type=_path, required=True)
    cosmos_project.add_argument("--manifest", type=_path, required=True)

    cosmos_verify = subcommands.add_parser(
        "cosmos-verify", help="reconcile PostgreSQL and Cosmos raw documents"
    )
    cosmos_verify.add_argument("--source", type=_path, required=True)
    cosmos_verify.add_argument("--manifest", type=_path, required=True)

    subcommands.add_parser("gremlin-validate", help="validate bounded Gremlin connectivity")

    graph_project = subcommands.add_parser(
        "gremlin-project-initial", help="project initial raw documents into graph evidence"
    )
    graph_project.add_argument("--source", type=_path, required=True)
    graph_project.add_argument("--manifest", type=_path, required=True)

    graph_verify = subcommands.add_parser(
        "gremlin-verify", help="reconcile PostgreSQL, raw documents, and graph evidence"
    )
    graph_verify.add_argument("--source", type=_path, required=True)
    graph_verify.add_argument("--manifest", type=_path, required=True)

    graph_analyze = subcommands.add_parser(
        "graph-analyze", help="compute and persist a strict historical graph snapshot"
    )
    graph_analyze.add_argument("--source", type=_path, required=True)
    graph_analyze.add_argument("--manifest", type=_path, required=True)
    graph_analyze.add_argument("--cutoff-step", type=int, required=True)

    subcommands.add_parser(
        "embedding-validate", help="validate the separate embedding vector container"
    )

    templates_build = subcommands.add_parser(
        "templates-build", help="build strict prior-step behavioral templates"
    )
    templates_build.add_argument("--source", type=_path, required=True)
    templates_build.add_argument("--manifest", type=_path, required=True)
    templates_build.add_argument("--output", type=_path)

    embeddings_generate = subcommands.add_parser(
        "embeddings-generate", help="generate real MiniLM embeddings to an ignored artifact"
    )
    embeddings_generate.add_argument("--template-artifact", type=_path)
    embeddings_generate.add_argument("--output", type=_path)
    embeddings_generate.add_argument("--project", action="store_true")

    embeddings_verify = subcommands.add_parser(
        "embeddings-verify", help="reconcile local or Cosmos embedding evidence"
    )
    embeddings_verify.add_argument("--source", type=_path, required=True)
    embeddings_verify.add_argument("--manifest", type=_path, required=True)
    embeddings_verify.add_argument("--artifact", type=_path)

    similar = subcommands.add_parser(
        "similar-behavior", help="run a bounded leakage-safe similarity query"
    )
    similar.add_argument("--artifact", type=_path)
    similar.add_argument("--transaction-id", required=True)
    similar.add_argument("--top-k", type=int)
    return parser


def _run(arguments: argparse.Namespace) -> Any:
    if arguments.command == "acquire":
        return {"source_filename": acquire_paysim(arguments.output_directory).name}
    if arguments.command == "validate-source":
        return validate_source(arguments.source)
    if arguments.command == "sample":
        return generate_sample(
            arguments.source,
            arguments.sample_out,
            arguments.manifest_out,
            target_size=arguments.target_size,
            sampling_seed=arguments.sampling_seed,
            enrichment_seed=arguments.enrichment_seed,
        )
    if arguments.command == "validate-manifest":
        return validate_manifest_artifacts(arguments.source, arguments.manifest)
    if arguments.command == "load-initial":
        return load_initial_population(
            arguments.source,
            arguments.manifest,
            _database_url(),
            batch_size=arguments.batch_size,
        )
    if arguments.command == "verify-load":
        return reconcile_initial_load(arguments.source, arguments.manifest, _database_url())
    if arguments.command == "cosmos-validate":
        _cosmos_adapter().validate_container()
        return {"status": "healthy", "container": get_settings().cosmos_raw_container}
    if arguments.command == "cosmos-project-initial":
        settings = get_settings()
        return project_initial_raw_documents(
            arguments.source,
            arguments.manifest,
            _database_url(),
            _cosmos_adapter(),
            batch_size=settings.cosmos_batch_size,
        )
    if arguments.command == "cosmos-verify":
        settings = get_settings()
        return reconcile_raw_projection(
            arguments.source,
            arguments.manifest,
            _database_url(),
            _cosmos_adapter(),
            batch_size=settings.cosmos_batch_size,
            page_size=settings.cosmos_page_size,
        )
    if arguments.command == "gremlin-validate":
        adapter = _graph_adapter()
        try:
            adapter.validate_graph()
            return {"status": "healthy", "graph": get_settings().gremlin_graph}
        finally:
            adapter.close()
    if arguments.command == "gremlin-project-initial":
        settings = get_settings()
        adapter = _graph_adapter()
        try:
            return project_initial_graph(
                arguments.source,
                arguments.manifest,
                _database_url(),
                _cosmos_adapter(),
                adapter,
                shard_count=settings.gremlin_partition_shards,
                page_size=settings.cosmos_page_size,
            )
        finally:
            adapter.close()
    if arguments.command == "gremlin-verify":
        settings = get_settings()
        adapter = _graph_adapter()
        try:
            return reconcile_graph_projection(
                arguments.source,
                arguments.manifest,
                _database_url(),
                _cosmos_adapter(),
                adapter,
                shard_count=settings.gremlin_partition_shards,
                chunk_size=settings.gremlin_projection_batch_size,
                page_size=settings.cosmos_page_size,
            )
        finally:
            adapter.close()
    if arguments.command == "graph-analyze":
        settings = get_settings()
        manifest = validate_manifest_artifacts(arguments.source, arguments.manifest)
        adapter = _graph_adapter()
        try:
            evidence = [
                GraphEdge.model_validate(dict(payload))
                for payload in adapter.iter_evidence_before(
                    arguments.cutoff_step,
                    page_size=settings.cosmos_page_size,
                )
            ]
            result = analyze_historical_evidence(
                evidence,
                batch_id=initial_batch_id(manifest),
                cutoff_step=arguments.cutoff_step,
                louvain_seed=settings.louvain_seed,
                resolution=settings.louvain_resolution,
            )
            return persist_historical_analysis(_database_url(), result)
        finally:
            adapter.close()
    if arguments.command == "embedding-validate":
        embedding_adapter = _embedding_adapter()
        embedding_adapter.validate_embedding_container()
        return {
            "status": "healthy",
            "container": get_settings().cosmos_embedding_container,
        }
    if arguments.command == "templates-build":
        settings = get_settings()
        return build_template_artifact(
            arguments.source,
            arguments.manifest,
            _database_url(),
            arguments.output or settings.behavioral_template_artifact,
            settings,
        )
    if arguments.command == "embeddings-generate":
        settings = get_settings()
        output = arguments.output or settings.embedding_local_artifact
        embedding_result = generate_embeddings(
            arguments.template_artifact or settings.behavioral_template_artifact,
            output,
            settings,
        )
        if arguments.project:
            projection = project_embedding_artifact(
                output,
                _database_url(),
                _embedding_adapter(),
            )
            return {
                "artifact": embedding_result.model_dump(mode="json"),
                "projection": projection.model_dump(mode="json"),
            }
        return embedding_result
    if arguments.command == "embeddings-verify":
        settings = get_settings()
        if settings.embedding_configured and arguments.artifact is None:
            return reconcile_embedding_store(
                arguments.source,
                arguments.manifest,
                _database_url(),
                _embedding_adapter(),
                page_size=settings.embedding_page_size,
                chunk_size=settings.cosmos_batch_size,
                expected_model_id=EMBEDDING_MODEL_ID,
                expected_model_revision=EMBEDDING_MODEL_REVISION,
                expected_embedding_version=settings.embedding_version,
                expected_template_version=settings.behavioral_template_version,
            )
        artifact = arguments.artifact or settings.embedding_local_artifact
        return reconcile_embedding_documents(
            arguments.source,
            arguments.manifest,
            _database_url(),
            iter_embedding_artifact(artifact),
            expected_model_id=EMBEDDING_MODEL_ID,
            expected_model_revision=EMBEDDING_MODEL_REVISION,
            expected_embedding_version=settings.embedding_version,
            expected_template_version=settings.behavioral_template_version,
            expected_logical_artifact_digest=embedding_artifact_logical_digest(artifact),
        )
    if arguments.command == "similar-behavior":
        settings = get_settings()
        return similar_behavior_from_artifact(
            arguments.artifact or settings.embedding_local_artifact,
            arguments.transaction_id,
            top_k=arguments.top_k or settings.vector_search_limit,
            max_top_k=settings.vector_search_limit,
        )
    raise IngestionError("unsupported command")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    configure_logging(get_settings().log_level)
    try:
        result = _run(arguments)
    except IngestionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SQLAlchemyError:
        print("error: database operation failed", file=sys.stderr)
        return 3
    except CosmosAdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except GremlinAdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5
    if hasattr(result, "model_dump_json"):
        print(result.model_dump_json(indent=2))
    else:
        import json

        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
