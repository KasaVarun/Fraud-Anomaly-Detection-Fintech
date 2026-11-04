"""Safe operational CLI for Phase 8 incremental locked scoring."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fraud_detection.config import get_settings
from fraud_detection.incremental.evidence import prepare_offline_evidence
from fraud_detection.incremental.reconciliation import (
    attempt_publication,
    evaluate_pooled_locked_test,
    reconcile_chunk,
)
from fraud_detection.incremental.workflow import (
    default_plan,
    list_chunk_status,
    next_eligible_chunk,
    run_chunk,
)


def _database_url() -> str:
    dsn = get_settings().postgres_dsn
    if dsn is None:
        raise ValueError("FRAUD_POSTGRES_DSN is required")
    return dsn.get_secret_value()


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-incremental", description="Phase 8 locked-model incremental workflow"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("plan", "generate the deterministic whole-step test chunk plan"),
        ("list", "list planned chunks and local PostgreSQL progress"),
        ("prepare-offline", "build ignored raw, graph, template, embedding, and feature artifacts"),
        ("run-chunk", "run one specified chunk idempotently"),
        ("run-next", "run the next chronologically eligible incomplete chunk"),
        ("resume", "resume one interrupted chunk idempotently"),
        ("reconcile", "reconcile one chunk and report publication blockers"),
        ("evaluate", "compute the pooled locked-test evaluation"),
        ("publish", "attempt publication only after live-store reconciliation"),
    ):
        command = commands.add_parser(name, help=help_text)
        _common(command)
        if name in {"run-chunk", "resume", "reconcile", "publish"}:
            command.add_argument("--chunk-index", type=int, required=True)
    return parser


def _run(arguments: argparse.Namespace) -> Any:
    settings = get_settings()
    plan = default_plan(arguments.source, arguments.manifest, settings)
    if arguments.command == "plan":
        return plan
    if arguments.command == "list":
        return list_chunk_status(_database_url(), plan)
    if arguments.command == "prepare-offline":
        return prepare_offline_evidence(
            arguments.source,
            arguments.manifest,
            settings.embedding_local_artifact,
            settings.incremental_data_root,
            settings,
        )
    if arguments.command in {"run-chunk", "resume"}:
        chunk = plan.chunks[arguments.chunk_index]
        return run_chunk(
            arguments.source,
            arguments.manifest,
            _database_url(),
            settings,
            plan,
            chunk,
        )
    if arguments.command == "run-next":
        chunk = next_eligible_chunk(_database_url(), plan)
        if chunk is None:
            return {"status": "complete", "message": "all chunks are complete"}
        return run_chunk(
            arguments.source,
            arguments.manifest,
            _database_url(),
            settings,
            plan,
            chunk,
        )
    if arguments.command == "reconcile":
        return reconcile_chunk(
            _database_url(),
            settings.incremental_data_root / "locked_test_features_v2.parquet",
            plan.chunks[arguments.chunk_index],
            settings,
        )
    if arguments.command == "evaluate":
        return evaluate_pooled_locked_test(
            _database_url(),
            plan,
            settings.incremental_artifact_root / "locked_test_pooled_v1.json",
        )
    if arguments.command == "publish":
        return attempt_publication(
            _database_url(),
            settings.incremental_data_root / "locked_test_features_v2.parquet",
            plan,
            plan.chunks[arguments.chunk_index],
            settings,
        )
    raise ValueError("unsupported incremental command")


def main(argv: list[str] | None = None) -> int:
    try:
        result = _run(build_parser().parse_args(argv))
        if hasattr(result, "model_dump"):
            print(json.dumps(result.model_dump(mode="json"), indent=2))
        else:
            print(json.dumps(result, indent=2))
        return 0
    except (IndexError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
