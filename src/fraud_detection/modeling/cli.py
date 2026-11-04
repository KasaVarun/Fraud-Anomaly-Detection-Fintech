"""Bounded Phase 7 feature-build and model-release commands."""

import argparse
import json
import sys
from pathlib import Path

from fraud_detection.config import get_settings
from fraud_detection.modeling.features import build_feature_artifact
from fraud_detection.modeling.training import train_and_release_models


def _database_url() -> str:
    settings = get_settings()
    if settings.postgres_dsn is None:
        raise ValueError("FRAUD_POSTGRES_DSN is required")
    return settings.postgres_dsn.get_secret_value()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud-model")
    commands = parser.add_subparsers(dest="command", required=True)
    features = commands.add_parser("features-build")
    features.add_argument("--source", type=Path, required=True)
    features.add_argument("--manifest", type=Path, required=True)
    features.add_argument("--embeddings", type=Path)
    features.add_argument("--output", type=Path)
    train = commands.add_parser("train-release")
    train.add_argument("--features", type=Path)
    train.add_argument("--metadata", type=Path)
    train.add_argument("--artifact-root", type=Path)
    return parser


def _run(arguments: argparse.Namespace) -> object:
    settings = get_settings()
    if arguments.command == "features-build":
        return build_feature_artifact(
            arguments.source,
            arguments.manifest,
            arguments.embeddings or settings.embedding_local_artifact,
            _database_url(),
            arguments.output or settings.model_feature_artifact,
        )
    feature_path = arguments.features or settings.model_feature_artifact
    return train_and_release_models(
        feature_path,
        arguments.metadata or feature_path.with_suffix(".metadata.json"),
        _database_url(),
        arguments.artifact_root or settings.model_artifact_root,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        result = _run(build_parser().parse_args(argv))
        if hasattr(result, "model_dump"):
            print(json.dumps(result.model_dump(mode="json"), indent=2))
        else:
            print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
