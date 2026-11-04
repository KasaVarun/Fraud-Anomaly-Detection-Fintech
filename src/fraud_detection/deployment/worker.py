"""Private-network manual pipeline worker with hash-verified staging and publication guards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from pydantic import BaseModel, ConfigDict, Field

PIPELINE_STAGES = (
    "schema-migration",
    "initial-relational-load",
    "raw-projection",
    "graph-projection",
    "behavioral-embeddings",
    "graph-snapshot",
    "feature-model",
    "incremental",
    "reconciliation",
    "evaluation",
    "publication",
)
WORK_ROOT = Path(os.environ.get("FRAUD_WORK_ROOT", "/work"))
SOURCE = WORK_ROOT / "data/raw/PS_20174392719_1491204439457_log.csv"
MANIFEST = WORK_ROOT / "data/processed/paysim_manifest.json"
SAMPLE = WORK_ROOT / "data/processed/paysim_sample.parquet"
EMBEDDINGS = WORK_ROOT / "data/processed/behavior_embeddings.parquet"
LOCK_ID = 7_375_111
INGEST_COMMAND = ["python", "-m", "fraud_detection.ingestion"]
MODEL_COMMAND = ["python", "-m", "fraud_detection.modeling.cli"]
INCREMENTAL_COMMAND = ["python", "-m", "fraud_detection.incremental.cli"]


class StagedArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    blob_name: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    restricted_source: bool
    required_for_stages: tuple[str, ...]
    extract_to: str | None = None


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "private-staging-artifacts-v1"
    source_provenance: str = Field(min_length=1)
    artifacts: tuple[StagedArtifact, ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_destination(relative: str) -> Path:
    destination = (WORK_ROOT / relative).resolve()
    root = WORK_ROOT.resolve()
    if destination != root and root not in destination.parents:
        raise ValueError("artifact destination escapes the worker root")
    return destination


def validate_local_artifacts(manifest_path: Path, stage: str) -> dict[str, str]:
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    verified: dict[str, str] = {}
    for artifact in manifest.artifacts:
        if stage not in artifact.required_for_stages:
            continue
        destination = _safe_destination(artifact.destination)
        if not destination.is_file():
            raise ValueError(f"required staged artifact is missing: {artifact.name}")
        actual = file_sha256(destination)
        if actual != artifact.sha256:
            raise ValueError(f"staged artifact hash mismatch: {artifact.name}")
        if artifact.restricted_source:
            expected = f"APPROVE_RESTRICTED_SOURCE:{artifact.sha256}"
            if os.environ.get("FRAUD_RESTRICTED_SOURCE_APPROVAL") != expected:
                raise ValueError("restricted source staging lacks exact approval")
        verified[artifact.name] = actual
    if stage != "schema-migration" and not verified:
        raise ValueError("no approved artifacts are defined for the requested stage")
    return verified


def _extract_approved_archives(manifest_path: Path, stage: str) -> None:
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest.artifacts:
        if stage not in artifact.required_for_stages or artifact.extract_to is None:
            continue
        archive = _safe_destination(artifact.destination)
        destination = _safe_destination(artifact.extract_to)
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, mode="r:*") as bundle:
            for member in bundle.getmembers():
                member_path = (destination / member.name).resolve()
                root = destination.resolve()
                if member_path != root and root not in member_path.parents:
                    raise ValueError(f"artifact archive contains an unsafe path: {artifact.name}")
            bundle.extractall(destination, filter="data")


def stage_artifacts(stage: str) -> tuple[Path, dict[str, str]]:
    local_manifest = os.environ.get("FRAUD_ARTIFACT_MANIFEST_PATH")
    if local_manifest:
        path = Path(local_manifest)
        verified = validate_local_artifacts(path, stage)
        _extract_approved_archives(path, stage)
        return path, verified
    account_url = os.environ.get("FRAUD_STAGING_ACCOUNT_URL")
    container_name = os.environ.get("FRAUD_STAGING_CONTAINER")
    manifest_blob = os.environ.get("FRAUD_ARTIFACT_MANIFEST_BLOB")
    manifest_sha256 = os.environ.get("FRAUD_ARTIFACT_MANIFEST_SHA256")
    if not all((account_url, container_name, manifest_blob, manifest_sha256)):
        raise ValueError("private artifact staging configuration is incomplete")
    assert account_url is not None
    assert container_name is not None
    assert manifest_blob is not None
    assert manifest_sha256 is not None
    credential = DefaultAzureCredential(
        managed_identity_client_id=os.environ.get("FRAUD_AZURE_MANAGED_IDENTITY_CLIENT_ID"),
        exclude_interactive_browser_credential=True,
    )
    container = BlobServiceClient(account_url, credential=credential).get_container_client(
        container_name
    )
    manifest_path = WORK_ROOT / "config/artifact-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(container.download_blob(manifest_blob).readall())
    if file_sha256(manifest_path) != manifest_sha256:
        raise ValueError("artifact manifest hash mismatch")
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest.artifacts:
        if stage not in artifact.required_for_stages:
            continue
        destination = _safe_destination(artifact.destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(container.download_blob(artifact.blob_name).readall())
    verified = validate_local_artifacts(manifest_path, stage)
    _extract_approved_archives(manifest_path, stage)
    return manifest_path, verified


def _staging_container() -> Any:
    account_url = os.environ.get("FRAUD_STAGING_ACCOUNT_URL")
    container_name = os.environ.get("FRAUD_STAGING_CONTAINER")
    if not account_url or not container_name:
        raise ValueError("private artifact staging configuration is incomplete")
    credential = DefaultAzureCredential(
        managed_identity_client_id=os.environ.get("FRAUD_AZURE_MANAGED_IDENTITY_CLIENT_ID"),
        exclude_interactive_browser_credential=True,
    )
    return BlobServiceClient(account_url, credential=credential).get_container_client(
        container_name
    )


def upload_stage_outputs(stage: str) -> dict[str, str]:
    output_roots = {
        "behavioral-embeddings": [WORK_ROOT / "data/processed"],
        "feature-model": [WORK_ROOT / "data/processed", WORK_ROOT / "artifacts/models/releases"],
        "incremental": [WORK_ROOT / "data/processed/phase8", WORK_ROOT / "artifacts/incremental"],
        "evaluation": [WORK_ROOT / "artifacts/incremental"],
    }
    roots = output_roots.get(stage, [])
    files = sorted(
        path
        for root in roots
        if root.exists()
        for path in (root.rglob("*") if root.is_dir() else [root])
        if path.is_file()
    )
    if not files:
        return {}
    container = _staging_container()
    uploaded: dict[str, str] = {}
    for path in files:
        relative = path.resolve().relative_to(WORK_ROOT.resolve()).as_posix()
        blob_name = f"generated/{stage}/{relative}"
        digest = file_sha256(path)
        blob = container.get_blob_client(blob_name)
        if blob.exists():
            existing = hashlib.sha256(blob.download_blob().readall()).hexdigest()
            if existing != digest:
                raise ValueError(f"immutable staged output conflicts: {relative}")
        else:
            with path.open("rb") as stream:
                blob.upload_blob(stream, overwrite=False, metadata={"sha256": digest})
        uploaded[blob_name] = digest
    return uploaded


@contextmanager
def pipeline_lock() -> Iterator[None]:
    dsn = os.environ.get("FRAUD_POSTGRES_DSN")
    if not dsn:
        raise ValueError("FRAUD_POSTGRES_DSN is required")
    with psycopg.connect(dsn, connect_timeout=10) as connection:
        acquired = connection.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_ID,)).fetchone()
        if acquired is None or acquired[0] is not True:
            raise ValueError("another private pipeline stage is already running")
        try:
            yield
        finally:
            connection.execute("SELECT pg_advisory_unlock(%s)", (LOCK_ID,))


def _run(command: list[str], timeout_seconds: int) -> dict[str, Any]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        cwd="/app",
    )
    if result.returncode != 0:
        raise ValueError(f"pipeline command failed with exit code {result.returncode}")
    output = result.stdout.strip()
    try:
        payload: Any = json.loads(output) if output else {"status": "complete"}
    except json.JSONDecodeError:
        payload = {"status": "complete", "output": output[-2000:]}
    return {"command": command[0], "result": payload}


def _chunk_index() -> int:
    value = os.environ.get("FRAUD_PIPELINE_CHUNK_INDEX")
    if value is None:
        raise ValueError("FRAUD_PIPELINE_CHUNK_INDEX is required")
    index = int(value)
    if index < 0:
        raise ValueError("chunk index must be nonnegative")
    return index


def _commands(stage: str) -> list[list[str]]:
    source = str(SOURCE)
    manifest = str(MANIFEST)
    if stage == "schema-migration":
        return [["alembic", "upgrade", "head"]]
    if stage == "initial-relational-load":
        return [[*INGEST_COMMAND, "load-initial", "--source", source, "--manifest", manifest]]
    if stage == "raw-projection":
        return [
            [*INGEST_COMMAND, "cosmos-project-initial", "--source", source, "--manifest", manifest]
        ]
    if stage == "graph-projection":
        return [
            [*INGEST_COMMAND, "gremlin-project-initial", "--source", source, "--manifest", manifest]
        ]
    if stage == "behavioral-embeddings":
        return [
            [*INGEST_COMMAND, "templates-build", "--source", source, "--manifest", manifest],
            [*INGEST_COMMAND, "embeddings-generate", "--project"],
            [*INGEST_COMMAND, "embeddings-verify", "--source", source, "--manifest", manifest],
        ]
    if stage == "graph-snapshot":
        cutoff = os.environ.get("FRAUD_GRAPH_CUTOFF_STEP")
        if not cutoff:
            raise ValueError("FRAUD_GRAPH_CUTOFF_STEP is required")
        return [
            [
                *INGEST_COMMAND,
                "graph-analyze",
                "--source",
                source,
                "--manifest",
                manifest,
                "--cutoff-step",
                cutoff,
            ]
        ]
    if stage == "feature-model":
        return [
            [
                *MODEL_COMMAND,
                "features-build",
                "--source",
                source,
                "--manifest",
                manifest,
                "--embeddings",
                str(EMBEDDINGS),
            ],
            [*MODEL_COMMAND, "train-release"],
        ]
    if stage == "incremental":
        return [
            [
                *INCREMENTAL_COMMAND,
                "run-chunk",
                "--source",
                source,
                "--manifest",
                manifest,
                "--chunk-index",
                str(_chunk_index()),
            ]
        ]
    if stage == "reconciliation":
        return [
            [
                *INCREMENTAL_COMMAND,
                "reconcile",
                "--source",
                source,
                "--manifest",
                manifest,
                "--chunk-index",
                str(_chunk_index()),
            ]
        ]
    if stage == "evaluation":
        return [[*INCREMENTAL_COMMAND, "evaluate", "--source", source, "--manifest", manifest]]
    raise ValueError("publication uses the guarded publication workflow")


def _publication_commands(timeout_seconds: int) -> list[dict[str, Any]]:
    batch_id = os.environ.get("FRAUD_PUBLICATION_BATCH_ID", "")
    confirmation = os.environ.get("FRAUD_PUBLICATION_CONFIRMATION", "")
    if not batch_id or confirmation != f"PUBLISH:{batch_id}":
        raise ValueError("publication requires an exact batch identifier and confirmation")
    index = _chunk_index()
    plan = _run(
        [*INCREMENTAL_COMMAND, "plan", "--source", str(SOURCE), "--manifest", str(MANIFEST)],
        timeout_seconds,
    )["result"]
    if not isinstance(plan, dict):
        raise ValueError("incremental plan is invalid")
    chunks = plan.get("chunks", [])
    if not isinstance(chunks, list) or index >= len(chunks):
        raise ValueError("publication chunk is absent from the approved plan")
    chunk = chunks[index]
    if not isinstance(chunk, dict) or chunk.get("batch_id") != batch_id:
        raise ValueError("publication batch does not match the approved chunk plan")
    reconcile = _run(
        [
            *INCREMENTAL_COMMAND,
            "reconcile",
            "--source",
            str(SOURCE),
            "--manifest",
            str(MANIFEST),
            "--chunk-index",
            str(index),
        ],
        timeout_seconds,
    )
    result = reconcile["result"]
    if not isinstance(result, dict) or result.get("publication_eligible") is not True:
        raise ValueError("reconciliation did not prove publication eligibility")
    publish = _run(
        [
            *INCREMENTAL_COMMAND,
            "publish",
            "--source",
            str(SOURCE),
            "--manifest",
            str(MANIFEST),
            "--chunk-index",
            str(index),
        ],
        timeout_seconds,
    )
    return [reconcile, publish]


def run_stage(stage: str, timeout_seconds: int) -> dict[str, Any]:
    if stage not in PIPELINE_STAGES:
        raise ValueError("unknown private pipeline stage")
    verified: dict[str, str] = {}
    if stage != "schema-migration":
        _manifest_path, verified = stage_artifacts(stage)
    with pipeline_lock():
        if stage == "publication":
            results = _publication_commands(timeout_seconds)
        else:
            results = [_run(command, timeout_seconds) for command in _commands(stage)]
        staged_outputs = upload_stage_outputs(stage)
    return {
        "status": "complete",
        "stage": stage,
        "verified_artifact_hashes": verified,
        "results": results,
        "staged_output_hashes": staged_outputs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=PIPELINE_STAGES)
    parser.add_argument("--timeout-seconds", type=int, default=14_400)
    args = parser.parse_args(argv)
    try:
        result = run_stage(args.stage, args.timeout_seconds)
    except (OSError, ValueError, subprocess.SubprocessError, psycopg.Error) as exc:
        print(json.dumps({"status": "failed", "stage": args.stage, "reason": str(exc)}))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
