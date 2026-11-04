"""Safe Gate A/Gate B operations command surface."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from fraud_detection.deployment.preflight import write_preflight_report
from fraud_detection.reporting.contracts import contract_drift, write_contracts

ROOT = Path(__file__).resolve().parents[3]
TARGET_PLATFORM = "linux/amd64"
STAGE_JOB_SUFFIX = {
    "schema-migration": "migrate",
    "initial-relational-load": "initial-load",
    "raw-projection": "raw-project",
    "graph-projection": "graph-project",
    "behavioral-embeddings": "embeddings",
    "graph-snapshot": "graph-snapshot",
    "feature-model": "feature-model",
    "incremental": "incremental",
    "reconciliation": "reconcile",
    "evaluation": "evaluate",
    "publication": "publish",
}
RESTRICTED_SOURCE_STAGES = frozenset(STAGE_JOB_SUFFIX) - {"schema-migration"}
SAFE_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
SAFE_BATCH_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,254}[A-Za-z0-9])?$")


def _run(command: list[str]) -> int:
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _validate_name(value: str, label: str) -> str:
    if not SAFE_NAME.fullmatch(value):
        raise RuntimeError(f"{label} contains unsupported characters")
    return value


def _job_name(environment: str, unique_suffix: str, stage: str) -> str:
    _validate_name(environment, "environment")
    _validate_name(unique_suffix, "unique suffix")
    if stage not in STAGE_JOB_SUFFIX:
        raise RuntimeError("stage is not in the approved pipeline map")
    prefix = f"fd-{environment}-{unique_suffix}"[:15].rstrip("-")
    name = f"{prefix}-{STAGE_JOB_SUFFIX[stage]}"
    if len(name) >= 32 or not SAFE_NAME.fullmatch(name):
        raise RuntimeError("derived Container Apps job name is invalid")
    return name


def _inspect_target_image(docker: str, image: str) -> dict[str, object]:
    result = _capture([docker, "image", "inspect", image])
    if result.returncode != 0:
        raise RuntimeError("image inspection failed")
    values = json.loads(result.stdout)
    if not isinstance(values, list) or not values or not isinstance(values[0], dict):
        raise RuntimeError("image inspection returned an invalid payload")
    details = values[0]
    user = str(details.get("Config", {}).get("User", ""))
    if details.get("Os") != "linux" or details.get("Architecture") != "amd64":
        raise RuntimeError("deployment image must be linux/amd64")
    if not user or user in {"0", "0:0", "root", "root:root"}:
        raise RuntimeError("deployment image must use a non-root runtime user")
    return {
        "id": details.get("Id"),
        "os": details.get("Os"),
        "architecture": details.get("Architecture"),
        "user": user,
        "size": details.get("Size"),
    }


def _staging_network_check(account: str, private_cidr: str) -> dict[str, object]:
    _validate_name(account, "staging account")
    network = ipaddress.ip_network(private_cidr, strict=False)
    hostname = f"{account}.blob.core.windows.net"
    try:
        addresses = sorted(
            {item[4][0] for item in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)}
        )
    except OSError as exc:
        raise RuntimeError("staging Blob private DNS resolution failed") from exc
    if not addresses:
        raise RuntimeError("staging Blob private DNS returned no addresses")
    parsed = [ipaddress.ip_address(address) for address in addresses]
    if any(not address.is_private or address not in network for address in parsed):
        raise RuntimeError("staging Blob endpoint did not resolve through the private endpoint")
    try:
        urllib.request.urlopen(f"https://{hostname}/?comp=list", timeout=10)
    except urllib.error.HTTPError as exc:
        if exc.code not in {400, 401, 403}:
            raise RuntimeError("staging Blob private HTTPS check failed") from exc
    except OSError as exc:
        raise RuntimeError("staging Blob private HTTPS endpoint is unreachable") from exc
    return {"hostname": hostname, "private_addresses": addresses, "https": "reachable"}


def _restricted_source_value(sha256: str | None, confirmation: str | None) -> str:
    if sha256 is None or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise RuntimeError("restricted-source SHA-256 is required")
    expected = f"APPROVE_RESTRICTED_SOURCE:{sha256}"
    if confirmation != expected:
        raise RuntimeError(f"restricted source requires exact confirmation {expected}")
    return expected


def _stage_overrides(
    stage: str,
    *,
    graph_cutoff_step: int | None = None,
    chunk_index: int | None = None,
    restricted_source_sha256: str | None = None,
    restricted_source_confirmation: str | None = None,
) -> list[str]:
    if stage not in STAGE_JOB_SUFFIX:
        raise RuntimeError("stage is not in the approved pipeline map")
    values: list[str] = []
    if stage == "publication":
        raise RuntimeError("start-stage cannot start publication")
    if stage == "graph-snapshot":
        if graph_cutoff_step is None or graph_cutoff_step < 1:
            raise RuntimeError("graph-snapshot requires a positive cutoff step")
        values.append(f"FRAUD_GRAPH_CUTOFF_STEP={graph_cutoff_step}")
    if stage in {"incremental", "reconciliation"}:
        if chunk_index is None or chunk_index < 0:
            raise RuntimeError(f"{stage} requires a nonnegative chunk index")
        values.append(f"FRAUD_PIPELINE_CHUNK_INDEX={chunk_index}")
    if stage in RESTRICTED_SOURCE_STAGES:
        values.append(
            "FRAUD_RESTRICTED_SOURCE_APPROVAL="
            + _restricted_source_value(restricted_source_sha256, restricted_source_confirmation)
        )
    return values


def _job_start_command(
    az: str,
    *,
    resource_group: str,
    environment: str,
    unique_suffix: str,
    stage: str,
    env_vars: list[str],
) -> list[str]:
    _validate_name(resource_group, "resource group")
    job_name = _job_name(environment, unique_suffix, stage)
    command = [
        az,
        "containerapp",
        "job",
        "start",
        "--resource-group",
        resource_group,
        "--name",
        job_name,
        "--container-name",
        "worker",
    ]
    if env_vars:
        command.extend(["--env-vars", *env_vars])
    command.extend(["--query", "name", "--output", "tsv"])
    return command


def _start_job(command: list[str]) -> str:
    result = _capture(command)
    if result.returncode != 0:
        raise RuntimeError("Container Apps job start failed")
    execution_name = result.stdout.strip()
    _validate_name(execution_name, "execution name")
    print(json.dumps({"execution_name": execution_name}))
    return execution_name


def _upload_and_verify_blob(
    az: str,
    *,
    account: str,
    container: str,
    blob_name: str,
    path: Path,
    expected_sha256: str,
) -> None:
    base = [
        "--auth-mode",
        "login",
        "--account-name",
        account,
        "--container-name",
        container,
        "--name",
        blob_name,
    ]
    upload = [
        az,
        "storage",
        "blob",
        "upload",
        *base,
        "--file",
        str(path),
        "--metadata",
        f"sha256={expected_sha256}",
        "--overwrite",
        "false",
        "--output",
        "none",
    ]
    if _run(upload) != 0:
        raise RuntimeError(f"artifact upload failed: {blob_name}")
    metadata = _capture(
        [
            az,
            "storage",
            "blob",
            "show",
            *base,
            "--query",
            "metadata.sha256",
            "--output",
            "tsv",
        ]
    )
    if metadata.returncode != 0 or metadata.stdout.strip() != expected_sha256:
        raise RuntimeError(f"artifact metadata verification failed: {blob_name}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temporary:
            temporary_path = Path(temporary.name)
        download = [
            az,
            "storage",
            "blob",
            "download",
            *base,
            "--file",
            str(temporary_path),
            "--overwrite",
            "true",
            "--output",
            "none",
        ]
        if _run(download) != 0 or _file_sha256(temporary_path) != expected_sha256:
            raise RuntimeError(f"artifact content verification failed: {blob_name}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_immutable_deployment_images(parameters_path: Path) -> None:
    document = json.loads(parameters_path.read_text(encoding="utf-8"))
    parameters = document.get("parameters", {})
    if not isinstance(parameters, dict):
        raise RuntimeError("deployment parameters are invalid")
    checks = (
        ("deployContainerApp", "containerImageDigest"),
        ("deployPipelineJobs", "workerImageDigest"),
    )
    digest_pattern = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")
    for toggle, image_parameter in checks:
        enabled = parameters.get(toggle, {}).get("value") is True
        image = parameters.get(image_parameter, {}).get("value", "")
        if enabled and (not isinstance(image, str) or not digest_pattern.fullmatch(image)):
            raise RuntimeError(f"{image_parameter} must be an immutable sha256 image reference")


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None and name == "docker":
        candidate = Path("/Applications/Docker.app/Contents/Resources/bin/docker")
        if candidate.exists():
            return str(candidate)
    if path is None:
        raise RuntimeError(f"required tool is unavailable: {name}")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--output", type=Path, default=ROOT / "deployment/preflight-report.json")
    sub.add_parser("bicep-validate")
    sub.add_parser("cost-review")
    build = sub.add_parser("container-build")
    build.add_argument("--tag", default="fraud-investigator:gate-a")
    inspect = sub.add_parser("container-inspect")
    inspect.add_argument("--tag", default="fraud-investigator:gate-a")
    worker_build = sub.add_parser("worker-build")
    worker_build.add_argument("--tag", default="fraud-worker:gate-a")
    worker_inspect = sub.add_parser("worker-inspect")
    worker_inspect.add_argument("--tag", default="fraud-worker:gate-a")
    image_push = sub.add_parser("image-push")
    image_push.add_argument("--image", required=True)
    image_push.add_argument("--confirm", required=True)
    contracts = sub.add_parser("contracts")
    contracts.add_argument("mode", choices=("generate", "check"))
    health = sub.add_parser("health")
    health.add_argument("--base-url", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--base-url", required=True)
    smoke.add_argument("--token", default=None)
    for name in ("powerbi-validate", "copilot-validate"):
        sub.add_parser(name)
    deploy = sub.add_parser("deploy")
    deploy.add_argument("--subscription", required=True)
    deploy.add_argument("--location", required=True)
    deploy.add_argument("--parameters", type=Path, required=True)
    deploy.add_argument("--postgres-password-env", default="FRAUD_POSTGRES_ADMIN_PASSWORD")
    deploy.add_argument("--confirm", required=True)
    api = sub.add_parser("api-deploy")
    api.add_argument("--subscription", required=True)
    api.add_argument("--location", required=True)
    api.add_argument("--parameters", type=Path, required=True)
    api.add_argument("--postgres-password-env", default="FRAUD_POSTGRES_ADMIN_PASSWORD")
    api.add_argument("--confirm", required=True)
    jobs = sub.add_parser("jobs-update")
    jobs.add_argument("--subscription", required=True)
    jobs.add_argument("--location", required=True)
    jobs.add_argument("--parameters", type=Path, required=True)
    jobs.add_argument("--postgres-password-env", default="FRAUD_POSTGRES_ADMIN_PASSWORD")
    jobs.add_argument("--confirm", required=True)
    backup = sub.add_parser("backup")
    backup.add_argument("--dsn-env", default="FRAUD_POSTGRES_DSN")
    backup.add_argument("--output", type=Path, required=True)
    artifacts_validate = sub.add_parser("artifacts-validate")
    artifacts_validate.add_argument("--manifest", type=Path, required=True)
    artifacts_validate.add_argument("--root", type=Path, required=True)
    artifacts_upload = sub.add_parser("artifacts-upload")
    artifacts_upload.add_argument("--manifest", type=Path, required=True)
    artifacts_upload.add_argument("--root", type=Path, required=True)
    artifacts_upload.add_argument("--account", required=True)
    artifacts_upload.add_argument("--container", required=True)
    artifacts_upload.add_argument("--manifest-blob", default="deployment/artifact-manifest.json")
    artifacts_upload.add_argument("--private-cidr", default="10.42.3.0/24")
    artifacts_upload.add_argument("--confirm", required=True)
    staging_check = sub.add_parser("staging-network-check")
    staging_check.add_argument("--account", required=True)
    staging_check.add_argument("--private-cidr", default="10.42.3.0/24")
    start = sub.add_parser("start-stage")
    start.add_argument("--resource-group", required=True)
    start.add_argument("--environment", required=True)
    start.add_argument("--unique-suffix", required=True)
    start.add_argument("--stage", choices=tuple(STAGE_JOB_SUFFIX), required=True)
    start.add_argument("--graph-cutoff-step", type=int)
    start.add_argument("--chunk-index", type=int)
    start.add_argument("--restricted-source-sha256")
    start.add_argument("--restricted-source-confirmation")
    wait = sub.add_parser("wait-stage")
    wait.add_argument("--resource-group", required=True)
    wait.add_argument("--environment", required=True)
    wait.add_argument("--unique-suffix", required=True)
    wait.add_argument("--stage", choices=tuple(STAGE_JOB_SUFFIX), required=True)
    wait.add_argument("--execution-name", required=True)
    wait.add_argument("--timeout-seconds", type=int, default=3600)
    wait.add_argument("--poll-seconds", type=int, default=15)
    job_logs = sub.add_parser("job-logs")
    job_logs.add_argument("--resource-group", required=True)
    job_logs.add_argument("--environment", required=True)
    job_logs.add_argument("--unique-suffix", required=True)
    job_logs.add_argument("--stage", choices=tuple(STAGE_JOB_SUFFIX), required=True)
    job_logs.add_argument("--execution-name", required=True)
    preview = sub.add_parser("publication-preview")
    preview.add_argument("--resource-group", required=True)
    preview.add_argument("--environment", required=True)
    preview.add_argument("--unique-suffix", required=True)
    preview.add_argument("--chunk-index", type=int, required=True)
    preview.add_argument("--restricted-source-sha256", required=True)
    preview.add_argument("--restricted-source-confirmation", required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--resource-group", required=True)
    publish.add_argument("--environment", required=True)
    publish.add_argument("--unique-suffix", required=True)
    publish.add_argument("--batch-id", required=True)
    publish.add_argument("--chunk-index", type=int, required=True)
    publish.add_argument("--restricted-source-sha256", required=True)
    publish.add_argument("--restricted-source-confirmation", required=True)
    publish.add_argument("--confirm", required=True)
    counts = sub.add_parser("record-counts")
    counts.add_argument("--dsn-env", default="FRAUD_POSTGRES_DSN")
    sub.add_parser("secret-scan")
    preview = sub.add_parser("teardown-preview")
    preview.add_argument("--environment", required=True)
    preview.add_argument("--resource-group", required=True)
    teardown = sub.add_parser("teardown")
    teardown.add_argument("--environment", required=True)
    teardown.add_argument("--resource-group", required=True)
    teardown.add_argument("--confirm", required=True)
    return parser


def _get_json(url: str, token: str | None = None) -> tuple[int, str]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=10
        ) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation == "preflight":
        report = write_preflight_report(args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.operation == "bicep-validate":
        bicep = _require_tool("bicep")
        return _run([bicep, "build", str(ROOT / "infra/main.bicep"), "--stdout"])
    if args.operation == "cost-review":
        print((ROOT / "deployment/cost-review.json").read_text(encoding="utf-8"))
        return 0
    if args.operation == "container-build":
        docker = _require_tool("docker")
        return _run(
            [
                docker,
                "buildx",
                "build",
                "--platform",
                TARGET_PLATFORM,
                "--load",
                "--pull=false",
                "--tag",
                args.tag,
                ".",
            ]
        )
    if args.operation == "container-inspect":
        docker = _require_tool("docker")
        print(json.dumps(_inspect_target_image(docker, args.tag), sort_keys=True))
        return _run([docker, "history", "--no-trunc", args.tag])
    if args.operation == "worker-build":
        docker = _require_tool("docker")
        return _run(
            [
                docker,
                "buildx",
                "build",
                "--platform",
                TARGET_PLATFORM,
                "--load",
                "--file",
                "Dockerfile.worker",
                "--tag",
                args.tag,
                ".",
            ]
        )
    if args.operation == "worker-inspect":
        docker = _require_tool("docker")
        print(json.dumps(_inspect_target_image(docker, args.tag), sort_keys=True))
        return _run([docker, "history", "--no-trunc", args.tag])
    if args.operation == "image-push":
        if args.confirm != f"PUSH:{args.image}":
            raise RuntimeError(f"image push requires --confirm PUSH:{args.image}")
        docker = _require_tool("docker")
        _inspect_target_image(docker, args.image)
        if _run([docker, "push", args.image]) != 0:
            return 1
        result = _capture([docker, "image", "inspect", args.image])
        values = json.loads(result.stdout)
        repo_digests = values[0].get("RepoDigests", []) if values else []
        if not repo_digests or not any("@sha256:" in value for value in repo_digests):
            raise RuntimeError("pushed image lacks an immutable repository digest")
        print(json.dumps({"image": args.image, "repo_digests": repo_digests}))
        return 0
    if args.operation == "contracts":
        if args.mode == "generate":
            write_contracts()
            return 0
        return 1 if contract_drift() else 0
    if args.operation in {"health", "smoke"}:
        live_status, _ = _get_json(f"{args.base_url.rstrip('/')}/health/live")
        ready_status, _ = _get_json(f"{args.base_url.rstrip('/')}/health/ready")
        print(json.dumps({"live": live_status, "ready": ready_status}, sort_keys=True))
        if args.operation == "health":
            return 0 if live_status == 200 and ready_status in {200, 503} else 1
        schema_status, schema = _get_json(f"{args.base_url.rstrip('/')}/openapi.json")
        summary_status, _ = _get_json(
            f"{args.base_url.rstrip('/')}/v1/dashboard/summary", args.token
        )
        safe = "access_token" not in schema.lower() and "client_secret" not in schema.lower()
        print(json.dumps({"openapi": schema_status, "v1_summary": summary_status, "safe": safe}))
        return 0 if schema_status == 200 and safe else 1
    if args.operation == "powerbi-validate":
        return 0 if not contract_drift() else 1
    if args.operation == "copilot-validate":
        return 0 if not contract_drift() else 1
    if args.operation in {"deploy", "api-deploy", "jobs-update"}:
        if args.confirm != "DEPLOY":
            raise RuntimeError("deployment requires --confirm DEPLOY")
        _validate_immutable_deployment_images(args.parameters)
        postgres_password = os.environ.get(args.postgres_password_env)
        if not postgres_password:
            raise RuntimeError(
                "PostgreSQL administrator password is missing from the configured "
                "environment variable"
            )
        az = _require_tool("az")
        secret_parameters = {
            "$schema": (
                "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#"
            ),
            "contentVersion": "1.0.0.0",
            "parameters": {"postgresAdministratorPassword": {"value": postgres_password}},
        }
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".json", delete=False
            ) as temporary:
                json.dump(secret_parameters, temporary)
                temporary_path = Path(temporary.name)
            temporary_path.chmod(0o600)
            return _run(
                [
                    az,
                    "deployment",
                    "sub",
                    "create",
                    "--subscription",
                    args.subscription,
                    "--location",
                    args.location,
                    "--template-file",
                    str(ROOT / "infra/main.bicep"),
                    "--parameters",
                    f"@{args.parameters}",
                    f"@{temporary_path}",
                ]
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    if args.operation == "backup":
        dsn = os.environ.get(args.dsn_env)
        if not dsn:
            raise RuntimeError(f"database DSN is missing from environment variable {args.dsn_env}")
        pg_dump = _require_tool("pg_dump")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        return _run([pg_dump, "--dbname", dsn, "--format", "custom", "--file", str(args.output)])
    if args.operation in {"artifacts-validate", "artifacts-upload"}:
        from fraud_detection.deployment.worker import ArtifactManifest

        manifest = ArtifactManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
        for artifact in manifest.artifacts:
            path = (args.root / artifact.destination).resolve()
            if not path.is_file() or _file_sha256(path) != artifact.sha256:
                raise RuntimeError(f"artifact is missing or mismatched: {artifact.name}")
        if args.operation == "artifacts-validate":
            print(
                json.dumps(
                    {
                        "artifact_count": len(manifest.artifacts),
                        "manifest_sha256": _file_sha256(args.manifest),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.confirm != "UPLOAD":
            raise RuntimeError("artifact upload requires --confirm UPLOAD")
        _staging_network_check(args.account, args.private_cidr)
        az = _require_tool("az")
        for artifact in manifest.artifacts:
            path = (args.root / artifact.destination).resolve()
            _upload_and_verify_blob(
                az,
                account=args.account,
                container=args.container,
                blob_name=artifact.blob_name,
                path=path,
                expected_sha256=artifact.sha256,
            )
        manifest_path = args.manifest.resolve()
        _upload_and_verify_blob(
            az,
            account=args.account,
            container=args.container,
            blob_name=args.manifest_blob,
            path=manifest_path,
            expected_sha256=_file_sha256(manifest_path),
        )
        return 0
    if args.operation == "staging-network-check":
        print(json.dumps(_staging_network_check(args.account, args.private_cidr), sort_keys=True))
        return 0
    if args.operation == "start-stage":
        env_vars = _stage_overrides(
            args.stage,
            graph_cutoff_step=args.graph_cutoff_step,
            chunk_index=args.chunk_index,
            restricted_source_sha256=args.restricted_source_sha256,
            restricted_source_confirmation=args.restricted_source_confirmation,
        )
        az = _require_tool("az")
        _start_job(
            _job_start_command(
                az,
                resource_group=args.resource_group,
                environment=args.environment,
                unique_suffix=args.unique_suffix,
                stage=args.stage,
                env_vars=env_vars,
            )
        )
        return 0
    if args.operation == "publication-preview":
        if args.chunk_index < 0:
            raise RuntimeError("publication-preview requires a nonnegative chunk index")
        approval = _restricted_source_value(
            args.restricted_source_sha256, args.restricted_source_confirmation
        )
        az = _require_tool("az")
        _start_job(
            _job_start_command(
                az,
                resource_group=args.resource_group,
                environment=args.environment,
                unique_suffix=args.unique_suffix,
                stage="reconciliation",
                env_vars=[
                    f"FRAUD_PIPELINE_CHUNK_INDEX={args.chunk_index}",
                    f"FRAUD_RESTRICTED_SOURCE_APPROVAL={approval}",
                ],
            )
        )
        return 0
    if args.operation == "wait-stage":
        az = _require_tool("az")
        _validate_name(args.execution_name, "execution name")
        job_name = _job_name(args.environment, args.unique_suffix, args.stage)
        deadline = time.monotonic() + args.timeout_seconds
        while time.monotonic() < deadline:
            execution_result = subprocess.run(
                [
                    az,
                    "containerapp",
                    "job",
                    "execution",
                    "list",
                    "--resource-group",
                    args.resource_group,
                    "--name",
                    job_name,
                    "--query",
                    f"[?name=='{args.execution_name}'] | [0].properties.status",
                    "--output",
                    "tsv",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            status = execution_result.stdout.strip()
            if status in {"Succeeded", "Failed"}:
                print(
                    json.dumps(
                        {
                            "job": job_name,
                            "execution_name": args.execution_name,
                            "status": status,
                        }
                    )
                )
                return 0 if status == "Succeeded" else 1
            time.sleep(max(1, min(args.poll_seconds, 60)))
        print(
            json.dumps(
                {
                    "job": job_name,
                    "execution_name": args.execution_name,
                    "status": "TimedOut",
                }
            )
        )
        return 1
    if args.operation == "job-logs":
        az = _require_tool("az")
        _validate_name(args.execution_name, "execution name")
        job_name = _job_name(args.environment, args.unique_suffix, args.stage)
        return _run(
            [
                az,
                "containerapp",
                "job",
                "logs",
                "show",
                "--resource-group",
                args.resource_group,
                "--name",
                job_name,
                "--execution",
                args.execution_name,
                "--container",
                "worker",
                "--follow",
                "false",
                "--tail",
                "200",
            ]
        )
    if args.operation == "record-counts":
        import psycopg

        dsn = os.environ.get(args.dsn_env)
        if not dsn:
            raise RuntimeError("record-counts requires a database DSN environment variable")
        queries = {
            "total_transactions": 'SELECT count(*) FROM "transaction"',
            "test_transactions": "SELECT count(*) FROM \"transaction\" WHERE data_split = 'test'",
            "scores": "SELECT count(*) FROM transaction_score",
            "flags": "SELECT count(*) FROM fraud_flag",
            "test_evaluations": "SELECT count(*) FROM model_evaluation WHERE split_name = 'test'",
            "validation_evaluations": (
                "SELECT count(*) FROM model_evaluation WHERE split_name = 'validation'"
            ),
            "pending_batches": "SELECT count(*) FROM ingestion_batch WHERE status = 'pending'",
            "published_batches": "SELECT count(*) FROM ingestion_batch WHERE status = 'published'",
        }
        with psycopg.connect(dsn, connect_timeout=10) as connection:
            values = {}
            for name, query in queries.items():
                row = connection.execute(query).fetchone()
                if row is None:
                    raise RuntimeError(f"count query returned no row: {name}")
                values[name] = row[0]
        print(json.dumps(values, sort_keys=True))
        return 0
    if args.operation == "publish":
        if not SAFE_BATCH_ID.fullmatch(args.batch_id):
            raise RuntimeError("publication batch ID contains unsupported characters")
        if args.chunk_index < 0:
            raise RuntimeError("publication requires a nonnegative chunk index")
        expected = f"PUBLISH:{args.batch_id}"
        if args.confirm != expected:
            raise RuntimeError(f"publication requires --confirm {expected}")
        approval = _restricted_source_value(
            args.restricted_source_sha256, args.restricted_source_confirmation
        )
        az = _require_tool("az")
        _start_job(
            _job_start_command(
                az,
                resource_group=args.resource_group,
                environment=args.environment,
                unique_suffix=args.unique_suffix,
                stage="publication",
                env_vars=[
                    f"FRAUD_PUBLICATION_BATCH_ID={args.batch_id}",
                    f"FRAUD_PUBLICATION_CONFIRMATION={expected}",
                    f"FRAUD_PIPELINE_CHUNK_INDEX={args.chunk_index}",
                    f"FRAUD_RESTRICTED_SOURCE_APPROVAL={approval}",
                ],
            )
        )
        return 0
    if args.operation == "teardown-preview":
        az = _require_tool("az")
        print(json.dumps({"environment": args.environment, "resource_group": args.resource_group}))
        return _run(
            [az, "resource", "list", "--resource-group", args.resource_group, "--output", "table"]
        )
    if args.operation == "teardown":
        expected = f"DELETE:{args.resource_group}:{args.environment}"
        if args.confirm != expected:
            raise RuntimeError(f"teardown requires --confirm {expected}")
        az = _require_tool("az")
        return _run([az, "group", "delete", "--name", args.resource_group, "--yes", "--no-wait"])
    if args.operation == "secret-scan":
        candidates = [
            ROOT / "Dockerfile",
            ROOT / "Dockerfile.worker",
            ROOT / ".dockerignore",
            *sorted((ROOT / "infra").rglob("*.bicep")),
            *sorted((ROOT / "infra").rglob("*.json")),
            *sorted((ROOT / "deployment").glob("*.in")),
            *sorted((ROOT / "deployment").glob("*.json")),
            *sorted((ROOT / "contracts/copilot").glob("*.json")),
        ]
        patterns = {
            "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
            "azure_account_key": re.compile(r"AccountKey=[A-Za-z0-9+/=]{16,}"),
            "bearer_token": re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}"),
            "postgres_password_url": re.compile(
                r"postgres(?:ql)?(?:\+psycopg)?://[^:\s'\"]+:"
                r"(?!\$\{)[^@\s'\"]{8,}@"
            ),
            "literal_client_secret": re.compile(
                r"(?i)client[_-]?secret\s*[:=]\s*['\"][^<$\[{\s]{8,}"
            ),
        }
        findings: list[str] = []
        for path in candidates:
            content = path.read_text(encoding="utf-8")
            for name, pattern in patterns.items():
                if pattern.search(content):
                    findings.append(f"{path.relative_to(ROOT)}:{name}")
        if findings:
            print("\n".join(findings), file=sys.stderr)
            return 1
        print(f"secret scan passed for {len(candidates)} deployment files")
        return 0
    print("unsupported operation", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
