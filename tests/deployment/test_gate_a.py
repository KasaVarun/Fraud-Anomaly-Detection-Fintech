"""Gate A infrastructure, preflight, and destructive-command safety tests."""

import json
import subprocess
import tarfile
import urllib.error
from contextlib import nullcontext
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fraud_detection.deployment import cli as deployment_cli
from fraud_detection.deployment import preflight, worker
from fraud_detection.deployment.cli import main
from fraud_detection.deployment.preflight import _mask_identifier, _sanitize, build_preflight_report

ROOT = Path(__file__).resolve().parents[2]


def test_preflight_is_read_only_and_secret_safe() -> None:
    report = build_preflight_report()
    assert report["mode"] == "read_only"
    assert report["cloud_mutations_performed"] is False
    serialized = json.dumps(report).lower()
    for forbidden in (
        "access_token",
        "client_secret",
        "primarymasterkey",
        "secondarymasterkey",
        "postgresql+psycopg://",
    ):
        assert forbidden not in serialized
    assert _sanitize(
        {
            "accessToken": "unsafe",
            "clientSecret": "unsafe",
            "connectionString": "unsafe",
            "safe": "visible",
        }
    ) == {
        "accessToken": "[redacted]",
        "clientSecret": "[redacted]",
        "connectionString": "[redacted]",
        "safe": "visible",
    }
    assert _mask_identifier("11111111-2222-3333-4444-555555555555") == "[masked-id:5555]"
    sanitized_ids = _sanitize(
        {
            "id": "11111111-2222-3333-4444-555555555555",
            "policy": "allowed for 11111111-2222-3333-4444-555555555555",
        }
    )
    assert sanitized_ids == {
        "id": "[masked-id:5555]",
        "policy": "allowed for [masked-id:5555]",
    }
    assert _sanitize(
        {
            "outer": {
                "tenantId": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeee1111",
                "principalId": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeee2222",
                "roleAssignmentId": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeee3333",
                "label": "policy aaaaaaaa-bbbb-cccc-dddd-eeeeeeee4444",
            }
        }
    ) == {
        "outer": {
            "tenantId": "[masked-id:1111]",
            "principalId": "[masked-id:2222]",
            "roleAssignmentId": "[masked-id:3333]",
            "label": "policy [masked-id:4444]",
        }
    }


def test_committed_preflight_report_is_sanitized() -> None:
    report = json.loads((ROOT / "deployment/preflight-report.json").read_text(encoding="utf-8"))
    assert report["cloud_mutations_performed"] is False
    assert report["azure"]["status"] == "authenticated_read_only"
    assert report["azure"]["default_subscription"]["name"] == "Azure for Students"
    assert report["azure"]["default_subscription"]["id"].startswith("[masked-id:")
    assert report["azure"]["tenant_id"].startswith("[masked-id:")
    assert report["azure"]["resource_inventory"]["total_resource_group_count"] == 0
    assert report["azure"]["resource_inventory"]["total_resource_count"] == 0
    assert report["candidate_regions"]["candidates_allowed_by_policy"] == sorted(
        report["azure"]["policies"]["allowed_locations"]
    )
    assert "Microsoft.Storage" in report["azure"]["provider_registration"]
    assert "[redacted]" not in json.dumps(report["azure"]["authorization_assessment"])


def test_deploy_and_teardown_require_exact_confirmation() -> None:
    with pytest.raises(RuntimeError, match="--confirm DEPLOY"):
        main(
            [
                "deploy",
                "--subscription",
                "sub",
                "--location",
                "eastus",
                "--parameters",
                "infra/parameters/gate-b.example.json",
                "--confirm",
                "NO",
            ]
        )
    with pytest.raises(RuntimeError, match="DELETE:rg-demo:demo"):
        main(
            [
                "teardown",
                "--environment",
                "demo",
                "--resource-group",
                "rg-demo",
                "--confirm",
                "NO",
            ]
        )


def test_deployment_secret_scan_passes() -> None:
    assert main(["secret-scan"]) == 0


def test_deploy_passes_secret_through_temporary_file_not_process_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def capture_command(command: list[str]) -> int:
        commands.append(command)
        return 0

    monkeypatch.setenv("FRAUD_POSTGRES_ADMIN_PASSWORD", "temporary-test-value")
    monkeypatch.setattr(deployment_cli, "_require_tool", lambda _name: "az")
    monkeypatch.setattr(deployment_cli, "_run", capture_command)
    assert (
        main(
            [
                "deploy",
                "--subscription",
                "sub",
                "--location",
                "eastus",
                "--parameters",
                "infra/parameters/gate-b.example.json",
                "--confirm",
                "DEPLOY",
            ]
        )
        == 0
    )
    assert len(commands) == 1
    command_text = " ".join(commands[0])
    assert "temporary-test-value" not in command_text
    secret_file = Path(commands[0][-1].removeprefix("@"))
    assert not secret_file.exists()


def test_read_only_local_operations_and_backup_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["contracts", "check"]) == 0
    assert main(["powerbi-validate"]) == 0
    assert main(["copilot-validate"]) == 0
    assert main(["cost-review"]) == 0
    monkeypatch.delenv("FRAUD_POSTGRES_DSN", raising=False)
    with pytest.raises(RuntimeError, match="database DSN is missing"):
        main(["backup", "--output", "artifacts/reports/not-created.dump"])


def test_cost_review_matches_current_consumption_container_apps_topology() -> None:
    review = json.loads((ROOT / "deployment/cost-review.json").read_text(encoding="utf-8"))
    estimate = review["estimated_monthly_usd"]
    assert estimate == {
        "base_architecture_high": 67.0,
        "base_architecture_low": 55.0,
        "credit_duration_months_high": 1.82,
        "credit_duration_months_low": 1.49,
    }
    container_apps = next(
        item for item in review["resources"] if item["category"] == "Container Apps API and Jobs"
    )
    assert container_apps["estimate_monthly_usd"] == "0-5 at light scale-to-zero usage"
    assert "private endpoint" in container_apps["future_repricing_trigger"]
    assert "dedicated or flexible workload profile" in container_apps["future_repricing_trigger"]
    assert "planned maintenance" in container_apps["future_repricing_trigger"]

    bicep = (ROOT / "infra/modules/platform.bicep").read_text(encoding="utf-8")
    environment = bicep.split("resource environment", maxsplit=1)[1].split(
        "var postgresDsn", maxsplit=1
    )[0]
    assert "workloadProfiles" not in environment
    assert "plannedMaintenance" not in environment
    assert "Microsoft.Network/privateEndpoints" not in environment


def test_acceptance_plan_covers_every_gate_b_requirement() -> None:
    plan = json.loads((ROOT / "deployment/acceptance-plan.json").read_text(encoding="utf-8"))
    assert {check["id"] for check in plan["checks"]} == {
        "https",
        "auth",
        "postgres",
        "alembic",
        "views",
        "artifact_staging",
        "pipeline_jobs",
        "raw",
        "embedding",
        "gremlin",
        "reconciliation",
        "publication",
        "dashboard",
        "powerbi",
        "copilot_import",
        "copilot_grounding",
        "partial",
        "monitoring",
        "secrets",
        "counts",
        "cost",
        "teardown",
    }
    assert all(check["evidence"] for check in plan["checks"])


def test_power_bi_private_connectivity_has_live_and_no_capacity_outcomes() -> None:
    contract = json.loads(
        (ROOT / "deployment/power-bi-connectivity.json").read_text(encoding="utf-8")
    )
    live = contract["live_service_outcome"]
    fallback = contract["no_capacity_fallback"]
    assert live["route"].startswith("Power BI virtual network data gateway")
    assert live["status"].startswith("unknown")
    assert "capacity" in live["requirements"]
    assert "gateway_region" in live["requirements"]
    assert "credential_storage" in live["requirements"]
    assert len(live["gate_b_evidence"]) >= 7
    assert fallback["status"] == "live_power_bi_service_blocked"
    assert "Making PostgreSQL public" in fallback["prohibited"]
    report = json.loads((ROOT / "deployment/preflight-report.json").read_text(encoding="utf-8"))
    assert report["power_bi"]["vnet_data_gateway_availability"] == "unknown"
    assert report["power_bi"]["fabric_or_premium_capacity_availability"] == "unknown"


def test_private_staging_connectivity_contract_is_fail_closed() -> None:
    contract = json.loads(
        (ROOT / "deployment/artifact-staging-connectivity.json").read_text(encoding="utf-8")
    )
    assert contract["status"] == "blocked_until_private_upload_route_is_verified"
    assert contract["selected_upload_runner_type"] == "unselected"
    assert contract["network"]["public_network_access"] == "disabled"
    assert contract["private_dns"]["required_zone"] == "privatelink.blob.core.windows.net"
    assert len(contract["gate_b_evidence"]) >= 7
    assert "must not be enabled" in contract["fallback"]


def test_bicep_preserves_approved_architecture_and_private_data_plane() -> None:
    source = (ROOT / "infra/modules/platform.bicep").read_text(encoding="utf-8")
    for required in (
        "Microsoft.App/containerApps",
        "Microsoft.DBforPostgreSQL/flexibleServers",
        "Microsoft.DocumentDB/databaseAccounts",
        "Microsoft.KeyVault/vaults",
        "Microsoft.OperationalInsights/workspaces",
        "Microsoft.Insights/components",
        "Microsoft.ContainerRegistry/registries",
        "transactions_raw",
        "behavior_embeddings",
        "'/sender_id'",
        "'/pk'",
        "dimensions: 384",
        "distanceFunction: 'cosine'",
        "publicNetworkAccess: 'Disabled'",
        "minimalTlsVersion: 'Tls12'",
        "enableRbacAuthorization: true",
        "disableLocalAuth: true",
        "CosmosDataContributor",
        "FRAUD_COSMOS_USE_MANAGED_IDENTITY",
        "minReplicas: containerMinReplicas",
    ):
        assert required in source
    assert "administratorLoginPassword: postgresAdministratorPassword" in source
    assert "output postgresAdministratorPassword" not in source


def test_container_contract_is_nonroot_bounded_and_excludes_sensitive_context() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for required in (
        "USER 10001:10001",
        "HEALTHCHECK",
        "EXPOSE 8000",
        "STOPSIGNAL SIGTERM",
        '"--timeout-graceful-shutdown", "30"',
        '"--no-access-log"',
    ):
        assert required in dockerfile
    assert "--reload" not in dockerfile
    for excluded in (".env", "data", "artifacts", "tests", "secrets", ".git"):
        assert excluded in dockerignore


def test_worker_image_and_all_manual_pipeline_stages_are_defined() -> None:
    jobs = json.loads((ROOT / "deployment/pipeline-jobs.json").read_text(encoding="utf-8"))
    assert {stage["worker_stage"] for stage in jobs["stages"]} == set(worker.PIPELINE_STAGES)
    assert all(stage["timeout_seconds"] <= 14_400 for stage in jobs["stages"])
    assert all(stage["retry_limit"] <= 1 for stage in jobs["stages"])
    publication = next(stage for stage in jobs["stages"] if stage["publication"])
    assert publication["name"] == "publication"
    assert publication["confirmation"] == "PUBLISH:<exact-batch-id>"

    dockerfile = (ROOT / "Dockerfile.worker").read_text(encoding="utf-8")
    assert "USER 10002:10002" in dockerfile
    assert "worker-requirements.lock" in dockerfile
    assert "COPY data" not in dockerfile
    assert "COPY artifacts" not in dockerfile
    assert "FRAUD_POSTGRES_DSN=" not in dockerfile

    bicep = (ROOT / "infra/modules/platform.bicep").read_text(encoding="utf-8")
    assert "resource pipelineJobs 'Microsoft.App/jobs" in bicep
    assert "triggerType: 'Manual'" in bicep
    assert "manualTriggerConfig: { parallelism: 1, replicaCompletionCount: 1 }" in bicep
    assert "FRAUD_STAGING_ACCOUNT_URL" in bicep
    assert "publicNetworkAccess: 'Disabled'" in bicep
    assert "containerImageDigest" in (ROOT / "infra/main.bicep").read_text(encoding="utf-8")
    assert "workerImageDigest" in (ROOT / "infra/main.bicep").read_text(encoding="utf-8")
    for stage in jobs["stages"]:
        stage_suffix = stage["job_suffix"]
        assert stage_suffix
        assert f"suffix: '{stage_suffix}'" in bicep
        assert f"stage: '{stage['worker_stage']}'" in bicep
    worker_env = (
        "FRAUD_WORK_ROOT",
        "FRAUD_BEHAVIORAL_TEMPLATE_ARTIFACT",
        "FRAUD_EMBEDDING_LOCAL_ARTIFACT",
        "FRAUD_MODEL_FEATURE_ARTIFACT",
        "FRAUD_MODEL_ARTIFACT_ROOT",
        "FRAUD_EMBEDDING_MODEL_CACHE",
        "FRAUD_INCREMENTAL_ARTIFACT_ROOT",
        "FRAUD_INCREMENTAL_DATA_ROOT",
    )
    assert all(name in bicep and "/work" in bicep for name in worker_env)


def test_worker_artifact_hash_validation_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "WORK_ROOT", tmp_path)
    artifact = tmp_path / "data/sample.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"approved")
    digest = worker.file_sha256(artifact)
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": "private-staging-artifacts-v1",
        "source_provenance": "test",
        "artifacts": [
            {
                "name": "sample",
                "blob_name": "approved/sample.bin",
                "destination": "data/sample.bin",
                "sha256": digest,
                "restricted_source": False,
                "required_for_stages": ["raw-projection"],
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert worker.validate_local_artifacts(manifest_path, "raw-projection") == {"sample": digest}
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        worker.validate_local_artifacts(manifest_path, "raw-projection")
    artifact.unlink()
    with pytest.raises(ValueError, match="missing"):
        worker.validate_local_artifacts(manifest_path, "raw-projection")


def test_publication_requires_exact_confirmation_and_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FRAUD_PUBLICATION_BATCH_ID", raising=False)
    monkeypatch.delenv("FRAUD_PUBLICATION_CONFIRMATION", raising=False)
    with pytest.raises(ValueError, match="exact batch"):
        worker._publication_commands(10)

    monkeypatch.setenv("FRAUD_PUBLICATION_BATCH_ID", "batch-1")
    monkeypatch.setenv("FRAUD_PUBLICATION_CONFIRMATION", "PUBLISH:batch-1")
    monkeypatch.setenv("FRAUD_PIPELINE_CHUNK_INDEX", "0")
    calls: list[list[str]] = []

    def fake_run(command: list[str], _timeout: int) -> dict[str, object]:
        calls.append(command)
        if "plan" in command:
            return {"result": {"chunks": [{"batch_id": "batch-1"}]}}
        return {"result": {"publication_eligible": False}}

    monkeypatch.setattr(worker, "_run", fake_run)
    with pytest.raises(ValueError, match="did not prove"):
        worker._publication_commands(10)
    assert not any("publish" in command for command in calls)


def test_deployment_cli_publication_guard_is_exact() -> None:
    with pytest.raises(RuntimeError, match="PUBLISH:batch-1"):
        main(
            [
                "publish",
                "--resource-group",
                "rg",
                "--environment",
                "demo",
                "--unique-suffix",
                "abc1",
                "--batch-id",
                "batch-1",
                "--chunk-index",
                "0",
                "--restricted-source-sha256",
                "a" * 64,
                "--restricted-source-confirmation",
                f"APPROVE_RESTRICTED_SOURCE:{'a' * 64}",
                "--confirm",
                "NO",
            ]
        )


def test_worker_publication_rejects_batch_chunk_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRAUD_PUBLICATION_BATCH_ID", "batch-1")
    monkeypatch.setenv("FRAUD_PUBLICATION_CONFIRMATION", "PUBLISH:batch-1")
    monkeypatch.setenv("FRAUD_PIPELINE_CHUNK_INDEX", "0")
    calls: list[list[str]] = []

    def fake_run(command: list[str], _timeout: int) -> dict[str, object]:
        calls.append(command)
        return {"result": {"chunks": [{"batch_id": "different-batch"}]}}

    monkeypatch.setattr(worker, "_run", fake_run)
    with pytest.raises(ValueError, match="does not match"):
        worker._publication_commands(10)
    assert len(calls) == 1


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["tool"], returncode, stdout=stdout, stderr="")


def test_amd64_build_commands_and_image_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def capture_build(command: list[str]) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(deployment_cli, "_require_tool", lambda _name: "docker")
    monkeypatch.setattr(deployment_cli, "_run", capture_build)
    assert main(["container-build", "--tag", "api:test"]) == 0
    assert main(["worker-build", "--tag", "worker:test"]) == 0
    for command in commands:
        assert command[:3] == ["docker", "buildx", "build"]
        assert command[3:5] == ["--platform", "linux/amd64"]
        assert "--load" in command

    amd64 = [
        {
            "Id": "sha256:1",
            "Os": "linux",
            "Architecture": "amd64",
            "Size": 1,
            "Config": {"User": "10001:10001"},
        }
    ]
    monkeypatch.setattr(deployment_cli, "_capture", lambda _command: _completed(json.dumps(amd64)))
    assert deployment_cli._inspect_target_image("docker", "api:test")["architecture"] == "amd64"
    for bad in (
        [{**amd64[0], "Architecture": "arm64"}],
        [{**amd64[0], "Os": "windows"}],
        [{**amd64[0], "Config": {"User": "root"}}],
    ):
        monkeypatch.setattr(
            deployment_cli, "_capture", lambda _command, value=bad: _completed(json.dumps(value))
        )
        with pytest.raises(RuntimeError):
            deployment_cli._inspect_target_image("docker", "bad:test")


def test_enabled_deployments_require_immutable_image_digests(tmp_path: Path) -> None:
    path = tmp_path / "parameters.json"
    parameters: dict[str, Any] = {
        "parameters": {
            "deployContainerApp": {"value": True},
            "deployPipelineJobs": {"value": True},
            "containerImageDigest": {"value": "registry/api:latest"},
            "workerImageDigest": {"value": "registry/worker:latest"},
        }
    }
    path.write_text(json.dumps(parameters), encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable"):
        deployment_cli._validate_immutable_deployment_images(path)
    parameters["parameters"]["containerImageDigest"]["value"] = f"registry/api@sha256:{'a' * 64}"
    parameters["parameters"]["workerImageDigest"]["value"] = f"registry/worker@sha256:{'b' * 64}"
    path.write_text(json.dumps(parameters), encoding="utf-8")
    deployment_cli._validate_immutable_deployment_images(path)


def test_stage_aware_azure_commands_capture_exact_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(deployment_cli, "_require_tool", lambda _name: "az")

    def capture(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _completed("execution-123\n")

    monkeypatch.setattr(deployment_cli, "_capture", capture)
    sha = "a" * 64
    approval = f"APPROVE_RESTRICTED_SOURCE:{sha}"
    common = [
        "--resource-group",
        "rg-demo",
        "--environment",
        "demo",
        "--unique-suffix",
        "abc1",
    ]
    assert main(["start-stage", *common, "--stage", "schema-migration"]) == 0
    assert (
        main(
            [
                "start-stage",
                *common,
                "--stage",
                "graph-snapshot",
                "--graph-cutoff-step",
                "282",
                "--restricted-source-sha256",
                sha,
                "--restricted-source-confirmation",
                approval,
            ]
        )
        == 0
    )
    for stage in ("incremental", "reconciliation"):
        assert (
            main(
                [
                    "start-stage",
                    *common,
                    "--stage",
                    stage,
                    "--chunk-index",
                    "3",
                    "--restricted-source-sha256",
                    sha,
                    "--restricted-source-confirmation",
                    approval,
                ]
            )
            == 0
        )
    assert (
        main(
            [
                "publication-preview",
                *common,
                "--chunk-index",
                "3",
                "--restricted-source-sha256",
                sha,
                "--restricted-source-confirmation",
                approval,
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "publish",
                *common,
                "--batch-id",
                "batch-3",
                "--chunk-index",
                "3",
                "--restricted-source-sha256",
                sha,
                "--restricted-source-confirmation",
                approval,
                "--confirm",
                "PUBLISH:batch-3",
            ]
        )
        == 0
    )

    assert all("--container-name" in command and "worker" in command for command in commands)
    assert all(command[-4:] == ["--query", "name", "--output", "tsv"] for command in commands)
    assert "fd-demo-abc1-migrate" in commands[0]
    assert "FRAUD_GRAPH_CUTOFF_STEP=282" in commands[1]
    assert "FRAUD_PIPELINE_CHUNK_INDEX=3" in commands[2]
    assert "fd-demo-abc1-reconcile" in commands[4]
    assert "fd-demo-abc1-publish" in commands[5]
    assert not any("PASSWORD" in " ".join(command) for command in commands)


def test_stage_parameter_validation_fails_before_azure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deployment_cli,
        "_require_tool",
        lambda _name: pytest.fail("Azure CLI must not be invoked"),
    )
    sha = "a" * 64
    common = [
        "start-stage",
        "--resource-group",
        "rg",
        "--environment",
        "demo",
        "--unique-suffix",
        "abc1",
    ]
    with pytest.raises(RuntimeError, match="cannot start publication"):
        main([*common, "--stage", "publication"])
    with pytest.raises(RuntimeError, match="positive cutoff"):
        main([*common, "--stage", "graph-snapshot", "--graph-cutoff-step", "0"])
    with pytest.raises(RuntimeError, match="chunk index"):
        main([*common, "--stage", "incremental", "--chunk-index", "-1"])
    with pytest.raises(RuntimeError, match="restricted-source SHA"):
        main([*common, "--stage", "raw-projection"])
    with pytest.raises(RuntimeError, match="exact confirmation"):
        main(
            [
                *common,
                "--stage",
                "raw-projection",
                "--restricted-source-sha256",
                sha,
                "--restricted-source-confirmation",
                "wrong",
            ]
        )
    with pytest.raises(RuntimeError, match="unsupported characters"):
        deployment_cli._job_name("BAD!", "abc1", "schema-migration")


def test_staging_network_check_requires_private_dns_and_https(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_result = [(None, None, None, None, ("10.42.3.8", 443))]
    monkeypatch.setattr(
        "fraud_detection.deployment.cli.socket.getaddrinfo",
        lambda *_args, **_kwargs: private_result,
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise urllib.error.HTTPError("https://blob", 403, "forbidden", Message(), None)

    monkeypatch.setattr("fraud_detection.deployment.cli.urllib.request.urlopen", forbidden)
    result = deployment_cli._staging_network_check("fraudstage", "10.42.3.0/24")
    assert result["private_addresses"] == ["10.42.3.8"]
    monkeypatch.setattr(
        "fraud_detection.deployment.cli.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("20.1.2.3", 443))],
    )
    with pytest.raises(RuntimeError, match="private endpoint"):
        deployment_cli._staging_network_check("fraudstage", "10.42.3.0/24")


def test_wait_and_logs_bind_to_exact_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(deployment_cli, "_require_tool", lambda _name: "az")

    def execution_status(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _completed("Succeeded\n")

    monkeypatch.setattr("fraud_detection.deployment.cli.subprocess.run", execution_status)
    common = [
        "--resource-group",
        "rg-demo",
        "--environment",
        "demo",
        "--unique-suffix",
        "abc1",
        "--stage",
        "reconciliation",
        "--execution-name",
        "execution-123",
    ]
    assert main(["wait-stage", *common, "--timeout-seconds", "1"]) == 0
    assert "[?name=='execution-123'] | [0].properties.status" in commands[0]
    commands.clear()

    def capture_log(command: list[str]) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(deployment_cli, "_run", capture_log)
    assert main(["job-logs", *common]) == 0
    assert "--execution" in commands[0] and "execution-123" in commands[0]
    assert "--container" in commands[0] and "worker" in commands[0]


def test_artifact_upload_orders_manifest_last_and_requires_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    first = root / "data/one.bin"
    second = root / "data/two.bin"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    manifest_path = tmp_path / "manifest.json"
    artifacts = []
    for name, path in (("one", first), ("two", second)):
        artifacts.append(
            {
                "name": name,
                "blob_name": f"approved/{name}.bin",
                "destination": f"data/{name}.bin",
                "sha256": deployment_cli._file_sha256(path),
                "restricted_source": False,
                "required_for_stages": ["raw-projection"],
                "extract_to": None,
            }
        )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "private-staging-artifacts-v1",
                "source_provenance": "test",
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    order: list[str] = []
    monkeypatch.setattr(deployment_cli, "_staging_network_check", lambda *_args: {})
    monkeypatch.setattr(deployment_cli, "_require_tool", lambda _name: "az")
    monkeypatch.setattr(
        deployment_cli,
        "_upload_and_verify_blob",
        lambda _az, **kwargs: order.append(str(kwargs["blob_name"])),
    )
    assert (
        main(
            [
                "artifacts-upload",
                "--manifest",
                str(manifest_path),
                "--root",
                str(root),
                "--account",
                "fraudstage",
                "--container",
                "deployment-artifacts",
                "--confirm",
                "UPLOAD",
            ]
        )
        == 0
    )
    assert order == ["approved/one.bin", "approved/two.bin", "deployment/artifact-manifest.json"]


def test_blob_upload_verifies_metadata_and_downloaded_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"approved")
    digest = deployment_cli._file_sha256(source)
    commands: list[list[str]] = []

    def run(command: list[str]) -> int:
        commands.append(command)
        if "download" in command:
            destination = Path(command[command.index("--file") + 1])
            destination.write_bytes(source.read_bytes())
        return 0

    monkeypatch.setattr(deployment_cli, "_run", run)
    monkeypatch.setattr(
        deployment_cli,
        "_capture",
        lambda _command: _completed(digest + "\n"),
    )
    deployment_cli._upload_and_verify_blob(
        "az",
        account="stage",
        container="artifacts",
        blob_name="approved/source.bin",
        path=source,
        expected_sha256=digest,
    )
    assert "upload" in commands[0]
    assert "download" in commands[1]


def test_worker_destination_archive_and_restricted_source_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "WORK_ROOT", tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        worker._safe_destination("../escape")

    archive = tmp_path / "bundle.tar"
    with tarfile.open(archive, "w") as bundle:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        bundle.addfile(root)
        payload = tarfile.TarInfo("safe/file.txt")
        payload.size = 2
        import io

        bundle.addfile(payload, io.BytesIO(b"ok"))
    manifest_path = tmp_path / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "private-staging-artifacts-v1",
        "source_provenance": "test",
        "artifacts": [
            {
                "name": "bundle",
                "blob_name": "bundle.tar",
                "destination": "bundle.tar",
                "sha256": worker.file_sha256(archive),
                "restricted_source": False,
                "required_for_stages": ["raw-projection"],
                "extract_to": "extracted",
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    worker._extract_approved_archives(manifest_path, "raw-projection")
    assert (tmp_path / "extracted/safe/file.txt").read_bytes() == b"ok"

    unsafe = tmp_path / "unsafe.tar"
    with tarfile.open(unsafe, "w") as bundle:
        member = tarfile.TarInfo("../escape")
        member.size = 1
        bundle.addfile(member, io.BytesIO(b"x"))
    manifest["artifacts"][0]["destination"] = "unsafe.tar"
    manifest["artifacts"][0]["sha256"] = worker.file_sha256(unsafe)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe path"):
        worker._extract_approved_archives(manifest_path, "raw-projection")

    source = tmp_path / "source.csv"
    source.write_bytes(b"source")
    digest = worker.file_sha256(source)
    manifest["artifacts"] = [
        {
            "name": "source",
            "blob_name": "source.csv",
            "destination": "source.csv",
            "sha256": digest,
            "restricted_source": True,
            "required_for_stages": ["raw-projection"],
            "extract_to": None,
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.delenv("FRAUD_RESTRICTED_SOURCE_APPROVAL", raising=False)
    with pytest.raises(ValueError, match="lacks exact approval"):
        worker.validate_local_artifacts(manifest_path, "raw-projection")
    monkeypatch.setenv("FRAUD_RESTRICTED_SOURCE_APPROVAL", f"APPROVE_RESTRICTED_SOURCE:{digest}")
    assert worker.validate_local_artifacts(manifest_path, "raw-projection") == {"source": digest}


def test_worker_configuration_lock_and_command_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRAUD_POSTGRES_DSN", raising=False)
    with pytest.raises(ValueError, match="POSTGRES_DSN"), worker.pipeline_lock():
        pass
    monkeypatch.delenv("FRAUD_GRAPH_CUTOFF_STEP", raising=False)
    with pytest.raises(ValueError, match="CUTOFF"):
        worker._commands("graph-snapshot")
    monkeypatch.delenv("FRAUD_PIPELINE_CHUNK_INDEX", raising=False)
    with pytest.raises(ValueError, match="CHUNK_INDEX"):
        worker._chunk_index()
    monkeypatch.setenv("FRAUD_PIPELINE_CHUNK_INDEX", "-1")
    with pytest.raises(ValueError, match="nonnegative"):
        worker._chunk_index()
    monkeypatch.setenv("FRAUD_GRAPH_CUTOFF_STEP", "282")
    monkeypatch.setenv("FRAUD_PIPELINE_CHUNK_INDEX", "0")
    for stage in set(worker.PIPELINE_STAGES) - {"publication"}:
        for command in worker._commands(stage):
            assert all("/work" in value for value in command if value.startswith("/"))


def test_worker_advisory_lock_contention(monkeypatch: pytest.MonkeyPatch) -> None:
    class Connection:
        def __enter__(self) -> "Connection":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, _query: str, _params: object) -> Any:
            return SimpleNamespace(fetchone=lambda: (False,))

    monkeypatch.setenv("FRAUD_POSTGRES_DSN", "postgresql://example")
    monkeypatch.setattr(
        "fraud_detection.deployment.worker.psycopg.connect",
        lambda *_args, **_kwargs: Connection(),
    )
    with pytest.raises(ValueError, match="already running"), worker.pipeline_lock():
        pass


def test_worker_manifest_hash_and_staged_output_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Download:
        def readall(self) -> bytes:
            return b"wrong-manifest"

    class Container:
        def download_blob(self, _name: str) -> Download:
            return Download()

    class Service:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def get_container_client(self, _name: str) -> Container:
            return Container()

    monkeypatch.setattr(worker, "WORK_ROOT", tmp_path)
    monkeypatch.setattr(worker, "BlobServiceClient", Service)
    monkeypatch.setattr(worker, "DefaultAzureCredential", lambda **_kwargs: object())
    monkeypatch.setenv("FRAUD_STAGING_ACCOUNT_URL", "https://private")
    monkeypatch.setenv("FRAUD_STAGING_CONTAINER", "artifacts")
    monkeypatch.setenv("FRAUD_ARTIFACT_MANIFEST_BLOB", "manifest.json")
    monkeypatch.setenv("FRAUD_ARTIFACT_MANIFEST_SHA256", "a" * 64)
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        worker.stage_artifacts("raw-projection")

    output = tmp_path / "data/processed/behavior_embeddings.parquet"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"new")

    class Blob:
        def exists(self) -> bool:
            return True

        def download_blob(self) -> Any:
            return SimpleNamespace(readall=lambda: b"different")

    monkeypatch.setattr(
        worker,
        "_staging_container",
        lambda: SimpleNamespace(get_blob_client=lambda _name: Blob()),
    )
    with pytest.raises(ValueError, match="conflicts"):
        worker.upload_stage_outputs("behavioral-embeddings")


def test_preflight_authenticated_and_command_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight, "_command_path", lambda name: f"/{name}")
    commands: list[list[str]] = []
    signed_in_object_id = "11111111-2222-3333-4444-555555559999"

    def fake_json(command: list[str], timeout: int = 20) -> tuple[Any, str | None]:
        del timeout
        commands.append(command)
        joined = " ".join(command)
        if "az version" in joined:
            return {"azure-cli": "test"}, None
        if "extension list" in joined:
            return [{"name": "containerapp", "version": "1"}], None
        if "account show" in joined:
            return {
                "id": "sub",
                "name": "demo",
                "tenantId": "tenant",
                "state": "Enabled",
                "user": {"type": "user"},
            }, None
        if "account list" in joined:
            return [
                {
                    "id": "sub",
                    "name": "demo",
                    "tenantId": "tenant",
                    "state": "Enabled",
                    "isDefault": True,
                }
            ], None
        if "ad signed-in-user show" in joined:
            return signed_in_object_id, None
        if "role assignment list" in joined:
            assert "--assignee" in command
            assert command[command.index("--assignee") + 1] == signed_in_object_id
            assert "--include-inherited" in command
            assert "--include-groups" in command
            return [
                {
                    "roleDefinitionName": "Owner",
                    "principalId": signed_in_object_id,
                    "id": "unpersisted-assignment-one",
                },
                {
                    "roleDefinitionName": "Owner",
                    "principalId": signed_in_object_id,
                    "id": "unpersisted-assignment-two",
                },
            ], None
        if "group list" in joined:
            return [{"name": "rg-fraud", "location": "eastus"}], None
        if "resource list" in joined:
            return [
                {"name": "fraud", "resourceGroup": "rg-fraud", "type": "type", "location": "eastus"}
            ], None
        if "policy assignment list" in joined:
            return [
                {
                    "displayName": "Allowed regions aaaaaaaa-bbbb-cccc-dddd-eeeeeeee4444",
                    "parameters": {"listOfAllowedLocations": {"value": ["centralus", "eastus"]}},
                }
            ], None
        if "az rest" in joined:
            return {
                "state": "Enabled",
                "subscriptionPolicies": {
                    "quotaId": "Student",
                    "spendingLimit": "On",
                    "locationPlacementId": "Public",
                },
            }, None
        if "consumption budget list" in joined:
            return [], None
        if "postgres flexible-server list-skus" in joined:
            if "eastus" in command:
                return [{"reason": "restricted"}], None
            return [
                {
                    "supportedServerVersions": [{"name": "16"}],
                    "supportedServerEditions": [
                        {
                            "name": "Burstable",
                            "supportedServerSkus": [{"name": "Standard_B1ms"}],
                        }
                    ],
                }
            ], None
        if "cosmosdb locations list" in joined:
            return None, "CalledProcessError"
        if "check-name-exists" in joined:
            return False, None
        if any(
            value in joined
            for value in ("acr check-name", "keyvault check-name", "storage account check-name")
        ):
            return {"nameAvailable": True}, None
        if "provider show" in joined:
            return {
                "registrationState": "Registered",
                "resourceTypes": [
                    {"resourceType": name, "locations": ["Central US", "East US"]}
                    for name in ("managedEnvironments", "containerApps", "jobs")
                ],
            }, None
        if "docker version" in joined:
            return {"Server": {"Version": "1"}}, None
        return {}, None

    monkeypatch.setattr(preflight, "_safe_json_command", fake_json)
    report = preflight.build_preflight_report()
    assert report["azure"]["authenticated"] is True
    assert report["azure"]["default_subscription"]["id"] == "[masked-id:masked]"
    assert report["azure"]["tenant_id"] == "[masked-id:nant]"
    assert report["azure"]["required_extension_status"]["containerapp"] == "installed"
    assert report["azure"]["resource_inventory"]["project_resource_groups"][0]["name"] == "rg-fraud"
    assert set(report["azure"]["provider_registration"].values()) == {"Registered"}
    assert "Microsoft.Storage" in report["azure"]["provider_registration"]
    assert report["azure"]["effective_roles"] == [
        {"role": "Owner", "assignment_count": 2, "scope": "selected_subscription_or_inherited"}
    ]
    assert report["azure"]["effective_role_discovery"] == {
        "status": "verified",
        "identity_scope": "signed_in_identity",
        "group_derived_roles": "included",
    }
    serialized = json.dumps(report)
    assert signed_in_object_id not in serialized
    assert "unpersisted-assignment" not in serialized
    role_commands = [command for command in commands if command[1:3] == ["role", "assignment"]]
    assert len(role_commands) == 1
    assert "--assignee" in role_commands[0]
    assert report["azure"]["policies"]["allowed_locations"] == ["centralus", "eastus"]
    assert report["azure"]["policies"]["assignments"] == ["Allowed regions [masked-id:4444]"]
    assert report["azure"]["subscription_offer"]["spending_limit"] == "On"
    assert report["azure"]["budget_count"] == 0
    assert report["azure"]["capabilities"]["postgresql_by_region"]["centralus"] == {
        "status": "supported",
        "postgresql_16": True,
        "standard_b1ms": True,
        "reason": None,
    }
    assert report["azure"]["capabilities"]["postgresql_by_region"]["eastus"]["status"] == (
        "restricted"
    )
    assert report["azure"]["capabilities"]["container_apps"]["status"] == "supported"
    assert report["azure"]["capabilities"]["cosmos_specialized"]["status"] == "unresolved"
    assert set(report["azure"]["name_availability"].values()) == {
        "available",
        "not_selected",
    }
    assert report["candidate_regions"]["recommended_for_parameter_review"] == "centralus"


def test_effective_role_discovery_is_identity_scoped_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeee7777"
    calls: list[list[str]] = []

    def identity_scoped(command: list[str], timeout: int = 45) -> tuple[Any, str | None]:
        del timeout
        calls.append(command)
        if "signed-in-user" in command:
            return object_id, None
        if "--assignee" in command:
            return [{"roleDefinitionName": "Owner"}, {"roleDefinitionName": "Owner"}], None
        return [{"roleDefinitionName": "Unrelated Tenant Administrator"}], None

    monkeypatch.setattr(preflight, "_safe_json_command", identity_scoped)
    roles, discovery = preflight._discover_effective_roles("az")
    assert roles == [
        {"role": "Owner", "assignment_count": 2, "scope": "selected_subscription_or_inherited"}
    ]
    assert discovery["group_derived_roles"] == "included"
    assert all("--assignee" in command for command in calls if "assignment" in command)
    assert all(role["role"] != "Unrelated Tenant Administrator" for role in roles)

    monkeypatch.setattr(
        preflight,
        "_safe_json_command",
        lambda _command, timeout=45: (None, "signed_in_user_unavailable"),
    )
    roles, discovery = preflight._discover_effective_roles("az")
    assert roles == []
    assert discovery["status"] == "unverified"
    assert discovery["group_derived_roles"] == "unverified"
    assessment = preflight._authorization_assessment(roles, discovery)
    assert assessment["resource_groups"] == "unverified"
    assert assessment["role_assignments"] == "unverified"
    assert assessment["service_resources"] == "unverified"

    lookup_count = 0

    def role_lookup_failure(command: list[str], timeout: int = 45) -> tuple[Any, str | None]:
        nonlocal lookup_count
        del timeout
        lookup_count += 1
        if lookup_count == 1:
            return object_id, None
        assert "--include-groups" in command
        return None, "include_groups_unavailable"

    monkeypatch.setattr(preflight, "_safe_json_command", role_lookup_failure)
    roles, discovery = preflight._discover_effective_roles("az")
    assert roles == []
    assert discovery == {
        "status": "unverified",
        "identity_scope": "signed_in_identity",
        "group_derived_roles": "unverified",
        "reason": "identity_role_lookup_failed:include_groups_unavailable",
    }
    assert lookup_count == 2


def test_preflight_cli_missing_and_invalid_capability_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight, "_command_path", lambda _name: None)
    report = preflight.build_preflight_report()
    assert report["azure"]["status"] == "azure_cli_not_installed"
    assert report["candidate_regions"]["candidates_allowed_by_policy"] == []
    assert preflight._postgres_capability(None, "timeout") == {
        "status": "unavailable",
        "postgresql_16": False,
        "standard_b1ms": False,
        "reason": "timeout",
    }
    assert preflight._postgres_capability([], None)["reason"] == "empty_response"
    assert preflight._provider_locations(None, "jobs") == []


def test_cli_health_contract_image_push_and_deploy_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        deployment_cli,
        "_get_json",
        lambda url, token=None: (503, "{}") if url.endswith("/health/ready") else (200, "{}"),
    )
    assert main(["health", "--base-url", "https://example.test"]) == 0
    assert main(["smoke", "--base-url", "https://example.test"]) == 0
    assert main(["contracts", "generate"]) == 0

    monkeypatch.setattr(deployment_cli, "_require_tool", lambda _name: "docker")
    monkeypatch.setattr(deployment_cli, "_inspect_target_image", lambda *_args: {})
    commands: list[list[str]] = []

    def capture_push(command: list[str]) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(deployment_cli, "_run", capture_push)
    inspect_payload = [{"RepoDigests": ["registry/api@sha256:" + "a" * 64]}]
    monkeypatch.setattr(
        deployment_cli,
        "_capture",
        lambda _command: _completed(json.dumps(inspect_payload)),
    )
    assert (
        main(
            [
                "image-push",
                "--image",
                "registry/api:gate-b",
                "--confirm",
                "PUSH:registry/api:gate-b",
            ]
        )
        == 0
    )
    assert commands == [["docker", "push", "registry/api:gate-b"]]

    parameters = tmp_path / "parameters.json"
    parameters.write_text(
        json.dumps(
            {
                "parameters": {
                    "deployContainerApp": {"value": False},
                    "deployPipelineJobs": {"value": False},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("FRAUD_POSTGRES_ADMIN_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="password is missing"):
        main(
            [
                "deploy",
                "--subscription",
                "sub",
                "--location",
                "eastus",
                "--parameters",
                str(parameters),
                "--confirm",
                "DEPLOY",
            ]
        )


def test_worker_success_paths_and_sanitized_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FRAUD_PUBLICATION_BATCH_ID", "batch-1")
    monkeypatch.setenv("FRAUD_PUBLICATION_CONFIRMATION", "PUBLISH:batch-1")
    monkeypatch.setenv("FRAUD_PIPELINE_CHUNK_INDEX", "0")
    calls: list[list[str]] = []

    def publication_run(command: list[str], _timeout: int) -> dict[str, object]:
        calls.append(command)
        if "plan" in command:
            return {"result": {"chunks": [{"batch_id": "batch-1"}]}}
        if "reconcile" in command:
            return {"result": {"publication_eligible": True}}
        return {"result": {"published": True}}

    monkeypatch.setattr(worker, "_run", publication_run)
    results = worker._publication_commands(10)
    assert len(results) == 2
    assert any("publish" in command for command in calls)

    monkeypatch.setattr(worker, "stage_artifacts", lambda _stage: (tmp_path, {"a": "b"}))
    monkeypatch.setattr(worker, "pipeline_lock", lambda: nullcontext())
    monkeypatch.setattr(worker, "_commands", lambda _stage: [["command"]])
    monkeypatch.setattr(
        worker,
        "_run",
        lambda _command, _timeout: {"result": {"status": "complete"}},
    )
    monkeypatch.setattr(worker, "upload_stage_outputs", lambda _stage: {"blob": "digest"})
    result = worker.run_stage("raw-projection", 10)
    assert result["status"] == "complete"
    assert result["staged_output_hashes"] == {"blob": "digest"}
    monkeypatch.setattr(worker, "run_stage", lambda _stage, _timeout: result)
    assert worker.main(["raw-projection", "--timeout-seconds", "10"]) == 0
    assert "complete" in capsys.readouterr().out
    monkeypatch.setattr(
        worker,
        "run_stage",
        lambda _stage, _timeout: (_ for _ in ()).throw(ValueError("sanitized failure")),
    )
    assert worker.main(["raw-projection"]) == 2
    output = capsys.readouterr().out
    assert "sanitized failure" in output
