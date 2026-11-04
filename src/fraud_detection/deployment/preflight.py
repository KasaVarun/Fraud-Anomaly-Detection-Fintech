"""Read-only, secret-safe Azure Gate B0 discovery."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REQUIRED_PROVIDERS = (
    "Microsoft.App",
    "Microsoft.ContainerRegistry",
    "Microsoft.DBforPostgreSQL",
    "Microsoft.DocumentDB",
    "Microsoft.Insights",
    "Microsoft.KeyVault",
    "Microsoft.ManagedIdentity",
    "Microsoft.Network",
    "Microsoft.OperationalInsights",
    "Microsoft.Storage",
)
REQUIRED_EXTENSIONS = ("containerapp",)
PROJECT_TOKENS = ("fraud", "fintech")
SENSITIVE_FRAGMENTS = (
    "accesstoken",
    "clientsecret",
    "connectionstring",
    "credential",
    "password",
    "primarykey",
    "secondarykey",
    "secret",
    "token",
)
UUID_PATTERN = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
IDENTIFIER_KEYS = {
    "id",
    "principalid",
    "principal_id",
    "roleassignmentid",
    "role_assignment_id",
    "subscriptionid",
    "subscription_id",
    "tenantid",
    "tenant_id",
}
NAME_CHECKS = {
    "container_registry_candidate": ("acr", "fraudgateb0preflight"),
    "key_vault_candidate": ("keyvault", "fraudgateb0preflight"),
    "cosmos_nosql_candidate": ("cosmosdb", "fraudgateb0nosql"),
    "cosmos_gremlin_candidate": ("cosmosdb", "fraudgateb0gremlin"),
    "storage_candidate": ("storage", "fraudgateb0preflight"),
}


def _command_path(name: str) -> str | None:
    path = shutil.which(name)
    if path is None and name in {"az", "bicep"}:
        candidate = Path(__file__).resolve().parents[3] / ".tools/bin" / name
        if candidate.exists():
            return str(candidate)
    if path is None and name == "docker":
        candidate = Path("/Applications/Docker.app/Contents/Resources/bin/docker")
        if candidate.exists():
            return str(candidate)
    return path


def _safe_json_command(command: list[str], timeout: int = 45) -> tuple[Any | None, str | None]:
    """Run one bounded read-only command without retaining stderr or command output on failure."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, type(exc).__name__
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError:
        return None, "non_json_output"


def _mask_identifier(value: str | None) -> str | None:
    """Return a stable masked representation without retaining a complete cloud identifier."""
    if value is None:
        return None
    if value.startswith("[masked-id:") and value.endswith("]"):
        return value
    suffix = value[-4:] if len(value) >= 4 else "masked"
    return f"[masked-id:{suffix}]"


def _sanitize(value: Any) -> Any:
    """Recursively remove secrets and mask cloud identifiers, including UUIDs in free text."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, nested in value.items():
            normalized = key.lower().replace("-", "").replace("_", "")
            if any(fragment in normalized for fragment in SENSITIVE_FRAGMENTS):
                cleaned[key] = "[redacted]"
            elif key.lower() in IDENTIFIER_KEYS and isinstance(nested, str):
                cleaned[key] = _mask_identifier(nested)
            else:
                cleaned[key] = _sanitize(nested)
        return cleaned
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return UUID_PATTERN.sub(lambda match: f"[masked-id:{match.group(0)[-4:]}]", value)
    return value


def _project_related(value: str) -> bool:
    return any(token in value.lower() for token in PROJECT_TOKENS)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for nested in value for item in _strings(nested)]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _strings(nested)]
    return []


def _allowed_locations(assignments: list[dict[str, Any]]) -> list[str]:
    for assignment in assignments:
        label = f"{assignment.get('displayName', '')} {assignment.get('name', '')}".lower()
        if "allowed" not in label or not any(term in label for term in ("location", "region")):
            continue
        parameters = assignment.get("parameters", {})
        candidates = {
            item.lower() for item in _strings(parameters) if re.fullmatch(r"[a-z]+[a-z0-9]*", item)
        }
        if candidates:
            return sorted(candidates)
    return []


def _subscription_offer(az: str, subscription_id: str) -> dict[str, Any]:
    url = f"https://management.azure.com/subscriptions/{subscription_id}?api-version=2022-12-01"
    payload, error = _safe_json_command([az, "rest", "--method", "get", "--url", url])
    if not isinstance(payload, dict):
        return {"status": f"unknown:{error}"}
    policies = payload.get("subscriptionPolicies", {})
    if not isinstance(policies, dict):
        policies = {}
    return {
        "state": payload.get("state"),
        "quota_offer": policies.get("quotaId"),
        "spending_limit": policies.get("spendingLimit"),
        "location_placement": policies.get("locationPlacementId"),
    }


def _postgres_capability(payload: Any, error: str | None) -> dict[str, Any]:
    if error is not None or not isinstance(payload, list) or not payload:
        return {
            "status": "unavailable",
            "postgresql_16": False,
            "standard_b1ms": False,
            "reason": error or "empty_response",
        }
    root = payload[0] if isinstance(payload[0], dict) else {}
    reason = root.get("reason")
    versions = {
        str(item.get("name"))
        for item in root.get("supportedServerVersions", [])
        if isinstance(item, dict)
    }
    sku_names: set[str] = set()
    for edition in root.get("supportedServerEditions", []):
        if not isinstance(edition, dict) or edition.get("name") != "Burstable":
            continue
        sku_names.update(
            str(item.get("name"))
            for item in edition.get("supportedServerSkus", [])
            if isinstance(item, dict)
        )
    postgres_16 = "16" in versions
    b1ms = "Standard_B1ms" in sku_names
    return {
        "status": "restricted"
        if reason
        else "supported"
        if postgres_16 and b1ms
        else "unavailable",
        "postgresql_16": postgres_16,
        "standard_b1ms": b1ms,
        "reason": reason,
    }


def _provider_locations(payload: Any, resource_type: str) -> list[str]:
    if not isinstance(payload, dict):
        return []
    for item in payload.get("resourceTypes", []):
        if isinstance(item, dict) and str(item.get("resourceType", "")).lower() == resource_type:
            return sorted(
                str(location).lower().replace(" ", "") for location in item.get("locations", [])
            )
    return []


def _name_availability(az: str) -> dict[str, str]:
    results: dict[str, str] = {}
    for label, (service, candidate) in NAME_CHECKS.items():
        if service == "acr":
            command = [az, "acr", "check-name", "--name", candidate, "--output", "json"]
        elif service == "keyvault":
            command = [az, "keyvault", "check-name", "--name", candidate, "--output", "json"]
        elif service == "storage":
            command = [
                az,
                "storage",
                "account",
                "check-name",
                "--name",
                candidate,
                "--output",
                "json",
            ]
        else:
            command = [
                az,
                "cosmosdb",
                "check-name-exists",
                "--name",
                candidate,
                "--output",
                "json",
            ]
        payload, error = _safe_json_command(command)
        if isinstance(payload, bool):
            results[label] = "not_available" if payload else "available"
        elif isinstance(payload, dict) and isinstance(payload.get("nameAvailable"), bool):
            results[label] = "available" if payload["nameAvailable"] else "not_available"
        else:
            results[label] = f"unknown:{error or 'invalid_response'}"
    results["final_names"] = "not_selected"
    return results


def _discover_effective_roles(az: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Discover only the signed-in identity's direct and group-derived effective roles."""
    identity, identity_error = _safe_json_command(
        [az, "ad", "signed-in-user", "show", "--query", "id", "--output", "json"]
    )
    object_id = identity if isinstance(identity, str) else None
    if not object_id:
        return [], {
            "status": "unverified",
            "identity_scope": "signed_in_identity",
            "group_derived_roles": "unverified",
            "reason": f"signed_in_user_lookup_failed:{identity_error or 'invalid_response'}",
        }

    roles, role_error = _safe_json_command(
        [
            az,
            "role",
            "assignment",
            "list",
            "--assignee",
            object_id,
            "--all",
            "--include-inherited",
            "--include-groups",
            "--output",
            "json",
        ]
    )
    if role_error is not None or not isinstance(roles, list):
        return [], {
            "status": "unverified",
            "identity_scope": "signed_in_identity",
            "group_derived_roles": "unverified",
            "reason": f"identity_role_lookup_failed:{role_error or 'invalid_response'}",
        }

    role_counter = Counter(
        str(item.get("roleDefinitionName"))
        for item in roles
        if isinstance(item, dict) and item.get("roleDefinitionName")
    )
    effective_roles = [
        {"role": role, "assignment_count": count, "scope": "selected_subscription_or_inherited"}
        for role, count in sorted(role_counter.items())
    ]
    return effective_roles, {
        "status": "verified",
        "identity_scope": "signed_in_identity",
        "group_derived_roles": "included",
    }


def _authorization_assessment(
    effective_roles: list[dict[str, Any]], role_discovery: dict[str, str]
) -> dict[str, str]:
    verified = role_discovery.get("status") == "verified"
    owner = verified and any(item.get("role") == "Owner" for item in effective_roles)
    return {
        "resource_groups": "authorized_by_subscription_owner_role" if owner else "unverified",
        "role_assignments": "authorized_by_subscription_owner_role_subject_to_policy"
        if owner
        else "unverified",
        "app_registrations": "unknown_directory_permission_not_implied_by_azure_owner",
        "service_resources": (
            "authorized_by_subscription_owner_role_subject_to_policy_quota_and_"
            "provider_registration"
        )
        if owner
        else "unverified",
    }


def _base_report(
    az: str | None, docker: str | None, pac: str | None, pbi: str | None
) -> dict[str, Any]:
    return {
        "schema_version": "phase11-gate-b0-read-only-preflight-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "read_only",
        "cloud_mutations_performed": False,
        "tools": {
            "azure_cli": {"installed": az is not None},
            "docker": {"installed": docker is not None},
            "power_platform_cli": {"installed": pac is not None},
            "power_bi_tooling": {"installed": pbi is not None},
        },
        "azure": {
            "authenticated": False,
            "status": "not_authenticated",
            "tenant_id": None,
            "default_subscription": None,
            "available_subscriptions": [],
            "effective_roles": [],
            "effective_role_discovery": {
                "status": "unverified",
                "identity_scope": "signed_in_identity",
                "group_derived_roles": "unverified",
                "reason": "not_authenticated",
            },
            "resource_inventory": {
                "total_resource_group_count": 0,
                "project_resource_group_count": 0,
                "project_resource_groups": [],
                "total_resource_count": 0,
                "project_resource_count": 0,
                "project_resources": [],
            },
            "policies": {"assignments": [], "allowed_locations": []},
            "subscription_offer": {"status": "unknown"},
            "budget_count": "unknown",
            "provider_registration": {},
            "installed_extensions": [],
            "required_extensions": list(REQUIRED_EXTENSIONS),
            "required_extension_status": {
                extension: "unknown" for extension in REQUIRED_EXTENSIONS
            },
            "capabilities": {
                "postgresql_by_region": {},
                "container_apps": {"status": "unknown"},
                "cosmos_specialized": {"status": "unknown"},
            },
            "name_availability": {},
            "authorization_assessment": {
                "resource_groups": "unknown",
                "role_assignments": "unknown",
                "app_registrations": "unknown",
                "service_resources": "unknown",
            },
        },
        "candidate_regions": {
            "candidates_allowed_by_policy": [],
            "recommended_for_parameter_review": None,
            "status": "unverified_without_authenticated_policy_evidence",
        },
        "power_bi": {
            "authentication": "not_detectable" if pbi is None else "tool_present_unverified",
            "workspace_availability": "unknown",
            "licensing": "unknown",
            "vnet_data_gateway_availability": "unknown",
            "fabric_or_premium_capacity_availability": "unknown",
            "live_service_deployment_status": (
                "blocked_pending_capacity_workspace_and_vnet_gateway_validation"
            ),
        },
        "copilot_studio": {
            "power_platform_environment": "unknown",
            "licensing": "unknown",
            "authentication": "not_detectable" if pac is None else "tool_present_unverified",
        },
        "artifact_staging": {
            "selected_upload_runner_type": "unselected",
            "private_route_availability": "not_configured",
            "private_dns_validation": "not_run_without_deployed_endpoint",
            "https_validation": "not_run_without_deployed_endpoint",
            "upload_status": "blocked_until_private_route_is_verified",
        },
        "what_if": {
            "executed": False,
            "reason": "Gate B0 permits authenticated read-only discovery only.",
        },
        "required_user_actions": [],
    }


def build_preflight_report() -> dict[str, Any]:
    """Build complete Gate B0 evidence using only bounded, read-only commands."""
    az = _command_path("az")
    docker = _command_path("docker")
    pac = _command_path("pac")
    pbi = _command_path("pbi-tools")
    report = _base_report(az, docker, pac, pbi)

    if az is None:
        report["azure"]["status"] = "azure_cli_not_installed"
        report["required_user_actions"] = [
            "Install Azure CLI, authenticate explicitly, and select the approved subscription."
        ]
        sanitized = _sanitize(report)
        assert isinstance(sanitized, dict)
        return sanitized

    versions, version_error = _safe_json_command([az, "version", "--output", "json"])
    report["tools"]["azure_cli"].update({"version": _sanitize(versions), "error": version_error})
    extensions, _ = _safe_json_command([az, "extension", "list", "--output", "json"])
    if isinstance(extensions, list):
        installed = sorted(
            (
                {"name": item.get("name"), "version": item.get("version")}
                for item in extensions
                if isinstance(item, dict)
            ),
            key=lambda item: str(item["name"]),
        )
        report["azure"]["installed_extensions"] = installed
        names = {str(item["name"]) for item in installed}
        report["azure"]["required_extension_status"] = {
            extension: "installed" if extension in names else "not_installed"
            for extension in REQUIRED_EXTENSIONS
        }

    account, account_error = _safe_json_command([az, "account", "show", "--output", "json"])
    if account_error is not None or not isinstance(account, dict):
        report["required_user_actions"] = [
            "Authenticate Azure CLI and explicitly select the approved subscription."
        ]
        sanitized = _sanitize(report)
        assert isinstance(sanitized, dict)
        return sanitized

    subscription_id = str(account.get("id", ""))
    report["mode"] = "authenticated_read_only"
    report["azure"].update(
        {
            "status": "authenticated_read_only",
            "authenticated": True,
            "tenant_id": _mask_identifier(str(account.get("tenantId", ""))),
            "default_subscription": {
                "id": _mask_identifier(subscription_id),
                "name": account.get("name"),
                "state": account.get("state"),
                "is_default": bool(account.get("isDefault", True)),
                "account_type": account.get("user", {}).get("type"),
            },
        }
    )
    subscriptions, _ = _safe_json_command([az, "account", "list", "--all", "--output", "json"])
    if isinstance(subscriptions, list):
        report["azure"]["available_subscriptions"] = sorted(
            (
                {
                    "id": _mask_identifier(str(item.get("id", ""))),
                    "name": item.get("name"),
                    "state": item.get("state"),
                    "is_default": item.get("isDefault"),
                    "tenant_id": _mask_identifier(str(item.get("tenantId", ""))),
                }
                for item in subscriptions
                if isinstance(item, dict)
            ),
            key=lambda item: (not bool(item["is_default"]), str(item["name"])),
        )

    effective_roles, role_discovery = _discover_effective_roles(az)
    report["azure"]["effective_roles"] = effective_roles
    report["azure"]["effective_role_discovery"] = role_discovery

    groups, _ = _safe_json_command([az, "group", "list", "--output", "json"])
    safe_groups = [item for item in groups or [] if isinstance(item, dict)]
    project_groups = sorted(
        (
            {"name": item.get("name"), "location": item.get("location")}
            for item in safe_groups
            if _project_related(str(item.get("name", "")))
        ),
        key=lambda item: str(item["name"]),
    )
    resources, _ = _safe_json_command([az, "resource", "list", "--output", "json"])
    safe_resources = [item for item in resources or [] if isinstance(item, dict)]
    project_resources = sorted(
        (
            {
                "name": item.get("name"),
                "type": item.get("type"),
                "resource_group": item.get("resourceGroup"),
                "location": item.get("location"),
            }
            for item in safe_resources
            if _project_related(f"{item.get('name', '')} {item.get('resourceGroup', '')}")
        ),
        key=lambda item: (str(item["resource_group"]), str(item["name"])),
    )
    report["azure"]["resource_inventory"] = {
        "total_resource_group_count": len(safe_groups),
        "project_resource_group_count": len(project_groups),
        "project_resource_groups": project_groups,
        "total_resource_count": len(safe_resources),
        "project_resource_count": len(project_resources),
        "project_resources": project_resources,
    }

    assignments, _ = _safe_json_command([az, "policy", "assignment", "list", "--output", "json"])
    safe_assignments = [item for item in assignments or [] if isinstance(item, dict)]
    allowed_locations = _allowed_locations(safe_assignments)
    report["azure"]["policies"] = {
        "assignments": sorted(
            str(_sanitize(item.get("displayName") or item.get("name") or "unnamed"))
            for item in safe_assignments
        ),
        "allowed_locations": allowed_locations,
    }
    report["candidate_regions"]["candidates_allowed_by_policy"] = allowed_locations
    report["azure"]["subscription_offer"] = _subscription_offer(az, subscription_id)

    budgets, budget_error = _safe_json_command(
        [az, "consumption", "budget", "list", "--output", "json"]
    )
    report["azure"]["budget_count"] = (
        len(budgets) if isinstance(budgets, list) else f"unknown:{budget_error}"
    )

    provider_payloads: dict[str, Any] = {}
    provider_states: dict[str, str] = {}
    for provider in REQUIRED_PROVIDERS:
        payload, error = _safe_json_command(
            [az, "provider", "show", "--namespace", provider, "--output", "json"]
        )
        provider_payloads[provider] = payload
        provider_states[provider] = (
            str(payload.get("registrationState"))
            if isinstance(payload, dict)
            else f"unknown:{error}"
        )
    report["azure"]["provider_registration"] = provider_states

    postgres_by_region: dict[str, Any] = {}
    for region in allowed_locations:
        payload, error = _safe_json_command(
            [
                az,
                "postgres",
                "flexible-server",
                "list-skus",
                "--location",
                region,
                "--output",
                "json",
            ],
            timeout=60,
        )
        postgres_by_region[region] = _postgres_capability(payload, error)
    report["azure"]["capabilities"]["postgresql_by_region"] = postgres_by_region

    app_state = provider_states.get("Microsoft.App", "unknown")
    app_payload = provider_payloads.get("Microsoft.App")
    app_types = {
        resource_type: sorted(
            region
            for region in allowed_locations
            if region in _provider_locations(app_payload, resource_type)
        )
        for resource_type in ("managedenvironments", "containerapps", "jobs")
    }
    all_app_types_supported = all(
        set(allowed_locations).issubset(values) for values in app_types.values()
    )
    report["azure"]["capabilities"]["container_apps"] = {
        "status": "provider_not_registered" if app_state != "Registered" else "supported",
        "provider_registration": app_state,
        "allowed_region_resource_type_support": app_types,
        "all_required_types_advertised_in_allowed_regions": all_app_types_supported,
        "quota_check": "blocked_until_provider_registration_is_explicitly_approved"
        if app_state != "Registered"
        else "requires_gate_b_quota_validation",
    }

    cosmos_payload, cosmos_error = _safe_json_command(
        [az, "cosmosdb", "locations", "list", "--output", "json"]
    )
    report["azure"]["capabilities"]["cosmos_specialized"] = {
        "status": "unresolved" if cosmos_error else "location_metadata_available",
        "provider_registration": provider_states.get("Microsoft.DocumentDB", "unknown"),
        "location_metadata_result": cosmos_error or "available",
        "serverless_nosql_vector": "unverified",
        "serverless_gremlin": "unverified",
        "returned_location_count": len(cosmos_payload) if isinstance(cosmos_payload, list) else 0,
    }
    report["azure"]["name_availability"] = _name_availability(az)

    recommended = next(
        (
            region
            for region in ("centralus", *allowed_locations)
            if region in allowed_locations
            and postgres_by_region.get(region, {}).get("status") == "supported"
        ),
        None,
    )
    report["candidate_regions"].update(
        {
            "recommended_for_parameter_review": recommended,
            "status": "provisional_pending_cosmos_and_container_apps_quota_checks",
        }
    )
    report["azure"]["authorization_assessment"] = _authorization_assessment(
        effective_roles, role_discovery
    )
    report["required_user_actions"] = [
        "Approve a policy-allowed region after reviewing the authenticated cost evidence.",
        "Explicitly approve any future provider registration before specialized quota validation.",
        "Confirm Power BI/Fabric capacity and VNet data gateway entitlement or accept the "
        "Desktop-only fallback.",
        "Confirm Copilot Studio licensing and an eligible Power Platform environment.",
        "Select and prove a private artifact upload route before any upload.",
    ]
    sanitized = _sanitize(report)
    assert isinstance(sanitized, dict)
    return sanitized


def write_preflight_report(path: Path) -> dict[str, Any]:
    report = build_preflight_report()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
