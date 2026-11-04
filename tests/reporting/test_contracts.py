"""Deterministic Power BI and Copilot Studio contract tests."""

import json
from pathlib import Path
from typing import Any

import pytest

from fraud_detection.reporting.cli import main
from fraud_detection.reporting.contracts import (
    ACTION_OPERATIONS,
    APPROVED_LINEAGE,
    COPILOT_INSTRUCTIONS_PATH,
    COPILOT_OPENAPI_PATH,
    COPILOT_SWAGGER_PATH,
    POWER_BI_CONTRACT_PATH,
    REPORTING_VIEW_COLUMNS,
    build_copilot_instructions,
    build_copilot_openapi_contract,
    build_copilot_swagger_contract,
    build_deployment_copilot_swagger,
    build_power_bi_contract,
    canonical_json,
    contract_drift,
)

REQUIRED_PAGES = {
    "Executive Overview": {
        "Published Transactions",
        "Scored Transactions",
        "Flagged Transactions",
        "Flag Rate",
        "Total Transaction Amount",
        "Latest Published Batch",
        "Latest Published Cutoff Step",
        "Pending Batch Count",
    },
    "Transaction Trends": {
        "Transaction count by PaySim step",
        "Flag count by step",
        "Flag rate by step",
        "Transaction-type distribution",
        "Score distribution",
        "Transaction amount by type",
    },
    "Investigation Queue": {"Published flagged transaction queue"},
    "Model Performance": {
        "TP, FP, TN, and FN",
        "Precision, Recall, and F1",
        "PR-AUC",
        "Evaluation lineage",
    },
    "Freshness and Lineage": {
        "Latest published batch",
        "Pending batch count",
        "Published minimum and maximum step",
        "Published transaction, score, and flag counts",
        "Latest score time",
        "Approved system lineage",
    },
}
REQUIRED_MEASURES = {
    "Published Transactions",
    "Scored Transactions",
    "Flagged Transactions",
    "Flag Rate",
    "Total Transaction Amount",
    "Flagged Exposure",
    "Latest Published Batch",
    "Latest Published Cutoff Step",
    "Pending Batch Count",
    "Latest Score Time",
}
REQUIRED_QUEUE_FIELDS = {
    "transaction_id",
    "step",
    "transaction_type",
    "amount",
    "sender_id",
    "receiver_id",
    "fraud_probability",
    "threshold",
    "flag_reason",
    "review_status",
    "model_version_id",
    "input_version",
    "snapshot_id",
    "batch_id",
    "scored_at",
}
REQUIRED_PLAYBOOKS = {
    "published_transaction_found",
    "unknown_or_unpublished_transaction",
    "no_published_data",
    "dependency_not_configured",
    "partial_evidence",
    "version_mismatch",
    "temporary_service_failure",
    "similarity_results",
    "related_account_results",
    "dashboard_summary",
    "fraud_flag_list",
}


def _load(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return value


def test_generated_contracts_are_deterministic_and_current() -> None:
    assert contract_drift() == ()
    assert main(["check"]) == 0
    assert POWER_BI_CONTRACT_PATH.read_text(encoding="utf-8") == canonical_json(
        build_power_bi_contract()
    )
    assert COPILOT_OPENAPI_PATH.read_text(encoding="utf-8") == canonical_json(
        build_copilot_openapi_contract()
    )
    assert COPILOT_INSTRUCTIONS_PATH.read_text(encoding="utf-8") == canonical_json(
        build_copilot_instructions()
    )
    assert COPILOT_SWAGGER_PATH.read_text(encoding="utf-8") == canonical_json(
        build_copilot_swagger_contract()
    )


def test_power_bi_contract_names_exact_views_columns_and_safe_sources() -> None:
    contract = _load(POWER_BI_CONTRACT_PATH)
    views = {view["name"]: view for view in contract["views"]}
    assert set(views) == set(REPORTING_VIEW_COLUMNS)
    for name, columns in REPORTING_VIEW_COLUMNS.items():
        assert tuple(field["name"] for field in views[name]["fields"]) == columns
        assert all(field["powerBiType"] for field in views[name]["fields"])
    assert contract["storageMode"]["default"] == "Import"
    assert contract["storageMode"]["directQuery"]["allowed"] is True
    assert len(contract["relationships"]) == 3
    assert APPROVED_LINEAGE.items() <= contract["activeReportingModel"].items() or (
        contract["activeReportingModel"]["modelVersionId"]
        == APPROVED_LINEAGE["active_model_version"]
    )
    serialized = json.dumps(contract).lower()
    for forbidden in (
        "is_fraud",
        "source_flag",
        "oldbalanceorg",
        "newbalanceorig",
        "oldbalancedest",
        "newbalancedest",
        "credential",
        "connection string",
        "384-dimensional vectors",
    ):
        assert forbidden not in serialized


def test_power_bi_pages_visuals_measures_and_empty_states_are_complete() -> None:
    contract = _load(POWER_BI_CONTRACT_PATH)
    measures = {measure["name"]: measure for measure in contract["measures"]}
    assert set(measures) == REQUIRED_MEASURES
    for measure in measures.values():
        assert measure["sourceView"] in REPORTING_VIEW_COLUMNS
        assert measure["expression"]
        assert measure["denominator"]

    pages = {page["name"]: page for page in contract["pages"]}
    assert set(pages) == set(REQUIRED_PAGES)
    visual_keys = {
        "title",
        "visualType",
        "sourceView",
        "categoryFields",
        "valueFields",
        "measure",
        "filters",
        "sort",
        "drillThroughField",
        "warning",
        "emptyState",
    }
    for page_name, required_titles in REQUIRED_PAGES.items():
        visuals = pages[page_name]["visuals"]
        assert all(isinstance(visual, dict) for visual in visuals)
        assert {visual["title"] for visual in visuals} == required_titles
        for visual in visuals:
            assert visual_keys <= set(visual)
            assert visual["sourceView"] in REPORTING_VIEW_COLUMNS
            assert visual["emptyState"]

    queue = pages["Investigation Queue"]["visuals"][0]
    assert set(queue["valueFields"]) == REQUIRED_QUEUE_FIELDS
    assert queue["drillThroughField"] == "transaction_id"
    assert queue["apiHandoff"] == "/v1/investigations/transactions/{transaction_id}"

    lineage = next(
        visual
        for visual in pages["Freshness and Lineage"]["visuals"]
        if visual["title"] == "Approved system lineage"
    )
    assert set(lineage["valueFields"]) == set(APPROVED_LINEAGE)


def test_copilot_openapi_contains_exact_six_get_actions_and_structured_errors() -> None:
    contract = _load(COPILOT_OPENAPI_PATH)
    assert set(contract["paths"]) == {path for path, _ in ACTION_OPERATIONS}
    assert "security" not in contract
    assert contract["components"]["securitySchemes"] == {
        "entraOpenId": {
            "description": ("Gate B replaces the tenant placeholder and grants Fraud.Investigate."),
            "openIdConnectUrl": (
                "https://login.microsoftonline.com/{tenant-id}/v2.0/"
                ".well-known/openid-configuration"
            ),
            "type": "openIdConnect",
        }
    }
    for path, operation_id in ACTION_OPERATIONS:
        assert set(contract["paths"][path]) == {"get"}
        operation = contract["paths"][path]["get"]
        assert operation["operationId"] == operation_id
        assert operation["summary"]
        assert operation["description"]
        assert operation["security"] == [{"entraOpenId": []}]
        assert "200" in operation["responses"]
        assert "401" in operation["responses"]
        assert "403" in operation["responses"]
        assert "500" in operation["responses"]
        assert "ErrorResponse" in json.dumps(operation["responses"])
    assert "/health/live" not in contract["paths"]
    assert "/health/ready" not in contract["paths"]
    assert not any(
        method.lower() in {"post", "put", "patch", "delete"}
        for methods in contract["paths"].values()
        for method in methods
    )


def test_copilot_swagger_two_is_connector_compatible_and_refs_resolve() -> None:
    contract = _load(COPILOT_SWAGGER_PATH)
    assert contract["swagger"] == "2.0"
    assert contract["schemes"] == ["https"]
    assert set(contract["paths"]) == {path for path, _ in ACTION_OPERATIONS}
    assert contract["securityDefinitions"]["entraOAuth2"]["flow"] == "accessCode"
    assert "clientCredentials" not in json.dumps(contract)
    for path, operation_id in ACTION_OPERATIONS:
        assert set(contract["paths"][path]) == {"get"}
        operation = contract["paths"][path]["get"]
        assert operation["operationId"] == operation_id
        assert operation["security"] and "entraOAuth2" in operation["security"][0]

    unsupported = {"openapi", "components", "content", "requestBody", "anyOf", "oneOf", "const"}

    def walk(value: object) -> None:
        if isinstance(value, dict):
            assert not unsupported & set(value)
            reference = value.get("$ref")
            if isinstance(reference, str):
                assert reference.startswith("#/definitions/")
                assert reference.rsplit("/", maxsplit=1)[-1] in contract["definitions"]
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(contract)
    serialized = json.dumps(contract).lower()
    assert "client_secret" not in serialized
    assert "client secret" not in serialized


def test_deployment_connector_requires_all_values_and_contains_no_placeholders() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        build_deployment_copilot_swagger(
            host="",
            tenant_id="tenant",
            api_application_id="api",
            connector_application_id="connector",
            scope="Fraud.Investigate",
        )
    contract = build_deployment_copilot_swagger(
        host="fraud.example.test",
        tenant_id="11111111-1111-1111-1111-111111111111",
        api_application_id="22222222-2222-2222-2222-222222222222",
        connector_application_id="33333333-3333-3333-3333-333333333333",
        scope="Fraud.Investigate",
    )
    serialized = json.dumps(contract)
    assert "{" not in contract["host"]
    for placeholder in (
        "{deployed-api-host}",
        "{entra-tenant-id}",
        "{api-application-client-id}",
        "{connector-application-client-id}",
        "{Fraud.Investigate-scope}",
    ):
        assert placeholder not in serialized


def test_copilot_instructions_enforce_grounding_safety_and_human_review() -> None:
    instructions = _load(COPILOT_INSTRUCTIONS_PATH)
    text = json.dumps(instructions).lower()
    for required in (
        "only the six approved fastapi action responses",
        "prompt",
        "untrusted data",
        "partial_evidence",
        "freshness watermark",
        "next_cursor",
        "not proof of fraud",
        "never fraud probability",
        "humanreviewrequired",
        "account closure",
        "payment denial",
        "legal",
        "direct postgresql, cosmos db, or gremlin access",
        "credentials",
        "phase 11",
    ):
        assert required in text
    assert instructions["approvedActions"] == [
        operation_id for _, operation_id in ACTION_OPERATIONS
    ]


def test_copilot_response_playbooks_are_complete_and_safe() -> None:
    instructions = _load(COPILOT_INSTRUCTIONS_PATH)
    playbooks = instructions["responsePlaybooks"]
    assert set(playbooks) == REQUIRED_PLAYBOOKS
    required_fields = {
        "triggerOrApiCondition",
        "requiredResponseContent",
        "requiredLineageCitations",
        "requiredWarning",
        "prohibitedClaims",
        "recommendedNextAction",
    }
    for playbook in playbooks.values():
        assert set(playbook) == required_fields
        assert all(playbook[field] for field in required_fields)

    serialized = json.dumps(playbooks).lower()
    for required in (
        "a 404 means no published evidence was found",
        "honest empty-state response",
        "list each unavailable, missing, or mismatched section",
        "must never be summarized as valid evidence",
        "never fraud probability",
        "graph evidence, never proof of coordination",
        "flags require human review",
        "next_cursor",
        "never invent a cursor",
        "temporary failure must not be converted into no fraud",
        "cite freshness and the active model",
    ):
        assert required in serialized
