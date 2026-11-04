"""Offline generation/check commands for Phase 10 contracts."""

from __future__ import annotations

import argparse

from fraud_detection.reporting.contracts import (
    build_deployment_copilot_swagger,
    canonical_json,
    contract_drift,
    write_contracts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("generate", "check", "connector"))
    parser.add_argument("--host")
    parser.add_argument("--tenant-id")
    parser.add_argument("--api-application-id")
    parser.add_argument("--connector-application-id")
    parser.add_argument("--scope")
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation == "generate":
        for path in write_contracts():
            print(f"generated {path}")
        return 0
    if args.operation == "connector":
        required = (
            args.host,
            args.tenant_id,
            args.api_application_id,
            args.connector_application_id,
            args.scope,
            args.output,
        )
        if not all(required):
            raise ValueError("deployment connector generation requires every connector parameter")
        document = build_deployment_copilot_swagger(
            host=args.host,
            tenant_id=args.tenant_id,
            api_application_id=args.api_application_id,
            connector_application_id=args.connector_application_id,
            scope=args.scope,
        )
        from pathlib import Path

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(document), encoding="utf-8")
        print(f"generated ignored deployment connector {output}")
        return 0
    drifted = contract_drift()
    if drifted:
        for path in drifted:
            print(f"contract drift: {path}")
        return 1
    print("reporting contracts are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
