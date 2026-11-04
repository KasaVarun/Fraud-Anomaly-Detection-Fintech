# 0001 Project Foundation

- **Status:** Accepted for Phase 1
- **Date:** 2026-10-31

## Context

The approved design requires several persistence technologies and later analytical workflows, but
Phase 1 must remain runnable without cloud services and must not introduce a partial version of the
approved 12-table relational schema.

## Decision

Use Python 3.12 with `uv` and a `src` package layout. FastAPI provides the application boundary.
Pydantic Settings loads environment configuration. Standard-library JSON logging provides
structured events without coupling the runtime to a hosted logging product.

Persistence, model-artifact, and reconciliation dependencies are expressed as typed protocols.
Phase 1 wires honest unavailable adapters: an omitted setting is `not_configured`, while a setting
that names an external dependency is `unavailable` until a real adapter exists and passes its
probe. Readiness is HTTP 200 when every configured dependency is healthy or optional/unconfigured,
and HTTP 503 when a configured dependency is unavailable. Liveness never depends on a database.

SQLAlchemy exposes empty metadata and Alembic is configured to consume it. Phase 2 will add all 12
tables, constraints, indexes, and SQL tests in one atomic migration.

## Consequences

- A new developer can run the API and its tests without PostgreSQL or Azure.
- Readiness cannot be mistaken for a successful database connection.
- Later adapters can enter through dependency injection without changing route contracts.
- Phase 1 intentionally has no application tables, production persistence behavior, PaySim data,
  model implementation, or Azure deployment.
