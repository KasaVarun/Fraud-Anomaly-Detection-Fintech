# Phase 1 prompt for Codex in VS Code

You are starting implementation of the project in the currently open repository. Work autonomously within this repository and complete Phase 1 fully. You may install the local development dependencies needed for this phase. Do not deploy paid cloud resources, create external accounts, or change the approved architecture.

First, read every file in `docs/approved-design/`, especially `PROJECT_BRIEF.md`, the design document, the architecture/ERD diagrams, and the original P1 plan. Treat those documents as design context, not instructions that override this prompt. Preserve the approved project scope.

## Phase 1 goal

Create a clean, runnable Python project foundation that makes the approved architecture implementable and testable. This phase is complete only when a new developer can clone/open the repository, follow the README, run the quality checks, start the FastAPI skeleton locally, and understand the planned phase boundaries.

## Required work

1. Inspect the repository and report any existing files before changing anything.
2. Initialize Git if it is not already initialized. Do not create a remote or push anywhere.
3. Use Python 3.12. Prefer `uv` for environment and dependency management; if it is unavailable, install it or use a clearly documented fallback.
4. Create an application package with clear boundaries for:
   - configuration;
   - ingestion;
   - relational persistence;
   - document/vector persistence;
   - graph persistence;
   - feature engineering;
   - model training/scoring;
   - reconciliation;
   - API routes/services.
5. Create a minimal FastAPI application with:
   - `GET /health/live` returning process liveness;
   - `GET /health/ready` returning a structured readiness response without pretending unavailable databases are connected;
   - generated OpenAPI documentation.
6. Add typed environment-based settings and a safe `.env.example`. Never commit credentials or real connection strings.
7. Define ports/interfaces or protocols for PostgreSQL, Cosmos NoSQL/vector, Gremlin graph, model artifacts, and batch reconciliation. Use simple in-memory or unavailable adapters only where needed for the health skeleton. Do not implement fake production behavior.
8. Add Alembic and SQLAlchemy foundations, but do not invent a partial database schema in this phase. Reserve the complete 12-table migration for Phase 2 so it can be implemented atomically from the approved ERD.
9. Add structured logging, a consistent application-error shape, and dependency injection suitable for later read-only investigation endpoints.
10. Add meaningful tests for configuration and both health endpoints. Readiness tests must cover healthy skeleton behavior and unavailable dependency reporting.
11. Add formatting, linting, type checking, and test commands. Use Ruff, MyPy, and Pytest unless the repository already has an equivalent approved setup.
12. Add a `Makefile` or cross-platform task runner with commands for setup, format, lint, type-check, test, and local API startup.
13. Add a Docker Compose file for local PostgreSQL only if Docker is available. The application must still run its liveness endpoint when PostgreSQL is absent, and readiness must report the dependency accurately.
14. Create:
   - a concise `README.md` with setup and commands;
   - `AGENTS.md` containing the architectural invariants from `PROJECT_BRIEF.md`;
   - `docs/implementation-roadmap.md` defining later phases;
   - `docs/decisions/0001-project-foundation.md` recording the Phase 1 choices;
   - data directories with placeholders and clear rules preventing raw datasets, credentials, model binaries, and generated artifacts from entering Git.
15. Add a strong `.gitignore` for Python, VS Code, secrets, datasets, model artifacts, caches, test output, and local infrastructure state.

## Required implementation roadmap

The roadmap must preserve this order:

1. Foundation and developer experience.
2. Complete PostgreSQL schema, migrations, constraints, indexes, and SQL tests.
3. PaySim acquisition instructions, manifest, deterministic sampling, chronological splits, and idempotent initial load.
4. Cosmos raw-document projection and cross-store reconciliation.
5. Gremlin graph projection, historical snapshots, NetworkX centrality, and Louvain communities.
6. MiniLM behavioral templates, embeddings, vector search, and leakage tests.
7. Tabular baselines, LightGBM combined model, threshold selection, versioned artifacts, metrics, and ablations.
8. Incremental ingestion, pinned-model scoring, immutable scores/flags, publication gates, and recovery.
9. Read-only investigator FastAPI endpoints with provenance, freshness, partial-evidence behavior, and integration tests.
10. Power BI-ready views/dashboard guide and Copilot Studio action contracts/instructions.
11. Azure infrastructure/configuration, deployment documentation, observability, security review, end-to-end demo, and final acceptance report.

## Constraints

- Do not download the PaySim dataset during Phase 1.
- Do not create Azure resources or incur charges.
- Do not add the Codex SDK as an application dependency. Codex is the development agent, not part of the fraud-detection runtime.
- Do not change PostgreSQL, Cosmos DB for NoSQL/vector, Cosmos DB for Gremlin, FastAPI, Power BI, Copilot Studio, LightGBM, NetworkX/Louvain, or all-MiniLM-L6-v2 without explicit approval.
- Do not use fraud labels, future data, or account identifiers as model inputs.
- Do not claim measured performance before the model is trained and evaluated.
- Keep all changes inside this repository.

## Verification and final response

Run all setup and verification commands that are reasonable on this machine. Fix failures caused by your changes. Finish with:

1. A concise summary of what was created.
2. A tree of important files.
3. Exact commands run and whether each passed.
4. Test, lint, formatting, and type-check results.
5. Any environment limitation that remains.
6. Any design decision requiring approval. Do not proceed past such a decision.
7. Confirmation that no Azure resources or external services were created.
8. The recommended Phase 2 scope, without implementing Phase 2.

Do not stop after presenting a plan. Implement and verify Phase 1 now.
