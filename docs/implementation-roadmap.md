# Implementation Roadmap

The project proceeds in reviewable phases. Each phase must pass its relevant tests and preserve the
approved architecture before the next begins.

1. **Foundation and developer experience (implemented).** Python 3.12 project, typed configuration, FastAPI
   health endpoints, ports, unavailable adapters, SQLAlchemy/Alembic foundations, logging, tests,
   and local quality commands.
2. **Complete PostgreSQL schema (implemented and validated on PostgreSQL 16).** All 12
   approved tables are in one migration with primary keys, foreign keys, checks, uniqueness,
   indexes, trigger-backed cross-row rules, metadata tests, and PostgreSQL integration tests.
3. **PaySim acquisition and initial relational load (implemented).** Acquisition instructions,
   source validation, stable original-row identities, deterministic fraud-preserving sampling,
   whole-step chronological splits, manifest integrity checks, an idempotent train/validation load,
   and PostgreSQL reconciliation are complete. The approved real Kaggle PaySim source was acquired,
   sampled to 200,000 rows with all fraud retained, loaded twice idempotently, and reconciled on
   PostgreSQL 16. Raw and generated files remain ignored and are not committed to Git.
4. **Cosmos raw-document projection and reconciliation (implemented; live Azure validation not
   run).** The versioned `transactions_raw` contract, `/sender_id` partition validation,
   deterministic conflict-detecting writes, bounded retries, point reads, paginated enumeration,
   restart-safe initial projection, exact cross-store reconciliation, readiness, and CLI operations
   are implemented and covered with controlled SDK tests. No Cosmos credentials or emulator were
   configured, so no live service claim is made.
5. **Gremlin graph projection and historical analysis (implemented; live Gremlin validation not
   run).** Versioned account, merchant, device, IP, money-flow, and observation contracts;
   deterministic `/pk` sharding; label-independent synthetic enrichment; restart-safe projection;
   exact reconciliation; strict historical NetworkX export; centrality; weighted evidence; seeded
   Louvain; and immutable PostgreSQL snapshots are implemented. One real-data snapshot at cutoff
   step 282 produced 224,915 account nodes, 777,590 collapsed weighted edges, and 282 deterministic
   communities. No Gremlin credentials or emulator were configured, so no live service claim is
   made.
6. **Behavioral templates and vector evidence (implemented; live Cosmos vector validation not
   run).** Deterministic strict-prior-step templates, fixed amount bands, the exact pinned
   `all-MiniLM-L6-v2` revision, 160,000 normalized 384-dimensional vectors, a separate embedding
   document contract/container adapter, local Parquet fallback, exact reconciliation, and
   earlier-training-fraud similarity evidence are implemented. The real local artifact reconciled
   with all 40,000 reserved test IDs absent. No Cosmos vector credentials were configured, so no
   live container claim is made.
7. **Model development and validation evaluation (implemented; corrected v2 release).** A versioned 160,000-row
   train/validation-only feature artifact combines safe tabular values, strict-prior-step causal
   graph evidence, and strict-earlier-training-fraud vector distance. A prevalence baseline,
   logistic baseline, and four equal-budget deterministic LightGBM feature-family variants are
   trained with an internal whole-step tuning split. The combined threshold is selected and locked
   on validation, immutable local artifacts and lineage are recorded, and model-version plus
   validation-evaluation rows are persisted idempotently. The v2 causal graph family exactly matches
   the Phase 5 collapsed weighted graph, and model artifacts use atomic immutable finalization. The
   earlier v1 artifact and six records remain superseded local development evidence. Reserved test
   rows remain untouched; test scoring and evaluation belong to Phase 8.
8. **Incremental ingestion and locked scoring (implemented and locally validated; live publication
   blocked).** The 40,000 reserved test rows are planned into 17 deterministic whole-step chunks,
   ingested chronologically, scored once with the hash-verified Phase 7 v2 release, assigned truthful
   strict-prior graph snapshots, and persisted as immutable scores, threshold-consistent flags, and
   per-chunk evaluations. Pooled locked-test metrics are stored in an ignored lineage artifact.
   Failure recovery and full rerun idempotency are tested. All batches remain pending because Cosmos
   NoSQL and Gremlin were not configured; the publication gate does not treat local artifacts as live
   Azure reconciliation. The corrected release lifecycle uses an authoritative v2 offline-evidence
   manifest, immutable staged artifacts, hash-anchored feature metadata, exact live-content
   reconciliation through persistence ports, published-batch-safe reruns, and an explicit dependency
   on the initial train/validation batch before chunk 0 can publish. The historical v1 evidence
   manifest is preserved and the authoritative v2 manifest is the release-integrity contract.
9. **Read-only investigator API (implemented and locally validated; live Cosmos/Gremlin not
   configured).** Versioned transaction, related-account, similar-behavior, dashboard summary,
   paginated flag, and freshness endpoints enforce published batches inside repository queries.
   Typed envelopes expose provenance, freshness, explicit partial/unavailable/version-mismatch
   states, bounded filter-bound cursors, sanitized failures, restrictive CORS, and query budgets.
   Optional raw, embedding, and graph evidence is contract-validated and reconciled field-by-field
   against published PostgreSQL lineage; strict whitelisted related/similarity contracts prevent
   protected or arbitrary store properties from reaching responses.
   The real local database exposes zero investigator transactions because all 18 batches remain
   pending; controlled published fixtures validate complete and partial cross-store behavior.
10. **Power BI and Copilot Studio contracts (implemented and locally validated; no live Power BI or
    Copilot deployment).** A reversible view-only migration exposes published transaction/score/flag
    facts, PaySim step/type trends, published aggregate evaluations, batch freshness, and pinned
    system lineage while keeping pending transaction facts and protected source truth out of
    reporting. A deterministic JSON Power BI semantic/dashboard contract defines stable types,
    relationships, ten explicit measures, Import/DirectQuery guidance, post-publication refresh,
    implementation-ready structured visuals, honest empty states, filters, warnings, and API
    drill-through. A
    generated OpenAPI action contract contains exactly the six approved read-only Phase 9 GET paths,
    and a versioned investigator instruction contract enforces API-only grounding, prompt-injection
    resistance, evidence distinctions, partial/freshness/cursor semantics, secret protection,
    human review, and eleven explicit response playbooks. Local migration, view-boundary,
    schema-drift, and no-mutation tests pass. No Power
    BI dataset, Copilot Studio agent, authentication connection, Azure endpoint, or refresh schedule
    is claimed; those remain Phase 11 deployment work.
11. **Azure deployment and acceptance (Gate A prepared locally; Gate B blocked pending account and
    cost approval).** Gate A provides read-only sanitized preflight, Bicep for the approved private
    PostgreSQL/two-account Cosmos/Container Apps architecture, managed identity and Key Vault,
    monitoring, a non-root hash-pinned production image, Entra JWT enforcement for `/v1`, guarded
    operations commands, a Swagger 2.0 Copilot connector template, an explicit Power BI VNet gateway
    or Desktop-only fallback decision, a private managed-identity artifact-staging account, a
    non-root analytical worker, eleven independently triggered and serialized pipeline jobs, cost
    categories, and a machine-readable Gate B acceptance plan. Azure CLI is installed locally but
    remains unauthenticated. Deployment images are built and inspected as Linux AMD64, Bicep accepts
    only immutable digest parameters for enabled workloads, and stage-aware job commands bind waits
    and logs to exact execution IDs. Private staging is blocked until a VPN/private runner route is
    selected and its Blob DNS/TLS checks pass. No Azure,
    Power BI, or Copilot Studio resource was created or connected. Gate B requires an authenticated
    Azure CLI, explicit subscription and region selection, provider/SKU/quota/policy validation,
    Power BI and Copilot licensing confirmation, subscription-specific cost approval, and separate
    authorization before provisioning or publication.

No phase may claim Azure resources, measured performance, or fraud-ring findings that have not been
created and verified.
