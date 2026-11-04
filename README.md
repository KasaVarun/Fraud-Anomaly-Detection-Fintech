# Fraud and Anomaly Detection Network

The repository contains the validated Python foundation, PostgreSQL relational design, real PaySim
ingestion workflow, and Phase 4 Cosmos raw-document projection for the approved fraud-detection
architecture. It includes deterministic sampling, an idempotent train/validation PostgreSQL load,
the versioned `transactions_raw` document contract, conflict-detecting Cosmos writes, Gremlin
evidence contracts, leakage-safe historical NetworkX analysis, deterministic MiniLM behavioral
embeddings, immutable model releases, locked incremental scoring, and the published-only read-only
investigator API. Raw data remains ignored. Phase 10 reporting/Copilot work is represented by
deterministic local contracts; Azure deployment remains unimplemented.

## Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Docker with Compose for local PostgreSQL 16 integration tests

## Setup

```bash
uv sync --all-groups
cp .env.example .env
```

Do not add credentials or real cloud connection strings to `.env.example`. Local `.env` files are
ignored by Git.

## Local PostgreSQL and migrations

```bash
make db-up
make db-test-create
make db-migrate
make db-status
```

The Compose credentials are local-only. The application database is `fraud_detection_dev`; the
integration suite accepts only a localhost database whose name ends in `_test`. Stop PostgreSQL
with `make db-down`. Reset its named development volume only with
`make db-reset CONFIRM=reset`.

```bash
uv run alembic upgrade head
uv run alembic downgrade base
uv run alembic upgrade head
```

## Run the API

```bash
make run
```

Open `http://127.0.0.1:8000/docs` for generated OpenAPI documentation.

- `GET /health/live` reports process liveness.
- `GET /health/ready` reports dependency readiness. Unconfigured stores are named explicitly and
  do not block the Phase 1 skeleton. A configured dependency with only the Phase 1 unavailable
  adapter returns HTTP 503 until a later phase supplies and verifies its real adapter.

## Obtain and validate PaySim

The expected Kaggle source is `ealaxi/paysim1`, commonly distributed as
`PS_20174392719_1491204439457_log.csv`. Keep Kaggle credentials outside this repository. After
installing and authenticating the Kaggle CLI, either use the safe wrapper:

```bash
uv run fraud-ingest acquire --output-directory data/raw
```

or download the dataset yourself and place the unchanged CSV at
`data/raw/PS_20174392719_1491204439457_log.csv`.

Validate the original source before generating any derived files:

```bash
uv run fraud-ingest validate-source \
  --source data/raw/PS_20174392719_1491204439457_log.csv
```

Validation checks the exact PaySim columns, parsing, nonnegative steps and amounts, binary fraud
fields, participant identifiers, and nonempty content. It reports the source SHA-256, byte size,
row/fraud counts, prevalence, step range, detected types, timestamp, and code version without
logging credentials.

## Generate and validate the deterministic sample

```bash
uv run fraud-ingest sample \
  --source data/raw/PS_20174392719_1491204439457_log.csv \
  --sample-out data/processed/paysim_sample.parquet \
  --manifest-out data/processed/paysim_manifest.json \
  --target-size 200000 \
  --sampling-seed 73750 \
  --enrichment-seed 73751

uv run fraud-ingest validate-manifest \
  --source data/raw/PS_20174392719_1491204439457_log.csv \
  --manifest data/processed/paysim_manifest.json
```

Manifest schema `1.0.0` records source identity, whole-step boundaries, split counts, seeds,
selected-row digest, and generated sample hash. `source_row` is the zero-based original CSV data-row
position. Transaction IDs use `paysim:<full-source-sha256>:<source-row>`. Non-fraud selection uses a
stable modular rank derived from the source hash, sampling seed, original row, and split. Every
fraud row is retained. Generated Parquet and manifest files remain ignored by Git.

## Initial relational load and reconciliation

Start PostgreSQL and apply the schema first. If port 5432 is occupied, use the documented override:

```bash
POSTGRES_PORT=55433 docker compose up -d postgres
export FRAUD_POSTGRES_DSN='postgresql+psycopg://fraud_dev:fraud_dev_local_only@127.0.0.1:55433/fraud_detection_dev'
uv run alembic upgrade head

uv run fraud-ingest load-initial \
  --source data/raw/PS_20174392719_1491204439457_log.csv \
  --manifest data/processed/paysim_manifest.json

uv run fraud-ingest verify-load \
  --source data/raw/PS_20174392719_1491204439457_log.csv \
  --manifest data/processed/paysim_manifest.json
```

The loader uses bounded PostgreSQL inserts and conflict checks. Repeating the same command is
idempotent; conflicting reuse of a stable identity fails. Only train and validation rows load in
Phase 3. Test rows remain in the sample for later incremental-arrival simulation and never enter the
initial database population. The ingestion batch remains `pending` because final scoring and
publication gates belong to later phases.

## Cosmos raw-document projection

Phase 4 uses Azure Cosmos DB for NoSQL with an existing database and a `transactions_raw`
container whose partition key is exactly `/sender_id`. The project does not provision Azure
resources. Create or select the development database/container outside this phase, then configure
only local secrets in `.env` or environment variables:

```bash
FRAUD_COSMOS_NOSQL_ENDPOINT=https://<account>.documents.azure.com:443/
FRAUD_COSMOS_NOSQL_KEY=<local-secret>
FRAUD_COSMOS_NOSQL_DATABASE=<development-database>
FRAUD_COSMOS_RAW_CONTAINER=transactions_raw
FRAUD_COSMOS_REQUEST_TIMEOUT_SECONDS=10
FRAUD_COSMOS_BATCH_SIZE=500
FRAUD_COSMOS_PAGE_SIZE=500
FRAUD_COSMOS_MAX_RETRIES=3
```

Never commit the key or a populated `.env`. Validate the real database/container contract before
projection:

```bash
uv run fraud-ingest cosmos-validate

uv run fraud-ingest cosmos-project-initial \
  --source data/raw/PS_20174392719_1491204439457_log.csv \
  --manifest data/processed/paysim_manifest.json

uv run fraud-ingest cosmos-verify \
  --source data/raw/PS_20174392719_1491204439457_log.csv \
  --manifest data/processed/paysim_manifest.json
```

Raw document schema `1.0.0` and projection version `raw-paysim-v1` preserve the source event,
forensic balance fields, protected source truth, source/seed provenance, stable IDs, versions, and
a deterministic immutable-content digest. Identifiers, balances, source labels, and source flags
are explicitly forbidden predictive inputs.

Only the 160,000 train and validation transactions are eligible for initial projection. The 40,000
reserved test IDs remain absent. Repeated projection accepts identical documents, rejects immutable
conflicts, and resumes safely after partial failure. Cross-store verification categorizes missing,
orphaned, conflicting, invalid, duplicate, and reserved-test documents separately. The PostgreSQL
batch remains `pending`; Phase 4 does not publish it.

## Gremlin evidence projection and historical analysis

Phase 5 targets an existing Azure Cosmos DB for Apache Gremlin graph with partition key `/pk`. The
project does not provision the account, database, or graph. Configure local secrets only:

```bash
FRAUD_GREMLIN_ENDPOINT=wss://<account>.gremlin.cosmos.azure.com:443/
FRAUD_GREMLIN_USERNAME=/dbs/<database>/colls/<graph>
FRAUD_GREMLIN_DATABASE=<database>
FRAUD_GREMLIN_GRAPH=<graph>
FRAUD_GREMLIN_KEY=<local-secret>
FRAUD_GREMLIN_REQUEST_TIMEOUT_SECONDS=10
FRAUD_GREMLIN_MAX_RETRIES=3
FRAUD_GREMLIN_PROJECTION_BATCH_SIZE=250
FRAUD_GREMLIN_PARTITION_SHARDS=64
FRAUD_LOUVAIN_SEED=73751
FRAUD_LOUVAIN_RESOLUTION=1.0
```

Commands:

```bash
uv run fraud-ingest gremlin-validate

uv run fraud-ingest gremlin-project-initial \
  --source data/raw/PS_20174392719_1491204439457_log.csv \
  --manifest data/processed/paysim_manifest.json

uv run fraud-ingest gremlin-verify \
  --source data/raw/PS_20174392719_1491204439457_log.csv \
  --manifest data/processed/paysim_manifest.json

uv run fraud-ingest graph-analyze \
  --source data/raw/PS_20174392719_1491204439457_log.csv \
  --manifest data/processed/paysim_manifest.json \
  --cutoff-step 282
```

Graph schema `1.0.0` and projection `gremlin-evidence-v1` use PostgreSQL account IDs directly.
Vertex partition keys use `sha256-shard-v1`: SHA-256 of the version plus vertex ID, modulo the
configured shard count. Transaction edges use `PAYS` for `PAYMENT`; `CASH_IN`, `CASH_OUT`, `DEBIT`,
and `TRANSFER` use `TRANSFER_TO`, while retaining the original PaySim type.

Synthetic device and IP IDs use `synthetic-artifact-v1`, seed `73751`, stable transaction ID,
source row, and step, then map into fixed artifact pools to create controlled sharing. The algorithm
never reads source labels, source flags, balances, model output, or future events. These synthetic
links illustrate investigation paths and cannot establish real-world device/IP ownership or fraud.

Historical analysis includes only evidence with `step < cutoff_step`. It builds a simple undirected
account graph, removes self-loops, and collapses repeated pairs. Edge weight is:

```text
1.0 * transaction_count
+ 0.5 * distinct_shared_device_count
+ 0.25 * distinct_shared_ip_count
```

Degree centrality is `degree / (N - 1)`, or zero for fewer than two nodes. Weighted degree sums
incident edge weights. NetworkX Louvain uses the configured seed and resolution; communities are
numbered by the lexicographically smallest account in each sorted community. Community membership
is investigation evidence, never proof of fraud. Snapshot IDs bind the batch, strict cutoff, graph
projection version, and complete algorithm version. Identical reruns are accepted; conflicting
historical records fail without overwriting them. The ingestion batch remains `pending`.

## Behavioral templates and vector evidence

Phase 6 uses `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, Sentence Transformers `6.2.0`, normalized float32
output, and exactly 384 dimensions. Model files cache only under ignored
`artifacts/models/sentence-transformers/`.

Template version `behavior-template-v1` uses this exact canonical form:

```text
transaction_type=<TYPE>;amount_band=<BAND>;prior_sender_count=<N>;prior_receiver_count=<N>;prior_sender_type_count=<N>;prior_sender_distinct_counterparties=<N>
```

Fixed amount-band version `fixed-paysim-amount-bands-v1` defines `zero`, `gt0_le10`,
`gt10_le100`, `gt100_le1000`, `gt1000_le10000`, and `gt10000`. Templates process transactions by
whole step: all templates at step `S` use history with `historical_step < S`, then history updates
only after every transaction in step `S` is complete. Same-step source-row order never represents
elapsed time.

Account IDs are internal history keys and never appear in template text. Labels, source flags,
balances, model outputs, and future events are rejected as template inputs. Source labels only set
the protected `reference_eligible` metadata for training fraud.

Local commands, used when Cosmos is not configured:

```bash
uv run fraud-ingest templates-build \
  --source data/raw/PS_20174392719_1491204439457_log.csv \
  --manifest data/processed/paysim_manifest.json

uv run fraud-ingest embeddings-generate

uv run fraud-ingest embeddings-verify \
  --source data/raw/PS_20174392719_1491204439457_log.csv \
  --manifest data/processed/paysim_manifest.json \
  --artifact data/processed/behavior_embeddings.parquet

uv run fraud-ingest similar-behavior \
  --artifact data/processed/behavior_embeddings.parquet \
  --transaction-id <stable-transaction-id> \
  --top-k 10
```

The ignored local artifacts are `data/processed/behavior_templates.parquet` and
`data/processed/behavior_embeddings.parquet`. Embedding document schema `1.0.0` stores raw ID and
partition references, template/model versions, normalized vector metadata, source version, and
deterministic digests. Projection timestamps are excluded from immutable digests.

For Cosmos projection, create a separate `behavior_embeddings` container with partition key
`/sender_id` and vector policy `/vector`, `float32`, 384 dimensions, cosine distance. Configure:

```bash
FRAUD_COSMOS_EMBEDDING_CONTAINER=behavior_embeddings
FRAUD_EMBEDDING_BATCH_SIZE=256
FRAUD_EMBEDDING_PAGE_SIZE=500
FRAUD_VECTOR_SEARCH_LIMIT=10
FRAUD_VECTOR_SIMILARITY_METRIC=cosine
FRAUD_BEHAVIORAL_TEMPLATE_VERSION=behavior-template-v1
FRAUD_EMBEDDING_VERSION=minilm-l6-v2-1110a243-template-v1-normalized-float32
FRAUD_EMBEDDING_MODEL_CACHE=artifacts/models/sentence-transformers
FRAUD_BEHAVIORAL_TEMPLATE_ARTIFACT=data/processed/behavior_templates.parquet
FRAUD_EMBEDDING_LOCAL_ARTIFACT=data/processed/behavior_embeddings.parquet
```

Use `fraud-ingest embedding-validate`, then `embeddings-generate --project` only with an explicitly
configured development container. Raw and embedding containers remain separate, so embedding writes
cannot retrigger raw change-feed processing.

Similarity references must be training fraud, strictly earlier than the query, on matching model,
embedding, and template versions, and never the query itself. Validation and test labels cannot join
the reference pool. When no reference is eligible, distance is null and the missingness indicator is
true. Cosine distance is evidence, not a fraud probability or performance metric. The batch remains
`pending`.

## Phase 7 model development

Phase 7 materializes only the 120,000 training and 40,000 validation rows. Steps 355–743 and all
40,000 reserved test rows remain untouched until Phase 8. The corrected ignored feature artifact defaults to
`data/processed/model_features_v2.parquet`; versioned trusted model releases default to
`artifacts/models/releases/`.

The `paysim-safe-tabular-v1` family contains `log1p(amount)`, simulation hour/day, and fixed one-hot
PaySim transaction types. `causal-graph-prior-step-v2` exactly follows the Phase 5 graph: repeated
money-flow pairs contribute `1.0 × count`, each distinct shared device contributes `0.5`, and each
distinct shared IP contributes `0.25`; self-loops are removed and repeated pairs collapse. Weighted
degree sums these incident weights and degree centrality is `degree / (N - 1)`, or zero below two
prior nodes. It also adds prior pair count, current synthetic device/IP prior-account counts, and
new-account indicators. Every row at step S is computed before step S updates graph state. The
`minilm-earlier-training-fraud-distance-v1` family uses the nearest cosine distance to a strictly
earlier training-fraud embedding plus a missing-reference indicator. It requires Phase 6 logical
digest `50c904d3a82dc916eb389196ce35e8becb2b3bef747ec79cd91f893503d1f767`.

Identifiers, labels, source flags, balances, community IDs, and reference-eligibility claims are
non-predictive. Predictor matrices are constructed only from explicit versioned allowlists. Feature
preprocessing and class weights fit training rows only. LightGBM tuning uses whole steps 1–224 for
fit and 225–281 for internal selection, with the same three-configuration search budget for every
feature-family variant. Final validation is steps 282–354.

Build features and train the validation release locally:

```bash
fraud-model features-build \
  --source data/raw/PS_20174392719_1491204439457_log.csv \
  --manifest data/processed/paysim_manifest.json \
  --embeddings data/processed/behavior_embeddings.parquet
fraud-model train-release
```

The combined-model threshold uses maximum validation F1, ties prefer the higher threshold, and the
result is quantized to PostgreSQL `NUMERIC(5,4)` before confusion counts and metrics are recomputed.
This is validation performance, not test performance. Phase 8 must load reserved chronological
chunks, score them once with the pinned release, and report locked test metrics without retuning.

The earlier `model_features_v1.parquet` and its six model/evaluation records are preserved as
superseded local development evidence. The corrected v2 release uses feature logical digest
`57606ca504fd0b5d50a16e66ee56a2c16c949563df07fc51be34b1bed84d7190` and Phase 8 must pin combined
model `model:lightgbm_combined:1e9a3fc0a52dbac541bf4f92` with threshold `0.9496`. Model releases are
finalized atomically; identical reruns validate and reuse existing files, while conflicting bytes or
immutable metadata fail without overwriting the release. Run durations remain outside immutable
metadata and model identity.

## Phase 8 incremental locked scoring

Phase 8 divides reserved steps 355–743 into deterministic, gapless 24-whole-step chunks. Stable
chunk and batch IDs bind the source/sample hashes, sampling and enrichment seeds, split policy,
step range, and selected source-row digest. Chunks load and score chronologically; a later chunk
cannot complete before every required earlier chunk has immutable scores and an evaluation.

The workflow verifies and loads only
`model:lightgbm_combined:1e9a3fc0a52dbac541bf4f92`, model SHA-256
`4893e1a67ea0c8eb290a71332264842c001a879d2487a1ba08578cf21dfda498`, metadata SHA-256
`e5be2044c14f0a5381907f4fac9b1136be8a807b835f2d4c700c8d639d0a0cc0`, and threshold `0.9496`.
It never retrains or recalibrates. Test feature artifacts contain no labels. Labels join only after
probabilities and immutable score rows exist.

Local workflow commands:

```bash
fraud-incremental plan --source <PaySim.csv> --manifest <manifest.json>
fraud-incremental list --source <PaySim.csv> --manifest <manifest.json>
fraud-incremental prepare-offline --source <PaySim.csv> --manifest <manifest.json>
fraud-incremental run-next --source <PaySim.csv> --manifest <manifest.json>
fraud-incremental run-chunk --chunk-index 0 --source <PaySim.csv> --manifest <manifest.json>
fraud-incremental resume --chunk-index 0 --source <PaySim.csv> --manifest <manifest.json>
fraud-incremental reconcile --chunk-index 0 --source <PaySim.csv> --manifest <manifest.json>
fraud-incremental evaluate --source <PaySim.csv> --manifest <manifest.json>
fraud-incremental publish --chunk-index 0 --source <PaySim.csv> --manifest <manifest.json>
```

Each score references a deterministic per-step snapshot whose stored cutoff is one step before the
transaction; its algorithm lineage states that only evidence with `evidence.step < transaction.step` was
used. Same-step transactions are scored before that step updates graph history. Vector similarity
continues to use only Phase 6 training-fraud references.

The verified local locked-test evaluation contains 40,000 scores and 2,422 threshold-qualified
flags: TP 2,213, FP 209, TN 35,533, FN 2,045, precision 0.913708, recall 0.519728, F1 0.662575,
and pooled PR-AUC 0.788792. These are locked PaySim test metrics, not validation metrics or a
real-world performance claim. All 17 batches remain `pending`: Cosmos NoSQL raw/vector and Cosmos
Gremlin were not configured, so the live cross-store publication gate correctly blocks publication.
Local artifacts and controlled adapters are not live Azure evidence.

Offline evidence is immutable. Historical manifest
`data/processed/phase8/offline_evidence_manifest_v1.json` remains preserved; authoritative manifest
`data/processed/phase8/offline_evidence_manifest_v2.json` additionally anchors the feature-metadata
artifact, independently recomputed feature logical digest, test and Phase 6 embedding hashes and
logical digests, ordered allowlist, lineage hashes, row count, input/feature versions, and the
explicit no-label-input assertion. Preparation validates every existing artifact before any write,
reuses an identical release without touching timestamps, rejects tampering or missing files, and
stages a new release before finalizing its manifest last.

The preserved historical manifest SHA-256 is
`4c509d920ca4c3910656ef467108130c129a7fee8bc83af21256abfaffcc11a4`. The authoritative v2
manifest SHA-256 is `40196d9eac5f206cf346fed5093e536fee75a2afeb07b2ae6ddbabdc13e1d82f`,
and its anchored feature-metadata SHA-256 is
`63ac345d2aa4a2a96cbe2a74df5b3ce8865f291b0d86608e4978502fa1d4491d`.

Publication requires the initial train/validation batch to be published before chunk 0. Every later
chunk requires all earlier Phase 8 chunks to be published. Batch status is lifecycle state: a
published batch can rerun idempotently and is never downgraded to pending. Live reconciliation
compares complete validated raw documents, embeddings, graph vertices, and all three graph edges per
transaction against deterministic expected immutable content; IDs and counts alone are insufficient.
In the current local database, chunk 0 is blocked first by `initial_batch_not_published`; the initial
batch is never automatically promoted merely to permit Phase 8 publication.

## Phase 9 read-only investigator API

The API exposes only transactions belonging to ingestion batches whose status is `published`.
Unknown and unpublished transaction/account identities return the same safe 404 response. There is
no pending-data override and no route can score, flag, review, update, or publish data.

Routes:

- `GET /v1/investigations/transactions/{transaction_id}` combines published relational facts,
  locked-model prediction and threshold decision, optional fraud flag, snapshot lineage, and
  explicit raw/graph/vector evidence states. Source labels and source flags are never queried or
  returned.
- `GET /v1/investigations/accounts/{account_id}/related` performs one bounded parameterized Gremlin
  traversal using a published snapshot and returns transaction/shared-device/shared-IP evidence
  weights. Shared infrastructure and communities are not proof of coordination.
- `GET /v1/investigations/transactions/{transaction_id}/similar` performs bounded compatible-version
  vector search and reports cosine distance as evidence, not fraud probability.
- `GET /v1/dashboard/summary` returns bounded published-only transaction, amount, score, flag, and
  transaction-type aggregates without source-label performance metrics.
- `GET /v1/dashboard/flags` provides deterministic filter-bound cursor pagination over published
  flags, with a maximum page size of 100.
- `GET /v1/meta/freshness` returns published cutoff, pending count, pinned model/version lineage,
  sanitized dependency states, and explicit evidence completeness.

Investigation responses use a consistent envelope containing `data`, `provenance`, `freshness`,
`partial_evidence`, `evidence_status`, `warnings`, and optional `next_cursor`. Optional stores report
`available`, `not_configured`, `unavailable`, `not_found`, or `version_mismatch`; dependency failures
are never presented as clean negative evidence. PostgreSQL failure returns a sanitized 503.

Optional evidence is never trusted by ID alone. Raw documents must match the published batch,
source row, sender partition, receiver, step, split, transaction type, decimal amount, source hash,
seeds, schema/projection versions, and immutable digest. Embeddings must additionally match raw IDs,
batch/sender lineage, pinned MiniLM model and revision, template and embedding versions, normalized
384-dimensional content, and non-reference status for validation/test rows. Graph evidence must be
the exact deterministic three-edge bundle—one money-flow, one device observation, and one IP
observation—with matching endpoints, partitions, synthetic provenance, versions, and digests.
Valid but incompatible evidence reports `version_mismatch`; malformed or protected evidence reports
`unavailable`; neither exposes the conflicting payload.

Related-account and similarity store results pass strict `extra="forbid"` response contracts before
pagination or serialization. Related weights must equal transaction count plus `0.5 ×` distinct
shared devices plus `0.25 ×` distinct shared IPs. Similar matches must be strictly earlier than the
query, version-compatible, self-excluding, finite, and within cosine distance 0–2. Protected labels,
source flags, balance fields, arbitrary properties, and invalid store records cannot reach responses.

Identifiers, numeric inputs, traversal breadth, vector limits, and cursors are bounded. Cursors bind
their ordering position to the active filters. SQLAlchemy queries and Gremlin/Cosmos calls are
parameterized, deterministic, and query-budget tested. CORS is disabled by default and accepts only
explicit `FRAUD_CORS_ALLOWED_ORIGINS` values.

The real local database has zero published and 18 pending batches. Verification therefore returns
404 for a real pending transaction, zero published dashboard transactions and flags, and freshness
showing 18 pending batches with Cosmos/Gremlin `not_configured`. No live Azure validation is claimed.

## Phase 10 reporting and Copilot Studio contracts

Alembic revision `20261031_0002` adds five read-only reporting views without changing the approved
12-table schema:

- `reporting_published_transaction` selects the latest immutable score per transaction and model
  version using score time, input version, and score ID as deterministic tie-breakers. It includes
  published transaction, score, flag, model, and graph-snapshot lineage only.
- `reporting_published_transaction_trend` groups published facts by batch, model version, PaySim
  simulation step, and transaction type. It reports distinct transaction volume, scored/flagged
  counts, dataset-unit amount, flagged exposure, score range/mean, and flag rate.
- `reporting_published_model_evaluation` exposes aggregate confusion counts and stored metrics only
  for published evaluation batches. It never joins transaction labels.
- `reporting_batch_freshness` exposes one row per batch. Pending batches expose lifecycle status but
  zero transaction, score, or flag facts; published rows provide the safe freshness watermark.
- `reporting_system_lineage` exposes exactly one row containing the approved reporting contract,
  active model, feature, embedding, raw projection, and graph projection versions. The values are
  pinned to existing repository contracts and contain no service-health or deployment claim.

The migration downgrade removes only these views. The views do not expose protected source labels,
source flags, PaySim balance fields, model artifacts, embeddings, credentials, or connection data.
The current local database remains honestly empty at the published reporting boundary because all
18 batches are pending.

The versioned Power BI contract is JSON at
`contracts/power-bi/dashboard-contract.v1.json`. It defines stable data types, relationships,
published-only filters, the pinned reporting model, Import as the default storage mode, the bounded
DirectQuery alternative, post-publication refresh semantics, measures, pages, visuals, warnings,
and transaction drill-through to the read-only API. Its ten named measures include explicit source
views, formulas, and denominators. Executive Overview, Transaction Trends, Investigation Queue,
Model Performance, and Freshness and Lineage use structured visual definitions with sources,
fields, filters, sorting, warnings, and honest zero-published-data states. PaySim amount is labeled
as synthetic dataset units; flags, communities, and synthetic device/IP evidence are investigation
signals rather than proof.

Copilot Studio uses two reviewable JSON contracts:

- `contracts/copilot/investigator-actions.openapi.json` is generated from FastAPI and contains
  exactly the six approved Phase 9 GET actions. Health and write operations are excluded. The
  schema preserves bounded parameters, response envelopes, sanitized structured errors, and stable
  operation IDs. Authentication and the deployed HTTPS server remain Phase 11 work.
- `contracts/copilot/investigator-instructions.v1.json` grounds answers only in those API responses,
  treats returned/user text as untrusted data, distinguishes relational/model/graph/vector evidence,
  preserves partial and freshness semantics, requires cursor discipline and human review, and
  forbids secrets, direct store access, arbitrary properties, legal conclusions, guilt claims, and
  adverse account/customer decisions. Eleven explicit response playbooks cover found and unpublished
  transactions, empty published data, unconfigured dependencies, partial or mismatched evidence,
  temporary failures, similarity, related accounts, dashboard summaries, and paginated fraud flags.

Generate or verify all three contracts offline; neither command starts the API or contacts a cloud
service:

```bash
uv run fraud-reporting-contracts generate
uv run fraud-reporting-contracts check
```

The committed JSON is canonical, sorted, and deterministic. CI should run `check` so any FastAPI
schema or instruction drift fails review. Power BI workspace configuration, refresh credentials,
Copilot connection/authentication, and deployment are intentionally deferred to Phase 11.

## Phase 11 Gate A deployment readiness

Gate A prepares deployment without creating or changing Azure, Power BI, or Copilot Studio
resources. The sanitized report at `deployment/preflight-report.json` records the actual local
tool/account state without tokens, keys, passwords, or connection strings. Regenerate it read-only:

```bash
uv run fraud-deploy preflight
```

The current workstation has Azure CLI 2.91.0 and Docker, but no Power Platform CLI or Power BI CLI
tooling. Azure CLI is not authenticated; Gate A did not invoke `az login`, open a browser, select a
subscription, or install a cloud extension. Tenant/subscription identity, provider registration,
roles, quotas, policies, regional SKU availability, Power BI workspace/license/capacity, VNet data
gateway, and Copilot Studio environment/license are therefore unverified. Before Gate B, run
`az login` yourself, then choose the explicitly approved subscription with
`az account set --subscription <subscription-id>`. Also confirm Power BI access at
`https://app.powerbi.com/` and Copilot Studio environment/licensing at
`https://copilotstudio.microsoft.com/`. Gate A never initiates interactive login.

The subscription-scope template is `infra/main.bicep`, with platform resources in
`infra/modules/platform.bicep` and a secret-free review sample in
`infra/parameters/gate-b.example.json`. It preserves the approved architecture: ACR, Log Analytics,
Application Insights, VNet-integrated Container Apps, private PostgreSQL 16, a private Cosmos NoSQL
account with `/sender_id` raw and 384-dimensional cosine-vector containers, a separate private
serverless Gremlin account with `/pk`, Key Vault, managed identity, narrow ACR/Key Vault roles, and
eleven manually triggered private-network pipeline jobs. The NoSQL account disables local key authentication and grants the API
identity the built-in Cosmos data-contributor role. PostgreSQL and Gremlin credentials remain
technical requirements of the current clients and are referenced through Key Vault rather than
image settings or outputs. Cost-affecting capacity, storage, retention, scaling, and deployment
toggles are parameters. No secret is an output.

The FastAPI-generated OpenAPI 3 contract remains API documentation and is not represented as a
Power Platform import file. Copilot Studio uses the deterministic Swagger 2.0 placeholder template
at `contracts/copilot/investigator-actions.swagger.json`. It contains exactly six HTTPS GET actions
and an Entra OAuth 2.0 authorization-code/OBO-compatible definition. Generate a tenant-specific,
ignored connector only after Gate B app registrations exist:

```bash
uv run fraud-reporting-contracts connector \
  --host <deployed-host> \
  --tenant-id <tenant-id> \
  --api-application-id <api-client-id> \
  --connector-application-id <connector-client-id> \
  --scope Fraud.Investigate \
  --output contracts/copilot/generated/investigator-actions.swagger.json
```

The exporter rejects missing values and placeholders. Connector client secrets are never stored in
either contract.

Private Power BI Service connectivity requires an eligible Premium/Fabric capacity and a Power BI
virtual network data gateway attached to the dedicated delegated subnet. The complete decision and
evidence contract is `deployment/power-bi-connectivity.json`. Until tenant, workspace, capacity,
region, administrators, private DNS, TLS connector, and gateway cost are proven, live Power BI
status is `unknown`. Without eligible capacity, live Power BI Service is blocked; only an approved
Power BI Desktop demonstration over a secure private route is permitted. PostgreSQL must never be
made public as a workaround.

Operational artifact staging uses a private Standard_LRS Blob Storage account with lifecycle
expiry and managed-identity read access. It is deployment staging, not a new analytical store. The
worker image contains no PaySim source, sample, model, or credentials. A deployment-specific ignored
manifest supplies exact SHA-256 values and provenance. The full raw PaySim source is accepted only
when its exact source hash has an explicit `APPROVE_RESTRICTED_SOURCE:<sha256>` approval.
The upload runner must have a verified VPN, ExpressRoute, self-hosted VNet runner, or administrative
VM route described by `deployment/artifact-staging-connectivity.json`. Upload refuses public DNS or
unreachable endpoints, verifies every uploaded blob by metadata and content readback, and uploads
the manifest last. No private upload route is currently selected, so artifact staging remains a
Gate B blocker; public Storage access is not an allowed fallback.

`Dockerfile.worker` and `deployment/worker-requirements.lock` define the non-root worker. The eleven
manual jobs are schema migration, initial relational load, raw projection, graph projection,
behavioral embeddings, graph snapshot, feature/model preparation, incremental execution,
reconciliation, evaluation, and controlled publication. All jobs share one PostgreSQL advisory
lock, one-replica manual execution, bounded timeouts/retries, private endpoints, managed identity,
and Key Vault references. Publication is separate and requires both live reconciliation eligibility
and exact `PUBLISH:<batch-id>` confirmation.

The candidate parameter-review region is `eastus`, with `eastus2` and `westus2` alternatives. This
is not a deployment decision: Gate B must verify the approved subscription supports Container Apps,
PostgreSQL Flexible Server 16 `Standard_B1ms`, Cosmos NoSQL vector search/serverless, Cosmos Gremlin
serverless, private endpoints, and the selected licenses in the chosen region. Review resource
categories and cost drivers in `deployment/cost-review.json` before approval.

The production image uses `Dockerfile`, `.dockerignore`, and the hash-pinned API-only dependency set
in `deployment/api-requirements.lock`. It runs Python 3.12 as UID/GID 10001, listens on port 8000,
uses explicit liveness, handles SIGTERM with a bounded graceful shutdown, and never uses reload.
Data, models, reports, tests, local databases, caches, credentials, and `.env` files are excluded.
Both API and worker deployment builds use BuildKit with explicit `--platform linux/amd64 --load`.
Inspection rejects non-Linux, non-AMD64, and root images. Final Bicep deployment parameters are
named `containerImageDigest` and `workerImageDigest`; deployment commands reject enabled workloads
unless each value is an immutable `@sha256:` reference.

Microsoft Entra authentication protects every `/v1` route in non-local environments. Health and
OpenAPI remain anonymous. Production validates RS256 signature, issuer, audience, expiry, tenant,
and the `Fraud.Investigate` scope or app role. `local_bypass` is accepted only in `local` and `test`;
configuration validation prevents it in development, staging, or production. Gate B supplies the
real tenant/application registration, and the six-action Copilot OpenAPI contract already declares
the OpenID Connect security scheme.

Safe operational commands:

```bash
uv run fraud-deploy preflight
uv run fraud-deploy bicep-validate
uv run fraud-deploy cost-review
uv run fraud-deploy container-build
uv run fraud-deploy container-inspect
uv run fraud-deploy worker-build
uv run fraud-deploy worker-inspect
uv run fraud-deploy artifacts-validate --manifest <private-manifest> --root <artifact-root>
uv run fraud-deploy staging-network-check --account <staging-account> \
  --private-cidr 10.42.3.0/24
uv run fraud-deploy artifacts-upload --manifest <private-manifest> --root <artifact-root> \
  --account <staging-account> --container deployment-artifacts \
  --private-cidr 10.42.3.0/24 --confirm UPLOAD
uv run fraud-deploy jobs-update <approved deployment arguments> --confirm DEPLOY
uv run fraud-deploy start-stage --resource-group <rg> --environment <env> \
  --unique-suffix <suffix> --stage <approved-stage> <required-stage-parameters>
uv run fraud-deploy wait-stage --resource-group <rg> --environment <env> \
  --unique-suffix <suffix> --stage <approved-stage> --execution-name <execution>
uv run fraud-deploy job-logs --resource-group <rg> --environment <env> \
  --unique-suffix <suffix> --stage <approved-stage> --execution-name <execution>
uv run fraud-deploy record-counts
uv run fraud-deploy publication-preview --resource-group <rg> --environment <env> \
  --unique-suffix <suffix> --chunk-index <index> \
  --restricted-source-sha256 <sha256> \
  --restricted-source-confirmation APPROVE_RESTRICTED_SOURCE:<sha256>
uv run fraud-deploy publish --resource-group <rg> --environment <env> \
  --unique-suffix <suffix> --batch-id <exact-batch-id> --chunk-index <index> \
  --restricted-source-sha256 <sha256> \
  --restricted-source-confirmation APPROVE_RESTRICTED_SOURCE:<sha256> \
  --confirm PUBLISH:<exact-batch-id>
uv run fraud-deploy contracts check
uv run fraud-deploy health --base-url <https-url>
uv run fraud-deploy smoke --base-url <https-url> --token <short-lived-token>
uv run fraud-deploy powerbi-validate
uv run fraud-deploy copilot-validate
uv run fraud-deploy teardown-preview --environment <env> --resource-group <rg>
```

Create/update commands are deliberately confirmation-gated. Deployment requires `--confirm DEPLOY`,
publication requires exact batch/source confirmations, and teardown requires the exact
`--confirm DELETE:<resource-group>:<environment>` value. Do not execute them until Gate A findings,
subscription, region, licenses, and estimated costs receive explicit approval. The machine-readable
Gate B acceptance sequence is `deployment/acceptance-plan.json`.

## Quality commands

```bash
make format
make format-check
make lint
make type-check
make test
make test-schema
make check
```

## Configuration

Settings use the `FRAUD_` environment prefix. See `.env.example` for current settings. Secret
values use Pydantic secret types and are never included in readiness output or logs.

## Architecture boundaries

The package separates configuration, ingestion, relational persistence, document/vector
persistence, graph persistence, feature engineering, model operations, reconciliation, and API
orchestration. Phase 1 exposes protocols and honest unavailable adapters only. It does not simulate
production database behavior.

Alembic revision `20261031_0001` creates all 12 approved tables atomically. PostgreSQL enforces
relational integrity and trigger-backed merchant/flag rules; services retain temporal,
immutability, cross-store, and publication responsibilities. See `docs/postgresql-schema.md` for
the full contract and `docs/implementation-roadmap.md` for later phase boundaries.

## Data and artifact safety

The `data/` and `artifacts/` trees retain only rules and placeholders in Git. Raw datasets,
generated samples, model binaries, reports, credentials, and local infrastructure state are ignored.
Raw PaySim data, deterministic samples, and generated manifests stay under ignored `data/`
directories and must never be committed.
