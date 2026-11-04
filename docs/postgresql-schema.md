# PostgreSQL Relational Schema

Phase 2 creates the approved relational design in Alembic revision `20261031_0001`. The revision
creates exactly 12 application tables. It does not create KPI tables or reporting views.

## Local PostgreSQL

The development service uses PostgreSQL 16 and local-only credentials.

```bash
cp .env.example .env
make db-up
make db-test-create
make db-status
```

Run migrations against the development database:

```bash
make db-migrate
```

Run the full suite, including PostgreSQL integration tests, after creating the dedicated test
database:

```bash
make test
```

Stop the service with `make db-down`. Resetting data requires the explicit
`make db-reset CONFIRM=reset` command. That target removes only the named local development volume
`fraud_anomaly_detection_postgres_data`.

## Tables and ownership

- Ingestion owns `ingestion_batch`, `account`, `merchant`, `transaction`, and
  `transaction_label`.
- Model release and scoring workflows own `model_version`, `transaction_score`, `fraud_flag`, and
  `model_evaluation`.
- Historical graph analysis owns `graph_snapshot`, `community`, and `account_risk_profile`.

Ingestion supplies every identifier. ORM models do not create random IDs. Published lineage,
model versions, scores, graph snapshots, profiles, and evaluations are immutable application
records. Corrections produce new source, input, model, or snapshot versions rather than overwriting
history.

## Database-enforced invariants

PostgreSQL enforces all primary keys, foreign keys, unique constraints, approved enumerations,
nonnegative values, unit-interval metrics, and the composite community membership reference. Most
historical foreign keys use `RESTRICT`. `merchant` and `transaction_label` use `CASCADE` because
they are true dependent records of an account and transaction.

Two trigger-backed cross-row rules complete the approved contract:

- A `merchant` row requires its account to have `account_kind = 'merchant'`, and that kind cannot
  change while the subtype exists.
- A `fraud_flag` can be created only when its score meets or exceeds the threshold on the score's
  model version.

## Service-level invariants

Application services remain responsible for immutability, restricting table ownership, requiring
historical graph cutoffs before scored transaction steps, selecting a snapshot when graph features
are present, and publishing only after cross-store reconciliation. PostgreSQL cannot enforce
references into Cosmos DB.

## Separate lifecycles

`transaction_label` stores source truth from PaySim for evaluation. `transaction_score` stores an
immutable prediction tied to a model and input version. `fraud_flag` stores the threshold decision
for one score and its review state. Keeping them separate prevents source labels from becoming
predictors and allows rescoring without rewriting source truth or earlier decisions.
