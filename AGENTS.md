# Repository Instructions

## Architectural invariants

- Preserve the approved hybrid local/Azure architecture and PaySim scope.
- Use PostgreSQL for relational records, model history, scores, flags, risk profiles, evaluations,
  and reporting views.
- Use Cosmos DB for NoSQL for raw transaction documents and versioned 384-dimensional behavioral
  embeddings.
- Use Cosmos DB for Apache Gremlin for account, merchant, device, IP, and transaction-edge
  evidence.
- Keep stable account and transaction identifiers consistent across every store. Reconcile
  cross-store references in application code because PostgreSQL cannot enforce Cosmos references.
- Keep the complete 12-table PostgreSQL migration atomic in Phase 2. Do not introduce a partial
  schema.
- Generate device/IP enrichment deterministically with a fixed seed and label it synthetic.
- Establish whole-step chronological train, validation, and test ranges before fitting models.
- Never use fraud labels, `isFlaggedFraud`, future events, account identifiers, future graph edges,
  or the four PaySim balance fields as inputs to the initial predictive model.
- Build graph and vector evidence only from information available before the scored transaction.
- Use `all-MiniLM-L6-v2` with a deterministic behavioral template and earlier labeled-fraud
  training examples as eligible similarity references.
- Treat graph community membership as investigation evidence, not proof of fraud.
- Routine refresh uses a pinned released model. Retraining is explicit and versioned, with no
  tuning on test data.
- Create fraud flags only when stored scores reach the threshold recorded by their model version.
- Publish batches only after required IDs, counts, and versions reconcile. Preserve the prior
  published batch after failures.
- Keep investigator APIs read-only and include provenance, model/input versions, freshness, and
  explicit partial-evidence behavior.
- Do not claim deployed infrastructure, measured model performance, or confirmed fraud rings
  without evidence.

## Development rules

- Use Python 3.12 and `uv`.
- Run formatting, linting, MyPy, and Pytest before completing a phase.
- Never commit credentials, raw datasets, generated samples, model binaries, or generated reports.
- Keep phase boundaries in `docs/implementation-roadmap.md`; do not implement a later phase early.
