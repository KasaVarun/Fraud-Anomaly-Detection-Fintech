# Fraud & Anomaly Detection Network for Fintech Transactions

## Approved scope

Build a hybrid local/Azure prototype using the PaySim synthetic mobile-money dataset. Use approximately 200,000 transactions while retaining every fraud case. Do not replace the dataset, cloud platform, data models, or promised project objectives without explicit team approval.

## Objectives

1. Detect fraudulent transactions using tabular, graph, and behavioral-similarity signals.
2. Discover candidate fraud rings using graph community detection and centrality.
3. Find transactions behaviorally similar to earlier confirmed fraud.
4. Provide Power BI reporting and transaction-level forensic drill-down.
5. Provide a natural-language investigator through Copilot Studio backed by a read-only FastAPI service.
6. Evaluate precision, recall, F1, PR-AUC, confusion counts, and feature-family ablations.
7. Maintain data through deterministic initial and scheduled incremental loads.

## Approved architecture

- Local workstation: source download, validation, reproducible sampling, development, model training, and tests.
- Azure Database for PostgreSQL Flexible Server: relational entities, model versions, scores, flags, evaluation results, account risk profiles, and reporting views.
- Azure Cosmos DB for NoSQL: raw transaction documents and versioned 384-dimensional behavioral embeddings.
- Azure Cosmos DB for Apache Gremlin: account, merchant, device, IP, and transaction-edge evidence graph.
- Azure Functions and Cosmos DB change feed: deterministic document-to-vector and document-to-graph projections.
- Timer Function or Azure Data Factory: scheduled batch ingestion and scoring.
- Python analysis: NetworkX centrality and Louvain community detection; LightGBM as the primary fraud classifier; scikit-learn for preprocessing, baselines, metrics, and ablations.
- Sentence Transformers `all-MiniLM-L6-v2`: behavioral embeddings.
- FastAPI: read-only, controlled orchestration across stores, including provenance, model/input versions, freshness, and explicit partial-evidence responses.
- Power BI: dashboards and drill-through.
- Copilot Studio: conversational investigator that calls FastAPI actions and explains returned evidence.

## Database contract

The connected relational design contains these 12 tables:

1. `ingestion_batch`
2. `account`
3. `merchant`
4. `transaction`
5. `transaction_label`
6. `model_version`
7. `transaction_score`
8. `fraud_flag`
9. `graph_snapshot`
10. `community`
11. `account_risk_profile`
12. `model_evaluation`

Shared stable account and transaction identifiers connect PostgreSQL rows, Cosmos documents, embeddings, and Gremlin vertices/edges. Cross-store references are checked by application reconciliation because PostgreSQL cannot enforce foreign keys into Cosmos DB.

## Data and modeling rules

- Generate any device/IP enrichment deterministically with a fixed seed and mark it as synthetic.
- Establish whole-step chronological train, validation, and test ranges before fitting models.
- Keep later test-period chunks reserved to simulate scheduled arrivals.
- Never use `isFraud`, `isFlaggedFraud`, future events, account identifiers, or future graph edges as predictive inputs.
- Keep PaySim balance fields for forensic display, but exclude them from the first predictive model because they may leak simulator outcomes.
- Build graph features and vector similarity references only from information available before the transaction being scored.
- Embed a deterministic behavioral text template containing transaction type, amount band, and prior activity; store 384 floats plus template/model version.
- Eligible similarity references are earlier labeled-fraud training examples only.
- Treat graph community membership as investigation evidence, not proof of fraud.
- Routine refresh uses the pinned released model. Retraining is a deliberate versioned workflow and never tunes on test data.
- Create a fraud flag only when a stored score reaches the threshold recorded by its model version.
- Publish a batch only after expected IDs, counts, and versions reconcile across stores. Failed processing must leave the last published batch available.

## Delivery definition

The final prototype should provide:

- Reproducible PaySim sampling and chronological splits.
- Initial and incremental idempotent ingestion.
- The 12-table PostgreSQL schema with migrations, constraints, and indexes.
- Raw-document and embedding containers plus Gremlin graph projections.
- Versioned feature engineering and model training/evaluation.
- Combined tabular, graph, and vector scoring with ablation results.
- Stored scores, flags, model versions, graph snapshots, communities, risk profiles, and evaluations.
- Read-only FastAPI health, transaction-evidence, related-accounts, similar-behavior, and dashboard-support endpoints.
- Power BI-ready SQL views and documentation for the dashboard.
- Copilot Studio action contracts and investigator prompts grounded only in FastAPI responses.
- Automated tests, local developer setup, Azure deployment/configuration instructions, observability, reconciliation, and a final demonstration walkthrough.

## Working agreement

Implement in reviewable phases. At the end of every phase, run the appropriate checks and report changed files, commands run, results, unresolved risks, and the exact next recommended phase. Do not claim Azure resources, model performance, or fraud-ring findings that have not actually been created or measured.
