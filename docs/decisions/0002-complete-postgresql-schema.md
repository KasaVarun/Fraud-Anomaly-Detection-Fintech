# 0002 Complete PostgreSQL Schema

- **Status:** Accepted for Phase 2
- **Date:** 2026-10-31

## Context

The approved ERD defines 12 connected relational tables. Partial migrations would leave invalid
intermediate states and make cross-table constraints difficult to review.

## Decision

Alembic revision `20261031_0001` creates all 12 tables atomically. SQLAlchemy 2 typed declarative
models mirror every column, key, check, unique constraint, and index. The reserved `transaction`
table name is quoted explicitly.

Historical references use restrictive deletion. `merchant` and `transaction_label` cascade only
when their parent record is removed because both are true dependents. PostgreSQL triggers enforce
merchant subtype consistency and the fraud-flag threshold rule. The service layer owns broader
immutability, temporal graph cutoffs, cross-store reconciliation, and publication behavior.

## Consequences

- PostgreSQL can reject invalid relational records before later ingestion code exists.
- Source labels, model scores, and threshold flags retain independent histories.
- The migration can downgrade in dependency-safe order and re-upgrade from an empty database.
- KPI views remain deferred to the reporting phase; no extra application tables exist.
- Integration tests require a dedicated localhost database whose name ends in `_test`.
