# Alembic Migrations

Revision `20261031_0001` creates the complete approved 12-table relational design atomically. The
downgrade removes trigger functions and tables in dependency-safe order. Future autogeneration sees
all models through `fraud_detection.persistence.relational`.

Do not add KPI tables. Reporting views belong to a later phase.
