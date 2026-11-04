"""Safe, user-facing ingestion errors."""


class IngestionError(Exception):
    """Base error for deterministic ingestion operations."""


class SourceValidationError(IngestionError):
    """The source CSV does not satisfy the PaySim contract."""


class ManifestValidationError(IngestionError):
    """A manifest or generated sample failed integrity checks."""


class LoadConflictError(IngestionError):
    """An immutable relational identity conflicts with stored data."""


class ReconciliationError(IngestionError):
    """Loaded PostgreSQL data does not reconcile with the manifest."""
