"""Sanitized Cosmos Gremlin errors."""


class GremlinAdapterError(Exception):
    """A bounded graph operation failed without exposing secrets or payloads."""


class GremlinConfigurationError(GremlinAdapterError):
    """Gremlin settings or graph structure are invalid."""


class GremlinEvidenceConflictError(GremlinAdapterError):
    """A stable graph identity has conflicting immutable evidence."""
