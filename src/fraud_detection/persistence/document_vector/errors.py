"""Sanitized Cosmos raw-document adapter errors."""


class CosmosAdapterError(Exception):
    """A Cosmos operation failed without exposing credentials or payloads."""


class CosmosConfigurationError(CosmosAdapterError):
    """The database/container contract differs from the approved configuration."""


class CosmosDocumentConflictError(CosmosAdapterError):
    """An existing stable identity has different immutable content."""
