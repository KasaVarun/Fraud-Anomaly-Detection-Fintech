"""Configuration behavior tests."""

import pytest
from pydantic import SecretStr, ValidationError

from fraud_detection.config import Settings


def test_settings_use_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment == "local"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert not settings.postgres_configured
    assert not settings.document_vector_configured
    assert not settings.graph_configured
    assert settings.auth_mode == "local_bypass"


def test_settings_load_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAUD_ENVIRONMENT", "test")
    monkeypatch.setenv("FRAUD_PORT", "9000")
    monkeypatch.setenv("FRAUD_POSTGRES_DSN", "postgresql+psycopg://local:local@localhost/db")

    settings = Settings(_env_file=None)

    assert settings.environment == "test"
    assert settings.port == 9000
    assert isinstance(settings.postgres_dsn, SecretStr)
    assert settings.postgres_configured
    assert "local:local" not in str(settings.postgres_dsn)


def test_invalid_port_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, port=0)


def test_database_secrets_are_masked() -> None:
    secret = "postgresql+psycopg://user:do-not-print@127.0.0.1/fraud_detection_test"
    settings = Settings(_env_file=None, test_postgres_dsn=secret)

    assert "do-not-print" not in repr(settings)
    assert "do-not-print" not in str(settings.test_postgres_dsn)


def test_cosmos_managed_identity_configuration_does_not_require_a_key() -> None:
    settings = Settings(
        _env_file=None,
        cosmos_nosql_endpoint="https://example.documents.azure.com/",
        cosmos_nosql_database="fraud_evidence",
        cosmos_use_managed_identity=True,
        azure_managed_identity_client_id="11111111-1111-1111-1111-111111111111",
    )

    assert settings.document_vector_configured
    assert settings.embedding_configured
    assert settings.cosmos_nosql_key is None
