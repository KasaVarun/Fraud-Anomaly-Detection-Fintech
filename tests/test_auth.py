"""Gate A Microsoft Entra authentication boundary tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fraud_detection.api.app import create_app
from fraud_detection.config import Settings
from fraud_detection.security.auth import (
    AuthenticationFailure,
    EntraJwtValidator,
)

TENANT = "11111111-1111-1111-1111-111111111111"
AUDIENCE = "api://fraud-investigator"
ISSUER = f"https://login.microsoftonline.com/{TENANT}/v2.0"


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "production",
        "auth_mode": "entra",
        "entra_tenant_id": TENANT,
        "entra_audience": AUDIENCE,
        "entra_issuer": ISSUER,
        "entra_jwks_url": f"{ISSUER}/.well-known/openid-configuration/jwks",
        "entra_required_scope": "Fraud.Investigate",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture(scope="module")
def keys() -> tuple[Any, Any]:
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private, private.public_key()


def _token(private_key: Any, **overrides: Any) -> str:
    now = datetime.now(UTC)
    claims: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "tid": TENANT,
        "oid": "22222222-2222-2222-2222-222222222222",
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "scp": "Fraud.Investigate",
    }
    claims.update(overrides)
    return str(jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-key"}))


def _validator(public_key: Any, settings: Settings | None = None) -> EntraJwtValidator:
    return EntraJwtValidator(
        settings or _settings(),
        signing_key_resolver=lambda _token_value: public_key,
    )


def test_valid_entra_token_checks_signature_issuer_audience_tenant_expiry_and_scope(
    keys: tuple[Any, Any],
) -> None:
    private, public = keys
    claims = _validator(public).validate(_token(private))
    assert claims["tid"] == TENANT
    assert claims["aud"] == AUDIENCE


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        ({"aud": "api://wrong"}, "invalid_token"),
        ({"iss": "https://login.microsoftonline.com/wrong/v2.0"}, "invalid_token"),
        ({"tid": "33333333-3333-3333-3333-333333333333"}, "invalid_tenant"),
        ({"exp": datetime.now(UTC) - timedelta(minutes=1)}, "invalid_token"),
        ({"scp": "Something.Else"}, "insufficient_scope"),
    ],
)
def test_invalid_entra_claims_are_rejected(
    keys: tuple[Any, Any], overrides: dict[str, Any], expected_code: str
) -> None:
    private, public = keys
    with pytest.raises(AuthenticationFailure) as captured:
        _validator(public).validate(_token(private, **overrides))
    assert captured.value.code == expected_code
    assert captured.value.status_code == (403 if expected_code == "insufficient_scope" else 401)


def test_malformed_and_wrong_signature_tokens_are_rejected(keys: tuple[Any, Any]) -> None:
    private, public = keys
    other_private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    validator = _validator(public)
    for token in ("not-a-jwt", _token(other_private)):
        with pytest.raises(AuthenticationFailure, match="invalid or expired"):
            validator.validate(token)
    assert private is not None


def test_nonlocal_environments_cannot_enable_local_bypass() -> None:
    for environment in ("development", "staging", "production"):
        with pytest.raises(ValidationError, match="require Entra"):
            Settings(_env_file=None, environment=environment, auth_mode="local_bypass")


def test_entra_configuration_must_be_complete() -> None:
    with pytest.raises(ValidationError, match="incomplete"):
        Settings(_env_file=None, environment="production", auth_mode="entra")


def test_health_and_openapi_are_anonymous_but_v1_requires_bearer(
    keys: tuple[Any, Any],
) -> None:
    private, public = keys
    settings = _settings()
    with TestClient(create_app(settings, auth_validator=_validator(public))) as client:
        live = client.get("/health/live")
        openapi = client.get("/openapi.json")
        missing = client.get("/v1/dashboard/summary")
        forbidden = client.get(
            "/v1/dashboard/summary",
            headers={"Authorization": f"Bearer {_token(private, scp='Something.Else')}"},
        )
        valid = client.get(
            "/v1/dashboard/summary",
            headers={"Authorization": f"Bearer {_token(private)}"},
        )
    assert live.status_code == 200
    assert openapi.status_code == 200
    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "authentication_required"
    assert missing.headers["www-authenticate"] == "Bearer"
    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["code"] == "insufficient_scope"
    assert valid.status_code == 503
    schema = openapi.json()
    assert "security" not in schema["paths"]["/health/live"]["get"]
    assert schema["paths"]["/v1/dashboard/summary"]["get"]["security"] == [{"entraOpenId": []}]


def test_bearer_token_is_absent_from_logs_responses_and_openapi(
    keys: tuple[Any, Any], caplog: pytest.LogCaptureFixture
) -> None:
    private, public = keys
    token = _token(private)
    with TestClient(create_app(_settings(), auth_validator=_validator(public))) as client:
        response = client.get(
            "/v1/dashboard/summary",
            headers={"Authorization": f"Bearer {token}"},
        )
        schema = client.get("/openapi.json")
    assert token not in caplog.text
    assert token not in response.text
    assert token not in schema.text
