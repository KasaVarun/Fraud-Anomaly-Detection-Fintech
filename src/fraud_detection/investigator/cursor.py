"""Deterministic filter-bound cursor encoding."""

import base64
import hashlib
import json
from typing import Any

from fraud_detection.ingestion.hashing import canonical_json_bytes


class InvalidCursorError(ValueError):
    """Cursor is malformed or belongs to different filters."""


def _fingerprint(filters: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(filters)).hexdigest()


def encode_cursor(position: dict[str, Any], filters: dict[str, Any]) -> str:
    body = {"v": 1, "filters": _fingerprint(filters), "position": position}
    encoded = base64.urlsafe_b64encode(canonical_json_bytes(body)).decode().rstrip("=")
    return encoded


def decode_cursor(
    value: str,
    filters: dict[str, Any],
    *,
    expected_position_fields: set[str],
) -> dict[str, Any]:
    if len(value) > 1024:
        raise InvalidCursorError("cursor is invalid")
    try:
        padding = "=" * (-len(value) % 4)
        body = json.loads(base64.urlsafe_b64decode(value + padding))
    except (ValueError, json.JSONDecodeError) as exc:
        raise InvalidCursorError("cursor is invalid") from exc
    if (
        not isinstance(body, dict)
        or body.get("v") != 1
        or body.get("filters") != _fingerprint(filters)
        or not isinstance(body.get("position"), dict)
        or set(body["position"]) != expected_position_fields
    ):
        raise InvalidCursorError("cursor is incompatible with request filters")
    return dict(body["position"])
