"""Shared dependency health contract."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class DependencyState(StrEnum):
    HEALTHY = "healthy"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"


class DependencyHealth(BaseModel):
    """Sanitized readiness evidence for one dependency."""

    model_config = ConfigDict(frozen=True)

    name: str
    state: DependencyState
    required: bool
    detail: str
