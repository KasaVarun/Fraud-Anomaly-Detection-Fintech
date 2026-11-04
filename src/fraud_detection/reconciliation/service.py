"""Readiness aggregation for the approved system boundaries."""

import asyncio
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from fraud_detection.ports.health import DependencyHealth, DependencyState


class ReadinessProbe(Protocol):
    async def readiness(self) -> DependencyHealth:
        """Return sanitized dependency readiness."""


class ReadinessResult(BaseModel):
    """Aggregated process readiness."""

    model_config = ConfigDict(frozen=True)

    ready: bool
    status: str
    dependencies: list[DependencyHealth]


class ReadinessService:
    """Evaluate independent dependency probes concurrently."""

    def __init__(self, probes: Sequence[ReadinessProbe]) -> None:
        self._probes = tuple(probes)

    async def check(self) -> ReadinessResult:
        dependencies = list(await asyncio.gather(*(probe.readiness() for probe in self._probes)))
        ready = all(
            dependency.state is not DependencyState.UNAVAILABLE or not dependency.required
            for dependency in dependencies
        )
        return ReadinessResult(
            ready=ready,
            status="ready" if ready else "not_ready",
            dependencies=dependencies,
        )
