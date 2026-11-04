"""Honest Phase 1 adapters with no production behavior."""

from dataclasses import dataclass

from fraud_detection.ports.health import DependencyHealth, DependencyState


@dataclass(frozen=True, slots=True)
class UnavailableAdapter:
    """Report configuration state until a verified adapter exists in a later phase."""

    name: str
    configured: bool
    phase: str

    async def readiness(self) -> DependencyHealth:
        if self.configured:
            return DependencyHealth(
                name=self.name,
                state=DependencyState.UNAVAILABLE,
                required=True,
                detail=f"configured but no {self.phase} production adapter is installed",
            )
        return DependencyHealth(
            name=self.name,
            state=DependencyState.NOT_CONFIGURED,
            required=False,
            detail="not configured for this process",
        )
