"""
Canary rollout controller.

Manages a gradual traffic ramp: 1% → 5% → 25% → 100%.
Each stage has a soak period; if guardrail metrics are clean, advance.
If any guardrail breaches, halt and rollback.

In production this integrates with a feature-flag / load-balancer system.
Here we provide the controller logic + state machine; the actual routing
is done by the caller (which checks `should_use_challenger()`).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Default traffic ramp stages: (pct, soak_seconds)
DEFAULT_RAMP = [
    (1, 3600),    #  1% for 1 hour
    (5, 7200),    #  5% for 2 hours
    (25, 14400),  # 25% for 4 hours
    (100, 0),     # 100% (final)
]


@dataclass
class CanaryStage:
    traffic_pct: int
    soak_seconds: int
    started_at: float = field(default_factory=time.time)
    requests_seen: int = 0
    guardrail_breaches: int = 0

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at

    @property
    def soak_complete(self) -> bool:
        return self.elapsed_seconds >= self.soak_seconds


@dataclass
class CanaryState:
    challenger_hash: str
    champion_hash: str
    stage_idx: int = 0
    status: str = "active"  # "active" | "halted" | "complete" | "rolledback"
    stages: list[CanaryStage] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    halt_reason: str | None = None


class CanaryController:
    """
    Traffic-split controller for gradual rollout.

    Usage::

        ctrl = CanaryController(challenger_hash, champion_hash)
        # On each request:
        if ctrl.should_use_challenger():
            response = challenger_prompt.generate(...)
        else:
            response = champion_prompt.generate(...)
        # Periodically:
        ctrl.maybe_advance()
    """

    def __init__(
        self,
        challenger_hash: str,
        champion_hash: str,
        ramp_schedule: list[tuple[int, int]] | None = None,
        state_path: str = "./data/canary_state.json",
        max_guardrail_breaches: int = 5,
    ):
        self.ramp = ramp_schedule or DEFAULT_RAMP
        self.state_path = Path(state_path)
        self.max_breaches = max_guardrail_breaches

        self.state = CanaryState(
            challenger_hash=challenger_hash,
            champion_hash=champion_hash,
            stages=[CanaryStage(pct, soak) for pct, soak in self.ramp],
        )
        self._save_state()
        import random
        self._rng = random.Random()

    @property
    def current_stage(self) -> CanaryStage:
        return self.state.stages[self.state.stage_idx]

    def should_use_challenger(self) -> bool:
        """
        Returns True if this request should be served by the challenger.

        Based on current traffic percentage.
        """
        if self.state.status != "active":
            return False
        self.current_stage.requests_seen += 1
        return self._rng.randint(1, 100) <= self.current_stage.traffic_pct

    def record_guardrail_breach(self) -> None:
        """Call this when a real-time guardrail metric fires."""
        self.current_stage.guardrail_breaches += 1
        logger.warning(
            "Canary guardrail breach #%d (stage=%d%%)",
            self.current_stage.guardrail_breaches,
            self.current_stage.traffic_pct,
        )
        if self.current_stage.guardrail_breaches >= self.max_breaches:
            self.halt("Too many guardrail breaches")

    def maybe_advance(self) -> bool:
        """
        Advance to the next stage if soak period is complete and no breaches.

        Returns True if advanced.
        """
        if self.state.status != "active":
            return False

        stage = self.current_stage
        if not stage.soak_complete:
            return False

        if stage.guardrail_breaches >= self.max_breaches:
            self.halt("Guardrail breach during soak period")
            return False

        next_idx = self.state.stage_idx + 1
        if next_idx >= len(self.state.stages):
            self.state.status = "complete"
            logger.info("Canary complete: challenger at 100%% traffic")
            self._save_state()
            return True

        self.state.stage_idx = next_idx
        logger.info(
            "Canary advanced to stage %d (%d%% traffic)",
            next_idx,
            self.state.stages[next_idx].traffic_pct,
        )
        self._save_state()
        return True

    def halt(self, reason: str) -> None:
        """Halt the canary rollout without rollback."""
        self.state.status = "halted"
        self.state.halt_reason = reason
        logger.warning("Canary HALTED: %s", reason)
        self._save_state()

    def rollback(self, reason: str = "") -> None:
        """Halt and signal rollback needed."""
        self.state.status = "rolledback"
        self.state.halt_reason = reason
        logger.error("Canary ROLLBACK: %s", reason)
        self._save_state()

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as fh:
            json.dump(
                {
                    "challenger_hash": self.state.challenger_hash,
                    "champion_hash": self.state.champion_hash,
                    "stage_idx": self.state.stage_idx,
                    "status": self.state.status,
                    "halt_reason": self.state.halt_reason,
                    "started_at": self.state.started_at,
                    "current_pct": self.current_stage.traffic_pct
                    if self.state.status == "active"
                    else 0,
                },
                fh,
                indent=2,
            )
