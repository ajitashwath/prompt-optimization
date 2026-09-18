"""
Canary rollout controller.

Manages a gradual traffic ramp: 1% → 5% → 25% → 100%.
Each stage has a soak period that starts when the stage BECOMES ACTIVE; once it
has elapsed (and the stage has fewer than `max_guardrail_breaches` breaches) the
rollout can advance. Too many guardrail breaches halt the rollout.

In production this integrates with a feature-flag / load-balancer system.
Here we provide the controller logic + state machine; the actual routing
is done by the caller (which checks `should_use_challenger()`).

The controller never touches the prompt registry. The champion pointer stays on
the old prompt during the rollout; the caller promotes the challenger once
`state.status == "complete"` (the CLI's `canary advance` does this).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
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
    started_at: float | None = None   # set when the stage becomes active
    requests_seen: int = 0
    guardrail_breaches: int = 0

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since this stage became active (0 if it has not started)."""
        return 0.0 if self.started_at is None else time.time() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        """Soak time still to go (the full soak if the stage has not started)."""
        return max(0.0, self.soak_seconds - self.elapsed_seconds)

    @property
    def soak_complete(self) -> bool:
        return self.started_at is not None and self.elapsed_seconds >= self.soak_seconds


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
        # On each request (pass a stable key, e.g. user or conversation id, so the same
        # caller always gets the same arm):
        if ctrl.should_use_challenger(request_key=user_id):
            response = challenger_prompt.generate(...)
        else:
            response = champion_prompt.generate(...)
        # Periodically:
        ctrl.maybe_advance()

    Request counters are persisted every `persist_every` requests (and on every state
    change, and on `flush()`), not on every request.
    """

    def __init__(
        self,
        challenger_hash: str,
        champion_hash: str,
        ramp_schedule: list[tuple[int, int]] | None = None,
        state_path: str = "./data/canary_state.json",
        max_guardrail_breaches: int = 5,
        persist_every: int = 50,
    ):
        self.ramp = ramp_schedule or DEFAULT_RAMP
        self.state_path = Path(state_path)
        self.max_breaches = max_guardrail_breaches
        self.persist_every = max(1, persist_every)
        self._unsaved = 0

        self.state = CanaryState(
            challenger_hash=challenger_hash,
            champion_hash=champion_hash,
            stages=[CanaryStage(pct, soak) for pct, soak in self.ramp],
        )
        # Only the first stage is live; every later stage's clock starts when it is reached.
        self.state.stages[0].started_at = time.time()
        self._save_state()
        import random
        self._rng = random.Random()

    # ── Persistence / resume ─────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        state_path: str = "./data/canary_state.json",
        max_guardrail_breaches: int = 5,
        persist_every: int = 50,
    ) -> CanaryController:
        """
        Reconstruct a controller from a persisted state file.

        Raises FileNotFoundError if no rollout was ever started at state_path.
        """
        path = Path(state_path)
        if not path.exists():
            raise FileNotFoundError(f"No canary state found at {path}")

        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        stage_idx = data["stage_idx"]
        stages = [
            CanaryStage(
                traffic_pct=s["traffic_pct"],
                soak_seconds=s["soak_seconds"],
                started_at=s.get("started_at"),
                requests_seen=s.get("requests_seen", 0),
                guardrail_breaches=s.get("guardrail_breaches", 0),
            )
            for s in data.get("stages", [])
        ]
        # State files written before soak clocks were per-stage stamped every stage with the
        # rollout's start time. A stage after the current one has not started yet.
        for stage in stages[stage_idx + 1:]:
            stage.started_at = None
        if stages and stages[stage_idx].started_at is None and data["status"] == "active":
            stages[stage_idx].started_at = time.time()

        ctrl = cls.__new__(cls)
        ctrl.ramp = [(s.traffic_pct, s.soak_seconds) for s in stages] or DEFAULT_RAMP
        ctrl.state_path = path
        ctrl.max_breaches = max_guardrail_breaches
        ctrl.persist_every = max(1, persist_every)
        ctrl._unsaved = 0
        ctrl.state = CanaryState(
            challenger_hash=data["challenger_hash"],
            champion_hash=data["champion_hash"],
            stage_idx=stage_idx,
            status=data["status"],
            stages=stages,
            started_at=data["started_at"],
            halt_reason=data.get("halt_reason"),
        )
        import random

        ctrl._rng = random.Random()
        logger.info("Resumed canary %s → %s (stage %d, status=%s)",
                    ctrl.state.champion_hash, ctrl.state.challenger_hash,
                    ctrl.state.stage_idx, ctrl.state.status)
        return ctrl

    @property
    def current_stage(self) -> CanaryStage:
        return self.state.stages[self.state.stage_idx]

    def _bucket(self, request_key: str) -> int:
        """Stable 0-99 bucket for a request key, specific to this rollout."""
        digest = hashlib.sha256(f"{self.state.challenger_hash}:{request_key}".encode()).hexdigest()
        return int(digest[:8], 16) % 100

    def should_use_challenger(self, request_key: str | None = None) -> bool:
        """
        Returns True if this request should be served by the challenger.

        Based on current traffic percentage. With a `request_key` (user / session /
        conversation id) the decision is deterministic: the same key always gets the same
        arm, so a conversation never flips prompts mid-way, and callers routed to the
        challenger at 5% stay on it at 25%. Without a key each request is an independent
        random draw.
        """
        if self.state.status != "active":
            return False
        stage = self.current_stage
        stage.requests_seen += 1
        if request_key is None:
            use_challenger = self._rng.randint(1, 100) <= stage.traffic_pct
        else:
            use_challenger = self._bucket(request_key) < stage.traffic_pct
        self._unsaved += 1
        if self._unsaved >= self.persist_every:
            self._save_state()
        return use_challenger

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
        else:
            self._save_state()

    def maybe_advance(self) -> bool:
        """
        Advance to the next stage if its soak period is complete and it has not hit the
        breach limit. The next stage's soak clock starts now.

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
        self.state.stages[next_idx].started_at = time.time()  # its soak starts now
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

    def flush(self) -> None:
        """Persist the current state (including request counters) now."""
        self._save_state()

    def _save_state(self) -> None:
        payload = {
            "challenger_hash": self.state.challenger_hash,
            "champion_hash": self.state.champion_hash,
            "stage_idx": self.state.stage_idx,
            "status": self.state.status,
            "halt_reason": self.state.halt_reason,
            "started_at": self.state.started_at,
            "current_pct": self.current_stage.traffic_pct
            if self.state.status == "active"
            else 0,
            "stages": [
                {
                    "traffic_pct": s.traffic_pct,
                    "soak_seconds": s.soak_seconds,
                    "started_at": s.started_at,
                    "requests_seen": s.requests_seen,
                    "guardrail_breaches": s.guardrail_breaches,
                }
                for s in self.state.stages
            ],
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a crash mid-write must never leave a corrupt state file
        tmp_path = self.state_path.with_name(self.state_path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, self.state_path)
        self._unsaved = 0
