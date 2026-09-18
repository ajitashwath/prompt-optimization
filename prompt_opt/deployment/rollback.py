"""
Automatic rollback triggers.

Monitors guardrail metrics and live signals; automatically reverts
the prompt registry to the previous champion and notifies humans.

Trigger conditions:
  - Guardrail metric breach (refusal rate, format compliance, regression suite)
  - Error rate spike above threshold
  - Latency regression above threshold
  - Negative live user-feedback shift
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from prompt_opt.registry.prompt_registry import PromptRegistry

if TYPE_CHECKING:
    from prompt_opt.deployment.canary import CanaryController

logger = logging.getLogger(__name__)


@dataclass
class RollbackTriggerConfig:
    max_error_rate: float = 0.05           # >5% error rate → rollback
    max_latency_regression_pct: float = 0.30  # >30% latency increase → rollback
    max_refusal_rate_delta: float = 0.15   # >15pp refusal rate change → rollback
    min_format_compliance_rate: float = 0.90  # <90% format compliance → rollback
    min_regression_pass_rate: float = 1.0  # any regression failure → rollback


@dataclass
class RollbackEvent:
    reason: str
    triggered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    previous_champion: str | None = None
    rolled_back_to: str | None = None
    metric_values: dict = field(default_factory=dict)
    # "registry":    the champion pointer was reverted to an earlier version.
    # "canary":      an active canary was rolled back; the champion is unchanged.
    # "suppressed":  the breach persists but the registry was rolled back moments ago and
    #                not re-promoted since; no further automatic rollback (a human decides).
    # "no_previous": a rollback was needed but there is no earlier version to return to.
    scope: str = "registry"


AlertFn = Callable[[RollbackEvent], None]


def _default_alert(event: RollbackEvent) -> None:
    """Default alerter: log to stderr. Replace with webhook/email in production."""
    logger.critical(
        "ROLLBACK TRIGGERED (%s): %s | rolled_back_to=%s | at=%s",
        event.scope,
        event.reason,
        event.rolled_back_to,
        event.triggered_at,
    )


class RollbackMonitor:
    """
    Evaluates live metrics against thresholds and triggers rollback if needed.

    If an *active* `canary` is supplied, a breach rolls back the canary and
    leaves the registry champion alone: the stable champion is not the
    suspect while only a slice of traffic is on the challenger. Otherwise the
    registry champion is reverted to the previously promoted version.

    Automatic rollbacks do not cascade: metrics lag a rollback, so a breach that persists
    right after one (with no promotion since) would otherwise walk the champion back
    through history one version per check. Within `cooldown_seconds` of a registry
    rollback that was not followed by a promotion, further breaches raise a
    `scope="suppressed"` event and alert instead of rolling back again.

    Usage::

        monitor = RollbackMonitor(registry)
        event = monitor.check(
            error_rate=0.01,
            latency_ms_current=250,
            latency_ms_baseline=200,
            refusal_rate=0.3,
            refusal_rate_baseline=0.1,
        )
        if event:
            print(f"Rolled back: {event.reason}")
    """

    def __init__(
        self,
        registry: PromptRegistry,
        config: RollbackTriggerConfig | None = None,
        alert_fn: AlertFn | None = None,
        canary: CanaryController | None = None,
        cooldown_seconds: float = 3600.0,
    ):
        self.registry = registry
        self.cooldown_seconds = cooldown_seconds
        self.config = config or RollbackTriggerConfig()
        self.alert_fn = alert_fn or _default_alert
        self.canary = canary
        self._rollback_events: list[RollbackEvent] = []

    def _recent_unpromoted_rollback(self) -> tuple[dict, float] | None:
        """(event, age in s) if the newest champion event is a rollback within the cooldown."""
        if self.cooldown_seconds <= 0:
            return None
        history = self.registry.champion_history(limit=1)  # newest first
        if not history or history[0]["action"] != "rollback":
            return None
        rolled_back_at = datetime.fromisoformat(history[0]["created_at"])
        age = (datetime.utcnow() - rolled_back_at).total_seconds()
        return (history[0], age) if age < self.cooldown_seconds else None

    def check(
        self,
        *,
        error_rate: float | None = None,
        latency_ms_current: float | None = None,
        latency_ms_baseline: float | None = None,
        refusal_rate: float | None = None,
        refusal_rate_baseline: float | None = None,
        format_compliance_rate: float | None = None,
        regression_pass_rate: float | None = None,
    ) -> RollbackEvent | None:
        """
        Check current live metrics against rollback thresholds.

        If any threshold is breached, automatically roll back and return a RollbackEvent.
        Returns None if all metrics are within acceptable bounds.
        """
        triggers: list[str] = []
        metric_values: dict = {}

        if error_rate is not None:
            metric_values["error_rate"] = error_rate
            if error_rate > self.config.max_error_rate:
                triggers.append(
                    f"error_rate={error_rate:.3f} > max={self.config.max_error_rate}"
                )

        if (
            latency_ms_current is not None
            and latency_ms_baseline is not None
            and latency_ms_baseline > 0
        ):
            pct_increase = (latency_ms_current - latency_ms_baseline) / latency_ms_baseline
            metric_values["latency_regression_pct"] = pct_increase
            if pct_increase > self.config.max_latency_regression_pct:
                triggers.append(
                    f"latency regression={pct_increase:.1%} > "
                    f"max={self.config.max_latency_regression_pct:.1%}"
                )

        if refusal_rate is not None and refusal_rate_baseline is not None:
            delta = abs(refusal_rate - refusal_rate_baseline)
            metric_values["refusal_rate_delta"] = delta
            if delta > self.config.max_refusal_rate_delta:
                triggers.append(
                    f"refusal_rate_delta={delta:.3f} > max={self.config.max_refusal_rate_delta}"
                )

        if format_compliance_rate is not None:
            metric_values["format_compliance_rate"] = format_compliance_rate
            if format_compliance_rate < self.config.min_format_compliance_rate:
                triggers.append(
                    f"format_compliance={format_compliance_rate:.3f} < "
                    f"min={self.config.min_format_compliance_rate}"
                )

        if regression_pass_rate is not None:
            metric_values["regression_pass_rate"] = regression_pass_rate
            if regression_pass_rate < self.config.min_regression_pass_rate:
                triggers.append(
                    f"regression_pass_rate={regression_pass_rate:.3f} < "
                    f"min={self.config.min_regression_pass_rate}"
                )

        if not triggers:
            return None

        reason = "Rollback triggered: " + "; ".join(triggers)
        current_champ = self.registry.get_champion()

        if self.canary is not None and self.canary.state.status == "active":
            self.canary.rollback(reason)
            challenger = self.registry.get_by_content_hash(self.canary.state.challenger_hash)
            event = RollbackEvent(
                reason=reason,
                previous_champion=(
                    challenger.version_id if challenger else self.canary.state.challenger_hash
                ),
                rolled_back_to=current_champ.version_id if current_champ else None,
                metric_values=metric_values,
                scope="canary",
            )
        elif (recent := self._recent_unpromoted_rollback()) is not None:
            last, age = recent
            event = RollbackEvent(
                reason=(
                    f"{reason} -- automatic rollback SUPPRESSED: the registry was rolled back to "
                    f"{last['version_id']} {age:.0f}s ago and nothing has been promoted since; "
                    f"the breach persists, investigate manually"
                ),
                previous_champion=current_champ.version_id if current_champ else None,
                rolled_back_to=None,
                metric_values=metric_values,
                scope="suppressed",
            )
        else:
            previous_id = self.registry.rollback()
            event = RollbackEvent(
                reason=(
                    reason
                    if previous_id is not None
                    else f"{reason} -- no previous version to roll back to; investigate manually"
                ),
                previous_champion=current_champ.version_id if current_champ else None,
                rolled_back_to=previous_id,
                metric_values=metric_values,
                scope="registry" if previous_id is not None else "no_previous",
            )
        self._rollback_events.append(event)
        self.alert_fn(event)
        return event

    @property
    def rollback_history(self) -> list[RollbackEvent]:
        return list(self._rollback_events)
