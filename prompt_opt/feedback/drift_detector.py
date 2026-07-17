"""
Drift detector for judge score distribution and guardrail metric trends.

Monitors:
  1. Judge score distribution drift (KL divergence from baseline)
  2. Guardrail metric EMA trends (refusal rate, format compliance, etc.)

Fires DriftEvent when any threshold is exceeded, triggering re-calibration
or re-optimization in the feedback loop.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

import numpy as np

from prompt_opt.config import settings

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    JUDGE_SCORE = "judge_score"
    REFUSAL_RATE = "refusal_rate"
    FORMAT_COMPLIANCE = "format_compliance"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"


@dataclass
class DriftEvent:
    drift_type: DriftType
    description: str
    severity: Literal["warning", "critical"]
    detected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metric_value: float = 0.0
    baseline_value: float = 0.0
    threshold: float = 0.0

    def to_dict(self) -> dict:
        return {
            "drift_type": self.drift_type.value,
            "description": self.description,
            "severity": self.severity,
            "detected_at": self.detected_at,
            "metric_value": self.metric_value,
            "baseline_value": self.baseline_value,
            "threshold": self.threshold,
        }


def _kl_divergence(p: list[float], q: list[float], n_bins: int = 20) -> float:
    """
    Estimate KL divergence D(P || Q) between two continuous distributions
    using histogram binning.

    Returns 0.0 if insufficient data.
    """
    if len(p) < 10 or len(q) < 10:
        return 0.0

    all_vals = p + q
    min_v, max_v = min(all_vals), max(all_vals)
    if min_v >= max_v:
        return 0.0

    bins = np.linspace(min_v, max_v, n_bins + 1)
    p_hist, _ = np.histogram(p, bins=bins)
    q_hist, _ = np.histogram(q, bins=bins)

    # Add small epsilon to avoid log(0)
    eps = 1e-9
    p_prob = (p_hist + eps) / (sum(p_hist) + eps * n_bins)
    q_prob = (q_hist + eps) / (sum(q_hist) + eps * n_bins)

    return float(np.sum(p_prob * np.log(p_prob / q_prob)))


class DriftDetector:
    """
    Detects distributional drift in judge scores and guardrail metrics.

    Maintains a baseline window and compares new observations against it.
    EMA smoothing is applied to guardrail metrics to reduce noise.

    Usage::

        detector = DriftDetector()
        # Establish baseline (e.g. first 100 production conversations)
        detector.update_baseline(judge_scores=[0.7, 0.8, ...])

        # As new data arrives:
        events = detector.check(new_scores=[0.5, 0.4, ...])
        for event in events:
            alert(event)
    """

    def __init__(
        self,
        kl_threshold: float | None = None,
        ema_alpha: float = 0.1,
        refusal_rate_max_delta: float = 0.1,
        format_compliance_min: float = 0.9,
    ):
        self.kl_threshold = kl_threshold or settings.drift_kl_threshold
        self.ema_alpha = ema_alpha
        self.refusal_rate_max_delta = refusal_rate_max_delta
        self.format_compliance_min = format_compliance_min

        # Baseline distributions
        self._baseline_scores: list[float] = []
        self._baseline_refusal_rate: float | None = None
        self._baseline_format_compliance: float | None = None

        # EMA state
        self._ema_refusal: float | None = None
        self._ema_format: float | None = None
        self._ema_latency: float | None = None

    def update_baseline(
        self,
        judge_scores: list[float],
        refusal_rate: float | None = None,
        format_compliance: float | None = None,
    ) -> None:
        """Establish the baseline distribution from an initial window of data."""
        self._baseline_scores = list(judge_scores)
        self._baseline_refusal_rate = refusal_rate
        self._baseline_format_compliance = format_compliance
        logger.info(
            "Drift detector baseline updated: n_scores=%d, refusal_rate=%s, format=%s",
            len(judge_scores),
            refusal_rate,
            format_compliance,
        )

    def _update_ema(self, current: float | None, new_val: float) -> float:
        if current is None:
            return new_val
        return self.ema_alpha * new_val + (1 - self.ema_alpha) * current

    def check(
        self,
        new_scores: list[float] | None = None,
        refusal_rate: float | None = None,
        format_compliance: float | None = None,
        latency_ms: float | None = None,
        baseline_latency_ms: float | None = None,
    ) -> list[DriftEvent]:
        """
        Check new observations against the baseline. Returns any DriftEvents detected.
        """
        events: list[DriftEvent] = []

        # ── Judge score KL divergence ─────────────────────────────────────────
        if new_scores and self._baseline_scores:
            kl = _kl_divergence(new_scores, self._baseline_scores)
            if kl > self.kl_threshold:
                severity = "critical" if kl > self.kl_threshold * 2 else "warning"
                events.append(
                    DriftEvent(
                        drift_type=DriftType.JUDGE_SCORE,
                        description=(
                            f"Judge score distribution drifted: KL={kl:.4f} > threshold={self.kl_threshold}"
                        ),
                        severity=severity,
                        metric_value=kl,
                        baseline_value=0.0,
                        threshold=self.kl_threshold,
                    )
                )
                logger.warning("Score drift detected (KL=%.4f)", kl)

        # ── Refusal rate EMA ──────────────────────────────────────────────────
        if refusal_rate is not None:
            self._ema_refusal = self._update_ema(self._ema_refusal, refusal_rate)
            if self._baseline_refusal_rate is not None:
                delta = abs(self._ema_refusal - self._baseline_refusal_rate)
                if delta > self.refusal_rate_max_delta:
                    events.append(
                        DriftEvent(
                            drift_type=DriftType.REFUSAL_RATE,
                            description=(
                                f"Refusal rate drifted: EMA={self._ema_refusal:.3f}, "
                                f"baseline={self._baseline_refusal_rate:.3f}, delta={delta:.3f}"
                            ),
                            severity="critical" if delta > self.refusal_rate_max_delta * 2 else "warning",
                            metric_value=self._ema_refusal,
                            baseline_value=self._baseline_refusal_rate,
                            threshold=self.refusal_rate_max_delta,
                        )
                    )

        # ── Format compliance EMA ─────────────────────────────────────────────
        if format_compliance is not None:
            self._ema_format = self._update_ema(self._ema_format, format_compliance)
            if self._ema_format < self.format_compliance_min:
                events.append(
                    DriftEvent(
                        drift_type=DriftType.FORMAT_COMPLIANCE,
                        description=(
                            f"Format compliance drifted: EMA={self._ema_format:.3f} < min={self.format_compliance_min}"
                        ),
                        severity="critical",
                        metric_value=self._ema_format,
                        baseline_value=1.0,
                        threshold=self.format_compliance_min,
                    )
                )

        return events
