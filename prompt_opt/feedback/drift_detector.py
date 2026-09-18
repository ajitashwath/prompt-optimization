"""
Drift detector for judge score distribution and guardrail metric trends.

Monitors:
  1. Judge score distribution drift: a two-sample KS test (is the difference real?) AND a
     smoothed KL divergence (is it large enough to matter?) — both must trip.
  2. Guardrail metric EMA trends (refusal rate, format compliance, latency)

A verdict is one of:
  * "drift"              — at least one event fired
  * "no_drift"           — enough data, nothing fired
  * "insufficient_data"  — not enough samples to say anything (never reported as "no drift")

Why two conditions for scores: KL divergence estimated from histograms is biased upwards on
small samples (identical distributions look "drifted" at n=100), so on its own it false-alarms.
The KS test controls the false-alarm rate (p < `ks_alpha`), and the KL threshold keeps
statistically-significant-but-negligible shifts on huge samples from firing.

Baselines can be persisted (`save` / `load`) so the scheduled loop and the CLI share one.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import stats

from prompt_opt.config import settings

logger = logging.getLogger(__name__)

MIN_SAMPLES = 30


class DriftType(str, Enum):
    JUDGE_SCORE = "judge_score"
    REFUSAL_RATE = "refusal_rate"
    FORMAT_COMPLIANCE = "format_compliance"
    LATENCY = "latency"


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


@dataclass
class DriftReport:
    """Outcome of one drift assessment."""

    verdict: Literal["drift", "no_drift", "insufficient_data"]
    events: list[DriftEvent] = field(default_factory=list)
    n_baseline: int = 0
    n_new: int = 0
    kl: float | None = None           # KL(new || baseline), smoothed
    ks_pvalue: float | None = None    # two-sample KS test
    notes: list[str] = field(default_factory=list)


# ─── Statistics ──────────────────────────────────────────────────────────────


def _adaptive_bins(n_p: int, n_q: int) -> int:
    """~sqrt(n) bins, so each bin holds a useful number of samples (4..20)."""
    return max(4, min(20, int(math.sqrt(min(n_p, n_q)))))


def _kl_divergence(p: list[float], q: list[float], n_bins: int | None = None) -> float:
    """
    KL(P || Q) between two samples, from histograms on a shared grid.

    Probabilities are additively smoothed (+0.5 per bin), so an empty bin in Q does not
    produce an astronomically large term. Returns 0.0 when either sample is empty or
    both samples are the same constant.
    """
    if not p or not q:
        return 0.0
    all_vals = list(p) + list(q)
    min_v, max_v = min(all_vals), max(all_vals)
    if min_v >= max_v:
        return 0.0

    k = n_bins or _adaptive_bins(len(p), len(q))
    bins = np.linspace(min_v, max_v, k + 1)
    p_hist, _ = np.histogram(p, bins=bins)
    q_hist, _ = np.histogram(q, bins=bins)

    alpha = 0.5
    p_prob = (p_hist + alpha) / (p_hist.sum() + alpha * k)
    q_prob = (q_hist + alpha) / (q_hist.sum() + alpha * k)
    return float(np.sum(p_prob * np.log(p_prob / q_prob)))


def _ks_pvalue(p: list[float], q: list[float]) -> float:
    if not p or not q:
        return 1.0
    return float(stats.ks_2samp(p, q).pvalue)


# ─── Detector ────────────────────────────────────────────────────────────────


class DriftDetector:
    """
    Detects distributional drift in judge scores and guardrail metrics.

    Maintains a baseline window and compares new observations against it.
    EMA smoothing is applied to guardrail metrics to reduce noise.

    Usage::

        detector = DriftDetector()
        # Establish baseline (e.g. the champion's judge scores on shadow traffic)
        detector.update_baseline(judge_scores=[0.7, 0.8, ...], refusal_rate=0.05)
        detector.save("./data/drift_baseline.json")

        # As new data arrives:
        report = detector.assess(new_scores=[0.5, 0.4, ...], refusal_rate=0.2)
        if report.verdict == "drift":
            for event in report.events:
                alert(event)
    """

    def __init__(
        self,
        kl_threshold: float | None = None,
        ema_alpha: float = 0.1,
        refusal_rate_max_delta: float = 0.1,
        format_compliance_min: float = 0.9,
        latency_max_increase: float = 0.3,
        ks_alpha: float = 0.01,
        min_samples: int = MIN_SAMPLES,
    ):
        self.kl_threshold = kl_threshold or settings.drift_kl_threshold
        self.ema_alpha = ema_alpha
        self.refusal_rate_max_delta = refusal_rate_max_delta
        self.format_compliance_min = format_compliance_min
        self.latency_max_increase = latency_max_increase
        self.ks_alpha = ks_alpha
        self.min_samples = min_samples

        # Baseline distributions
        self._baseline_scores: list[float] = []
        self._baseline_refusal_rate: float | None = None
        self._baseline_format_compliance: float | None = None
        self._baseline_latency_ms: float | None = None

        # EMA state
        self._ema_refusal: float | None = None
        self._ema_format: float | None = None
        self._ema_latency: float | None = None

    # ── baseline ────────────────────────────────────────────────────────────

    def update_baseline(
        self,
        judge_scores: list[float],
        refusal_rate: float | None = None,
        format_compliance: float | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Establish the baseline from an initial window of data (resets the EMA state)."""
        self._baseline_scores = [float(s) for s in judge_scores]
        self._baseline_refusal_rate = refusal_rate
        self._baseline_format_compliance = format_compliance
        self._baseline_latency_ms = latency_ms
        self._ema_refusal = self._ema_format = self._ema_latency = None
        logger.info(
            "Drift detector baseline updated: n_scores=%d, refusal_rate=%s, format=%s, "
            "latency_ms=%s",
            len(self._baseline_scores),
            refusal_rate,
            format_compliance,
            latency_ms,
        )

    @property
    def baseline_size(self) -> int:
        return len(self._baseline_scores)

    @property
    def baseline_refusal_rate(self) -> float | None:
        return self._baseline_refusal_rate

    @property
    def baseline_latency_ms(self) -> float | None:
        return self._baseline_latency_ms

    def set_baseline_metrics(
        self,
        refusal_rate: float | None = None,
        format_compliance: float | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Set/override baseline guardrail metrics without touching the score baseline."""
        if refusal_rate is not None:
            self._baseline_refusal_rate = refusal_rate
            self._ema_refusal = None
        if format_compliance is not None:
            self._baseline_format_compliance = format_compliance
            self._ema_format = None
        if latency_ms is not None:
            self._baseline_latency_ms = latency_ms
            self._ema_latency = None

    # ── persistence ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "baseline_scores": self._baseline_scores,
            "baseline_refusal_rate": self._baseline_refusal_rate,
            "baseline_format_compliance": self._baseline_format_compliance,
            "baseline_latency_ms": self._baseline_latency_ms,
            "ema_refusal": self._ema_refusal,
            "ema_format": self._ema_format,
            "ema_latency": self._ema_latency,
            "saved_at": datetime.utcnow().isoformat(),
        }

    def load_state(self, data: dict) -> None:
        self._baseline_scores = [float(s) for s in data.get("baseline_scores", [])]
        self._baseline_refusal_rate = data.get("baseline_refusal_rate")
        self._baseline_format_compliance = data.get("baseline_format_compliance")
        self._baseline_latency_ms = data.get("baseline_latency_ms")
        self._ema_refusal = data.get("ema_refusal")
        self._ema_format = data.get("ema_format")
        self._ema_latency = data.get("ema_latency")

    def save(self, path: str | Path) -> None:
        """Write the baseline and EMA state (atomically) so other processes can reuse it."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> DriftDetector:
        """A detector with the baseline/EMA state saved at *path*. Raises FileNotFoundError."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        detector = cls(**kwargs)
        detector.load_state(data)
        return detector

    # ── checks ──────────────────────────────────────────────────────────────

    def _update_ema(self, current: float | None, new_val: float) -> float:
        if current is None:
            return new_val
        return self.ema_alpha * new_val + (1 - self.ema_alpha) * current

    def assess(
        self,
        new_scores: list[float] | None = None,
        refusal_rate: float | None = None,
        format_compliance: float | None = None,
        latency_ms: float | None = None,
        baseline_latency_ms: float | None = None,
    ) -> DriftReport:
        """
        Compare new observations against the baseline.

        Args:
            new_scores:          Recent judge scores (compared as a distribution).
            refusal_rate:        Current refusal rate observation (needs a baseline refusal rate).
            format_compliance:   Current format-compliance observation (absolute threshold).
            latency_ms:          Current latency observation (compared with the baseline latency,
                                 or `baseline_latency_ms`).
            baseline_latency_ms: Override for the baseline latency.
        """
        events: list[DriftEvent] = []
        notes: list[str] = []
        report = DriftReport(verdict="no_drift", n_baseline=self.baseline_size)
        assessed_anything = False

        # ── Judge score distribution ─────────────────────────────────────────
        if new_scores is not None:
            new_scores = [float(s) for s in new_scores]
            report.n_new = len(new_scores)
            if self.baseline_size < self.min_samples or len(new_scores) < self.min_samples:
                notes.append(
                    f"Insufficient data for a score drift verdict: baseline "
                    f"n={self.baseline_size}, new n={len(new_scores)} "
                    f"(need at least {self.min_samples} each)"
                )
            else:
                assessed_anything = True
                kl = _kl_divergence(new_scores, self._baseline_scores)
                ks_p = _ks_pvalue(new_scores, self._baseline_scores)
                report.kl, report.ks_pvalue = kl, ks_p
                if ks_p < self.ks_alpha and kl > self.kl_threshold:
                    severity = "critical" if kl > self.kl_threshold * 2 else "warning"
                    events.append(
                        DriftEvent(
                            drift_type=DriftType.JUDGE_SCORE,
                            description=(
                                f"Judge score distribution drifted: KL={kl:.4f} > "
                                f"{self.kl_threshold} and KS p={ks_p:.4g} < {self.ks_alpha}"
                            ),
                            severity=severity,
                            metric_value=kl,
                            baseline_value=0.0,
                            threshold=self.kl_threshold,
                        )
                    )
                    logger.warning("Score drift detected (KL=%.4f, KS p=%.4g)", kl, ks_p)

        # ── Refusal rate EMA ──────────────────────────────────────────────────
        if refusal_rate is not None:
            self._ema_refusal = self._update_ema(self._ema_refusal, refusal_rate)
            if self._baseline_refusal_rate is None:
                notes.append(
                    "A refusal-rate observation was given but there is no baseline refusal rate; "
                    "it was recorded, not compared"
                )
            else:
                assessed_anything = True
                delta = abs(self._ema_refusal - self._baseline_refusal_rate)
                if delta > self.refusal_rate_max_delta:
                    events.append(
                        DriftEvent(
                            drift_type=DriftType.REFUSAL_RATE,
                            description=(
                                f"Refusal rate drifted: EMA={self._ema_refusal:.3f}, "
                                f"baseline={self._baseline_refusal_rate:.3f}, delta={delta:.3f}"
                            ),
                            severity=(
                                "critical"
                                if delta > self.refusal_rate_max_delta * 2
                                else "warning"
                            ),
                            metric_value=self._ema_refusal,
                            baseline_value=self._baseline_refusal_rate,
                            threshold=self.refusal_rate_max_delta,
                        )
                    )

        # ── Format compliance EMA ─────────────────────────────────────────────
        if format_compliance is not None:
            self._ema_format = self._update_ema(self._ema_format, format_compliance)
            assessed_anything = True
            if self._ema_format < self.format_compliance_min:
                events.append(
                    DriftEvent(
                        drift_type=DriftType.FORMAT_COMPLIANCE,
                        description=(
                            f"Format compliance drifted: EMA={self._ema_format:.3f} "
                            f"< min={self.format_compliance_min}"
                        ),
                        severity="critical",
                        metric_value=self._ema_format,
                        baseline_value=1.0,
                        threshold=self.format_compliance_min,
                    )
                )

        # ── Latency EMA ───────────────────────────────────────────────────────
        if latency_ms is not None:
            self._ema_latency = self._update_ema(self._ema_latency, latency_ms)
            base = (
                baseline_latency_ms if baseline_latency_ms is not None
                else self._baseline_latency_ms
            )
            if base is None or base <= 0:
                notes.append(
                    "A latency observation was given but there is no baseline latency; "
                    "it was recorded, not compared"
                )
            else:
                assessed_anything = True
                increase = (self._ema_latency - base) / base
                if increase > self.latency_max_increase:
                    events.append(
                        DriftEvent(
                            drift_type=DriftType.LATENCY,
                            description=(
                                f"Latency drifted: EMA={self._ema_latency:.0f}ms is "
                                f"{increase:.0%} above baseline {base:.0f}ms "
                                f"(max +{self.latency_max_increase:.0%})"
                            ),
                            severity=(
                                "critical" if increase > self.latency_max_increase * 2
                                else "warning"
                            ),
                            metric_value=self._ema_latency,
                            baseline_value=base,
                            threshold=self.latency_max_increase,
                        )
                    )

        report.events = events
        report.notes = notes
        if events:
            report.verdict = "drift"
        elif not assessed_anything:
            # Nothing could be compared (too little data, no baseline value, or no
            # observations): that is "no verdict", never "no drift".
            report.verdict = "insufficient_data"
            if not notes:
                notes.append("Nothing to assess: no observations were provided")
                report.notes = notes
            logger.warning("%s", "; ".join(notes))
        else:
            report.verdict = "no_drift"
        return report

    def check(
        self,
        new_scores: list[float] | None = None,
        refusal_rate: float | None = None,
        format_compliance: float | None = None,
        latency_ms: float | None = None,
        baseline_latency_ms: float | None = None,
    ) -> list[DriftEvent]:
        """`assess(...).events` (an empty list does NOT mean "no drift": see `assess`)."""
        return self.assess(
            new_scores=new_scores,
            refusal_rate=refusal_rate,
            format_compliance=format_compliance,
            latency_ms=latency_ms,
            baseline_latency_ms=baseline_latency_ms,
        ).events
