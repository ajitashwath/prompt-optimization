"""
Judge version registry — freezes and tracks judge configurations.

A "judge version" is the combination of:
  - model identifier (litellm model string)
  - rubric version
  - calibration report

Changing any of these MUST produce a new judge version because it changes
the yardstick against which all optimizer runs are evaluated. `load_judge`
rebuilds the exact frozen judge (model + rubric) and refuses to if the rubric
definition has changed since it was calibrated.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from prompt_opt.judge.calibrator import CalibrationReport
from prompt_opt.judge.judge import LLMJudge
from prompt_opt.labeling.rubric import DEFAULT_RUBRIC, Rubric

logger = logging.getLogger(__name__)


class JudgeNotFoundError(LookupError):
    """No (matching) frozen judge version exists."""


class JudgeMismatchError(RuntimeError):
    """The frozen judge can no longer be reproduced (its rubric changed)."""


def rubric_fingerprint(rubric: Rubric) -> str:
    """Stable hash of the full rubric definition (not just its version string)."""
    payload = json.dumps(rubric.to_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def resolve_rubric(rubric_version: str) -> Rubric:
    """Map a rubric version string to its definition."""
    if rubric_version == DEFAULT_RUBRIC.version:
        return DEFAULT_RUBRIC
    raise JudgeMismatchError(
        f"Unknown rubric version {rubric_version!r}; only {DEFAULT_RUBRIC.version!r} is defined"
    )


@dataclass
class JudgeVersion:
    version_id: str          # e.g. "judge_v3"
    model: str               # litellm model string
    rubric_version: str
    calibration_report: CalibrationReport | None = None
    frozen_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    content_hash: str = ""   # SHA256 of model+rubric definition for dedup
    rubric_fingerprint: str = ""  # hash of the rubric definition at freeze time

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "model": self.model,
            "rubric_version": self.rubric_version,
            "frozen_at": self.frozen_at,
            "content_hash": self.content_hash,
            "rubric_fingerprint": self.rubric_fingerprint,
            "calibration": self.calibration_report.to_dict() if self.calibration_report else None,
        }


class JudgeRegistry:
    """
    File-backed registry for frozen judge versions.

    Stored at `{base_path}/judge_registry.json`.
    """

    def __init__(self, base_path: str = "./data"):
        self._path = Path(base_path) / "judge_registry.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._versions: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path) as fh:
                self._versions = json.load(fh)

    def _save(self) -> None:
        with open(self._path, "w") as fh:
            json.dump(self._versions, fh, indent=2)

    def _content_hash(self, model: str, rubric_version: str, fingerprint: str = "") -> str:
        return hashlib.sha256(f"{model}|{rubric_version}|{fingerprint}".encode()).hexdigest()[:16]

    def freeze(
        self,
        model: str,
        rubric_version: str,
        calibration_report: CalibrationReport,
        rubric: Rubric | None = None,
    ) -> JudgeVersion:
        """
        Freeze a judge configuration. Returns the new JudgeVersion.

        `rubric` (default: the rubric registered under `rubric_version`) is
        fingerprinted so `load_judge` can detect later edits to it.

        If the exact same (model, rubric_version) pair already exists,
        a new version entry is still created (calibration may have improved).
        """
        if not calibration_report.passed:
            raise ValueError(
                f"Cannot freeze a judge that failed calibration: "
                f"{calibration_report.failure_reasons}"
            )

        n = len(self._versions) + 1
        version_id = f"judge_v{n}"
        if rubric is None:
            rubric = resolve_rubric(rubric_version)
        fingerprint = rubric_fingerprint(rubric)
        content_hash = self._content_hash(model, rubric_version, fingerprint)

        jv = JudgeVersion(
            version_id=version_id,
            model=model,
            rubric_version=rubric_version,
            calibration_report=calibration_report,
            content_hash=content_hash,
            rubric_fingerprint=fingerprint,
        )
        self._versions[version_id] = jv.to_dict()
        self._save()
        logger.info(
            "Frozen judge version %s (model=%s, rubric=%s)",
            version_id,
            model,
            rubric_version,
        )
        return jv

    def get(self, version_id: str) -> JudgeVersion | None:
        data = self._versions.get(version_id)
        if data is None:
            return None
        report = None
        if data.get("calibration"):
            # Reconstruct a minimal CalibrationReport (sufficient for display)
            c = data["calibration"]
            report = CalibrationReport(
                judge_version=c.get("judge_version", version_id),
                rubric_version=c.get("rubric_version", ""),
                n_items=c.get("n_items", 0),
                kappa=c.get("kappa", 0.0),
                spearman_rho=c.get("spearman_rho", 0.0),
                agreement_rate=c.get("agreement_rate", 0.0),
                passed=c.get("passed", False),
            )
        return JudgeVersion(
            version_id=data["version_id"],
            model=data["model"],
            rubric_version=data["rubric_version"],
            frozen_at=data["frozen_at"],
            content_hash=data.get("content_hash", ""),
            rubric_fingerprint=data.get("rubric_fingerprint", ""),
            calibration_report=report,
        )

    def load_judge(self, version_id: str | None = None) -> tuple[LLMJudge, JudgeVersion]:
        """
        Rebuild the frozen judge: its own model and rubric, never the current settings.

        Args:
            version_id: A frozen version (e.g. "judge_v2"); default: the latest.

        Raises:
            JudgeNotFoundError: nothing has been frozen / the version does not exist.
            JudgeMismatchError: the rubric definition changed since calibration, so
                                the judge would no longer be the one that was calibrated.
        """
        jv = self.get(version_id) if version_id else self.latest()
        if jv is None:
            raise JudgeNotFoundError(
                f"Judge version {version_id!r} not found"
                if version_id
                else "No calibrated judge has been frozen yet"
            )
        rubric = resolve_rubric(jv.rubric_version)
        if jv.rubric_fingerprint and jv.rubric_fingerprint != rubric_fingerprint(rubric):
            raise JudgeMismatchError(
                f"The rubric {jv.rubric_version!r} changed after {jv.version_id} was "
                f"calibrated; re-run `calibrate-judge` to freeze a new judge version"
            )
        return LLMJudge(model=jv.model, rubric=rubric), jv

    def latest(self) -> JudgeVersion | None:
        if not self._versions:
            return None
        last_key = list(self._versions.keys())[-1]
        return self.get(last_key)

    def list_versions(self) -> list[str]:
        return list(self._versions.keys())
