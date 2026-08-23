"""
Judge version registry — freezes and tracks judge configurations.

A "judge version" is the combination of:
  - model identifier (litellm model string)
  - rubric version
  - calibration report

Changing any of these MUST produce a new judge version because it changes
the yardstick against which all optimizer runs are evaluated.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from prompt_opt.judge.calibrator import CalibrationReport

logger = logging.getLogger(__name__)


@dataclass
class JudgeVersion:
    version_id: str          # e.g. "judge_v3"
    model: str               # litellm model string
    rubric_version: str
    calibration_report: CalibrationReport | None = None
    frozen_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    content_hash: str = ""   # SHA256 of model+rubric_version for dedup

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "model": self.model,
            "rubric_version": self.rubric_version,
            "frozen_at": self.frozen_at,
            "content_hash": self.content_hash,
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

    def _content_hash(self, model: str, rubric_version: str) -> str:
        return hashlib.sha256(f"{model}|{rubric_version}".encode()).hexdigest()[:16]

    def freeze(
        self,
        model: str,
        rubric_version: str,
        calibration_report: CalibrationReport,
    ) -> JudgeVersion:
        """
        Freeze a judge configuration. Returns the new JudgeVersion.

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
        content_hash = self._content_hash(model, rubric_version)

        jv = JudgeVersion(
            version_id=version_id,
            model=model,
            rubric_version=rubric_version,
            calibration_report=calibration_report,
            content_hash=content_hash,
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
            calibration_report=report,
        )

    def latest(self) -> JudgeVersion | None:
        if not self._versions:
            return None
        last_key = list(self._versions.keys())[-1]
        return self.get(last_key)

    def list_versions(self) -> list[str]:
        return list(self._versions.keys())
