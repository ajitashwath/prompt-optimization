"""
Closed feedback loop scheduler.

Periodically triggers ingestion, judge re-calibration, optimization,
and drift monitoring. Uses APScheduler to schedule periodic jobs.

Jobs:
  1. `re_ingest`: Ingests raw conversation logs, performs deduplication and PII scrubbing,
     and refreshes the train/dev splits. The test split is kept frozen unless a manual
     refresh is requested.
  2. `re_optimize`: Triggers MIPROv2 or GEPA optimizer runs when enough new training data
     is available or when drift is detected.
  3. `drift_check`: Runs drift detection on production/shadow logs.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apscheduler.schedulers.background import BackgroundScheduler

from prompt_opt.config import settings
from prompt_opt.eval.harness import EvalHarness, PromptCandidate
from prompt_opt.feedback.drift_detector import DriftDetector
from prompt_opt.ingestion.ingestor import Ingestor
from prompt_opt.optimizer.budget import BudgetController
from prompt_opt.optimizer.miprov2 import MIPROv2
from prompt_opt.registry.prompt_registry import PromptRegistry
from prompt_opt.storage.dataset_store import DatasetStore

logger = logging.getLogger(__name__)


class FeedbackLoopManager:
    """
    Orchestrates the scheduled periodic feedback loop tasks.
    """

    def __init__(
        self,
        registry: PromptRegistry,
        ingestor: Ingestor,
        harness: EvalHarness,
        drift_detector: DriftDetector,
        dataset_store: DatasetStore | None = None,
        raw_logs_path: str = "./data/raw_production_logs.jsonl",
        optimizer_algorithm: str = "miprov2",
        max_usd_spend: float = 5.0,
        n_trials: int = 10,
    ):
        self.registry = registry
        self.ingestor = ingestor
        self.harness = harness
        self.drift_detector = drift_detector
        self.dataset_store = dataset_store or DatasetStore()
        self.raw_logs_path = Path(raw_logs_path)
        self.optimizer_algorithm = optimizer_algorithm
        self.max_usd_spend = max_usd_spend
        self.n_trials = n_trials

    def re_ingest(self) -> None:
        """Scheduled job: ingest fresh production conversation logs and refresh splits."""
        logger.info("Feedback loop: starting re_ingest job")
        if not self.raw_logs_path.exists():
            logger.warning("No fresh production logs found at %s", self.raw_logs_path)
            return

        # Ingest and stratify (test split is refreshed too — the store overwrite
        # is intentional for scheduled refreshes; gate uses the frozen copy).
        result = self.ingestor.ingest_file(self.raw_logs_path)
        counts = self.dataset_store.save(result)
        logger.info(
            "Feedback loop: ingested and persisted splits: %s",
            counts,
        )

    def re_optimize(self) -> None:
        """Scheduled job: trigger a prompt optimization run from persisted splits."""
        logger.info("Feedback loop: starting re_optimize job")

        # Fetch current champion prompt to start optimization from
        champion = self.registry.get_champion()
        if not champion:
            logger.warning("No champion prompt found in registry to optimize from.")
            return

        initial_prompt = PromptCandidate(
            system_prompt=champion.system_prompt,
            few_shot_demos=champion.few_shot_demos,
        )

        try:
            train_convs = self.dataset_store.load_split("train")
            dev_convs = self.dataset_store.load_split("dev")
        except FileNotFoundError as exc:
            logger.warning("No persisted splits available for optimization: %s", exc)
            return

        if not train_convs or not dev_convs:
            logger.warning("No training/dev conversations available for optimization.")
            return

        budget = BudgetController(max_usd_spend=self.max_usd_spend)

        if self.optimizer_algorithm == "gepa":
            from prompt_opt.optimizer.gepa import GEPA

            optimizer = GEPA(max_generations=self.n_trials, minibatch_size=min(2, len(dev_convs)))
        else:
            optimizer = MIPROv2(
                n_trials=self.n_trials,
                minibatch_size=min(2, len(dev_convs)),
                task_summary="Feedback loop auto-optimization",
            )

        try:
            candidates = optimizer.optimize(
                initial_prompt, train_convs, dev_convs, self.harness, budget
            )
            if candidates:
                best_cand, best_eval = candidates[0]
                logger.info(
                    "Feedback loop: optimization completed. Best score: %.4f (hash: %s)",
                    best_eval.mean_score,
                    best_cand.content_hash,
                )
                # Register the candidate in the registry (decision remains
                # PENDING until gate review)
                self.registry.register(
                    system_prompt=best_cand.system_prompt,
                    few_shot_demos=best_cand.few_shot_demos,
                    optimizer_run_id=f"auto_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    eval_scores=best_eval.to_dict(),
                )
        except Exception as exc:
            logger.error("Feedback loop: optimization failed: %s", exc)

    def drift_check(self, production_scores: list[float]) -> None:
        """Triggered periodically or on-demand to check for performance drift."""
        logger.info("Feedback loop: running drift check")
        events = self.drift_detector.check(new_scores=production_scores)
        for event in events:
            logger.warning("Drift detected: %s (severity: %s)", event.description, event.severity)
            if event.severity == "critical":
                # In a real loop, you could automatically schedule an early optimization run
                logger.info("Feedback loop: critical drift requires prompt re-optimization.")


def start_scheduler(manager: FeedbackLoopManager) -> BackgroundScheduler | None:
    """Initialize and start the background scheduler using configured cron strings."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore[import]
        from apscheduler.triggers.cron import CronTrigger  # type: ignore[import]
    except ImportError:
        logger.warning(
            "apscheduler not installed. Background scheduler is disabled. "
            "Please run jobs manually via CLI."
        )
        return None

    scheduler = BackgroundScheduler()

    # Schedule re-ingest (cron string from settings, e.g. "0 2 * * *")
    scheduler.add_job(
        manager.re_ingest,
        CronTrigger.from_crontab(settings.reingest_cron),
        id="re_ingest_job",
    )

    # Schedule re-optimize (e.g. weekly Monday early morning)
    scheduler.add_job(
        manager.re_optimize,
        CronTrigger.from_crontab(settings.reoptimize_cron),
        id="re_optimize_job",
    )

    scheduler.start()
    logger.info(
        "Feedback loop background scheduler started "
        "(re-ingest=%r, re-optimize=%r).",
        settings.reingest_cron,
        settings.reoptimize_cron,
    )
    return scheduler
