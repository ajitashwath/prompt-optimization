"""Tests for the FeedbackLoopManager's persisted-split integration."""

import json
from unittest.mock import MagicMock

from prompt_opt.feedback.drift_detector import DriftDetector
from prompt_opt.feedback.loop import FeedbackLoopManager, start_scheduler
from prompt_opt.ingestion.ingestor import Ingestor
from prompt_opt.registry.prompt_registry import PromptRegistry
from prompt_opt.storage.dataset_store import DatasetStore

RAW_LOGS = [
    {"conversation_id": f"c{i}", "messages": [{"role": "user", "content": f"question number {i}"}]}
    for i in range(6)
]


def _manager(tmp_path) -> FeedbackLoopManager:
    raw_path = tmp_path / "raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as fh:
        for row in RAW_LOGS:
            fh.write(json.dumps(row) + "\n")

    return FeedbackLoopManager(
        registry=PromptRegistry(db_url="sqlite:///:memory:"),
        ingestor=Ingestor(use_ner=False),
        harness=MagicMock(),
        drift_detector=DriftDetector(),
        dataset_store=DatasetStore(base_path=tmp_path / "splits"),
        raw_logs_path=str(raw_path),
    )


def test_re_ingest_persists_splits(tmp_path):
    manager = _manager(tmp_path)
    manager.re_ingest()

    store = manager.dataset_store
    assert store.exists("train")
    assert store.exists("dev")
    assert store.exists("test")
    stats = store.stats()
    assert sum(stats.values()) == len(RAW_LOGS)


def test_re_ingest_without_logs_is_noop(tmp_path):
    manager = _manager(tmp_path)
    manager.raw_logs_path = tmp_path / "missing.jsonl"
    manager.re_ingest()  # must not raise or create files
    assert not (tmp_path / "splits" / "train.jsonl").exists()


def test_re_optimize_registers_candidate_from_persisted_splits(tmp_path):
    manager = _manager(tmp_path)
    manager.re_ingest()

    reg = manager.registry
    v1 = reg.register(system_prompt="Champion seed")
    reg.promote(v1)

    # Stub out the optimizer to avoid LLM calls; verify wiring end-to-end.
    from prompt_opt.eval.harness import PromptCandidate
    from prompt_opt.eval.metrics import EvalSummary

    best = PromptCandidate(system_prompt="Optimized prompt")
    summary = EvalSummary(
        prompt_hash=best.content_hash, split="dev", judge_version="v0",
        n_examples=2, mean_score=0.9, median_score=0.9, std_score=0.0,
    )
    fake_optimizer = MagicMock()
    fake_optimizer.optimize.return_value = [(best, summary)]

    import prompt_opt.feedback.loop as loop_mod
    original_mipro = loop_mod.MIPROv2
    loop_mod.MIPROv2 = MagicMock(return_value=fake_optimizer)
    try:
        manager.re_optimize()
    finally:
        loop_mod.MIPROv2 = original_mipro

    # Best candidate was registered with the auto run id and eval scores
    version = reg.get_by_content_hash(best.content_hash)
    assert version is not None
    assert version.optimizer_run_id.startswith("auto_")
    assert version.eval_scores.get("mean_score") == 0.9


def test_start_scheduler_uses_configured_crons():
    manager = FeedbackLoopManager(
        registry=PromptRegistry(db_url="sqlite:///:memory:"),
        ingestor=Ingestor(use_ner=False),
        harness=MagicMock(),
        drift_detector=DriftDetector(),
    )
    scheduler = start_scheduler(manager)
    assert scheduler is not None
    job_ids = {job.id for job in scheduler.get_jobs()}
    assert {"re_ingest_job", "re_optimize_job"} <= job_ids
    scheduler.shutdown(wait=False)
