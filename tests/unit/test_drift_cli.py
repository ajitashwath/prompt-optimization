"""drift-check exit codes and flags, score-shadow, and the loop's scheduled drift job."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from typer.testing import CliRunner

from prompt_opt.cli import main as cli
from prompt_opt.feedback.drift_detector import DriftDetector
from prompt_opt.feedback.loop import FeedbackLoopManager, start_scheduler
from prompt_opt.ingestion.ingestor import Ingestor
from prompt_opt.registry.prompt_registry import PromptRegistry

runner = CliRunner()


@pytest.fixture(autouse=True)
def quiet(caplog):
    caplog.set_level(logging.CRITICAL)


def write_json(path, values):
    path.write_text(json.dumps(list(values)), encoding="utf-8")
    return str(path)


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(11)
    return SimpleNamespace(
        dir=tmp_path,
        baseline=write_json(tmp_path / "baseline.json", rng.beta(5, 2, 200)),
        same=write_json(tmp_path / "same.json", rng.beta(5, 2, 200)),
        drifted=write_json(tmp_path / "drifted.json", rng.beta(2, 5, 200)),
        tiny=write_json(tmp_path / "tiny.json", [0.5] * 10),
        state=str(tmp_path / "state" / "baseline.json"),
    )


def check(*args):
    return runner.invoke(cli.app, ["drift-check", *args])


# ─── drift-check ─────────────────────────────────────────────────────────────


def test_no_drift_exits_zero_and_shows_the_evidence(data):
    result = check("--baseline-file", data.baseline, "--scores-file", data.same)

    assert result.exit_code == 0, result.output
    assert "No drift detected" in result.output
    assert "KS p=" in result.output


def test_drift_exits_one_with_a_table(data):
    result = check("--baseline-file", data.baseline, "--scores-file", data.drifted)

    assert result.exit_code == 1
    assert "Drift Events" in result.output and "judge_score" in result.output


def test_too_little_data_is_exit_two_and_never_claims_no_drift(data):
    """Regression: the CLI printed 'No drift detected.' for any input under 10 scores."""
    result = check("--baseline-file", data.baseline, "--scores-file", data.tiny)

    assert result.exit_code == 2
    assert "No verdict" in result.output
    assert "No drift detected" not in result.output


def test_refusal_rate_needs_a_baseline_instead_of_being_ignored(data):
    """Regression: --refusal-rate was accepted and silently did nothing."""
    result = check("--baseline-file", data.baseline, "--scores-file", data.same,
                   "--refusal-rate", "0.5")

    assert result.exit_code == 2
    assert "--baseline-refusal-rate" in result.output


def test_refusal_rate_drift_is_detected_with_a_baseline(data):
    args = ("--baseline-file", data.baseline, "--scores-file", data.same,
            "--baseline-refusal-rate", "0.05")

    assert check(*args, "--refusal-rate", "0.06").exit_code == 0
    drifted = check(*args, "--refusal-rate", "0.4")
    assert drifted.exit_code == 1 and "refusal_rate" in drifted.output


def test_latency_drift_needs_a_baseline_and_is_detected(data):
    base = ("--baseline-file", data.baseline, "--scores-file", data.same)

    assert check(*base, "--latency-ms", "500").exit_code == 2
    slow = check(*base, "--baseline-latency-ms", "200", "--latency-ms", "500")
    assert slow.exit_code == 1 and "latency" in slow.output.lower()


def test_a_saved_baseline_is_reused(data):
    saved = check("--baseline-file", data.baseline, "--scores-file", data.same,
                  "--baseline-refusal-rate", "0.05", "--baseline-state", data.state,
                  "--save-baseline")
    assert saved.exit_code == 0 and "Baseline saved" in saved.output

    # Later: no --baseline-file, the saved state (scores and refusal baseline) is used
    later = check("--scores-file", data.drifted, "--baseline-state", data.state)
    assert later.exit_code == 1
    refusal = check("--scores-file", data.same, "--baseline-state", data.state,
                    "--refusal-rate", "0.5")
    assert refusal.exit_code == 1 and "refusal_rate" in refusal.output


def test_a_missing_baseline_is_a_clear_error(data):
    result = check("--scores-file", data.same, "--baseline-state", data.state)
    assert result.exit_code == 2
    assert "No baseline" in result.output

    no_file = check("--scores-file", data.same, "--baseline-state", data.state, "--save-baseline")
    assert no_file.exit_code == 2 and "--baseline-file" in no_file.output


# ─── score-shadow ────────────────────────────────────────────────────────────


class FakeJudge:
    def __init__(self, fail_for=()):
        self.fail_for = set(fail_for)
        self.seen = []

    def score_absolute(self, conv, response):
        self.seen.append((conv, response))
        if conv.conversation_id in self.fail_for:
            raise RuntimeError("judge down")
        return SimpleNamespace(aggregate_score=0.8 if "good" in response else 0.4)


def log_record(request_id, champion="a good answer", challenger="a good answer",
               champion_error=None, latency=100.0):
    return {
        "request_id": request_id,
        "messages": [{"role": "user", "content": f"question {request_id}"}],
        "champion_response": champion,
        "challenger_response": challenger,
        "champion_error": champion_error,
        "challenger_error": None,
        "latency_champion_ms": latency,
        "latency_challenger_ms": latency * 2,
    }


def write_log(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


@pytest.fixture
def shadow_env(tmp_path, monkeypatch):
    judge = FakeJudge()
    monkeypatch.setattr(cli, "_build_harness", lambda *a, **k: SimpleNamespace(judge=judge))
    return SimpleNamespace(judge=judge, log=tmp_path / "shadow.jsonl", out=tmp_path / "scores.json")


def score(env, *args):
    return runner.invoke(
        cli.app,
        ["score-shadow", "--log-file", str(env.log), "--output", str(env.out), *args],
        env={"OPENAI_API_KEY": "sk-test"},
    )


def test_score_shadow_writes_scores_for_successful_responses_only(shadow_env):
    write_log(shadow_env.log, [
        log_record("r1"),
        log_record("r2", champion="I can't help with that."),
        log_record("r3", champion="", champion_error="RuntimeError: 503"),   # failed call
        log_record("r4", champion="another good one"),
    ])

    result = score(shadow_env)

    assert result.exit_code == 0, result.output
    assert json.loads(shadow_env.out.read_text()) == [0.8, 0.4, 0.8]
    assert "Scored 3 champion responses (1 skipped" in result.output
    assert "refusal rate: 0.3333" in result.output
    assert "mean latency: 100 ms" in result.output
    # the judge saw the real request as context, and never the failed call
    assert {conv.conversation_id for conv, _ in shadow_env.judge.seen} == {"r1", "r2", "r4"}


def test_score_shadow_can_score_the_challenger_and_respects_limit(shadow_env):
    write_log(shadow_env.log, [log_record(f"r{i}") for i in range(5)])

    result = score(shadow_env, "--which", "challenger", "--limit", "2")

    assert result.exit_code == 0
    assert len(json.loads(shadow_env.out.read_text())) == 2
    assert "mean latency: 200 ms" in result.output


def test_score_shadow_counts_judge_failures(shadow_env):
    shadow_env.judge.fail_for = {"r2"}
    write_log(shadow_env.log, [log_record("r1"), log_record("r2"), log_record("r3")])

    result = score(shadow_env)

    assert result.exit_code == 0
    assert len(json.loads(shadow_env.out.read_text())) == 2
    assert "1 judge failures" in result.output


def test_score_shadow_fails_when_nothing_can_be_scored(shadow_env):
    write_log(shadow_env.log, [log_record("r1", champion="", champion_error="boom")])
    assert score(shadow_env).exit_code == 1
    assert not shadow_env.out.exists()


def test_score_shadow_validates_its_inputs(shadow_env):
    assert score(shadow_env).exit_code == 1                       # no log yet
    write_log(shadow_env.log, [log_record("r1")])
    assert score(shadow_env, "--which", "nobody").exit_code == 1


def test_score_shadow_output_feeds_drift_check(shadow_env, data):
    rng = np.random.default_rng(2)
    records = [log_record(f"r{i}", champion="good" if rng.random() < 0.7 else "bad")
               for i in range(60)]
    write_log(shadow_env.log, records)
    assert score(shadow_env).exit_code == 0

    result = check("--baseline-file", str(shadow_env.out), "--scores-file", str(shadow_env.out))
    assert result.exit_code == 0        # a set of scores never drifts from itself


# ─── the loop's drift job ────────────────────────────────────────────────────


def manager(tmp_path):
    return FeedbackLoopManager(
        registry=PromptRegistry(db_url="sqlite:///:memory:"),
        ingestor=Ingestor(use_ner=False),
        harness=MagicMock(),
        drift_detector=DriftDetector(),
        drift_state_path=str(tmp_path / "drift_baseline.json"),
        recent_scores_path=str(tmp_path / "recent_scores.json"),
    )


def test_drift_job_is_a_logged_no_op_until_both_files_exist(tmp_path):
    m = manager(tmp_path)
    assert m.drift_check_job() is None                                   # no recent scores

    write_json(tmp_path / "recent_scores.json", [0.5] * 50)
    assert m.drift_check_job() is None                                   # no baseline yet


def test_drift_job_compares_recent_scores_with_the_saved_baseline(tmp_path):
    rng = np.random.default_rng(4)
    seed = DriftDetector()
    seed.update_baseline(list(rng.beta(5, 2, 200)))
    seed.save(tmp_path / "drift_baseline.json")
    write_json(tmp_path / "recent_scores.json", rng.beta(2, 5, 200))

    report = manager(tmp_path).drift_check_job()

    assert report.verdict == "drift"
    assert report.events[0].severity == "critical"


def test_drift_job_reports_insufficient_data(tmp_path):
    seed = DriftDetector()
    seed.update_baseline([0.7] * 100)
    seed.save(tmp_path / "drift_baseline.json")
    write_json(tmp_path / "recent_scores.json", [0.1] * 5)

    assert manager(tmp_path).drift_check_job().verdict == "insufficient_data"


def test_the_scheduler_now_runs_the_drift_job(tmp_path):
    scheduler = start_scheduler(manager(tmp_path))
    try:
        assert {"re_ingest_job", "re_optimize_job", "drift_check_job"} <= {
            job.id for job in scheduler.get_jobs()
        }
    finally:
        scheduler.shutdown(wait=False)
