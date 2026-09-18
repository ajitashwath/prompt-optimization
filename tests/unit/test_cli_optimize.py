"""CLI: the frozen judge is enforced, and optimize registers only real improvements."""

import pytest
import typer
from typer.testing import CliRunner

from prompt_opt.cli import main as cli
from prompt_opt.eval.harness import PromptCandidate
from prompt_opt.eval.metrics import PerExampleResult, compute_eval_summary
from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole
from prompt_opt.judge.calibrator import CalibrationReport
from prompt_opt.judge.registry import JudgeRegistry
from prompt_opt.optimizer import miprov2
from prompt_opt.registry.prompt_registry import PromptRegistry
from prompt_opt.storage.dataset_store import DatasetStore

runner = CliRunner()
ENV = {"OPENAI_API_KEY": "sk-test"}


# ─── judge enforcement ───────────────────────────────────────────────────────


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # judge registry, eval cache and DB paths are cwd-relative
    (tmp_path / "data").mkdir()
    return tmp_path


def _freeze(model="frozen/judge-model"):
    return JudgeRegistry().freeze(
        model,
        "v1.0",
        CalibrationReport(judge_version="auto", rubric_version="v1.0", n_items=10, passed=True),
    )


def test_evaluation_refuses_to_run_without_a_calibrated_judge(workdir):
    with pytest.raises(typer.Exit):
        cli._build_harness()


def test_allow_uncalibrated_is_an_explicit_escape_hatch(workdir):
    harness = cli._build_harness(allow_uncalibrated=True)
    assert harness.judge_version.startswith("uncalibrated:")


def test_the_frozen_judge_is_used_and_keys_the_cache(workdir):
    frozen = _freeze()

    harness = cli._build_harness()

    assert harness.judge_version == frozen.version_id == "judge_v1"
    assert harness.judge.model == "frozen/judge-model"  # not settings.judge_model


def test_an_explicit_missing_judge_version_is_an_error_even_with_the_escape_hatch(workdir):
    _freeze()
    with pytest.raises(typer.Exit):
        cli._build_harness("judge_v9", allow_uncalibrated=True)


def test_a_stale_rubric_stops_evaluation(workdir):
    _freeze()
    reg = JudgeRegistry()
    reg._versions["judge_v1"]["rubric_fingerprint"] = "stale"
    reg._save()
    with pytest.raises(typer.Exit):
        cli._build_harness()


# ─── optimize ────────────────────────────────────────────────────────────────


def _conv(cid):
    return ConversationRecord(
        conversation_id=cid,
        messages=[Message(role=MessageRole.USER, content=f"q {cid}")],
        category="general",
    )


def _summary(candidate, score, n=8):
    per_example = [
        PerExampleResult(
            example_id=f"d{i}", category="general", challenger_score=score,
            champion_score=None, outcome=None,
        )
        for i in range(n)
    ]
    return compute_eval_summary(candidate.content_hash, "dev", "jv", per_example)


@pytest.fixture
def optimize_env(tmp_path, monkeypatch):
    """A champion, persisted train/dev splits, a stub harness and MIPROv2 stubbed out."""
    registry = PromptRegistry(db_url="sqlite:///:memory:")
    registry.promote(registry.register(system_prompt="champion prompt"))
    store = DatasetStore(base_path=tmp_path / "splits")
    store.save_split("train", [_conv(f"t{i}") for i in range(10)])
    store.save_split("dev", [_conv(f"d{i}") for i in range(40)])

    monkeypatch.setattr(cli, "_registry", lambda: registry)
    monkeypatch.setattr(cli, "_dataset_store", lambda: store)
    monkeypatch.setattr(cli, "_build_harness", lambda *a, **k: object())
    monkeypatch.setattr(cli, "REGRESSION_SUITE_PATH", tmp_path / "none.jsonl")

    captured = {}

    def install(outcome):
        def fake_optimize(self, initial, train, dev, harness, budget):
            captured["minibatch"] = self.minibatch_size
            captured["initial_hash"] = initial.content_hash
            return outcome(initial)

        monkeypatch.setattr(miprov2.MIPROv2, "optimize", fake_optimize)

    return registry, captured, install


def test_optimize_registers_nothing_when_no_candidate_beats_the_current_prompt(optimize_env):
    registry, _, install = optimize_env
    install(lambda initial: [(initial, _summary(initial, 0.7))])

    result = runner.invoke(cli.app, ["optimize"], env=ENV)

    assert result.exit_code == 0
    assert "No candidate beat" in result.output
    assert len(registry.list_versions()) == 1  # still only the champion


def test_optimize_registers_a_candidate_that_beats_the_current_prompt(optimize_env):
    registry, _, install = optimize_env
    better = PromptCandidate(system_prompt="better prompt")
    install(lambda initial: [(better, _summary(better, 0.9)), (initial, _summary(initial, 0.7))])

    result = runner.invoke(cli.app, ["optimize"], env=ENV)

    assert result.exit_code == 0
    registered = registry.get_by_content_hash(better.content_hash)
    assert registered is not None
    assert registered.eval_scores["mean_score"] == pytest.approx(0.9)
    assert registered.eval_scores["n_examples"] == 8


def test_optimize_fails_clearly_when_the_budget_ran_out_before_the_baseline(optimize_env):
    registry, _, install = optimize_env
    install(lambda initial: [(initial, _summary(initial, 0.0, n=0))])

    result = runner.invoke(cli.app, ["optimize"], env=ENV)

    assert result.exit_code == 1
    assert len(registry.list_versions()) == 1


def test_optimize_minibatch_is_no_longer_hard_coded_to_two(optimize_env):
    _, captured, install = optimize_env
    install(lambda initial: [(initial, _summary(initial, 0.7))])

    runner.invoke(cli.app, ["optimize"], env=ENV)
    assert captured["minibatch"] == 16  # auto: min(dev size 40, 16)

    runner.invoke(cli.app, ["optimize", "--minibatch-size", "5"], env=ENV)
    assert captured["minibatch"] == 5

    runner.invoke(cli.app, ["optimize", "--minibatch-size", "500"], env=ENV)
    assert captured["minibatch"] == 40  # capped at the dev split


def test_optimize_surfaces_evaluation_errors_without_a_traceback(optimize_env):
    from prompt_opt.eval.harness import EvaluationError

    _, _, install = optimize_env

    def boom(initial):
        raise EvaluationError("9/10 examples failed")

    install(boom)
    result = runner.invoke(cli.app, ["optimize"], env=ENV)

    assert result.exit_code == 1
    assert "Optimization failed" in result.output
