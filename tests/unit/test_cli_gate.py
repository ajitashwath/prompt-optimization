"""CLI gate flow: immutable results, reviewable spot checks, slim registry verdicts."""

import json
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from prompt_opt.cli import main as cli
from prompt_opt.gate.champion_gate import ChampionGate, GateResult
from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole
from prompt_opt.registry.prompt_registry import PromptRegistry
from prompt_opt.storage.dataset_store import DatasetStore

runner = CliRunner()
ENV = {"OPENAI_API_KEY": "sk-test"}

SPOT_ITEMS = [
    {
        "example_id": "test_1",
        "category": "general",
        "user_query": "How do I reset my password? [bold]not markup[/bold]",
        "challenger_response": "Click the forgot-password link.",
        "champion_response": "Contact support.",
        "challenger_score": 0.9,
        "champion_score": 0.5,
        "outcome": "win",
        "judge_rationale": "More direct.",
    }
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    registry = PromptRegistry(db_url="sqlite:///:memory:")
    champ_id = registry.register(system_prompt="champion")
    chal_id = registry.register(system_prompt="challenger")
    registry.promote(champ_id)

    gate_dir = tmp_path / "gate_results"
    monkeypatch.setattr(cli, "_registry", lambda: registry)
    monkeypatch.setattr(cli, "GATE_RESULTS_DIR", gate_dir)
    monkeypatch.setattr(cli, "REGRESSION_SUITE_PATH", tmp_path / "none.jsonl")
    return SimpleEnv(registry, champ_id, chal_id, gate_dir, tmp_path)


class SimpleEnv:
    def __init__(self, registry, champ_id, chal_id, gate_dir, tmp_path):
        self.registry = registry
        self.champ_id = champ_id
        self.chal_id = chal_id
        self.gate_dir = gate_dir
        self.tmp_path = tmp_path

    @property
    def chal_hash(self):
        return self.registry.get(self.chal_id).content_hash

    @property
    def champ_hash(self):
        return self.registry.get(self.champ_id).content_hash

    def write_pending_result(self, name="gate_20260101_000000_x.json"):
        result = GateResult(
            decision="PENDING_HUMAN",
            challenger_hash=self.chal_hash,
            champion_hash=self.champ_hash,
            n_paired=40,
            criteria_checks=[
                {"criterion": "win_rate", "value": 0.7, "threshold": 0.55, "passed": True}
            ],
            spot_check_items=SPOT_ITEMS,
        )
        self.gate_dir.mkdir(parents=True, exist_ok=True)
        path = self.gate_dir / name
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return path


# ─── gate ────────────────────────────────────────────────────────────────────


def _install_gate(monkeypatch, env, decision="PENDING_HUMAN", captured=None):
    store = DatasetStore(base_path=env.tmp_path / "splits")
    store.save_split(
        "test",
        [
            ConversationRecord(
                conversation_id="test_1",
                messages=[Message(role=MessageRole.USER, content="q")],
            )
        ],
    )
    monkeypatch.setattr(cli, "_dataset_store", lambda: store)
    monkeypatch.setattr(cli, "_build_harness", lambda *a, **k: MagicMock())

    def fake_evaluate(self, challenger, champion, convs, **kwargs):
        if captured is not None:
            captured["criteria"] = self.criteria
        return GateResult(
            decision=decision,
            challenger_hash=challenger.content_hash,
            champion_hash=champion.content_hash,
            n_paired=40,
            criteria_checks=[
                {"criterion": "win_rate", "value": 0.7, "threshold": 0.55, "passed": True}
            ],
            spot_check_items=SPOT_ITEMS,
        )

    monkeypatch.setattr(ChampionGate, "evaluate", fake_evaluate)


def test_gate_saves_the_full_result_but_only_a_slim_verdict_in_the_registry(env, monkeypatch):
    _install_gate(monkeypatch, env)

    result = runner.invoke(cli.app, ["gate", "--challenger-hash", env.chal_hash], env=ENV)

    assert result.exit_code == 0, result.output
    files = list(env.gate_dir.glob("gate_*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["spot_check_items"] == SPOT_ITEMS  # the reviewer can read them later

    verdict = env.registry.get(env.chal_id).gate_result
    assert verdict["decision"] == "PENDING_HUMAN"
    assert "spot_check_items" not in verdict  # test-split content stays out of the registry


def test_gate_requires_the_regression_suite_unless_waived(env, monkeypatch):
    captured = {}
    _install_gate(monkeypatch, env, captured=captured)

    runner.invoke(cli.app, ["gate", "--challenger-hash", env.chal_hash], env=ENV)
    assert captured["criteria"].require_regression_suite is True

    runner.invoke(
        cli.app,
        ["gate", "--challenger-hash", env.chal_hash, "--allow-no-regression-suite"],
        env=ENV,
    )
    assert captured["criteria"].require_regression_suite is False


def test_gate_points_the_reviewer_to_gate_review(env, monkeypatch):
    _install_gate(monkeypatch, env)
    result = runner.invoke(cli.app, ["gate", "--challenger-hash", env.chal_hash], env=ENV)
    assert "gate-review" in result.output


# ─── gate-review ─────────────────────────────────────────────────────────────


def test_gate_review_shows_the_full_texts_without_interpreting_markup(env):
    env.write_pending_result()

    result = runner.invoke(cli.app, ["gate-review"])

    assert result.exit_code == 0, result.output
    assert "Click the forgot-password link." in result.output
    assert "Contact support." in result.output
    assert "More direct." in result.output
    assert "[bold]not markup[/bold]" in result.output  # shown literally, not rendered


def test_gate_review_without_pending_results_says_so(env):
    result = runner.invoke(cli.app, ["gate-review"])
    assert result.exit_code == 1


# ─── gate-approve ────────────────────────────────────────────────────────────


def test_approval_never_modifies_the_gate_result(env):
    path = env.write_pending_result()
    before = path.read_bytes()

    result = runner.invoke(cli.app, ["gate-approve", "--reviewer", "alice", "--notes", "ok"])

    assert result.exit_code == 0, result.output
    assert path.read_bytes() == before  # the recorded result is immutable

    review = json.loads((env.gate_dir / "gate_20260101_000000_x.review.json").read_text())
    assert review["reviewer"] == "alice"
    assert review["approved"] is True
    assert review["decision"] == "PASS"
    assert review["challenger_hash"] == env.chal_hash

    verdict = env.registry.get(env.chal_id).gate_result
    assert verdict["decision"] == "PASS"
    assert verdict["human_reviewer"] == "alice"
    assert "spot_check_items" not in verdict
    assert env.registry.get_champion().version_id == env.champ_id  # not promoted


def test_a_result_can_only_be_reviewed_once(env):
    env.write_pending_result()
    assert runner.invoke(cli.app, ["gate-approve", "--reviewer", "alice"]).exit_code == 0

    again = runner.invoke(
        cli.app,
        ["gate-approve", "--reviewer", "bob", "--reject",
         "--result-file", str(env.gate_dir / "gate_20260101_000000_x.json")],
    )
    assert again.exit_code == 1
    assert "already reviewed" in again.output
    assert env.registry.get(env.chal_id).gate_result["decision"] == "PASS"  # bob changed nothing


def test_reviewed_results_are_no_longer_pending(env):
    env.write_pending_result("gate_20260101_000000_a.json")
    runner.invoke(cli.app, ["gate-approve", "--reviewer", "alice"])

    result = runner.invoke(cli.app, ["gate-approve", "--reviewer", "bob"])
    assert result.exit_code == 1
    assert "No pending gate results" in result.output


def test_rejection_is_recorded_beside_the_result_and_in_the_registry(env):
    path = env.write_pending_result()
    before = path.read_bytes()

    result = runner.invoke(
        cli.app, ["gate-approve", "--reviewer", "bob", "--reject", "--notes", "tone is off"]
    )

    assert result.exit_code == 0
    assert path.read_bytes() == before
    verdict = env.registry.get(env.chal_id).gate_result
    assert verdict["decision"] == "FAIL"
    assert verdict["human_approved"] is False
    assert any("tone is off" in r for r in verdict["failure_reasons"])
    # ... and a rejected version cannot be promoted without --force
    assert runner.invoke(cli.app, ["promote", "--hash", env.chal_hash]).exit_code == 1


def test_approval_warns_when_the_champion_changed_since_the_gate_ran(env):
    env.write_pending_result()
    newer = env.registry.register(system_prompt="a newer champion")
    env.registry.promote(newer)

    result = runner.invoke(cli.app, ["gate-approve", "--reviewer", "alice"])

    assert result.exit_code == 0
    assert "champion has changed" in result.output


def test_promote_flag_still_promotes_on_approval(env):
    env.write_pending_result()
    result = runner.invoke(cli.app, ["gate-approve", "--reviewer", "alice", "--promote"])
    assert result.exit_code == 0
    assert env.registry.get_champion().version_id == env.chal_id
