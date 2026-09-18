"""CLI promotion flow: a gate PASS only stages a version.

Only `promote` or canary completion flips the champion.
"""

import json

import pytest
from typer.testing import CliRunner

from prompt_opt.cli import main as cli
from prompt_opt.registry.prompt_registry import PromptRegistry

runner = CliRunner()


@pytest.fixture
def registry(monkeypatch):
    reg = PromptRegistry(db_url="sqlite:///:memory:")
    monkeypatch.setattr(cli, "_registry", lambda: reg)
    return reg


def _seed(registry):
    champ_id = registry.register(system_prompt="champion")
    chal_id = registry.register(system_prompt="challenger")
    registry.promote(champ_id)
    return champ_id, chal_id


def test_promote_requires_gate_pass(registry):
    champ_id, chal_id = _seed(registry)
    chal_hash = registry.get(chal_id).content_hash

    result = runner.invoke(cli.app, ["promote", "--hash", chal_hash])
    assert result.exit_code == 1
    assert registry.get_champion().version_id == champ_id

    registry.attach_gate_result(chal_id, {"decision": "FAIL"})
    assert runner.invoke(cli.app, ["promote", "--hash", chal_hash]).exit_code == 1

    registry.attach_gate_result(chal_id, {"decision": "PASS"})
    result = runner.invoke(cli.app, ["promote", "--hash", chal_hash])
    assert result.exit_code == 0
    assert registry.get_champion().version_id == chal_id


def test_promote_force_and_unknown_hash(registry):
    _, chal_id = _seed(registry)
    assert runner.invoke(cli.app, ["promote", "--hash", "nope"]).exit_code == 1

    chal_hash = registry.get(chal_id).content_hash
    assert runner.invoke(cli.app, ["promote", "--hash", chal_hash, "--force"]).exit_code == 0
    assert registry.get_champion().version_id == chal_id


def test_canary_start_requires_gate_pass_and_defaults_champion(registry, tmp_path):
    champ_id, chal_id = _seed(registry)
    chal_hash = registry.get(chal_id).content_hash
    state = tmp_path / "canary.json"
    args = ["canary", "start", "--challenger-hash", chal_hash, "--state-path", str(state)]

    assert runner.invoke(cli.app, args).exit_code == 1  # not gated
    assert not state.exists()

    registry.attach_gate_result(chal_id, {"decision": "PASS"})
    assert runner.invoke(cli.app, args).exit_code == 0
    data = json.loads(state.read_text())
    assert data["champion_hash"] == registry.get(champ_id).content_hash
    assert data["challenger_hash"] == chal_hash
    assert registry.get_champion().version_id == champ_id  # still the old champion

    # A second start while one is active is refused
    assert runner.invoke(cli.app, args).exit_code == 1


def test_canary_completion_promotes_challenger(registry, tmp_path):
    champ_id, chal_id = _seed(registry)
    chal_hash = registry.get(chal_id).content_hash
    registry.attach_gate_result(chal_id, {"decision": "PASS"})
    state = tmp_path / "canary.json"
    base = ["--state-path", str(state)]
    start = runner.invoke(cli.app, ["canary", "start", "--challenger-hash", chal_hash, *base])
    assert start.exit_code == 0

    stages = len(json.loads(state.read_text())["stages"])

    def elapse_current_stage_soak():
        # A stage's soak clock starts when it becomes active, so each stage's soak has to
        # elapse in turn.
        data = json.loads(state.read_text())
        data["stages"][data["stage_idx"]]["started_at"] = 0
        state.write_text(json.dumps(data))

    for stage in range(stages):
        elapse_current_stage_soak()
        assert runner.invoke(cli.app, ["canary", "advance", *base]).exit_code == 0
        if stage < stages - 1:
            assert registry.get_champion().version_id == champ_id  # not promoted yet

    assert registry.get_champion().version_id == chal_id
    assert json.loads(state.read_text())["status"] == "complete"


def test_shadow_refuses_challenger_that_is_champion(registry):
    champ_id, _ = _seed(registry)
    result = runner.invoke(
        cli.app,
        ["shadow", "--challenger-hash", registry.get(champ_id).content_hash],
        env={"OPENAI_API_KEY": "sk-test"},
    )
    assert result.exit_code == 1
