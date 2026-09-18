"""Shadow mode: runs what was evaluated (demos included), records errors, replays cleanly."""

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from prompt_opt import llm
from prompt_opt.cli import main as cli
from prompt_opt.config import settings
from prompt_opt.deployment.shadow import ShadowRunner
from prompt_opt.eval.harness import PromptCandidate, replay_turns
from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole
from prompt_opt.registry.prompt_registry import PromptRegistry
from prompt_opt.storage.dataset_store import DatasetStore


def _reply(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


CHAMPION = PromptCandidate(system_prompt="CHAMPION SYSTEM")
CHALLENGER = PromptCandidate(
    system_prompt="CHALLENGER SYSTEM", few_shot_demos=[{"user": "demo q", "assistant": "demo a"}]
)
MESSAGES = [{"role": "user", "content": "the real question"}]


@pytest.fixture
def calls(monkeypatch):
    """Record every completion call; behaviour is set per test via `behaviour`."""
    class Calls(list):
        behaviour: dict

    recorded = Calls()
    behaviour = {"fail_on": set()}

    def fake(**kwargs):
        system = kwargs["messages"][0]["content"]
        recorded.append(kwargs["messages"])
        if system.split()[0] in behaviour["fail_on"]:
            raise RuntimeError("503 upstream unavailable")
        return _reply(f"answer from {system.split()[0]}")

    monkeypatch.setattr(llm, "completion", fake)
    recorded.behaviour = behaviour
    return recorded


def _runner(tmp_path):
    return ShadowRunner(CHALLENGER, CHAMPION, model="m", log_path=str(tmp_path))


def test_both_prompts_are_replayed_with_their_few_shot_demos(calls, tmp_path):
    result = _runner(tmp_path).run_request(MESSAGES)

    champion_call, challenger_call = calls
    assert [m["role"] for m in champion_call] == ["system", "user"]
    assert [m["role"] for m in challenger_call] == ["system", "user", "assistant", "user"]
    assert challenger_call[1]["content"] == "demo q"  # demos were silently dropped before
    assert challenger_call[-1]["content"] == "the real question"
    assert result.error is None


def test_a_failed_challenger_never_disturbs_the_served_response(calls, tmp_path):
    calls.behaviour["fail_on"] = {"CHALLENGER"}

    result = _runner(tmp_path).run_request(MESSAGES)

    assert result.champion_response == "answer from CHAMPION"
    assert result.challenger_response == ""       # not an "[ERROR: ...]" pseudo-answer
    assert result.challenger_error.startswith("RuntimeError: 503")
    assert result.champion_error is None
    assert result.error.startswith("challenger: ")


def test_a_failed_champion_is_reported_not_served_as_text(calls, tmp_path):
    calls.behaviour["fail_on"] = {"CHAMPION"}

    result = _runner(tmp_path).run_request(MESSAGES)

    assert result.champion_response == ""
    assert "ERROR" not in result.champion_response
    assert result.error.startswith("champion: ")
    assert result.challenger_response == "answer from CHALLENGER"


def test_errors_are_logged(calls, tmp_path):
    calls.behaviour["fail_on"] = {"CHALLENGER"}
    _runner(tmp_path).run_request(MESSAGES)

    record = json.loads((tmp_path / "shadow_log.jsonl").read_text().splitlines()[0])
    assert record["challenger_error"].startswith("RuntimeError")
    assert record["champion_error"] is None
    assert record["challenger_response"] == ""


# ─── replay_turns ────────────────────────────────────────────────────────────


def _conv(*messages):
    return ConversationRecord(
        conversation_id="c1", messages=[Message(role=r, content=c) for r, c in messages]
    )


def test_replay_turns_drops_system_tool_and_the_trailing_assistant_reply():
    conv = _conv(
        (MessageRole.SYSTEM, "original system"),
        (MessageRole.USER, "q1"),
        (MessageRole.ASSISTANT, "a1"),
        (MessageRole.USER, "q2"),
        (MessageRole.TOOL, "tool output"),
        (MessageRole.ASSISTANT, "logged final reply"),
    )
    assert replay_turns(conv) == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]


# ─── CLI ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def cli_env(tmp_path, monkeypatch, calls):
    registry = PromptRegistry(db_url="sqlite:///:memory:")
    champ_id = registry.register(system_prompt="CHAMPION SYSTEM")
    chal_id = registry.register(system_prompt="CHALLENGER SYSTEM")
    registry.promote(champ_id)

    store = DatasetStore(base_path=tmp_path / "splits")
    conv = _conv(
        (MessageRole.SYSTEM, "logged system"),
        (MessageRole.USER, "how do I reset my password?"),
        (MessageRole.ASSISTANT, "the reply that was logged in production"),
    )
    store.save_split("dev", [conv])
    store.save_split("test", [conv])

    monkeypatch.setattr(cli, "_registry", lambda: registry)
    monkeypatch.setattr(cli, "_dataset_store", lambda: store)
    monkeypatch.setattr(settings, "shadow_log_path", str(tmp_path / "shadow_logs"))
    return registry.get(chal_id).content_hash


ENV = {"OPENAI_API_KEY": "sk-test"}


def test_cli_does_not_feed_the_logged_reply_to_the_models(cli_env, calls):
    result = CliRunner().invoke(cli.app, ["shadow", "--challenger-hash", cli_env], env=ENV)

    assert result.exit_code == 0, result.output
    for sent in calls:
        assert sent[-1] == {"role": "user", "content": "how do I reset my password?"}
        assert not any("logged in production" in m["content"] for m in sent)
        assert not any(m["content"] == "logged system" for m in sent)


def test_cli_counts_errors_per_prompt(cli_env, calls):
    calls.behaviour["fail_on"] = {"CHALLENGER"}

    result = CliRunner().invoke(cli.app, ["shadow", "--challenger-hash", cli_env], env=ENV)

    assert result.exit_code == 0, result.output
    assert "errors" in result.output
    row = next(line for line in result.output.splitlines() if "errors" in line)
    assert row.replace("│", " ").split()[-2:] == ["0", "1"]  # champion 0, challenger 1
    assert "n/a" in result.output                            # no successful challenger latency


def test_cli_refuses_the_frozen_test_split(cli_env, calls):
    result = CliRunner().invoke(
        cli.app, ["shadow", "--challenger-hash", cli_env, "--split", "test"], env=ENV
    )
    assert result.exit_code == 1
    assert "reserved for the gate" in result.output
    assert calls == []
