"""RollbackMonitor must not walk the champion back through history on a persisting breach."""

from datetime import datetime, timedelta

import pytest
from typer.testing import CliRunner

from prompt_opt.cli import main as cli
from prompt_opt.deployment import rollback as rollback_mod
from prompt_opt.deployment.rollback import RollbackMonitor
from prompt_opt.registry.prompt_registry import PromptRegistry

BREACH = {"error_rate": 0.5}


def _registry(n=3):
    registry = PromptRegistry(db_url="sqlite:///:memory:")
    ids = [registry.register(system_prompt=f"V{i}") for i in range(1, n + 1)]
    for version_id in ids:
        registry.promote(version_id)
    return registry, ids


def _monitor(registry, **kwargs):
    alerts = []
    return RollbackMonitor(registry, alert_fn=alerts.append, **kwargs), alerts


def test_a_persisting_breach_rolls_back_once_not_repeatedly():
    registry, (v1, v2, v3) = _registry()
    monitor, alerts = _monitor(registry)

    first = monitor.check(**BREACH)
    assert first.scope == "registry"
    assert first.rolled_back_to == v2
    assert registry.get_champion().version_id == v2

    # Metrics lag a rollback: the breach is still visible on the next checks.
    for _ in range(3):
        again = monitor.check(**BREACH)
        assert again.scope == "suppressed"
        assert again.rolled_back_to is None
        assert "SUPPRESSED" in again.reason
        assert registry.get_champion().version_id == v2  # v1 was NOT reached
    assert len(alerts) == 4  # every breach still alerts a human


def test_a_new_promotion_re_arms_automatic_rollback():
    registry, (v1, v2, v3) = _registry()
    monitor, _ = _monitor(registry)
    monitor.check(**BREACH)                      # v3 -> v2
    assert monitor.check(**BREACH).scope == "suppressed"

    v4 = registry.register(system_prompt="V4")
    registry.promote(v4)                         # a deliberate deployment

    event = monitor.check(**BREACH)
    assert event.scope == "registry"
    assert event.rolled_back_to == v2            # v3 was rejected and is skipped
    assert registry.get_champion().version_id == v2


def test_suppression_expires_after_the_cooldown(monkeypatch):
    registry, (v1, v2, v3) = _registry()
    monitor, _ = _monitor(registry, cooldown_seconds=60)
    monitor.check(**BREACH)
    rolled_back_at = datetime.fromisoformat(registry.champion_history(limit=1)[0]["created_at"])

    def clock_at(seconds_after_rollback):
        class FakeDatetime(datetime):
            @classmethod
            def utcnow(cls):
                return rolled_back_at + timedelta(seconds=seconds_after_rollback)

        monkeypatch.setattr(rollback_mod, "datetime", FakeDatetime)

    clock_at(59)   # still inside the cooldown
    assert monitor.check(**BREACH).scope == "suppressed"

    clock_at(61)   # cooldown over: a persisting breach may roll back again
    event = monitor.check(**BREACH)
    assert event.scope == "registry"
    assert event.rolled_back_to == v1


def test_a_zero_cooldown_restores_the_old_cascading_behaviour():
    registry, (v1, v2, v3) = _registry()
    monitor, _ = _monitor(registry, cooldown_seconds=0)
    assert monitor.check(**BREACH).rolled_back_to == v2
    assert monitor.check(**BREACH).rolled_back_to == v1


def test_no_previous_version_is_reported_as_such():
    registry, (only,) = _registry(n=1)
    monitor, alerts = _monitor(registry)

    event = monitor.check(**BREACH)

    assert event.scope == "no_previous"
    assert event.rolled_back_to is None
    assert "no previous version" in event.reason
    assert registry.get_champion().version_id == only
    assert alerts == [event]


def test_healthy_metrics_never_trigger_anything():
    registry, _ = _registry()
    monitor, alerts = _monitor(registry)
    assert monitor.check(error_rate=0.01) is None
    assert alerts == []


# ─── CLI ─────────────────────────────────────────────────────────────────────


def test_monitor_cli_says_when_no_rollback_was_performed(tmp_path, monkeypatch):
    registry, (v1, v2, v3) = _registry()
    monkeypatch.setattr(cli, "_registry", lambda: registry)
    runner = CliRunner()
    args = ["monitor", "--error-rate", "0.5", "--canary-state", str(tmp_path / "none.json")]

    first = runner.invoke(cli.app, args)
    assert first.exit_code == 1
    assert "Rolled back to" in first.output

    second = runner.invoke(cli.app, args)
    assert second.exit_code == 1  # still a breach: the exit code stays non-zero
    assert "human attention needed" in second.output
    assert registry.get_champion().version_id == v2


@pytest.mark.parametrize("scope", ["suppressed", "no_previous"])
def test_event_scopes_are_documented_values(scope):
    from prompt_opt.deployment.rollback import RollbackEvent

    assert RollbackEvent(reason="x", scope=scope).scope == scope
