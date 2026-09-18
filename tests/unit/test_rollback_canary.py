"""RollbackMonitor behaviour while a canary is active."""

from prompt_opt.deployment.canary import CanaryController
from prompt_opt.deployment.rollback import RollbackMonitor
from prompt_opt.registry.prompt_registry import PromptRegistry


def _setup(tmp_path):
    registry = PromptRegistry(db_url="sqlite:///:memory:")
    stable_old = registry.register(system_prompt="old")
    stable = registry.register(system_prompt="stable")
    challenger = registry.register(system_prompt="challenger")
    registry.promote(stable_old)
    registry.promote(stable)
    canary = CanaryController(
        challenger_hash=registry.get(challenger).content_hash,
        champion_hash=registry.get(stable).content_hash,
        state_path=str(tmp_path / "canary.json"),
    )
    return registry, canary, stable_old, stable, challenger


def test_breach_during_canary_rolls_back_canary_not_champion(tmp_path):
    registry, canary, _, stable, challenger = _setup(tmp_path)
    monitor = RollbackMonitor(registry, canary=canary, alert_fn=lambda e: None)

    event = monitor.check(error_rate=0.2)

    assert event is not None
    assert event.scope == "canary"
    assert event.previous_champion == challenger
    assert event.rolled_back_to == stable
    assert canary.state.status == "rolledback"
    assert registry.get_champion().version_id == stable  # champion untouched


def test_breach_without_active_canary_rolls_back_registry(tmp_path):
    registry, canary, stable_old, stable, _ = _setup(tmp_path)
    canary.rollback("already done")  # canary no longer active
    monitor = RollbackMonitor(registry, canary=canary, alert_fn=lambda e: None)

    event = monitor.check(error_rate=0.2)

    assert event.scope == "registry"
    assert event.rolled_back_to == stable_old
    assert registry.get_champion().version_id == stable_old


def test_no_breach_leaves_canary_running(tmp_path):
    registry, canary, _, stable, _ = _setup(tmp_path)
    monitor = RollbackMonitor(registry, canary=canary, alert_fn=lambda e: None)

    assert monitor.check(error_rate=0.01) is None
    assert canary.state.status == "active"
