import json
import time
from unittest.mock import MagicMock, patch

from prompt_opt.deployment.canary import CanaryController
from prompt_opt.deployment.rollback import RollbackMonitor
from prompt_opt.deployment.shadow import ShadowRunner
from prompt_opt.eval.harness import PromptCandidate
from prompt_opt.registry.prompt_registry import PromptRegistry


def test_prompt_registry_lifecycle():
    # Use in-memory SQLite database
    registry = PromptRegistry(db_url="sqlite:///:memory:")

    # Initially no champion
    assert registry.get_champion() is None

    # Register first candidate
    v1_id = registry.register(
        system_prompt="System Prompt V1",
        few_shot_demos=[{"input": "hello", "output": "hi"}]
    )
    assert v1_id == "prompt_v1"

    # Registering same content should be idempotent
    v1_id_dup = registry.register(
        system_prompt="System Prompt V1",
        few_shot_demos=[{"input": "hello", "output": "hi"}]
    )
    assert v1_id == v1_id_dup

    # Promote V1
    registry.promote(v1_id)
    champ = registry.get_champion()
    assert champ is not None
    assert champ.version_id == v1_id
    assert champ.is_champion is True

    # Register second candidate
    v2_id = registry.register(
        system_prompt="System Prompt V2",
        few_shot_demos=None
    )
    assert v2_id == "prompt_v2"

    # Promote V2
    registry.promote(v2_id)
    champ = registry.get_champion()
    assert champ.version_id == v2_id

    # Rollback should revert to V1
    rolled_to = registry.rollback()
    assert rolled_to == v1_id

    champ = registry.get_champion()
    assert champ.version_id == v1_id

    # Rollback again should return None (no further history)
    assert registry.rollback() is None


def test_rollback_monitor():
    registry = PromptRegistry(db_url="sqlite:///:memory:")

    # Set up champion and previous champion
    v1_id = registry.register(system_prompt="V1")
    registry.promote(v1_id)
    v2_id = registry.register(system_prompt="V2")
    registry.promote(v2_id)

    monitor = RollbackMonitor(registry=registry)

    # 1. Normal check
    event = monitor.check(
        error_rate=0.01,
        latency_ms_current=200,
        latency_ms_baseline=200,
        refusal_rate=0.05,
        refusal_rate_baseline=0.05,
        format_compliance_rate=0.95,
        regression_pass_rate=1.0
    )
    assert event is None
    assert registry.get_champion().version_id == v2_id

    # 2. Breach error rate
    event = monitor.check(error_rate=0.06)
    assert event is not None
    assert "error_rate=0.060 > max=0.05" in event.reason
    assert event.rolled_back_to == v1_id
    assert registry.get_champion().version_id == v1_id

    # Reset champion to V2 for next tests
    registry.promote(v2_id)

    # 3. Breach latency regression
    event = monitor.check(latency_ms_current=270, latency_ms_baseline=200)
    assert event is not None
    assert "latency regression=35.0% > max=30.0%" in event.reason

    # Reset champion to V2
    registry.promote(v2_id)

    # 4. Breach refusal rate delta
    event = monitor.check(refusal_rate=0.3, refusal_rate_baseline=0.1)
    assert event is not None
    assert "refusal_rate_delta=0.200 > max=0.15" in event.reason

    # Reset champion to V2
    registry.promote(v2_id)

    # 5. Breach format compliance
    event = monitor.check(format_compliance_rate=0.85)
    assert event is not None
    assert "format_compliance=0.850 < min=0.9" in event.reason

    # Reset champion to V2
    registry.promote(v2_id)

    # 6. Breach regression pass rate
    event = monitor.check(regression_pass_rate=0.9)
    assert event is not None
    assert "regression_pass_rate=0.900 < min=1.0" in event.reason


def test_canary_controller(tmp_path):
    state_file = tmp_path / "canary_state.json"

    # Custom schedule for testing: 10% soak 10s, 50% soak 20s
    schedule = [(10, 10), (50, 20)]

    ctrl = CanaryController(
        challenger_hash="chal_123",
        champion_hash="champ_456",
        ramp_schedule=schedule,
        state_path=str(state_file),
        max_guardrail_breaches=2
    )

    assert ctrl.state.status == "active"
    assert ctrl.current_stage.traffic_pct == 10

    # test traffic routing
    ctrl._rng = MagicMock()
    # Mocking rng to return 5 (which is <= 10, so challenger should be used)
    ctrl._rng.randint.return_value = 5
    assert ctrl.should_use_challenger() is True
    assert ctrl.current_stage.requests_seen == 1

    # Mocking rng to return 15 (which is > 10, so champion should be used)
    ctrl._rng.randint.return_value = 15
    assert ctrl.should_use_challenger() is False
    assert ctrl.current_stage.requests_seen == 2

    # Advance when soak time is NOT complete
    assert ctrl.maybe_advance() is False
    assert ctrl.state.stage_idx == 0

    # Advance when soak time is complete
    ctrl.current_stage.started_at = time.time() - 15  # soak is 10s
    assert ctrl.maybe_advance() is True
    assert ctrl.state.stage_idx == 1
    assert ctrl.current_stage.traffic_pct == 50

    # Record guardrail breach
    ctrl.record_guardrail_breach()
    assert ctrl.current_stage.guardrail_breaches == 1
    assert ctrl.state.status == "active"

    # Second breach triggers halt since max_guardrail_breaches = 2
    ctrl.record_guardrail_breach()
    assert ctrl.state.status == "halted"
    assert ctrl.state.halt_reason == "Too many guardrail breaches"

    # Rollback state
    ctrl.rollback("Manual rollback requested")
    assert ctrl.state.status == "rolledback"


def test_canary_state_resume(tmp_path):
    state_file = tmp_path / "canary_state.json"
    schedule = [(10, 10), (50, 20), (100, 0)]

    ctrl = CanaryController(
        challenger_hash="chal_resume",
        champion_hash="champ_resume",
        ramp_schedule=schedule,
        state_path=str(state_file),
    )
    ctrl.should_use_challenger()
    ctrl.should_use_challenger()

    # Simulate a process restart: reload from the persisted state file
    resumed = CanaryController.load(state_path=str(state_file))
    assert resumed.state.challenger_hash == "chal_resume"
    assert resumed.state.champion_hash == "champ_resume"
    assert resumed.state.status == "active"
    assert resumed.state.stage_idx == 0
    assert resumed.current_stage.requests_seen == 2
    assert [s.traffic_pct for s in resumed.state.stages] == [10, 50, 100]

    # Advance past the soak period and verify persistence across another reload
    resumed.current_stage.started_at = time.time() - 15
    assert resumed.maybe_advance() is True

    reloaded = CanaryController.load(state_path=str(state_file))
    assert reloaded.state.stage_idx == 1
    assert reloaded.current_stage.traffic_pct == 50


@patch("litellm.completion")
def test_shadow_runner(mock_completion, tmp_path):
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content="Mocked LLM Response"))
    ]
    mock_completion.return_value = mock_resp

    challenger = PromptCandidate(system_prompt="Challenger prompt")
    champion = PromptCandidate(system_prompt="Champion prompt")

    runner = ShadowRunner(
        challenger=challenger,
        champion=champion,
        model="mock-model",
        log_path=str(tmp_path)
    )

    messages = [{"role": "user", "content": "How are you?"}]
    result = runner.run_request(messages)

    assert result.champion_response == "Mocked LLM Response"
    assert result.challenger_response == "Mocked LLM Response"
    assert result.latency_champion_ms > 0
    assert result.latency_challenger_ms > 0
    assert result.error is None

    # Check that log file is created and contains log entry
    log_file = tmp_path / "shadow_log.jsonl"
    assert log_file.exists()

    lines = log_file.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["champion_hash"] == champion.content_hash
    assert entry["challenger_hash"] == challenger.content_hash
    assert entry["champion_response"] == "Mocked LLM Response"
    assert entry["challenger_response"] == "Mocked LLM Response"
