"""Canary: per-stage soak clocks, sticky routing, batched persistence, atomic state files."""

import json

import pytest

from prompt_opt.deployment import canary as cmod
from prompt_opt.deployment.canary import CanaryController

HOUR = 3600


@pytest.fixture
def clock(monkeypatch):
    """A controllable time.time() for the canary module."""

    class Clock:
        now = 1_000_000.0

        def advance(self, seconds):
            self.now += seconds

    c = Clock()
    monkeypatch.setattr(cmod.time, "time", lambda: c.now)
    return c


def _ctrl(tmp_path, ramp=None, **kwargs):
    return CanaryController(
        "chal", "champ", ramp_schedule=ramp, state_path=str(tmp_path / "canary.json"), **kwargs
    )


# ─── soak clocks ─────────────────────────────────────────────────────────────


def test_each_stage_soaks_from_the_moment_it_becomes_active(clock, tmp_path):
    ctrl = _ctrl(tmp_path)  # default ramp: 1%/1h, 5%/2h, 25%/4h, 100%

    clock.advance(HOUR)
    assert ctrl.maybe_advance() is True  # 1% -> 5% at t+1h
    assert ctrl.current_stage.traffic_pct == 5
    assert ctrl.current_stage.started_at == clock.now  # its clock starts now, not at t0

    assert ctrl.maybe_advance() is False  # immediately: 0 of 2h soaked
    clock.advance(2 * HOUR - 1)
    assert ctrl.maybe_advance() is False  # one second short
    clock.advance(1)
    assert ctrl.maybe_advance() is True
    assert ctrl.current_stage.traffic_pct == 25


def test_an_operator_who_waits_long_cannot_skip_intermediate_stages(clock, tmp_path):
    """Regression: every stage's clock used to start at rollout start, so after a long wait
    the whole ramp could be advanced back-to-back with no soak at 5% or 25%."""
    ctrl = _ctrl(tmp_path)

    clock.advance(24 * HOUR)
    assert ctrl.maybe_advance() is True    # 1% -> 5%
    assert ctrl.maybe_advance() is False   # 5% has soaked for 0s, whatever the wall clock says
    assert ctrl.current_stage.traffic_pct == 5
    assert ctrl.state.status == "active"


def test_full_ramp_takes_the_sum_of_the_soaks(clock, tmp_path):
    ctrl = _ctrl(tmp_path)
    steps = 0
    while ctrl.state.status == "active":
        clock.advance(ctrl.current_stage.soak_seconds)
        assert ctrl.maybe_advance() is True
        steps += 1
    assert ctrl.state.status == "complete"
    assert steps == 4
    assert clock.now - 1_000_000.0 == 1 * HOUR + 2 * HOUR + 4 * HOUR + 0


def test_a_stage_with_no_soak_completes_as_soon_as_it_is_reached(clock, tmp_path):
    ctrl = _ctrl(tmp_path, ramp=[(50, 0), (100, 0)])
    assert ctrl.maybe_advance() is True   # -> 100%
    assert ctrl.maybe_advance() is True   # -> complete
    assert ctrl.state.status == "complete"


def test_pending_stages_report_their_full_soak_as_remaining(clock, tmp_path):
    ctrl = _ctrl(tmp_path)
    assert ctrl.state.stages[1].started_at is None
    assert ctrl.state.stages[1].remaining_seconds == 2 * HOUR
    assert ctrl.state.stages[1].elapsed_seconds == 0.0
    assert ctrl.current_stage.remaining_seconds == HOUR


def test_soak_clocks_survive_a_restart(clock, tmp_path):
    ctrl = _ctrl(tmp_path)
    clock.advance(HOUR)
    ctrl.maybe_advance()
    advanced_at = ctrl.current_stage.started_at

    clock.advance(30 * 60)
    resumed = CanaryController.load(str(tmp_path / "canary.json"))

    assert resumed.state.stage_idx == 1
    assert resumed.current_stage.started_at == advanced_at
    assert resumed.state.stages[2].started_at is None
    assert resumed.current_stage.elapsed_seconds == 30 * 60


def test_legacy_state_files_get_fresh_clocks_for_stages_not_yet_reached(clock, tmp_path):
    t0 = clock.now
    legacy = {
        "challenger_hash": "chal", "champion_hash": "champ", "stage_idx": 1,
        "status": "active", "halt_reason": None, "started_at": "2026-01-01T00:00:00",
        "current_pct": 5,
        "stages": [
            {"traffic_pct": pct, "soak_seconds": soak, "started_at": t0 - 5 * HOUR,
             "requests_seen": 0, "guardrail_breaches": 0}
            for pct, soak in [(1, HOUR), (5, 2 * HOUR), (25, 4 * HOUR), (100, 0)]
        ],
    }
    path = tmp_path / "canary.json"
    path.write_text(json.dumps(legacy))

    ctrl = CanaryController.load(str(path))

    assert ctrl.state.stages[0].started_at == t0 - 5 * HOUR   # already past: unchanged
    assert ctrl.state.stages[1].started_at == t0 - 5 * HOUR   # the current stage keeps its clock
    assert ctrl.state.stages[2].started_at is None            # not reached: was a bogus stamp
    assert ctrl.state.stages[3].started_at is None
    assert ctrl.maybe_advance() is True                       # 5% soaked long ago -> 25%
    assert ctrl.maybe_advance() is False                      # 25% must now soak for 4h


# ─── routing ─────────────────────────────────────────────────────────────────


def test_a_request_key_always_gets_the_same_arm(clock, tmp_path):
    ctrl = _ctrl(tmp_path, ramp=[(50, HOUR), (100, 0)])
    arms = {key: ctrl.should_use_challenger(request_key=key) for key in map(str, range(200))}

    for _ in range(3):
        assert all(ctrl.should_use_challenger(request_key=k) == arm for k, arm in arms.items())


def test_keyed_routing_matches_the_traffic_percentage(clock, tmp_path):
    ctrl = _ctrl(tmp_path, ramp=[(10, HOUR), (100, 0)], persist_every=10_000)
    share = sum(ctrl.should_use_challenger(request_key=f"user-{i}") for i in range(5000)) / 5000
    assert 0.07 < share < 0.13


def test_callers_on_the_challenger_stay_on_it_as_traffic_grows(clock, tmp_path):
    ctrl = _ctrl(tmp_path, ramp=[(5, 0), (25, 0), (100, 0)], persist_every=10_000)
    keys = [f"user-{i}" for i in range(2000)]

    at_5 = {k for k in keys if ctrl.should_use_challenger(request_key=k)}
    ctrl.maybe_advance()
    at_25 = {k for k in keys if ctrl.should_use_challenger(request_key=k)}
    ctrl.maybe_advance()
    at_100 = {k for k in keys if ctrl.should_use_challenger(request_key=k)}

    assert at_5 <= at_25 <= at_100
    assert len(at_5) < len(at_25) < len(at_100) == len(keys)


def test_different_rollouts_bucket_the_same_key_differently(clock, tmp_path):
    a = CanaryController("chal-A", "champ", ramp_schedule=[(50, 0)],
                         state_path=str(tmp_path / "a.json"))
    b = CanaryController("chal-B", "champ", ramp_schedule=[(50, 0)],
                         state_path=str(tmp_path / "b.json"))
    keys = [f"user-{i}" for i in range(500)]
    assert [a.should_use_challenger(k) for k in keys] != [b.should_use_challenger(k) for k in keys]


def test_no_challenger_traffic_unless_the_canary_is_active(clock, tmp_path):
    ctrl = _ctrl(tmp_path, ramp=[(100, HOUR)])
    ctrl.halt("manual")
    assert ctrl.should_use_challenger(request_key="anyone") is False
    assert ctrl.should_use_challenger() is False


# ─── persistence ─────────────────────────────────────────────────────────────


def _saved_requests(tmp_path):
    return json.loads((tmp_path / "canary.json").read_text())["stages"][0]["requests_seen"]


def test_request_counters_are_persisted_in_batches_not_per_request(clock, tmp_path):
    ctrl = _ctrl(tmp_path, persist_every=3)

    ctrl.should_use_challenger()
    ctrl.should_use_challenger()
    assert _saved_requests(tmp_path) == 0   # no disk write for the first two requests
    ctrl.should_use_challenger()
    assert _saved_requests(tmp_path) == 3   # the third completes a batch

    ctrl.should_use_challenger()
    ctrl.flush()
    assert _saved_requests(tmp_path) == 4


def test_state_changes_are_persisted_immediately(clock, tmp_path):
    ctrl = _ctrl(tmp_path, persist_every=10_000)
    ctrl.should_use_challenger()
    ctrl.record_guardrail_breach()

    saved = json.loads((tmp_path / "canary.json").read_text())
    assert saved["stages"][0]["guardrail_breaches"] == 1
    assert saved["stages"][0]["requests_seen"] == 1  # the pending counter rides along


def test_state_writes_are_atomic(clock, tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path)
    good = (tmp_path / "canary.json").read_text()

    def crash(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(cmod.os, "replace", crash)
    with pytest.raises(OSError):
        ctrl.halt("test")

    # The previous state file is intact and still parses
    assert (tmp_path / "canary.json").read_text() == good
    assert json.loads(good)["status"] == "active"


def test_no_temp_files_are_left_behind(clock, tmp_path):
    ctrl = _ctrl(tmp_path, persist_every=1)
    for _ in range(5):
        ctrl.should_use_challenger()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["canary.json"]
