"""DriftDetector: valid statistics, honest verdicts, latency, persistence."""

import json
import logging

import numpy as np
import pytest

from prompt_opt.feedback.drift_detector import (
    DriftDetector,
    DriftType,
    _kl_divergence,
)


@pytest.fixture(autouse=True)
def quiet(caplog):
    caplog.set_level(logging.CRITICAL)


def beta(rng, n, a=5.0, b=2.0):
    return list(rng.beta(a, b, n))


def detector_with_baseline(scores, **baseline):
    detector = DriftDetector()
    detector.update_baseline(scores, **baseline)
    return detector


# ─── score distribution ──────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [30, 50, 100, 300, 1000])
def test_identical_distributions_do_not_raise_false_alarms(n):
    """Regression: the old KL estimator (20 bins, eps=1e-9) exceeded its 0.1 threshold on
    100% of same-distribution comparisons at n=100 and 57% at n=300."""
    rng = np.random.default_rng(1234)
    trials = 200
    alarms = sum(
        detector_with_baseline(beta(rng, n)).assess(new_scores=beta(rng, n)).verdict == "drift"
        for _ in range(trials)
    )
    assert alarms / trials <= 0.03


def test_a_real_shift_is_detected():
    rng = np.random.default_rng(7)
    detected = sum(
        detector_with_baseline(beta(rng, 200)).assess(
            new_scores=beta(rng, 200, 3.5, 2.6)   # mean drops from 0.71 to 0.57
        ).verdict == "drift"
        for _ in range(100)
    )
    assert detected >= 95


def test_a_large_drop_is_detected_even_on_small_samples():
    rng = np.random.default_rng(8)
    detected = sum(
        detector_with_baseline(beta(rng, 40)).assess(new_scores=beta(rng, 40, 2, 5)).verdict
        == "drift"
        for _ in range(100)
    )
    assert detected >= 98


def test_a_statistically_real_but_negligible_shift_does_not_fire():
    rng = np.random.default_rng(3)
    baseline = beta(rng, 30_000)
    shifted = [min(1.0, s + 0.01) for s in beta(rng, 30_000)]     # a 0.01 nudge, huge n

    report = detector_with_baseline(baseline).assess(new_scores=shifted)

    assert report.ks_pvalue < 0.01           # the KS test can see it...
    assert report.kl < 0.1                   # ...but it is far too small to matter
    assert report.verdict == "no_drift"


def test_drift_events_carry_severity_and_the_evidence():
    rng = np.random.default_rng(5)
    report = detector_with_baseline(beta(rng, 200)).assess(new_scores=beta(rng, 200, 2, 5))

    (event,) = report.events
    assert event.drift_type == DriftType.JUDGE_SCORE
    assert event.severity == "critical"                 # KL far above twice the threshold
    assert "KS p=" in event.description
    assert report.n_baseline == 200 and report.n_new == 200


def test_constant_scores_are_handled():
    same = detector_with_baseline([0.8] * 40).assess(new_scores=[0.8] * 40)
    assert same.verdict == "no_drift"

    different = detector_with_baseline([0.9] * 40).assess(new_scores=[0.1] * 40)
    assert different.verdict == "drift"


def test_kl_is_finite_and_non_negative_even_with_empty_bins():
    kl = _kl_divergence([0.9] * 50, [0.1] * 50)      # the two samples share no bin at all
    assert np.isfinite(kl) and kl > 0
    assert _kl_divergence([], [0.5]) == 0.0


# ─── no verdict is not "no drift" ────────────────────────────────────────────


def test_too_few_samples_is_reported_as_insufficient_not_as_no_drift():
    """Regression: <10 samples silently returned 'no drift', even for 0.9 -> 0.1."""
    detector = detector_with_baseline([0.9] * 5)

    report = detector.assess(new_scores=[0.1] * 5)

    assert report.verdict == "insufficient_data"
    assert report.events == []
    assert "at least 30" in report.notes[0]


@pytest.mark.parametrize("n_baseline, n_new", [(100, 10), (10, 100), (29, 200)])
def test_both_samples_need_enough_data(n_baseline, n_new):
    rng = np.random.default_rng(0)
    report = detector_with_baseline(beta(rng, n_baseline)).assess(new_scores=beta(rng, n_new))
    assert report.verdict == "insufficient_data"


def test_the_minimum_sample_size_is_configurable():
    detector = DriftDetector(min_samples=5)
    detector.update_baseline([0.9] * 6)
    assert detector.assess(new_scores=[0.1] * 6).verdict == "drift"


def test_no_baseline_at_all_is_no_verdict():
    assert DriftDetector().assess(new_scores=[0.5] * 50).verdict == "insufficient_data"


def test_an_observation_that_cannot_be_compared_is_no_verdict():
    report = DriftDetector().assess(refusal_rate=0.4)      # no baseline refusal rate
    assert report.verdict == "insufficient_data"
    assert "no baseline refusal rate" in report.notes[0]
    assert DriftDetector().assess().verdict == "insufficient_data"      # nothing given at all


# ─── guardrail trends ────────────────────────────────────────────────────────


def test_refusal_rate_drift_is_detected_against_the_baseline():
    detector = detector_with_baseline([0.7] * 40, refusal_rate=0.05)

    report = detector.assess(refusal_rate=0.30)

    (event,) = report.events
    assert event.drift_type == DriftType.REFUSAL_RATE and event.severity == "critical"

    calm = detector_with_baseline([0.7] * 40, refusal_rate=0.05)
    assert calm.assess(refusal_rate=0.06).verdict == "no_drift"


def test_ema_smooths_observations_and_starts_from_the_first_one():
    detector = detector_with_baseline([0.7] * 40, refusal_rate=0.05)

    detector.assess(refusal_rate=0.05)
    detector.assess(refusal_rate=0.35)          # one bad reading: EMA = 0.1*0.35 + 0.9*0.05 = 0.08
    assert detector._ema_refusal == pytest.approx(0.08)
    assert detector.assess(refusal_rate=0.05).verdict == "no_drift"


def test_format_compliance_below_the_floor_fires():
    detector = detector_with_baseline([0.7] * 40)
    report = detector.assess(format_compliance=0.6)
    (event,) = report.events
    assert event.drift_type == DriftType.FORMAT_COMPLIANCE and event.severity == "critical"
    assert DriftDetector().assess(format_compliance=0.99).verdict == "no_drift"


def test_latency_drift_is_now_implemented():
    """Regression: `latency_ms` / `baseline_latency_ms` were accepted and ignored."""
    detector = detector_with_baseline([0.7] * 40, latency_ms=200.0)

    within = detector.assess(latency_ms=250.0)                    # +25%
    assert within.verdict == "no_drift"

    detector.update_baseline([0.7] * 40, latency_ms=200.0)        # reset the EMA
    slow = detector.assess(latency_ms=300.0)                      # +50%
    (event,) = slow.events
    assert event.drift_type == DriftType.LATENCY and event.severity == "warning"
    assert event.baseline_value == 200.0

    detector.update_baseline([0.7] * 40, latency_ms=200.0)
    assert detector.assess(latency_ms=500.0).events[0].severity == "critical"   # > +60%


def test_latency_baseline_can_be_passed_per_call_and_is_required_otherwise():
    assert DriftDetector().assess(latency_ms=300.0, baseline_latency_ms=100.0).verdict == "drift"
    report = DriftDetector().assess(latency_ms=300.0)
    assert report.verdict == "insufficient_data" and "no baseline latency" in report.notes[0]


def test_unimplemented_drift_types_are_gone():
    assert {t.value for t in DriftType} == {
        "judge_score", "refusal_rate", "format_compliance", "latency"
    }


def test_check_is_assess_events():
    detector = detector_with_baseline([0.9] * 40)
    events = detector.check(new_scores=[0.1] * 40)
    assert [e.drift_type for e in events] == [DriftType.JUDGE_SCORE]


# ─── persistence ─────────────────────────────────────────────────────────────


def test_baseline_and_ema_state_round_trip(tmp_path):
    detector = detector_with_baseline([0.7, 0.8] * 20, refusal_rate=0.05, latency_ms=200.0)
    detector.assess(refusal_rate=0.07)
    path = tmp_path / "state" / "drift.json"

    detector.save(path)
    loaded = DriftDetector.load(path)

    assert loaded.baseline_size == 40
    assert loaded.baseline_refusal_rate == 0.05
    assert loaded.baseline_latency_ms == 200.0
    assert loaded._ema_refusal == pytest.approx(0.07)
    assert loaded.assess(new_scores=[0.1] * 40).verdict == "drift"    # usable immediately
    assert [p.name for p in path.parent.iterdir()] == ["drift.json"]  # no temp file left behind


def test_loading_a_missing_baseline_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DriftDetector.load(tmp_path / "nope.json")


def test_a_new_baseline_resets_the_ema_state():
    detector = detector_with_baseline([0.7] * 40, refusal_rate=0.05)
    detector.assess(refusal_rate=0.5)
    detector.update_baseline([0.7] * 40, refusal_rate=0.5)
    assert detector._ema_refusal is None


def test_baseline_metrics_can_be_set_without_touching_the_scores():
    detector = detector_with_baseline([0.7] * 40)
    detector.set_baseline_metrics(refusal_rate=0.1, latency_ms=150.0)
    assert detector.baseline_size == 40
    assert (detector.baseline_refusal_rate, detector.baseline_latency_ms) == (0.1, 150.0)


def test_saved_state_is_plain_json(tmp_path):
    detector = detector_with_baseline([0.5] * 3)
    detector.save(tmp_path / "d.json")
    data = json.loads((tmp_path / "d.json").read_text())
    assert data["baseline_scores"] == [0.5, 0.5, 0.5]
