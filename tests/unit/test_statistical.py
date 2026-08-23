from prompt_opt.gate.statistical import paired_bootstrap_ci, run_statistical_tests


def test_bootstrap_ci_simple():
    outcomes = ["win", "win", "win", "win", "tie", "loss", "win", "win", "win", "tie"]
    win_rate, lower, upper = paired_bootstrap_ci(outcomes, n_resamples=1000, ci_level=0.95)

    assert 0.0 <= win_rate <= 1.0
    assert lower <= win_rate <= upper


def test_run_statistical_tests():
    outcomes = ["win"] * 8 + ["loss"] * 2
    challenger_scores = [0.9] * 8 + [0.3] * 2
    champion_scores = [0.4] * 8 + [0.8] * 2

    res = run_statistical_tests(
        outcomes,
        challenger_scores,
        champion_scores,
        n_resamples=1000,
    )

    assert res.n_examples == 10
    assert res.n_wins == 8
    assert res.win_rate == 0.8
    assert res.ci_lower <= 0.8 <= res.ci_upper
