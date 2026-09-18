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


# ─── ties count half; robustness ─────────────────────────────────────────────


def test_ties_count_as_half_a_win():
    # 30 wins / 10 losses / 60 ties: a clear improvement, mostly tied.
    outcomes = ["win"] * 30 + ["loss"] * 10 + ["tie"] * 60
    scores = [0.8 + i * 0.001 for i in range(30)] + [0.6] * 10 + [0.7] * 60
    champ = [0.6] * 30 + [0.8 + i * 0.001 for i in range(10)] + [0.7] * 60

    res = run_statistical_tests(outcomes, scores, champ)

    assert res.win_rate == 0.6  # (30 + 0.5 * 60) / 100, not 30 / 100
    assert res.decisive_win_rate == 0.75
    assert res.ci_lower > 0.5  # the improvement is real
    assert res.ci_excludes_null


def test_all_ties_is_exactly_no_difference():
    res = run_statistical_tests(["tie"] * 40, [0.7] * 40, [0.7] * 40)

    assert res.win_rate == 0.5
    assert res.decisive_win_rate is None
    assert not res.ci_excludes_null
    assert res.wilcoxon_pvalue == 1.0
    assert res.n_nonzero_diffs == 0


def test_a_handful_of_wins_among_ties_is_not_a_win_rate_of_one():
    outcomes = ["win"] * 5 + ["tie"] * 95
    win_rate, lower, _ = paired_bootstrap_ci(outcomes)
    assert win_rate == 0.525
    assert lower <= 0.5 + 0.03  # nowhere near excluding the null


def test_empty_input_does_not_crash():
    assert paired_bootstrap_ci([]) == (0.0, 0.0, 1.0)

    res = run_statistical_tests([], [], [])
    assert res.n_examples == 0
    assert res.win_rate == 0.0
    assert res.ci_excludes_null is False
    assert res.statistically_significant is False


def test_bootstrap_is_reproducible_and_the_ci_brackets_the_estimate():
    outcomes = ["win"] * 55 + ["loss"] * 30 + ["tie"] * 15
    first = paired_bootstrap_ci(outcomes, seed=7)
    assert first == paired_bootstrap_ci(outcomes, seed=7)

    win_rate, lower, upper = first
    assert win_rate == (55 + 7.5) / 100
    assert lower < win_rate < upper
    # roughly the normal-approximation interval: se = sqrt(var / n)
    assert 0.09 < (upper - lower) < 0.22


def test_unanimous_wins_give_a_degenerate_ci():
    win_rate, lower, upper = paired_bootstrap_ci(["win"] * 20)
    assert (win_rate, lower, upper) == (1.0, 1.0, 1.0)


def test_summary_reports_counts_and_nonzero_differences():
    res = run_statistical_tests(["win"] * 6 + ["tie"] * 2, [0.9] * 6 + [0.5] * 2, [0.5] * 8)
    summary = res.summary()
    assert (summary["n_wins"], summary["n_ties"], summary["n_losses"]) == (6, 2, 0)
    assert summary["n_nonzero_diffs"] == 6
