"""ChampionGate criteria: no vacuous passes, honest tie handling, real spot-check content."""

import json
from unittest.mock import MagicMock

import pytest

from prompt_opt.eval.guardrails import MODEL_CALL_FAILED, RegressionCase
from prompt_opt.eval.harness import EvalHarness, PromptCandidate
from prompt_opt.eval.metrics import PerExampleResult, compute_eval_summary
from prompt_opt.gate.champion_gate import ChampionGate, GateCriteria
from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole

CHALLENGER = PromptCandidate(system_prompt="Challenger")
CHAMPION = PromptCandidate(system_prompt="Champion")


def _conv(i):
    return ConversationRecord(
        conversation_id=f"test_{i}",
        messages=[Message(role=MessageRole.USER, content=f"question {i}")],
        category="general",
    )


def _examples(outcomes, chal, champ, chal_text=None):
    return [
        PerExampleResult(
            example_id=f"test_{i}",
            category="general",
            challenger_score=cs,
            champion_score=ms,
            outcome=oc,
            judge_rationale=f"rationale {i}",
            response_text=chal_text or f"chal response {i}",
            champion_response_text=f"champ response {i}" if oc is not None else None,
        )
        for i, (oc, cs, ms) in enumerate(zip(outcomes, chal, champ, strict=True))
    ]


def _wins(n):
    return ["win"] * n, [0.9 + i * 1e-4 for i in range(n)], [0.5] * n


def _crit(**overrides):
    base = {
        "min_test_examples": 10,
        "require_regression_suite": False,
        "require_human_spot_check": False,
    }
    return GateCriteria(**{**base, **overrides})


def _run(per_example, *, criteria=None, details=None, regression=None, cases=None, convs=None):
    summary = compute_eval_summary(CHALLENGER.content_hash, "test", "v0", per_example)
    summary.guardrail_details = details or {}
    harness = MagicMock(spec=EvalHarness)
    harness.evaluate.return_value = summary
    gate = ChampionGate(harness, criteria=criteria or _crit(), regression_cases=cases)
    n = len(convs) if convs is not None else len(per_example)
    convs = convs if convs is not None else [_conv(i) for i in range(n)]
    champion_responses = {c.conversation_id: "champ" for c in convs}
    return gate.evaluate(
        CHALLENGER, CHAMPION, convs,
        champion_responses=champion_responses, regression_responses=regression,
    )


def _failed(result):
    return {c["criterion"] for c in result.criteria_checks if not c["passed"]}


# ─── statistics ──────────────────────────────────────────────────────────────


def test_a_mostly_tied_but_real_improvement_passes():
    outcomes = ["win"] * 30 + ["loss"] * 10 + ["tie"] * 60
    chal = [0.8 + i * 1e-3 for i in range(30)] + [0.6] * 10 + [0.7] * 60
    champ = [0.6] * 30 + [0.8 + i * 1e-3 for i in range(10)] + [0.7] * 60

    result = _run(_examples(outcomes, chal, champ), criteria=_crit(min_test_examples=30))

    assert result.decision == "PASS", result.failure_reasons
    assert result.stat_test.win_rate == 0.6
    assert result.stat_test.ci_lower > 0.5


def test_all_ties_is_not_an_improvement():
    result = _run(_examples(["tie"] * 40, [0.7] * 40, [0.7] * 40))

    assert result.decision == "FAIL"
    assert {"win_rate", "ci_excludes_null", "score_shift_significance"} <= _failed(result)


def test_too_few_examples_fail_however_good_they_look():
    result = _run(_examples(*_wins(10)), criteria=_crit(min_test_examples=30))

    assert result.decision == "FAIL"
    assert _failed(result) == {"min_test_examples"}
    assert "Only 10 paired test examples (minimum 30)" in result.failure_reasons[0]


def test_the_wilcoxon_test_is_enforced():
    # Wins often, but only by a hair, while losing big: the win rate looks fine
    # (0.65, CI above 0.5) but the score differences do not support an improvement.
    outcomes = ["win"] * 130 + ["loss"] * 70
    chal = [0.53] * 130 + [0.2] * 70
    champ = [0.5] * 130 + [0.7] * 70
    per_example = _examples(outcomes, chal, champ)

    result = _run(per_example)
    assert result.decision == "FAIL"
    assert _failed(result) == {"score_shift_significance"}

    relaxed = _run(per_example, criteria=_crit(require_significant_score_shift=False))
    assert relaxed.decision == "PASS"


def test_empty_test_split_fails_cleanly():
    result = _run([], convs=[])

    assert result.decision == "FAIL"
    assert result.n_paired == 0
    assert result.stat_test.n_examples == 0
    assert "Only 0 paired" in result.failure_reasons[0]


def test_unpaired_examples_do_not_enter_the_statistics():
    paired = _examples(*_wins(12))
    unpaired = _examples([None] * 5, [0.1] * 5, [None] * 5)
    for i, ex in enumerate(unpaired):
        ex.example_id = f"extra_{i}"

    result = _run(paired + unpaired, convs=[_conv(i) for i in range(17)])

    assert result.n_paired == 12
    assert result.stat_test.n_examples == 12
    assert result.decision == "PASS"


# ─── guardrails ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "details, failing",
    [
        ({"refusal_rate_delta": 0.25}, "refusal_rate_shift"),
        ({"refusal_rate_delta": -0.25}, "refusal_rate_shift"),  # refusing far less also counts
        ({"token_count_delta_pct": 0.45}, "token_growth"),
        ({"length_ks_pvalue": 0.001}, "output_length_shift"),
        ({"format_compliance_rate": 0.5, "format_compliance_pass": False}, "format_compliance"),
    ],
)
def test_each_guardrail_fails_on_its_own_threshold(details, failing):
    result = _run(_examples(*_wins(20)), details=details)

    assert result.decision == "FAIL"
    assert _failed(result) == {failing}


def test_guardrails_within_their_thresholds_pass():
    details = {
        "refusal_rate_delta": 0.09,     # <= 0.10
        "token_count_delta_pct": 0.30,  # <= 0.30
        "length_ks_pvalue": 0.02,       # > 0.01
    }
    assert _run(_examples(*_wins(20)), details=details).decision == "PASS"


def test_thresholds_come_from_the_criteria_not_from_the_harness():
    details = {"refusal_rate_delta": 0.08}
    per_example = _examples(*_wins(20))

    assert _run(per_example, details=details).decision == "PASS"
    strict = _run(per_example, details=details, criteria=_crit(max_refusal_rate_shift=0.05))
    assert _failed(strict) == {"refusal_rate_shift"}

    long_output = {"token_count_delta_pct": 0.5}
    lenient = _run(per_example, details=long_output, criteria=_crit(max_token_growth_pct=0.6))
    assert lenient.decision == "PASS"

    ks = {"length_ks_pvalue": 0.0001}
    assert _run(per_example, details=ks, criteria=_crit(length_shift_alpha=None)).decision == "PASS"


def test_unmeasured_guardrails_do_not_fail_by_themselves():
    result = _run(_examples(*_wins(20)), details={})
    by_name = {c["criterion"]: c for c in result.criteria_checks}
    assert by_name["refusal_rate_shift"]["value"] == "n/a"
    assert result.decision == "PASS"


# ─── regression suite ────────────────────────────────────────────────────────


REFUSE = RegressionCase(
    conversation_id="reg_1", expected_behavior="refuse", expected_keywords=["can't"]
)


def test_a_regression_suite_that_never_ran_fails_the_gate():
    per_example = _examples(*_wins(20))

    no_suite = _run(per_example, criteria=_crit(require_regression_suite=True))
    assert no_suite.decision == "FAIL"
    assert "no regression suite was provided" in no_suite.failure_reasons[0]

    not_run = _run(per_example, criteria=_crit(require_regression_suite=True), cases=[REFUSE])
    assert "exists but was not run" in not_run.failure_reasons[0]


def test_the_regression_requirement_can_be_waived_explicitly():
    result = _run(_examples(*_wins(20)), criteria=_crit(require_regression_suite=False))
    by_name = {c["criterion"]: c for c in result.criteria_checks}
    assert by_name["regression_suite_pass_rate"]["value"] == "not run (allowed)"
    assert result.decision == "PASS"


def test_regression_results_are_judged_by_the_gate_itself():
    per_example = _examples(*_wins(20))
    criteria = _crit(require_regression_suite=True)

    ok = _run(per_example, criteria=criteria, regression=[(REFUSE, "I can't help with that")])
    assert ok.decision == "PASS"

    bad = _run(per_example, criteria=criteria, regression=[(REFUSE, "Sure, here you go!")])
    assert bad.decision == "FAIL"
    assert "reg_1" in bad.failure_reasons[0]


def test_a_regression_call_that_failed_counts_as_a_failure():
    result = _run(
        _examples(*_wins(20)),
        criteria=_crit(require_regression_suite=True),
        regression=[(REFUSE, MODEL_CALL_FAILED)],
    )
    assert result.decision == "FAIL"
    assert "Model call failed" in result.failure_reasons[0]


# ─── decision / audit record ─────────────────────────────────────────────────


def test_passing_statistics_wait_for_the_human_when_review_is_required():
    result = _run(_examples(*_wins(20)), criteria=_crit(require_human_spot_check=True))
    assert result.decision == "PENDING_HUMAN"
    assert result.spot_check_items


def test_the_result_records_the_thresholds_it_was_decided_under():
    result = _run(_examples(*_wins(20)), criteria=_crit(min_win_rate=0.6, max_token_growth_pct=0.2))

    assert result.criteria["min_win_rate"] == 0.6
    assert result.criteria["max_token_growth_pct"] == 0.2
    assert result.to_dict()["criteria"] == result.criteria


# ─── spot check ──────────────────────────────────────────────────────────────


def _mixed(n_win=20, n_loss=10, n_tie=10):
    outcomes = ["win"] * n_win + ["loss"] * n_loss + ["tie"] * n_tie
    chal = [0.9] * n_win + [0.3] * n_loss + [0.6] * n_tie
    champ = [0.5] * n_win + [0.8] * n_loss + [0.6] * n_tie
    return _examples(outcomes, chal, champ)


def _spot(result):
    return result.spot_check_items


def test_spot_check_items_contain_what_a_reviewer_needs():
    result = _run(_mixed(), criteria=_crit(require_significant_score_shift=False))

    assert len(_spot(result)) == 5
    for item in _spot(result):
        i = item["example_id"].split("_")[1]
        assert item["user_query"] == f"question {i}"
        assert item["challenger_response"] == f"chal response {i}"
        assert item["champion_response"] == f"champ response {i}"
        assert item["judge_rationale"] == f"rationale {i}"
        assert item["outcome"] in {"win", "loss", "tie"}


def test_spot_check_covers_losses_wins_and_ties():
    result = _run(_mixed(), criteria=_crit(require_significant_score_shift=False))

    assert {item["outcome"] for item in _spot(result)} == {"loss", "win", "tie"}
    assert sum(item["outcome"] == "loss" for item in _spot(result)) == 2  # L, W, T, L, W


def test_spot_check_is_a_reproducible_sample_not_the_first_n_examples():
    first = _spot(_run(_mixed()))
    again = _spot(_run(_mixed()))

    assert [i["example_id"] for i in first] == [i["example_id"] for i in again]
    assert [i["example_id"] for i in first] != [f"test_{i}" for i in range(5)]


def test_spot_check_respects_the_sample_size_and_available_examples():
    assert len(_spot(_run(_mixed(), criteria=_crit(spot_check_sample_size=2)))) == 2
    assert len(_spot(_run(_examples(*_wins(12)), criteria=_crit(spot_check_sample_size=50)))) == 12


def test_spot_check_texts_are_capped():
    huge = "x" * 5000
    result = _run(_examples(*_wins(12), chal_text=huge))
    text = _spot(result)[0]["challenger_response"]
    assert len(text) == 2000 and text.endswith("…")


def test_result_serialises_with_spot_checks_and_human_review():
    result = _run(_mixed(30, 5, 5), criteria=_crit(require_human_spot_check=True))
    assert result.decision == "PENDING_HUMAN", result.failure_reasons

    result.record_human_review(reviewer="alice", notes="looks fine", approved=True)
    data = json.loads(json.dumps(result.to_dict()))

    assert data["decision"] == "PASS"
    assert data["human_reviewer"] == "alice"
    assert data["human_notes"] == "looks fine"
    assert len(data["spot_check_items"]) == 5
    assert data["n_paired"] == 40
