"""JudgeCalibrator: real bias diagnostics, NaN kappa, failure accounting, conversation context."""

import pytest

from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole
from prompt_opt.judge.calibrator import JudgeCalibrator
from prompt_opt.judge.judge import PairwiseJudgeResult
from prompt_opt.labeling.gold_store import GoldLabel
from prompt_opt.labeling.rubric import DEFAULT_RUBRIC

RV = DEFAULT_RUBRIC.version


def _item(i, preferred, len_a=50, len_b=50, conv_id=None):
    return GoldLabel(
        id=f"item{i}",
        conversation_id=conv_id or f"c{i}",
        response_a="a" * len_a,
        response_b="b" * len_b,
        preferred=preferred,
        rationale="r",
        labeler_id="h",
        rubric_version=RV,
    )


class FakeStore:
    def __init__(self, items):
        self.items = items

    def get_labeled(self, *, rubric_version=None):
        return self.items


class FakeJudge:
    """`fn(index, conv, response_a, response_b)` returns a PairwiseJudgeResult or raises."""

    def __init__(self, fn):
        self.fn = fn
        self.convs = []

    def score_pairwise(self, conv, a, b):
        self.convs.append(conv)
        return self.fn(len(self.convs) - 1, conv, a, b)


def _result(preferred, score_a=None, score_b=None, verdict_ab="__auto__", verdict_ba="__auto__"):
    """A consistent judge result; verdicts default to agreeing with `preferred`."""
    score_a = {"A": 0.9, "B": 0.2, "tie": 0.5}[preferred] if score_a is None else score_a
    score_b = {"A": 0.2, "B": 0.9, "tie": 0.5}[preferred] if score_b is None else score_b
    if verdict_ab == "__auto__":
        verdict_ab = preferred
    if verdict_ba == "__auto__":
        verdict_ba = preferred
    return PairwiseJudgeResult(
        result_ab=None,
        result_ba=None,
        preferred=preferred,
        aggregate_score_a=score_a,
        aggregate_score_b=score_b,
        rationale="",
        verdict_ab=verdict_ab,
        verdict_ba=verdict_ba,
    )


def _calibrate(items, fn, conversations=None, **kwargs):
    judge = FakeJudge(fn)
    calibrator = JudgeCalibrator(judge, FakeStore(items), **kwargs)
    return calibrator.calibrate(RV, conversations=conversations), judge


def _alternating(i, conv, a, b):
    """A judge that agrees with items labeled A, B, A, B, ..."""
    return _result("A" if i % 2 == 0 else "B")


def _mixed_items(n=12, **kwargs):
    return [_item(i, "A" if i % 2 == 0 else "B", **kwargs) for i in range(n)]


# ─── position bias ───────────────────────────────────────────────────────────


def test_position_bias_is_not_confused_with_label_imbalance():
    """All-A gold labels with a perfectly consistent judge: a label imbalance, not bias."""
    items = [_item(i, "A") for i in range(12)]

    report, _ = _calibrate(items, lambda i, c, a, b: _result("A"))

    assert report.bias.position_bias_rate == pytest.approx(0.5)
    assert not report.bias.has_significant_position_bias
    assert not any("position" in r.lower() for r in report.failure_reasons)


def test_a_judge_that_always_favours_the_first_listed_response_is_flagged():
    # Pass 1 picks A (first-listed); pass 2 picks the first-listed again = original B.
    biased = lambda i, c, a, b: _result("tie", verdict_ab="A", verdict_ba="B")  # noqa: E731

    report, _ = _calibrate(_mixed_items(12), biased)

    assert report.bias.position_bias_rate == pytest.approx(1.0)
    assert report.bias.order_inconsistency_rate == pytest.approx(1.0)
    assert report.bias.has_significant_position_bias
    assert not report.passed
    assert any("position bias" in r for r in report.failure_reasons)


def test_position_bias_needs_enough_decisive_verdicts():
    biased = lambda i, c, a, b: _result("tie", verdict_ab="A", verdict_ba="B")  # noqa: E731

    # Ties carry no position information: an all-tie judge has nothing to assess.
    all_ties = lambda i, c, a, b: _result("tie", verdict_ab="tie", verdict_ba="tie")  # noqa: E731
    report, _ = _calibrate(_mixed_items(12), all_ties)
    assert report.bias.n_position_verdicts == 0
    assert not report.bias.has_significant_position_bias

    # 5 items x 2 orderings = 10 decisive verdicts: enough to call it (the report needs >= 5
    # items anyway); the bias is detected as soon as the threshold is reached.
    report, _ = _calibrate(_mixed_items(5), biased)
    assert report.bias.n_position_verdicts == 10
    assert report.bias.has_significant_position_bias


# ─── length bias ─────────────────────────────────────────────────────────────


def _length_items(human_follows_length):
    items = []
    for i in range(12):
        diff = i * 10 - 55  # A minus B length, spread over both signs
        len_a, len_b = 100 + max(diff, 0), 100 + max(-diff, 0)
        if human_follows_length:
            preferred = "A" if diff > 0 else "B"
        else:
            preferred = "A" if i % 2 == 0 else "B"  # unrelated to length
        items.append(_item(i, preferred, len_a=len_a, len_b=len_b))
    return items


def _length_judge(i, conv, a, b):
    # The judge's score difference simply tracks the length difference.
    diff = len(a) - len(b)
    return _result("A" if diff > 0 else "B", score_a=0.5 + diff / 1000, score_b=0.5)


def test_judge_length_preference_beyond_the_humans_is_flagged():
    report, _ = _calibrate(_length_items(human_follows_length=False), _length_judge)

    assert report.bias.length_bias_rho > 0.9
    assert abs(report.bias.human_length_rho) < 0.3
    assert report.bias.has_significant_length_bias
    assert any("length bias" in r for r in report.failure_reasons)


def test_length_preference_shared_with_the_humans_is_not_bias():
    report, _ = _calibrate(_length_items(human_follows_length=True), _length_judge)

    assert report.bias.length_bias_rho > 0.9
    assert report.bias.human_length_rho > 0.7
    assert not report.bias.has_significant_length_bias


# ─── agreement ───────────────────────────────────────────────────────────────


def test_undefined_kappa_fails_calibration_instead_of_passing_silently():
    items = [_item(i, "A") for i in range(8)]  # no variation in the human labels

    report, _ = _calibrate(items, lambda i, c, a, b: _result("A"))

    assert report.kappa == 0.0
    assert report.passed is False
    assert any("undefined" in r for r in report.failure_reasons)


def test_thresholds_default_to_the_configured_settings(monkeypatch):
    from prompt_opt.config import settings

    monkeypatch.setattr(settings, "calibration_min_kappa", 0.95)
    judge = FakeJudge(lambda i, c, a, b: _result("A" if i % 4 else "B"))  # frequently wrong
    calibrator = JudgeCalibrator(judge, FakeStore(_mixed_items(12)))
    assert calibrator.min_kappa == 0.95


# ─── unusable items ──────────────────────────────────────────────────────────


def test_judge_failures_are_counted_and_can_fail_calibration():
    def flaky(i, conv, a, b):
        if i % 3 == 0:
            raise RuntimeError("judge down")
        return _result("A" if i % 2 == 0 else "B")

    report, _ = _calibrate(_mixed_items(12), flaky)

    assert report.n_judge_failures == 4
    assert report.n_items == 8
    assert not report.passed
    assert any("4/12 judge calls failed" in r for r in report.failure_reasons)


def test_too_few_items_explains_why():
    def always_fails(i, conv, a, b):
        raise RuntimeError("401 unauthorized")

    report, _ = _calibrate(_mixed_items(8), always_fails)

    assert report.passed is False
    assert report.n_judge_failures == 8
    assert "8 judge call failures" in report.failure_reasons[0]


def test_items_without_conversation_context_are_skipped_and_counted():
    items = _mixed_items(10)
    conversations = {
        f"c{i}": ConversationRecord(
            conversation_id=f"c{i}",
            messages=[Message(role=MessageRole.USER, content=f"real question {i}")],
        )
        for i in range(6)  # c6..c9 are missing
    }

    report, judge = _calibrate(items, _alternating, conversations)

    assert report.n_skipped_no_context == 4
    assert report.n_items == 6
    assert len(judge.convs) == 6  # the judge was never called for the missing ones


def test_judge_sees_the_real_conversation_not_a_stub():
    conv = ConversationRecord(
        conversation_id="c0",
        messages=[Message(role=MessageRole.USER, content="the actual user question")],
    )
    conversations = {
        f"c{i}": conv.model_copy(update={"conversation_id": f"c{i}"}) for i in range(8)
    }

    _, judge = _calibrate(_mixed_items(8), _alternating, conversations)

    assert all(c.messages[0].content == "the actual user question" for c in judge.convs)


def test_report_serialises_the_new_fields():
    report, _ = _calibrate(_mixed_items(12), _alternating)
    data = report.to_dict()
    assert data["n_judge_failures"] == 0
    assert data["bias"]["n_position_verdicts"] == 24
    assert data["bias"]["order_inconsistency_rate"] == 0.0
    assert report.passed
