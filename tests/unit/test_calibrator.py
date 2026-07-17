import pytest
from unittest.mock import MagicMock, patch
from prompt_opt.judge.calibrator import JudgeCalibrator, CalibrationReport
from prompt_opt.labeling.gold_store import GoldLabel, GoldStore
from prompt_opt.judge.judge import LLMJudge, PairwiseJudgeResult, JudgeResult
from prompt_opt.labeling.rubric import DEFAULT_RUBRIC


@patch("prompt_opt.judge.judge.LLMJudge")
@patch("prompt_opt.labeling.gold_store.GoldStore")
def test_calibrator_agreement_metrics(mock_store, mock_judge):
    # Mock labeled items
    item1 = GoldLabel(
        id="item1",
        conversation_id="c1",
        response_a="A",
        response_b="B",
        preferred="A",
        rationale="reason A",
        labeler_id="human",
        rubric_version=DEFAULT_RUBRIC.version,
    )
    item2 = GoldLabel(
        id="item2",
        conversation_id="c2",
        response_a="A",
        response_b="B",
        preferred="B",
        rationale="reason B",
        labeler_id="human",
        rubric_version=DEFAULT_RUBRIC.version,
    )
    mock_store.get_labeled.return_value = [item1, item2]

    # Mock judge responses
    jr1 = PairwiseJudgeResult(
        result_ab=MagicMock(aggregate_score=0.9, dimension_scores=[]),
        result_ba=MagicMock(aggregate_score=0.9, dimension_scores=[]),
        preferred="A",
        aggregate_score_a=0.9,
        aggregate_score_b=0.2,
        rationale="judge rationale",
    )
    jr2 = PairwiseJudgeResult(
        result_ab=MagicMock(aggregate_score=0.2, dimension_scores=[]),
        result_ba=MagicMock(aggregate_score=0.2, dimension_scores=[]),
        preferred="B",
        aggregate_score_a=0.2,
        aggregate_score_b=0.8,
        rationale="judge rationale",
    )
    mock_judge.score_pairwise.side_effect = [jr1, jr2]

    calibrator = JudgeCalibrator(mock_judge, mock_store)
    # We calibrate with at least 2 items (normally requires >= 5, but we can override min check or pass dummy list)
    # Let's bypass minimum item check by providing a list of 5 copies of items to guarantee calculation
    mock_store.get_labeled.return_value = [item1, item1, item1, item2, item2, item2]
    mock_judge.score_pairwise.side_effect = [jr1, jr1, jr1, jr2, jr2, jr2]

    report = calibrator.calibrate(rubric_version=DEFAULT_RUBRIC.version)

    assert report.n_items == 6
    assert report.agreement_rate == 1.0  # Perfect agreement since mock labels and judge choices match perfectly
    assert report.kappa == 1.0
    assert report.passed is True
