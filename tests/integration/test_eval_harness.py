import pytest
from unittest.mock import MagicMock, patch
from prompt_opt.eval.harness import EvalHarness, PromptCandidate
from prompt_opt.eval.cache import EvalCache
from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole
from prompt_opt.judge.judge import LLMJudge, JudgeResult


@patch("litellm.completion")
@patch("prompt_opt.judge.judge.LLMJudge")
def test_harness_evaluation_flow(mock_judge, mock_litellm):
    # Mock Litellm completion response
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Mocked response text"))]
    mock_litellm.return_value = mock_resp

    # Mock Judge response
    mock_judge_result = JudgeResult(
        dimension_scores=[],
        aggregate_score=0.85,
        overall_verdict="score",
        rationale="mock rationale",
        raw_response="{}",
    )
    mock_judge.score_absolute.return_value = mock_judge_result

    # Mock Cache
    cache = MagicMock(spec=EvalCache)
    cache.get.return_value = None

    harness = EvalHarness(
        judge=mock_judge,
        judge_version="test_v1",
        cache=cache,
        optimizee_model="openai/gpt-4o-mini",
    )

    candidate = PromptCandidate(system_prompt="System instructions")
    convs = [
        ConversationRecord(
            conversation_id="conv_1",
            messages=[
                Message(role=MessageRole.USER, content="Hello"),
            ],
            category="general",
            split="dev",
        )
    ]

    summary = harness.evaluate(candidate, convs, split="dev")

    assert summary.n_examples == 1
    assert summary.mean_score == 0.85
    assert cache.put.called is True
