import pytest
from unittest.mock import MagicMock, patch
from prompt_opt.optimizer.miprov2 import MIPROv2
from prompt_opt.eval.harness import EvalHarness, PromptCandidate
from prompt_opt.eval.metrics import EvalSummary
from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole
from prompt_opt.optimizer.budget import BudgetController


@patch("prompt_opt.optimizer.miprov2._call_proposer")
def test_miprov2_optimization_smoke(mock_proposer):
    # Proposer returns 3 instruction variants
    mock_proposer.return_value = ["variant 1", "variant 2", "variant 3"]

    # Mock harness
    harness = MagicMock(spec=EvalHarness)
    summary = MagicMock(spec=EvalSummary)
    summary.mean_score = 0.82
    summary.guardrail_passed = True
    summary.per_example = []
    harness.evaluate.return_value = summary

    optimizer = MIPROv2(n_trials=3, minibatch_size=2)
    budget = BudgetController(max_usd_spend=1.0)

    initial_prompt = PromptCandidate(system_prompt="Initial prompt")
    train_convs = [
        ConversationRecord(
            conversation_id="c1",
            messages=[Message(role=MessageRole.USER, content="Hello")],
        )
    ]
    dev_convs = [
        ConversationRecord(
            conversation_id="c2",
            messages=[Message(role=MessageRole.USER, content="Hello")],
        )
    ]

    results = optimizer.optimize(
        initial_prompt,
        train_convs,
        dev_convs,
        harness,
        budget,
    )

    assert len(results) > 0
    best_cand, best_summary = results[0]
    assert best_summary.mean_score == 0.82
