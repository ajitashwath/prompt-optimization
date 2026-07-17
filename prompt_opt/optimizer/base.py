"""
OptimizerStrategy ABC — common interface for MIPROv2 and GEPA.

All optimizers take:
  - initial_prompt: current champion PromptCandidate
  - train/dev conversations
  - eval harness
  - budget controller

And return:
  - ranked list of PromptCandidate (best first)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from prompt_opt.eval.harness import EvalHarness, PromptCandidate
from prompt_opt.eval.metrics import EvalSummary
from prompt_opt.ingestion.schema import ConversationRecord
from prompt_opt.optimizer.budget import BudgetController


class OptimizerStrategy(ABC):
    """Abstract base class for prompt optimization strategies."""

    @abstractmethod
    def optimize(
        self,
        initial_prompt: PromptCandidate,
        train_conversations: list[ConversationRecord],
        dev_conversations: list[ConversationRecord],
        harness: EvalHarness,
        budget: BudgetController,
    ) -> list[tuple[PromptCandidate, EvalSummary]]:
        """
        Run the optimization loop.

        Args:
            initial_prompt:        Current champion system prompt.
            train_conversations:   Training split for demo bootstrapping.
            dev_conversations:     Dev split for internal scoring.
            harness:               Eval harness for scoring candidates.
            budget:                Budget controller (raises on cap).

        Returns:
            List of (PromptCandidate, EvalSummary) sorted by dev score descending.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this strategy."""
        ...
