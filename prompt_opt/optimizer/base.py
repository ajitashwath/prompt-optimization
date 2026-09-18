"""
OptimizerStrategy ABC — common interface for MIPROv2 and GEPA.

All optimizers take:
  - initial_prompt: current champion PromptCandidate
  - train/dev conversations
  - eval harness
  - budget controller

And return:
  - ranked list of (PromptCandidate, EvalSummary), best first

Contract shared by all strategies:
  - Every LLM call made during `optimize` is metered against the budget
    (real token usage / cost, checked before each call), see `prompt_opt.llm`.
  - The returned candidates were all evaluated on the FULL dev split under
    identical conditions, and the initial prompt is always among them, so the
    caller can tell whether anything actually beat it. On equal rank the
    initial prompt is listed first.
  - Candidates that fail the guardrails rank below those that pass.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from prompt_opt import llm
from prompt_opt.eval.guardrails import RegressionCase
from prompt_opt.eval.harness import EvalHarness, EvaluationError, PromptCandidate
from prompt_opt.eval.metrics import EvalSummary, compute_eval_summary
from prompt_opt.ingestion.schema import ConversationRecord
from prompt_opt.optimizer.budget import BudgetController, BudgetExceededError

logger = logging.getLogger(__name__)


class OptimizationError(RuntimeError):
    """The search could not produce any usable evaluation (e.g. every trial failed)."""


def default_minibatch_size(n_dev: int, cap: int = 16) -> int:
    """Dev examples per search step: enough to be a usable signal, capped for cost."""
    return max(1, min(n_dev, cap))


def response_texts(summary: EvalSummary) -> dict[str, str]:
    """example_id -> generated response, used as the guardrail baseline."""
    return {ex.example_id: ex.response_text for ex in summary.per_example if ex.response_text}


def rank_key(summary: EvalSummary) -> tuple[bool, float]:
    """Guardrail-passing candidates first, then by mean judge score."""
    return (bool(summary.guardrail_passed), float(summary.mean_score))


def empty_summary(harness: EvalHarness, prompt_hash: str, split: str = "dev") -> EvalSummary:
    """Placeholder for 'could not be evaluated' (e.g. budget spent before the first eval)."""
    return compute_eval_summary(
        prompt_hash=prompt_hash,
        split=split,
        judge_version=getattr(harness, "judge_version", ""),
        per_example=[],
    )


class OptimizerStrategy(ABC):
    """Abstract base class for prompt optimization strategies."""

    def optimize(
        self,
        initial_prompt: PromptCandidate,
        train_conversations: list[ConversationRecord],
        dev_conversations: list[ConversationRecord],
        harness: EvalHarness,
        budget: BudgetController,
    ) -> list[tuple[PromptCandidate, EvalSummary]]:
        """
        Run the optimization loop with all LLM calls metered against *budget*.

        Args:
            initial_prompt:        Current champion system prompt.
            train_conversations:   Training split (demo bootstrapping / reflection).
            dev_conversations:     Dev split for scoring and selection.
            harness:               Eval harness for scoring candidates.
            budget:                Budget controller (real usage; raises on cap).

        Returns:
            List of (PromptCandidate, EvalSummary) sorted best first; every entry has
            a full-dev EvalSummary and the initial prompt is one of them (unless the
            budget ran out before it could be evaluated, in which case it is returned
            alone with an empty summary).

        Raises:
            EvaluationError:   the initial prompt could not be evaluated reliably.
            OptimizationError: every search trial failed.
        """
        with llm.track_budget(budget):
            return self._optimize(
                initial_prompt, train_conversations, dev_conversations, harness, budget
            )

    @abstractmethod
    def _optimize(
        self,
        initial_prompt: PromptCandidate,
        train_conversations: list[ConversationRecord],
        dev_conversations: list[ConversationRecord],
        harness: EvalHarness,
        budget: BudgetController,
    ) -> list[tuple[PromptCandidate, EvalSummary]]:
        """Strategy-specific search; runs inside the budget-tracking context."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this strategy."""
        ...

    # ── Shared final stage ────────────────────────────────────────────────────

    def _final_stage(
        self,
        initial_prompt: PromptCandidate,
        finalists: list[PromptCandidate],
        dev_conversations: list[ConversationRecord],
        harness: EvalHarness,
        budget: BudgetController,
        regression_cases: list[RegressionCase] | None,
    ) -> list[tuple[PromptCandidate, EvalSummary]]:
        """
        Evaluate the initial prompt and every finalist on the FULL dev split.

        Search-time scores come from a small minibatch and are noisy; this is the
        apples-to-apples comparison. The initial prompt is evaluated first and its
        responses become the guardrail baseline (refusal / length / token drift) for
        the others; the regression suite (if any) is replayed live per candidate.
        """
        ordered = [initial_prompt] + [
            c for c in finalists if c.content_hash != initial_prompt.content_hash
        ]
        results: list[tuple[PromptCandidate, EvalSummary]] = []
        baseline_texts: dict[str, str] | None = None

        for cand in ordered:
            is_initial = cand is initial_prompt
            try:
                budget.check()
                regression = (
                    harness.run_regression(cand, regression_cases) if regression_cases else None
                )
                summary = harness.evaluate(
                    cand,
                    dev_conversations,
                    split="dev",
                    regression_data=regression,
                    guardrail_baseline_responses=baseline_texts,
                )
            except BudgetExceededError as exc:
                logger.warning("Budget exhausted during final evaluation: %s", exc.reason)
                break
            except EvaluationError as exc:
                if is_initial:
                    raise  # nothing to compare against
                logger.warning("Dropping finalist %s: %s", cand.content_hash, exc)
                continue
            results.append((cand, summary))
            if is_initial:
                baseline_texts = response_texts(summary)

        if not results:
            return [(initial_prompt, empty_summary(harness, initial_prompt.content_hash))]

        results.sort(key=lambda r: rank_key(r[1]), reverse=True)  # stable: initial wins ties
        return results
