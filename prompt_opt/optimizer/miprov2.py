"""
MIPROv2 — Multi-prompt Instruction Proposal Optimizer (Bayesian variant).

Algorithm:
  Phase 1 — Bootstrap demos
    Run current prompt over a train sample; keep the high-scoring *generated*
    responses (the ones the judge actually scored) as candidate few-shot demos.

  Phase 2 — Instruction proposal
    Ask a "proposer" LLM to generate N diverse instruction rewrites,
    guided by task summary + worst-performing categories.

  Phase 3 — Bayesian search (TPE via Optuna)
    Joint search over (instruction_variant_idx × demo_subset_idx).
    Each trial scores the candidate on ONE fixed dev minibatch (the same
    examples for every trial, so scores are comparable and cache-friendly),
    with guardrails checked against the initial prompt's responses.

  Phase 4 — Final evaluation (shared, see OptimizerStrategy._final_stage)
    The initial prompt and the top-K distinct candidates are evaluated on the
    FULL dev split; the initial prompt is always in the result so callers can
    tell whether anything beat it.

  Returns ranked list of (PromptCandidate, EvalSummary).
"""

from __future__ import annotations

import json
import logging
import random

from prompt_opt import llm
from prompt_opt.config import settings
from prompt_opt.eval.guardrails import RegressionCase
from prompt_opt.eval.harness import EvalHarness, EvaluationError, PromptCandidate
from prompt_opt.eval.metrics import EvalSummary
from prompt_opt.ingestion.schema import ConversationRecord
from prompt_opt.optimizer.base import (
    OptimizationError,
    OptimizerStrategy,
    empty_summary,
    response_texts,
)
from prompt_opt.optimizer.budget import BudgetController, BudgetExceededError

logger = logging.getLogger(__name__)


# ─── Proposer prompt templates ────────────────────────────────────────────────

_BOOTSTRAP_SCORE_THRESHOLD = 0.7  # keep demos with score >= this

_PROPOSE_INSTRUCTIONS_SYSTEM = """\
You are an expert prompt engineer. Your task is to improve an AI assistant's system prompt.
You will be given:
  1. The current system prompt
  2. A summary of the task the assistant performs
  3. Categories where the current prompt under-performs (with example failure patterns)

Generate {n_proposals} diverse, improved system prompt variants. Each variant should:
  - Address the identified failure patterns
  - Be clear, specific, and actionable
  - Maintain the overall task scope
  - Vary in approach (different framing, structure, emphasis)

Return a JSON array of strings: ["variant 1 text", "variant 2 text", ...]
"""

_PROPOSE_INSTRUCTIONS_USER = """\
## Current System Prompt
{current_prompt}

## Task Summary
{task_summary}

## Under-performing Categories
{failure_categories}

## Example Failure Patterns
{failure_examples}

Generate {n_proposals} improved system prompt variants as a JSON array.
"""


def _call_proposer(
    system: str,
    user: str,
    model: str | None = None,
) -> list[str]:
    """Ask the proposer LLM for instruction variants. Returns list of prompt strings."""
    model = model or settings.proposer_model
    try:
        resp = llm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.8,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "[]"
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
        # Sometimes the model wraps it: {"variants": [...]}
        for key in parsed:
            if isinstance(parsed[key], list):
                return [str(v) for v in parsed[key]]
    except BudgetExceededError:
        raise
    except Exception as exc:
        logger.warning("Proposer call failed: %s", exc)
    return []


class MIPROv2(OptimizerStrategy):
    """
    Bayesian instruction + demo search using Optuna's TPE sampler.

    Args:
        n_proposals:       Number of instruction variants to generate.
        n_demo_subsets:    How many different demo subsets to try.
        max_demos:         Max few-shot demos per candidate.
        n_trials:          Total Optuna trials (instruction × demo combos to try).
        top_k:             After trials, full-dev-evaluate the top-K distinct candidates.
        minibatch_size:    Dev examples per trial (one fixed sample shared by all trials).
        task_summary:      Human description of the task (injected into proposer prompt).
        regression_cases:  Regression suite replayed live for each finalist (optional).
        seed:              Seed for minibatch / demo sampling and the TPE sampler.
    """

    def __init__(
        self,
        *,
        n_proposals: int = 8,
        n_demo_subsets: int = 4,
        max_demos: int = 3,
        n_trials: int = 20,
        top_k: int = 3,
        minibatch_size: int = 10,
        task_summary: str = "General AI assistant task",
        regression_cases: list[RegressionCase] | None = None,
        seed: int = 42,
    ):
        self.n_proposals = n_proposals
        self.n_demo_subsets = n_demo_subsets
        self.max_demos = max_demos
        self.n_trials = n_trials
        self.top_k = top_k
        self.minibatch_size = minibatch_size
        self.task_summary = task_summary
        self.regression_cases = regression_cases
        self.seed = seed

    @property
    def name(self) -> str:
        return "miprov2"

    # ── Phase 1: Bootstrap demos ──────────────────────────────────────────────

    def _bootstrap_demos(
        self,
        initial_prompt: PromptCandidate,
        train_convs: list[ConversationRecord],
        harness: EvalHarness,
        budget: BudgetController,
        rng: random.Random,
    ) -> list[list[dict]]:
        """
        Run initial prompt on a train sample; keep the responses the judge scored highly.

        Returns a list of demo subsets (each a list of {user, assistant} dicts). The first
        subset is always the initial prompt's own demos, so combination (0, 0) of the
        search space is exactly the initial prompt.
        """
        subsets: list[list[dict]] = [list(initial_prompt.few_shot_demos or [])]
        if not train_convs:
            return subsets

        logger.info("Bootstrapping demos from %d train examples …", len(train_convs))
        budget.check()

        sample = rng.sample(train_convs, min(30, len(train_convs)))
        by_id = {c.conversation_id: c for c in sample}
        try:
            eval_summary = harness.evaluate(initial_prompt, sample, split="train")
        except EvaluationError as exc:
            logger.warning("Demo bootstrapping skipped: %s", exc)
            return subsets

        # The demo's assistant turn is the GENERATED response that earned the score,
        # not the historical log reply (which the judge never saw).
        good_examples: list[dict] = []
        for ex in eval_summary.per_example:
            if ex.challenger_score < _BOOTSTRAP_SCORE_THRESHOLD or not ex.response_text:
                continue
            conv = by_id.get(ex.example_id)
            user_turns = [m.content for m in conv.user_turns if m.content] if conv else []
            if user_turns:
                good_examples.append({"user": user_turns[-1], "assistant": ex.response_text})

        logger.info("Found %d high-quality demo candidates", len(good_examples))

        if good_examples:
            demo_rng = random.Random(self.seed)
            for _ in range(self.n_demo_subsets - 1):
                k = min(self.max_demos, len(good_examples))
                subsets.append(demo_rng.sample(good_examples, k))

        return subsets

    # ── Phase 2: Instruction proposal ────────────────────────────────────────

    def _propose_instructions(
        self,
        current_prompt: str,
        eval_summary: EvalSummary | None,
        budget: BudgetController,
    ) -> list[str]:
        """Generate diverse instruction variants via proposer LLM."""
        budget.check()

        failure_categories = "None identified (no prior eval available)"
        failure_examples = "None"

        if eval_summary:
            worst = eval_summary.worst_categories(3)
            failure_categories = "\n".join(
                f"- {c.category} (win_rate={c.win_rate:.2f})"
                if c.win_rate is not None
                else f"- {c.category} (mean_score={c.mean_score:.2f})"
                for c in worst
            )
            failing_ex = eval_summary.failing_examples(threshold=0.4)[:3]
            failure_examples = "\n".join(
                f"- [{ex.category}] score={ex.challenger_score:.2f}: {ex.judge_rationale[:100]}"
                for ex in failing_ex
            ) or "None"

        system = _PROPOSE_INSTRUCTIONS_SYSTEM.format(n_proposals=self.n_proposals)
        user = _PROPOSE_INSTRUCTIONS_USER.format(
            current_prompt=current_prompt,
            task_summary=self.task_summary,
            failure_categories=failure_categories,
            failure_examples=failure_examples,
            n_proposals=self.n_proposals,
        )

        variants = _call_proposer(system, user)

        if not variants:
            logger.warning("No instruction variants returned; using current prompt")
            return [current_prompt]

        logger.info("Proposer generated %d instruction variants", len(variants))
        return [current_prompt] + variants  # always include original

    # ── Phase 3: Bayesian search ──────────────────────────────────────────────

    def _optimize(
        self,
        initial_prompt: PromptCandidate,
        train_conversations: list[ConversationRecord],
        dev_conversations: list[ConversationRecord],
        harness: EvalHarness,
        budget: BudgetController,
    ) -> list[tuple[PromptCandidate, EvalSummary]]:
        try:
            import optuna  # type: ignore[import]

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.error("optuna not installed — cannot run MIPROv2")
            raise

        rng = random.Random(self.seed)
        # One fixed minibatch for every trial: candidate scores are comparable.
        minibatch = rng.sample(
            dev_conversations, min(self.minibatch_size, len(dev_conversations))
        )

        # Baseline on the minibatch: the guardrail baseline for every trial and the
        # failure summary the proposer sees.
        logger.info("Running initial baseline eval on the dev minibatch …")
        try:
            baseline_eval = harness.evaluate(initial_prompt, minibatch, split="dev")
        except BudgetExceededError:
            return [(initial_prompt, empty_summary(harness, initial_prompt.content_hash))]
        baseline_texts = response_texts(baseline_eval)

        # Phase 1: Bootstrap demos
        try:
            demo_subsets = self._bootstrap_demos(
                initial_prompt, train_conversations, harness, budget, rng
            )
        except BudgetExceededError:
            demo_subsets = [list(initial_prompt.few_shot_demos or [])]

        # Phase 2: Instruction proposals
        try:
            instruction_variants = self._propose_instructions(
                initial_prompt.system_prompt, baseline_eval, budget
            )
        except BudgetExceededError:
            instruction_variants = [initial_prompt.system_prompt]

        logger.info(
            "Search space: %d instructions × %d demo subsets = %d combos",
            len(instruction_variants),
            len(demo_subsets),
            len(instruction_variants) * len(demo_subsets),
        )

        # Distinct candidates evaluated so far: content_hash -> (candidate, search score)
        evaluated: dict[str, tuple[PromptCandidate, float]] = {}
        trial_errors: list[Exception] = []

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

        def objective(trial: optuna.Trial) -> float:
            instr_idx = trial.suggest_int("instr_idx", 0, len(instruction_variants) - 1)
            demo_idx = trial.suggest_int("demo_idx", 0, len(demo_subsets) - 1)

            candidate = PromptCandidate(
                system_prompt=instruction_variants[instr_idx],
                few_shot_demos=demo_subsets[demo_idx] or None,
                metadata={"instr_idx": instr_idx, "demo_idx": demo_idx, "optimizer": "miprov2"},
            )

            try:
                budget.check()
                summary = harness.evaluate(
                    candidate,
                    minibatch,
                    split="dev",
                    guardrail_baseline_responses=baseline_texts,
                )
                score = summary.mean_score
                if not summary.guardrail_passed:
                    score *= 0.5  # penalise guardrail violations

                evaluated[candidate.content_hash] = (candidate, score)
                logger.info(
                    "Trial %d: score=%.4f (instr=%d, demo=%d)",
                    trial.number,
                    score,
                    instr_idx,
                    demo_idx,
                )
                return score

            except BudgetExceededError as e:
                trial.set_user_attr("budget_exceeded", str(e))
                study.stop()
                raise optuna.TrialPruned() from e
            except Exception as exc:
                trial_errors.append(exc)
                raise

        study.optimize(objective, n_trials=self.n_trials, catch=(Exception,))

        if not evaluated and trial_errors:
            raise OptimizationError(
                f"All {len(trial_errors)} search trials failed; last error: {trial_errors[-1]}"
            ) from trial_errors[-1]

        # Full-dev evaluation of the initial prompt + the top-K distinct candidates
        logger.info("Full dev eval of the initial prompt + top-%d candidates …", self.top_k)
        ranked = sorted(evaluated.values(), key=lambda item: item[1], reverse=True)
        finalists = [
            cand for cand, _ in ranked if cand.content_hash != initial_prompt.content_hash
        ][: self.top_k]
        results = self._final_stage(
            initial_prompt,
            finalists,
            dev_conversations,
            harness,
            budget,
            self.regression_cases,
        )

        logger.info(
            "MIPROv2 complete: best score=%.4f (budget: %s)",
            results[0][1].mean_score,
            budget.summary(),
        )
        return results
