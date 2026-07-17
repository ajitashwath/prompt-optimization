"""
MIPROv2 — Multi-prompt Instruction Proposal Optimizer (Bayesian variant).

Algorithm:
  Phase 1 — Bootstrap demos
    Run current prompt over train split; keep high-scoring examples
    as candidate few-shot demonstrations.

  Phase 2 — Instruction proposal
    Ask a "proposer" LLM to generate N diverse instruction rewrites,
    guided by task summary + worst-performing categories.

  Phase 3 — Bayesian search (TPE via Optuna)
    Joint search over (instruction_variant_idx × demo_subset_idx).
    Each trial:
      a. Score on a random minibatch of dev examples (cheap).
      b. Track (candidate, score) in the Optuna study.
    After all trials, full-dev evaluation for top-K survivors.

  Returns ranked list of (PromptCandidate, EvalSummary).
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from prompt_opt.config import settings
from prompt_opt.eval.harness import EvalHarness, PromptCandidate
from prompt_opt.eval.metrics import EvalSummary
from prompt_opt.ingestion.schema import ConversationRecord
from prompt_opt.optimizer.base import OptimizerStrategy
from prompt_opt.optimizer.budget import BudgetExceededError, BudgetController

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
        import litellm  # type: ignore[import]

        resp = litellm.completion(
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
        top_k:             After trials, do full-dev eval for top-K candidates.
        minibatch_size:    Number of dev examples per trial (cheap pass).
        task_summary:      Human description of the task (injected into proposer prompt).
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
    ):
        self.n_proposals = n_proposals
        self.n_demo_subsets = n_demo_subsets
        self.max_demos = max_demos
        self.n_trials = n_trials
        self.top_k = top_k
        self.minibatch_size = minibatch_size
        self.task_summary = task_summary

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
    ) -> list[list[dict]]:
        """
        Run initial prompt on train split; keep high-scoring examples as demos.
        Returns a list of demo subsets (each subset is a list of {user, assistant} dicts).
        """
        logger.info("Bootstrapping demos from %d train examples …", len(train_convs))
        budget.check()

        sample = random.sample(train_convs, min(30, len(train_convs)))
        eval_summary = harness.evaluate(initial_prompt, sample, split="train")
        budget.record_call(n_calls=len(sample))

        # Collect high-scoring examples as candidate demos
        good_examples: list[dict] = []
        for ex in eval_summary.per_example:
            if ex.challenger_score >= _BOOTSTRAP_SCORE_THRESHOLD:
                # Find the conv to extract user/assistant turns
                for conv in sample:
                    if conv.conversation_id == ex.example_id:
                        user_turns = [m.content for m in conv.user_turns if m.content]
                        asst_turns = [m.content for m in conv.assistant_turns if m.content]
                        if user_turns and asst_turns:
                            good_examples.append({
                                "user": user_turns[-1],
                                "assistant": asst_turns[-1],
                            })
                        break

        logger.info("Found %d high-quality demo candidates", len(good_examples))

        # Build demo subsets
        demo_subsets: list[list[dict]] = [
            []  # always include the no-demo option
        ]
        if good_examples:
            rng = random.Random(42)
            for _ in range(self.n_demo_subsets - 1):
                k = min(self.max_demos, len(good_examples))
                subset = rng.sample(good_examples, k)
                demo_subsets.append(subset)

        return demo_subsets

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
        budget.record_call(n_calls=1, input_tokens=len(user) // 4, output_tokens=500)

        if not variants:
            logger.warning("No instruction variants returned; using current prompt")
            return [current_prompt]

        logger.info("Proposer generated %d instruction variants", len(variants))
        return [current_prompt] + variants  # always include original

    # ── Phase 3: Bayesian search ──────────────────────────────────────────────

    def optimize(
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

        # Initial baseline eval
        logger.info("Running initial baseline eval on dev split …")
        try:
            baseline_eval = harness.evaluate(
                initial_prompt, dev_conversations, split="dev", sample_size=self.minibatch_size
            )
            budget.record_call(n_calls=self.minibatch_size)
        except BudgetExceededError:
            return [(initial_prompt, harness.evaluate(initial_prompt, dev_conversations[:5], split="dev"))]

        # Phase 1: Bootstrap demos
        try:
            demo_subsets = self._bootstrap_demos(
                initial_prompt, train_conversations, harness, budget
            )
        except BudgetExceededError:
            demo_subsets = [[]]

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

        # Track all evaluated candidates
        all_candidates: list[tuple[PromptCandidate, float]] = []

        # ── Optuna TPE study ─────────────────────────────────────────────────
        def objective(trial: "optuna.Trial") -> float:
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
                    candidate, dev_conversations, split="dev", sample_size=self.minibatch_size
                )
                budget.record_call(n_calls=self.minibatch_size)
                score = summary.mean_score

                if not summary.guardrail_passed:
                    score *= 0.5  # penalise guardrail violations

                all_candidates.append((candidate, score))
                logger.info("Trial %d: score=%.4f (instr=%d, demo=%d)", trial.number, score, instr_idx, demo_idx)
                return score

            except BudgetExceededError as e:
                trial.set_user_attr("budget_exceeded", str(e))
                raise optuna.TrialPruned()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        try:
            study.optimize(objective, n_trials=self.n_trials, catch=(Exception,))
        except Exception as exc:
            logger.warning("Optuna study terminated early: %s", exc)

        # Top-K full evaluation
        logger.info("Running full dev eval for top-%d candidates …", self.top_k)
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = all_candidates[: self.top_k]

        results: list[tuple[PromptCandidate, EvalSummary]] = []
        for cand, _ in top_candidates:
            try:
                budget.check()
                full_eval = harness.evaluate(cand, dev_conversations, split="dev")
                budget.record_call(n_calls=len(dev_conversations))
                results.append((cand, full_eval))
            except BudgetExceededError:
                logger.warning("Budget exhausted during top-K full eval; stopping")
                break

        if not results:
            # Fallback: return initial prompt with baseline eval
            results = [(initial_prompt, baseline_eval)]

        results.sort(key=lambda x: x[1].mean_score, reverse=True)
        logger.info(
            "MIPROv2 complete: best score=%.4f (budget: %s)",
            results[0][1].mean_score,
            budget.summary(),
        )
        return results
