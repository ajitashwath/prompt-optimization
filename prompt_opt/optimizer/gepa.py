"""
GEPA — Genetic/reflective Evolution with Pareto selection.

Algorithm:
  1. Initialize population K from initial prompt (paraphrase + restructure variants).
  2. Evaluate each candidate on dev minibatch.
  3. Reflective mutation:
       - For each candidate, collect low-scoring examples + judge rationale.
       - Ask proposer LLM: "given these failures, how would you improve this prompt?"
       → produces a mutated candidate.
  4. Optional crossover: combine strongest sub-instructions from two Pareto-optimal candidates.
  5. Pareto selection: maintain non-dominated front across:
       (a) overall judge score
       (b) worst-category score
       (c) guardrail pass rate (0 or 1 here; use continuous proxy)
  6. Repeat until budget exhausted or N generations with no Pareto improvement.

Returns ranked list of (PromptCandidate, EvalSummary) from the Pareto front.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass

from prompt_opt.config import settings
from prompt_opt.eval.harness import EvalHarness, PromptCandidate
from prompt_opt.eval.metrics import EvalSummary
from prompt_opt.ingestion.schema import ConversationRecord
from prompt_opt.optimizer.base import OptimizerStrategy
from prompt_opt.optimizer.budget import BudgetController, BudgetExceededError

logger = logging.getLogger(__name__)


# ─── Proposer templates ───────────────────────────────────────────────────────

_INIT_VARIANTS_SYSTEM = """\
You are an expert prompt engineer. Given a system prompt, produce {k} diverse variants.
Variants should rephrase, restructure, or emphasize different aspects, while preserving intent.
Return a JSON array: ["variant1", "variant2", ...]
"""

_MUTATE_SYSTEM = """\
You are an expert prompt engineer improving an AI assistant's system prompt.
You will be given:
  1. Current system prompt
  2. Examples where the current prompt led to poor responses
  3. Judge's analysis of WHY each response was poor

Your task: produce ONE improved system prompt that specifically addresses the observed failures.
Return ONLY the improved prompt text (no JSON, no commentary).
"""

_MUTATE_USER = """\
## Current System Prompt
{current_prompt}

## Failing Examples (score < threshold)
{failing_examples}

## Judge Analysis
{judge_rationale}

Write an improved system prompt that fixes these issues:
"""

_CROSSOVER_SYSTEM = """\
You are an expert prompt engineer. Given two high-performing system prompt variants,
combine their strongest elements into one superior prompt.
Return ONLY the combined prompt text.
"""

_CROSSOVER_USER = """\
## Prompt A (strength: {strength_a})
{prompt_a}

## Prompt B (strength: {strength_b})
{prompt_b}

Combine the best elements of both into a single improved system prompt:
"""


def _call_proposer_text(system: str, user: str, model: str | None = None) -> str:
    """Call proposer LLM; return plain text response."""
    model = model or settings.proposer_model
    try:
        import litellm  # type: ignore[import]

        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("Proposer call failed: %s", exc)
        return ""


def _call_proposer_json_array(system: str, user: str, model: str | None = None) -> list[str]:
    """Call proposer LLM; parse JSON array response."""
    model = model or settings.proposer_model
    try:
        import litellm  # type: ignore[import]

        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "[]"
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
        for key in parsed:
            if isinstance(parsed[key], list):
                return [str(v) for v in parsed[key]]
    except Exception as exc:
        logger.warning("Proposer JSON call failed: %s", exc)
    return []


# ─── Pareto utilities ─────────────────────────────────────────────────────────


@dataclass
class Individual:
    candidate: PromptCandidate
    eval_summary: EvalSummary | None = None
    # Objective vector (all maximised)
    obj_overall: float = 0.0      # mean judge score
    obj_worst_cat: float = 0.0    # worst category mean score
    obj_guardrail: float = 0.0    # 1.0 if passed, 0.0 if failed
    generation: int = 0


def _dominates(a: Individual, b: Individual) -> bool:
    """True if a Pareto-dominates b (a is >= on all objectives and > on at least one)."""
    geq = (
        a.obj_overall >= b.obj_overall
        and a.obj_worst_cat >= b.obj_worst_cat
        and a.obj_guardrail >= b.obj_guardrail
    )
    gt = (
        a.obj_overall > b.obj_overall
        or a.obj_worst_cat > b.obj_worst_cat
        or a.obj_guardrail > b.obj_guardrail
    )
    return geq and gt


def _pareto_front(population: list[Individual]) -> list[Individual]:
    """Return the non-dominated front from the population."""
    front: list[Individual] = []
    for ind in population:
        dominated = False
        for other in population:
            if other is not ind and _dominates(other, ind):
                dominated = True
                break
        if not dominated:
            front.append(ind)
    return front


def _set_objectives(ind: Individual, summary: EvalSummary) -> None:
    ind.eval_summary = summary
    ind.obj_overall = summary.mean_score
    worst_cat_score = min(
        (c.mean_score for c in summary.category_breakdown), default=0.0
    )
    ind.obj_worst_cat = worst_cat_score
    ind.obj_guardrail = 1.0 if summary.guardrail_passed else 0.0


class GEPA(OptimizerStrategy):
    """
    Genetic-Pareto reflective evolution optimizer.

    Args:
        population_size:     Initial number of candidates.
        max_generations:     Maximum evolution generations.
        stagnation_limit:    Stop if Pareto front unchanged for this many generations.
        minibatch_size:      Dev examples per generation eval.
        n_init_variants:     Initial population size from prompt paraphrasing.
        crossover_prob:      Probability of crossover between two Pareto-front individuals.
    """

    def __init__(
        self,
        *,
        population_size: int = 6,
        max_generations: int = 10,
        stagnation_limit: int = 3,
        minibatch_size: int = 15,
        n_init_variants: int = 5,
        crossover_prob: float = 0.3,
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.stagnation_limit = stagnation_limit
        self.minibatch_size = minibatch_size
        self.n_init_variants = n_init_variants
        self.crossover_prob = crossover_prob

    @property
    def name(self) -> str:
        return "gepa"

    def _initialize_population(
        self,
        initial_prompt: PromptCandidate,
        budget: BudgetController,
    ) -> list[Individual]:
        """Generate initial population by paraphrasing the seed prompt."""
        budget.check()
        system = _INIT_VARIANTS_SYSTEM.format(k=self.n_init_variants)
        user = f"System prompt:\n{initial_prompt.system_prompt}"
        variants = _call_proposer_json_array(system, user)
        budget.record_call(n_calls=1, input_tokens=len(user) // 4, output_tokens=300)

        population = [Individual(candidate=initial_prompt, generation=0)]
        for v in variants[: self.n_init_variants]:
            if v and v != initial_prompt.system_prompt:
                cand = PromptCandidate(
                    system_prompt=v,
                    metadata={"origin": "gepa_init"},
                )
                population.append(Individual(candidate=cand, generation=0))

        logger.info("GEPA: initialized population of %d individuals", len(population))
        return population

    def _evaluate_population(
        self,
        population: list[Individual],
        dev_conversations: list[ConversationRecord],
        harness: EvalHarness,
        budget: BudgetController,
        generation: int,
    ) -> None:
        """Evaluate all individuals in the population (in-place)."""
        for ind in population:
            if ind.eval_summary is not None:
                continue  # already evaluated
            try:
                budget.check()
                summary = harness.evaluate(
                    ind.candidate,
                    dev_conversations,
                    split="dev",
                    sample_size=self.minibatch_size,
                )
                budget.record_call(n_calls=self.minibatch_size)
                _set_objectives(ind, summary)
                logger.debug(
                    "Gen %d — %s: overall=%.3f, worst_cat=%.3f, guardrail=%s",
                    generation,
                    ind.candidate.content_hash,
                    ind.obj_overall,
                    ind.obj_worst_cat,
                    "pass" if ind.obj_guardrail > 0.5 else "fail",
                )
            except BudgetExceededError:
                raise

    def _mutate(
        self,
        ind: Individual,
        budget: BudgetController,
    ) -> Individual | None:
        """Reflective mutation: use judge rationale from failing examples to improve prompt."""
        if ind.eval_summary is None:
            return None

        failing = ind.eval_summary.failing_examples(threshold=0.4)[:5]
        if not failing:
            # No clear failures — try a creative rewrite
            failing = ind.eval_summary.per_example[:3]

        failing_text = "\n".join(
            f"[{ex.category}] score={ex.challenger_score:.2f}"
            for ex in failing
        )
        rationale_text = "\n".join(
            f"- {ex.judge_rationale[:150]}"
            for ex in failing
            if ex.judge_rationale
        ) or "No specific rationale available."

        budget.check()
        user = _MUTATE_USER.format(
            current_prompt=ind.candidate.system_prompt,
            failing_examples=failing_text,
            judge_rationale=rationale_text,
        )
        new_prompt_text = _call_proposer_text(_MUTATE_SYSTEM, user)
        budget.record_call(n_calls=1, input_tokens=len(user) // 4, output_tokens=400)

        if not new_prompt_text or new_prompt_text == ind.candidate.system_prompt:
            return None

        return Individual(
            candidate=PromptCandidate(
                system_prompt=new_prompt_text,
                few_shot_demos=ind.candidate.few_shot_demos,
                metadata={"origin": "gepa_mutation", "parent": ind.candidate.content_hash},
            ),
            generation=ind.generation + 1,
        )

    def _crossover(
        self,
        a: Individual,
        b: Individual,
        budget: BudgetController,
    ) -> Individual | None:
        """Combine sub-instructions from two Pareto-optimal individuals."""
        budget.check()
        user = _CROSSOVER_USER.format(
            strength_a=f"overall={a.obj_overall:.2f}",
            prompt_a=a.candidate.system_prompt,
            strength_b=f"overall={b.obj_overall:.2f}",
            prompt_b=b.candidate.system_prompt,
        )
        combined = _call_proposer_text(_CROSSOVER_SYSTEM, user)
        budget.record_call(n_calls=1, input_tokens=len(user) // 4, output_tokens=400)

        if not combined:
            return None

        return Individual(
            candidate=PromptCandidate(
                system_prompt=combined,
                metadata={
                    "origin": "gepa_crossover",
                    "parent_a": a.candidate.content_hash,
                    "parent_b": b.candidate.content_hash,
                },
            )
        )

    def optimize(
        self,
        initial_prompt: PromptCandidate,
        train_conversations: list[ConversationRecord],
        dev_conversations: list[ConversationRecord],
        harness: EvalHarness,
        budget: BudgetController,
    ) -> list[tuple[PromptCandidate, EvalSummary]]:
        rng = random.Random(42)

        # Initialize
        try:
            population = self._initialize_population(initial_prompt, budget)
        except BudgetExceededError:
            logger.warning("Budget exhausted during GEPA init")
            fallback_eval = harness.evaluate(
                initial_prompt, dev_conversations[:5], split="dev"
            )
            return [(initial_prompt, fallback_eval)]

        all_individuals: list[Individual] = []
        stagnation_count = 0
        prev_front_size = 0

        for gen in range(self.max_generations):
            logger.info(
                "GEPA Generation %d/%d (pop=%d)",
                gen + 1,
                self.max_generations,
                len(population),
            )

            # Evaluate unevaluated individuals
            try:
                self._evaluate_population(population, dev_conversations, harness, budget, gen)
            except BudgetExceededError:
                logger.info("Budget exhausted during evaluation; stopping")
                break

            all_individuals.extend(population)

            # Pareto front
            evaluated = [i for i in all_individuals if i.eval_summary is not None]
            front = _pareto_front(evaluated)

            if len(front) == prev_front_size:
                stagnation_count += 1
                logger.info(
                    "GEPA: Pareto front stagnated (%d/%d)",
                    stagnation_count,
                    self.stagnation_limit,
                )
            else:
                stagnation_count = 0
                prev_front_size = len(front)

            if stagnation_count >= self.stagnation_limit:
                logger.info("GEPA: early stop (stagnation)")
                break

            # Generate next generation via mutation
            new_population: list[Individual] = []

            # Keep Pareto front (elitism)
            new_population.extend(front)

            # Mutate Pareto-front individuals
            for ind in front[:self.population_size]:
                try:
                    mutant = self._mutate(ind, budget)
                    if mutant:
                        new_population.append(mutant)
                except BudgetExceededError:
                    break

            # Crossover (with probability)
            if len(front) >= 2 and rng.random() < self.crossover_prob:
                a, b = rng.sample(front[:4] if len(front) >= 4 else front, 2)
                try:
                    child = self._crossover(a, b, budget)
                    if child:
                        new_population.append(child)
                except BudgetExceededError:
                    pass

            population = new_population[: self.population_size * 2]

        # Final Pareto front from all evaluated individuals
        evaluated = [i for i in all_individuals if i.eval_summary is not None]
        final_front = _pareto_front(evaluated)

        # Sort by overall score for output
        final_front.sort(key=lambda x: x.obj_overall, reverse=True)

        results = [
            (ind.candidate, ind.eval_summary)
            for ind in final_front
            if ind.eval_summary is not None
        ]

        if not results:
            fallback_eval = harness.evaluate(
                initial_prompt, dev_conversations[:5], split="dev"
            )
            results = [(initial_prompt, fallback_eval)]

        logger.info(
            "GEPA complete: %d on Pareto front, best overall=%.4f (budget: %s)",
            len(results),
            results[0][1].mean_score if results else 0.0,
            budget.summary(),
        )
        return results
