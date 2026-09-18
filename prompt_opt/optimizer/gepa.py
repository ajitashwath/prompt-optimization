"""
GEPA — Genetic/reflective Evolution with Pareto selection.

Algorithm:
  1. Initialize population K from initial prompt (paraphrase + restructure variants).
  2. Select: evaluate each candidate on ONE fixed dev minibatch (same examples for
     every candidate, so the Pareto comparison is fair), with guardrails checked
     against the initial prompt's responses.
  3. Reflective mutation (uses the TRAIN split, so dev stays a clean selection set):
       - Evaluate the candidate on a train minibatch and collect the low-scoring
         examples with the user query, the response and the judge's rationale.
       - Ask proposer LLM: "given these failures, how would you improve this prompt?"
       → produces a mutated candidate.
  4. Optional crossover: combine strongest sub-instructions from two Pareto-optimal candidates.
  5. Pareto selection: maintain the non-dominated front (unique candidates) across:
       (a) overall judge score
       (b) worst-category score
       (c) guardrail pass (0 or 1)
  6. Repeat until budget exhausted or N generations with no change of the front.
  7. Final evaluation (shared, see OptimizerStrategy._final_stage): the initial prompt and
     the best front members are evaluated on the FULL dev split; the initial prompt is
     always in the result so callers can tell whether anything beat it.

Returns ranked list of (PromptCandidate, EvalSummary).
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass

from prompt_opt import llm
from prompt_opt.config import settings
from prompt_opt.eval.guardrails import RegressionCase
from prompt_opt.eval.harness import EvalHarness, EvaluationError, PromptCandidate
from prompt_opt.eval.metrics import EvalSummary, PerExampleResult
from prompt_opt.ingestion.schema import ConversationRecord
from prompt_opt.optimizer.base import OptimizerStrategy, empty_summary, response_texts
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
        resp = llm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content or ""
    except BudgetExceededError:
        raise
    except Exception as exc:
        logger.warning("Proposer call failed: %s", exc)
        return ""


def _call_proposer_json_array(system: str, user: str, model: str | None = None) -> list[str]:
    """Call proposer LLM; parse JSON array response."""
    model = model or settings.proposer_model
    try:
        resp = llm.completion(
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
    except BudgetExceededError:
        raise
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
    reflection: EvalSummary | None = None  # train-minibatch eval used for reflective mutation


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


def _snippet(text: str | None, limit: int) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


class GEPA(OptimizerStrategy):
    """
    Genetic-Pareto reflective evolution optimizer.

    Args:
        population_size:     Max new candidates evaluated per generation.
        max_generations:     Maximum evolution generations.
        stagnation_limit:    Stop if the Pareto front is unchanged for this many generations.
        minibatch_size:      Examples per selection (dev) / reflection (train) minibatch.
        n_init_variants:     Initial population size from prompt paraphrasing.
        crossover_prob:      Probability of crossover between two Pareto-front individuals.
        final_top_k:         Front members to evaluate on the full dev split.
        regression_cases:    Regression suite replayed live for each finalist (optional).
        seed:                Seed for minibatch sampling and crossover.
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
        final_top_k: int = 3,
        regression_cases: list[RegressionCase] | None = None,
        seed: int = 42,
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.stagnation_limit = stagnation_limit
        self.minibatch_size = minibatch_size
        self.n_init_variants = n_init_variants
        self.crossover_prob = crossover_prob
        self.final_top_k = final_top_k
        self.regression_cases = regression_cases
        self.seed = seed

    @property
    def name(self) -> str:
        return "gepa"

    def _initialize_population(
        self,
        initial_prompt: PromptCandidate,
        budget: BudgetController,
    ) -> list[Individual]:
        """Generate initial population by paraphrasing the seed prompt (initial is index 0)."""
        budget.check()
        system = _INIT_VARIANTS_SYSTEM.format(k=self.n_init_variants)
        user = f"System prompt:\n{initial_prompt.system_prompt}"
        variants = _call_proposer_json_array(system, user)

        population = [Individual(candidate=initial_prompt, generation=0)]
        seen = {initial_prompt.content_hash}
        for v in variants[: self.n_init_variants]:
            if not v:
                continue
            cand = PromptCandidate(
                system_prompt=v,
                few_shot_demos=initial_prompt.few_shot_demos,
                metadata={"origin": "gepa_init"},
            )
            if cand.content_hash not in seen:
                seen.add(cand.content_hash)
                population.append(Individual(candidate=cand, generation=0))

        logger.info("GEPA: initialized population of %d individuals", len(population))
        return population

    def _evaluate_individual(
        self,
        ind: Individual,
        minibatch: list[ConversationRecord],
        harness: EvalHarness,
        baseline_texts: dict[str, str] | None,
    ) -> None:
        """Score one individual on the shared dev minibatch (in-place)."""
        summary = harness.evaluate(
            ind.candidate,
            minibatch,
            split="dev",
            guardrail_baseline_responses=baseline_texts,
        )
        _set_objectives(ind, summary)
        logger.debug(
            "%s: overall=%.3f, worst_cat=%.3f, guardrail=%s",
            ind.candidate.content_hash,
            ind.obj_overall,
            ind.obj_worst_cat,
            "pass" if ind.obj_guardrail > 0.5 else "fail",
        )

    def _mutate(
        self,
        ind: Individual,
        reflection_batch: list[ConversationRecord],
        convs_by_id: dict[str, ConversationRecord],
        harness: EvalHarness,
        budget: BudgetController,
    ) -> Individual | None:
        """
        Reflective mutation: show the proposer where the prompt fails (query, response and
        the judge's reasoning) on a TRAIN minibatch and ask for an improved prompt.
        """
        if ind.eval_summary is None:
            return None

        summary = ind.reflection
        if summary is None:
            if reflection_batch:
                budget.check()
                try:
                    summary = harness.evaluate(ind.candidate, reflection_batch, split="train")
                except EvaluationError as exc:
                    logger.warning(
                        "Reflection eval failed for %s: %s", ind.candidate.content_hash, exc
                    )
                    return None
            else:  # no train data: fall back to the selection evaluation
                summary = ind.eval_summary
            ind.reflection = summary

        failing = summary.failing_examples(threshold=0.4)[:5]
        if not failing:
            # No clear failures — reflect on the weakest examples anyway
            failing = sorted(summary.per_example, key=lambda e: e.challenger_score)[:3]

        def describe(ex: PerExampleResult) -> str:
            conv = convs_by_id.get(ex.example_id)
            query = ""
            if conv is not None:
                user_turns = [m.content for m in conv.user_turns if m.content]
                query = user_turns[-1] if user_turns else ""
            return (
                f"[{ex.category}] score={ex.challenger_score:.2f}\n"
                f"  user: {_snippet(query, 300)}\n"
                f"  response: {_snippet(ex.response_text, 300)}"
            )

        failing_text = "\n".join(describe(ex) for ex in failing) or "None"
        rationale_text = "\n".join(
            f"- {_snippet(ex.judge_rationale, 150)}" for ex in failing if ex.judge_rationale
        ) or "No specific rationale available."

        budget.check()
        user = _MUTATE_USER.format(
            current_prompt=ind.candidate.system_prompt,
            failing_examples=failing_text,
            judge_rationale=rationale_text,
        )
        new_prompt_text = _call_proposer_text(_MUTATE_SYSTEM, user)

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

        if not combined:
            return None

        return Individual(
            candidate=PromptCandidate(
                system_prompt=combined,
                few_shot_demos=a.candidate.few_shot_demos,
                metadata={
                    "origin": "gepa_crossover",
                    "parent_a": a.candidate.content_hash,
                    "parent_b": b.candidate.content_hash,
                },
            ),
            generation=max(a.generation, b.generation) + 1,
        )

    def _optimize(
        self,
        initial_prompt: PromptCandidate,
        train_conversations: list[ConversationRecord],
        dev_conversations: list[ConversationRecord],
        harness: EvalHarness,
        budget: BudgetController,
    ) -> list[tuple[PromptCandidate, EvalSummary]]:
        rng = random.Random(self.seed)

        # Fixed minibatches: dev for selection (shared by every candidate), train for
        # reflection.
        dev_batch = rng.sample(dev_conversations, min(self.minibatch_size, len(dev_conversations)))
        train_batch = (
            rng.sample(train_conversations, min(self.minibatch_size, len(train_conversations)))
            if train_conversations
            else []
        )
        convs_by_id = {c.conversation_id: c for c in dev_batch + train_batch}

        # Initialize
        try:
            population = self._initialize_population(initial_prompt, budget)
        except BudgetExceededError:
            logger.warning("Budget exhausted during GEPA init")
            return [(initial_prompt, empty_summary(harness, initial_prompt.content_hash))]

        # The initial prompt is evaluated first: its responses are the guardrail baseline.
        initial_ind = population[0]
        try:
            self._evaluate_individual(initial_ind, dev_batch, harness, None)
        except BudgetExceededError:
            return [(initial_prompt, empty_summary(harness, initial_prompt.content_hash))]
        baseline_texts = response_texts(initial_ind.eval_summary)  # type: ignore[arg-type]

        evaluated: dict[str, Individual] = {initial_prompt.content_hash: initial_ind}
        attempted: set[str] = {initial_prompt.content_hash}
        stagnation_count = 0
        prev_front: frozenset[str] = frozenset()

        for gen in range(self.max_generations):
            logger.info(
                "GEPA Generation %d/%d (%d new candidates)",
                gen + 1,
                self.max_generations,
                len(population),
            )

            # Evaluate the new individuals (each unique candidate is evaluated once)
            try:
                for ind in population:
                    h = ind.candidate.content_hash
                    if h in attempted:
                        continue
                    attempted.add(h)
                    try:
                        self._evaluate_individual(ind, dev_batch, harness, baseline_texts)
                    except EvaluationError as exc:
                        logger.warning("Dropping candidate %s: %s", h, exc)
                        continue
                    evaluated[h] = ind
            except BudgetExceededError:
                logger.info("Budget exhausted during evaluation; stopping")
                break

            # Pareto front over the unique evaluated candidates
            front = _pareto_front(list(evaluated.values()))
            front_hashes = frozenset(i.candidate.content_hash for i in front)

            if front_hashes == prev_front:
                stagnation_count += 1
                logger.info(
                    "GEPA: Pareto front unchanged (%d/%d)", stagnation_count, self.stagnation_limit
                )
            else:
                stagnation_count = 0
                prev_front = front_hashes

            if stagnation_count >= self.stagnation_limit:
                logger.info("GEPA: early stop (stagnation)")
                break
            if gen == self.max_generations - 1:
                break  # mutants of the last generation would never be evaluated

            # Next generation: reflective mutants of front members (+ optional crossover)
            new_individuals: list[Individual] = []
            new_hashes: set[str] = set()

            def _add(candidate_ind: Individual | None) -> None:
                if candidate_ind is None:
                    return
                h = candidate_ind.candidate.content_hash
                if h in attempted or h in new_hashes:
                    return
                new_hashes.add(h)
                new_individuals.append(candidate_ind)

            try:
                for ind in front[: self.population_size]:
                    _add(self._mutate(ind, train_batch, convs_by_id, harness, budget))

                if len(front) >= 2 and rng.random() < self.crossover_prob:
                    a, b = rng.sample(front[:4] if len(front) >= 4 else front, 2)
                    _add(self._crossover(a, b, budget))
            except BudgetExceededError:
                logger.info("Budget exhausted during mutation; stopping")
                break

            population = new_individuals[: self.population_size]
            if not population:
                logger.info("GEPA: no new candidates produced; stopping")
                break

        # Finalists: best unique front members (guardrail-passing first)
        front = _pareto_front(list(evaluated.values()))
        front.sort(key=lambda i: (i.obj_guardrail, i.obj_overall), reverse=True)
        finalists = [
            i.candidate
            for i in front
            if i.candidate.content_hash != initial_prompt.content_hash
        ][: self.final_top_k]

        results = self._final_stage(
            initial_prompt,
            finalists,
            dev_conversations,
            harness,
            budget,
            self.regression_cases,
        )

        logger.info(
            "GEPA complete: %d evaluated candidates, best full-dev score=%.4f (budget: %s)",
            len(evaluated),
            results[0][1].mean_score,
            budget.summary(),
        )
        return results
