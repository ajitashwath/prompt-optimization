# Self-Improving Prompt Optimization Pipeline
A closed-loop system that automatically evolves an LLM's system prompt (and few-shot
examples) using production conversation logs, calibrated LLM-judges, and Bayesian /
Genetic optimizers.

## Table of Contents
1. [Architecture](#architecture)
2. [How the Loop Works](#how-the-loop-works)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Quick Start (Full Walkthrough)](#quick-start-full-walkthrough)
7. [CLI Reference](#cli-reference)
8. [Data Formats](#data-formats)
9. [Storage Layout](#storage-layout)
10. [Python API Usage](#python-api-usage)
11. [Running Without API Keys](#running-without-api-keys)
12. [Testing](#testing)
13. [Project Structure](#project-structure)
14. [Troubleshooting](#troubleshooting)

## Architecture
```
 ┌────────────────┐     ┌───────────────────┐     ┌────────────────────┐
 │ Conversation    │     │  Human Labeling    │     │  Judge Calibration  │
 │ Ingestion &     │────▶│  Studio (seed set) │────▶│  & Bias Correction  │
 │ Curation        │     └───────────────────┘     └──────────┬─────────┘
 └───────┬─────────┘                                          │
         │                                                     ▼
         │                                          ┌─────────────────────┐
         │                                          │  Calibrated Judge   │
         │                                          │  (prompt+rubric+    │
         │                                          │   model version)    │
         │                                          └──────────┬──────────┘
         ▼                                                     │
 ┌────────────────┐     ┌───────────────────┐                 │
 │ Eval Dataset    │────▶│  Eval Harness      │◀────────────────┘
 │ (train/dev/test)│     │  (scores any prompt│
 └────────────────┘     │   against dataset) │
                        └─────────┬─────────┘
                                  │ score + per-example feedback
                                  ▼
                        ┌───────────────────┐
                        │ Optimizer Engine   │
                        │ (MIPROv2 / GEPA)   │
                        │  proposes new      │
                        │  candidate prompts │
                        └─────────┬─────────┘
                                  │ candidate prompt(s)
                                  ▼
                        ┌───────────────────┐
                        │ Champion/Challenger│
                        │ Gate (stat test +  │
                        │ guardrails)         │
                        └─────────┬─────────┘
                                  │ pass
                                  ▼
                        ┌───────────────────┐
                        │ Prompt Registry &   │
                        │ Shadow/Canary        │
                        │ Deployment           │
                        └───────────────────┘
```

## How the Loop Works

| Phase | Module | What happens |
|-------|--------|--------------|
| 0. Ingestion | `prompt_opt/ingestion` | Raw production logs are PII-scrubbed (validated patterns + optional spaCy NER), deduplicated (exact text, then near-duplicates via embeddings, with an offline lexical fallback; copies of frozen test items are dropped), enriched with category/difficulty labels, and stratified into **train / dev / test** splits. The test split stays frozen for gate evaluation only: once written, re-ingesting (including the scheduled loop) never changes it and keeps its conversations out of train/dev; `ingest --refresh-test` rebuilds it deliberately. |
| 1. Labeling & Calibration | `prompt_opt/labeling`, `prompt_opt/judge` | Humans annotate A/B preferences in the web Studio. The LLM judge is then calibrated against these gold labels (Cohen's kappa >= 0.6, Spearman >= 0.7, plus length/position bias checks). Passing judges are frozen into versioned registry entries. |
| 2. Eval Harness | `prompt_opt/eval` | Replays conversations through any candidate prompt, scores responses with the judge, computes guardrail metrics (refusal shift, KS length test, token growth, regression suite), and caches every result in SQLite so repeated evaluations are free. |
| 3. Optimizers | `prompt_opt/optimizer` | **MIPROv2**: bootstraps few-shot demos, asks a proposer LLM for instruction variants, then runs Optuna TPE Bayesian search over instruction x demo combinations. **GEPA**: genetic search with Pareto fronts and reflective mutation driven by judge rationales. Both respect hard budget caps (calls / wall-time / USD). |
| 4. Champion Gate | `prompt_opt/gate` | The only component allowed to touch the frozen test split. Runs a paired bootstrap CI (10,000 resamples) on the win rate (ties count half) and a one-sided Wilcoxon signed-rank test on the per-example score differences, and enforces a minimum sample size, the win-rate threshold, each guardrail against its own threshold, and a 100% regression-suite pass rate. A check whose input is missing fails instead of passing vacuously. Produces `PASS / FAIL / PENDING_HUMAN` verdicts that are written once and never edited. |
| 5. Registry & Deployment | `prompt_opt/registry`, `prompt_opt/deployment` | Content-addressed, append-only prompt version store with an audit log of champion transitions. Rollback reverts to the previous promoted version. Shadow mode serves the champion while logging challenger responses; canary mode ramps traffic 1% -> 5% -> 25% -> 100% with soak periods and breach halts. A gate PASS does not change the champion; the challenger is promoted when the canary completes (or via `promote`). |
| 6. Feedback Loop | `prompt_opt/feedback` | Cron-driven scheduler that re-ingests fresh logs and triggers re-optimization, plus drift detection over production judge scores (KS test + smoothed KL, and refusal-rate / latency trends) that reports "no verdict" instead of "no drift" when it lacks data. |

## Requirements

- **Python 3.11+**
- At least one LLM provider API key (**OpenAI**, **Anthropic**, or **Google**) for
  optimization / evaluation stages. Ingestion, canary, monitor, drift-check, and the
  labeling studio work offline.
- ~2 GB disk for dependencies (spaCy, sentence-transformers, FAISS, torch are pulled in).

## Installation

```bash
# 1. Clone and enter the project
git clone https://github.com/ajitashwath/prompt-optimization.git
cd prompt-optimization

# 2. (recommended) create a virtual environment
python -m venv .venv
.\.venv\Scripts\activate          # Windows
source .venv/bin/activate         # Linux/macOS

# 3. Install in editable mode with dev tools
pip install -e ".[dev]"

# 4. Configure environment (see next section)
cp .env.example .env              # Windows PowerShell: Copy-Item .env.example .env
# ...edit .env and set your API key...

# 5. Optional: enable spaCy NER-based PII scrubbing
python -m spacy download en_core_web_sm
```

Verify the installation:

```bash
python -m prompt_opt.cli.main --help
pytest tests/ -q                  # should end with "N passed"
```

## Configuration

All settings live in `.env` (see `.env.example`) or environment variables.
Every value has a sensible default; only an API key is mandatory for LLM stages.
A `.env` file in the current directory is loaded automatically, including the provider
keys (variables already set in your shell take precedence, and placeholder values such as
`sk-...` are ignored).

### Provider keys (at least one required for optimize/gate/shadow/calibrate)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI key (`openai/...` models) |
| `ANTHROPIC_API_KEY` | Anthropic key (`anthropic/...` models) |
| `GOOGLE_API_KEY` | Google key (`google/...` models) |

### Model selection

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPTIMIZEE_MODEL` | `openai/gpt-4o-mini` | Model whose system prompt is being optimized |
| `JUDGE_MODEL` | `anthropic/claude-3-5-sonnet-20241022` | Judge (use a different family than the optimizee to reduce self-preference bias) |
| `PROPOSER_MODEL` | `openai/gpt-4o` | Generates candidate instruction rewrites |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embeddings for dedup |

### Budgets & thresholds

| Variable | Default | Purpose |
|----------|---------|---------|
| `MAX_LLM_CALLS_PER_RUN` | `5000` | Hard cap per optimizer run; counts every real LLM call (optimizee, judge, proposer, retries), cached evaluations are free |
| `MAX_WALL_SECONDS_PER_RUN` | `3600` | Hard cap per optimizer run |
| `MAX_USD_SPEND_PER_RUN` | `10.0` | Estimated-cost cap per run |
| `GATE_MIN_WIN_RATE` | `0.55` | Min paired win rate to promote (a tie counts as half a win) |
| `GATE_CI_LEVEL` | `0.95` | Bootstrap CI level (also the Wilcoxon significance level, `1 - level`) |
| `GATE_MIN_TEST_EXAMPLES` | `30` | Min paired test examples for a gate verdict |
| `GATE_MAX_GUARDRAIL_REGRESSION_PCT` | `0.10` | Max tolerated shift in refusal rate (absolute) between challenger and champion; shared by the eval guardrails and the gate |
| `DEDUP_COSINE_THRESHOLD` | `0.95` | Near-duplicate similarity cutoff (embedding cosine) |
| `PII_USE_NER` | `false` | Add the spaCy NER pass (names/orgs/places) to PII scrubbing; also used by the scheduled loop |
| `CALIBRATION_MIN_KAPPA` | `0.6` | Min human-judge agreement |
| `CALIBRATION_MIN_CORRELATION` | `0.7` | Min Spearman correlation |
| `DRIFT_KL_THRESHOLD` | `0.1` | Smoothed KL divergence a score shift must exceed to count as drift (a KS test must also be significant) |
| `REINGEST_CRON` | `0 2 * * *` | Feedback-loop ingest schedule |
| `REOPTIMIZE_CRON` | `0 3 * * 1` | Feedback-loop optimize schedule |
| `DRIFT_CRON` | `30 * * * *` | Feedback-loop drift-check schedule (`data/recent_scores.json` vs the saved baseline) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/prompt_opt.db` | Main storage DB |
| `LOG_LEVEL` | `INFO` | Logging verbosity |


## Quick Start (Full Walkthrough)

This takes you from raw logs to a deployed, gated prompt. Commands below use
`python -m prompt_opt.cli.main`; if you installed the package you can use the shorter
`prompt-opt` entry point instead.

```bash
# Step 1: Ingest raw production logs
# PII-scrubs, deduplicates, classifies, and stratifies into persisted splits.
python -m prompt_opt.cli.main ingest --input-file data/sample_conversations.jsonl
# Output: train/dev/test counts + files written to ./data/splits/*.jsonl

# Step 2: Build a gold label set in the Studio
python -m prompt_opt.cli.main add-gold \
    --conversation-id conv_001 \
    --response-a "Answer A text..." \
    --response-b "Answer B text..."
python -m prompt_opt.cli.main studio --port 8000
# Open http://127.0.0.1:8000 -> Queue tab -> annotate pairs (choose A / B / tie).
# Each item shows the conversation the responses answer (looked up from the train/dev
# splits by --conversation-id, so ingest first). Aim for >= 20 labeled items.

# Step 3: Calibrate the LLM judge against human labels
python -m prompt_opt.cli.main calibrate-judge
# PASS  -> judge frozen into data/judge_registry.json (e.g. judge_v1)
# FAIL  -> failure reasons printed (kappa too low, position bias, etc.)
# `optimize`, `gate` and `loop` then score with this frozen judge (its model + rubric),
# and refuse to run without one unless you pass --allow-uncalibrated.

# Step 4: Run an optimizer over the persisted splits
python -m prompt_opt.cli.main optimize --algorithm miprov2 --n-trials 20 --max-usd 10.0
# or the genetic optimizer:
python -m prompt_opt.cli.main optimize --algorithm gepa --n-trials 10
# The current prompt and the best candidates are all scored on the full dev split.
# A candidate is registered only if it beats the current prompt; its hash is printed, e.g.:
#   Best Candidate Hash: 3d1fdd7c836a3303
# otherwise: "No candidate beat the current prompt ... nothing registered."

# Step 5: Run the champion/challenger gate
python -m prompt_opt.cli.main gate \
    --challenger-hash 3d1fdd7c836a3303 \
    --skip-human-review          # omit this flag to require human sign-off
# Prints the criteria table (sample size, win rate, CI, score-shift significance, each
# guardrail, regression suite) and saves the result to ./data/gate_results/. The
# verdict is also recorded on the registered version. A PASS does NOT change the
# champion: stage the rollout (steps 6-7) or run `promote` (below).

# With human review (default):
python -m prompt_opt.cli.main gate --challenger-hash 3d1fdd7c836a3303
# decision=PENDING_HUMAN -> read the spot-check items (full texts), then decide:
python -m prompt_opt.cli.main gate-review
python -m prompt_opt.cli.main gate-approve --reviewer alice            # approve
python -m prompt_opt.cli.main gate-approve --reviewer bob --reject     # or reject

# Step 6: Shadow-mode the new prompt before exposing traffic
python -m prompt_opt.cli.main shadow --challenger-hash 3d1fdd7c836a3303 --n 20
# Serves the champion; logs challenger responses side-by-side to
# ./data/shadow_logs/shadow_log.jsonl for offline scoring.

# Step 7: Staged canary rollout (champion defaults to the registry champion;
# the challenger must have a recorded gate PASS unless you pass --force)
python -m prompt_opt.cli.main canary start --challenger-hash 3d1fdd7c836a3303
python -m prompt_opt.cli.main canary status       # stage, traffic %, soak timer
python -m prompt_opt.cli.main canary advance      # next ramp stage when soaked; the final
                                                  # advance promotes the challenger to champion
python -m prompt_opt.cli.main canary breach       # record a live guardrail breach
python -m prompt_opt.cli.main canary rollback --reason "error spike"

# Skipping staging: promote a gate-passed version directly
python -m prompt_opt.cli.main promote --hash 3d1fdd7c836a3303

# Step 8: Live rollback monitoring
# Wire these flags to your real metrics. While a canary is active a breach rolls
# the canary back (the champion is untouched); otherwise it reverts the registry
# champion to the previously promoted version.
python -m prompt_opt.cli.main monitor \
    --error-rate 0.02 \
    --latency-current 250 --latency-baseline 200 \
    --refusal-rate 0.08 --refusal-baseline 0.05

# Step 9: Drift detection on production scores
# Score real responses from a shadow log (the shadow step writes one), for a baseline and
# for recent traffic; each run also prints the refusal rate and mean latency:
python -m prompt_opt.cli.main score-shadow --which champion --output baseline_scores.json
python -m prompt_opt.cli.main score-shadow --output data/recent_scores.json
python -m prompt_opt.cli.main drift-check \
    --baseline-file baseline_scores.json \
    --scores-file data/recent_scores.json \
    --baseline-refusal-rate 0.05 --save-baseline    # keep the baseline for the loop
# exit 0 = no drift, 1 = drift, 2 = no verdict (e.g. fewer than 30 scores)

# Step 10: Continuous loop
python -m prompt_opt.cli.main loop           # scheduled by REINGEST_CRON / REOPTIMIZE_CRON / DRIFT_CRON
python -m prompt_opt.cli.main loop --once    # single cycle, then exit
```

Typical end-state after one loop iteration:
```
data/
├── splits/{train,dev,test}.jsonl     # curated dataset
├── prompt_opt.db                     # registry versions + eval cache
├── judge_registry.json               # frozen judge configs
├── gate_results/gate_*.json          # gate verdicts (write-once) + gate_*.review.json (human decisions)
├── shadow_logs/shadow_log.jsonl      # side-by-side responses
└── canary_state.json                 # rollout state machine
```

## CLI Reference
Every command supports `--help` for the full option list.

### `ingest`
```bash
python -m prompt_opt.cli.main ingest --input-file PATH [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input-file` | *(required)* | Raw JSONL log file (format below) |
| `--train-ratio` | `0.60` | Train fraction |
| `--dev-ratio` | `0.20` | Dev fraction |
| `--test-ratio` | `0.20` | Test fraction (frozen for gate) |
| `--use-ner` | off | Enable spaCy NER PII scrubbing (slower, needs `en_core_web_sm`; also on when `PII_USE_NER=true`) |
| `--refresh-test` | off | Rebuild the frozen test split (old one is archived; earlier gate results stop being comparable) |

Pipeline: parse -> PII scrub -> heuristic category/difficulty enrichment -> dedup ->
stratified split -> persist to `data/splits/`. Split ratios must be non-negative and sum to 1.
The command ends with an **Ingestion Report**: nothing is dropped without being counted.

* **Parsing.** One bad record never aborts an ingest; it is skipped and listed with its reason
  (no messages, no user turn, unknown role, duplicate `conversation_id`, unreadable line). A
  conversation is skipped whole rather than losing a turn from the middle. Role aliases
  (`human`, `ai`, `function`, ...) and OpenAI-style content parts are normalised. Tool calls are
  kept, in the native format or OpenAI's `tool_calls` / `function_call` (string arguments are
  parsed, results linked by `tool_call_id`), and `has_tool_use` is set from them.
* **PII scrubbing.** All matching happens on the original text and is resolved in one pass.
  Numbers are redacted only when they validate: cards must pass the Luhn check, SSNs must be in
  real ranges (or follow an "SSN" label), IPs need octets <= 255, phones need separators or a
  "phone"/"call" label, ZIPs need a "zip code" label or a state code. So `1234567` and `10000` in
  a math question survive. Also redacted: emails, URLs, API keys/tokens/private keys and
  `password=...`, street addresses, and "my name is ...". Message content, message names and
  tool-call arguments/results are all scrubbed. Names and places elsewhere in free text need the
  NER pass. The report counts redactions per label, never the values.
* **Deduplication.** Exact duplicates (case/whitespace-normalised) are removed first. Near
  duplicates are found by comparing each conversation only against those already kept
  (embeddings, batched). If embeddings are unavailable the report says so and a deterministic
  lexical fallback is used: it catches near-identical text of at least 40 characters (spacing,
  punctuation, small edits), not paraphrases. Conversations that duplicate an item of the frozen
  test split are dropped, so paraphrases of test items cannot leak into train/dev.
* **Categories.** Keyword classification matches whole words (with inflections) and picks the
  category with the most distinct hits, so "whatever" is not `safety`.

### `add-gold`
```bash
python -m prompt_opt.cli.main add-gold --conversation-id ID --response-a TEXT --response-b TEXT
```
Adds an unlabeled pair to the annotation queue. The conversation is looked up in the
persisted train/dev splits so annotators (and the judge, at calibration time) see the same
context; a warning is printed if it cannot be found.

### `studio`
```bash
python -m prompt_opt.cli.main studio [--port 8000] [--host 127.0.0.1]
```
FastAPI labeling UI. Routes: `/` (stats), `/queue`, `/annotate/{id}`, `/calibration`.

### `calibrate-judge`
Runs the LLM judge over every labeled gold item (with the item's real conversation as
context; items whose conversation is not in the train/dev splits are skipped and counted)
and reports Cohen's kappa, Spearman/Pearson correlation, agreement rate, confusion matrix,
and bias diagnostics:

* **position bias**: across both A/B orderings of every pair, the share of decisive verdicts
  that picked the first-listed response (a fair judge is ~50%, whatever the gold set's label
  balance), plus how often the verdict flips when the order is swapped;
* **length bias**: how strongly the judge's score difference follows the length difference,
  compared with the human labels' own length correlation.

Calibration fails if kappa is undefined (no variation in the labels), if more than 20% of
judge calls fail, or if a threshold (`CALIBRATION_MIN_KAPPA`, `CALIBRATION_MIN_CORRELATION`)
or a bias check is missed. On success the judge (model + rubric, fingerprinted) is frozen
as a new version in `data/judge_registry.json`; if the rubric definition later changes,
evaluation refuses to use that version until you re-calibrate.

### `optimize`
```bash
python -m prompt_opt.cli.main optimize [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--algorithm` | `miprov2` | `miprov2` (Bayesian TPE) or `gepa` (genetic) |
| `--n-trials` | `10` | Optuna trials (MIPROv2) / generations (GEPA) |
| `--max-usd` | `10.0` | Spend cap for this run |
| `--minibatch-size` | auto | Dev examples per search step (auto = `min(dev size, 16)`) |
| `--judge-version` | latest | Frozen judge version to score with (e.g. `judge_v2`); also keys the eval cache |
| `--allow-uncalibrated` | off | Evaluate with an unvalidated judge if none has been frozen |

Loads `data/splits/{train,dev}.jsonl` (errors out if ingestion hasn't run or dev is empty)
and seeds the search from the current registry champion. Search steps score candidates on
one fixed dev minibatch (the same examples for every candidate; GEPA reflects on a train
minibatch and selects on dev), with refusal/length/token guardrails checked against the
current prompt's responses. Then the current prompt and the best few candidates are
scored on the **full dev split** under identical conditions, with the regression suite
replayed live. The winner is registered (with those eval scores) and its content hash
printed for `gate`, but only if it beats the current prompt; guardrail-failing candidates
rank below passing ones.

All LLM calls of a run (optimizee, judge and proposer, including retries) are metered against
the budget from the real response usage / litellm pricing, and each call is checked against
the caps *before* it is made. Cached evaluations cost nothing. A model or judge failure never
becomes a score: the example is excluded, and if more than 20% of the examples of an
evaluation fail the run stops with an error instead.

### `gate`
```bash
python -m prompt_opt.cli.main gate --challenger-hash HASH [--champion-hash HASH] [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--challenger-hash` | *(required)* | Candidate content hash from `optimize` |
| `--champion-hash` | registry champion | Compare against a specific champion |
| `--skip-human-review` | off | Produce direct PASS/FAIL without spot-check |
| `--allow-no-regression-suite` | off | Do not fail when no regression suite ran (by default the suite must run and pass) |
| `--auto-promote / --no-auto-promote` | off | Promote to champion immediately on PASS, skipping shadow/canary staging |
| `--judge-version` | latest | Frozen judge version to score with (also keys the eval cache) |
| `--allow-uncalibrated` | off | Evaluate with an unvalidated judge if none has been frozen |

Evaluation order: champion baseline first (paired responses captured), challenger
replayed against those responses, statistics, guardrail checks, regression suite replayed
live through the challenger prompt (system prompt and few-shot demos, from
`data/regression_suite.jsonl`). Exit codes: `0` = PASS/PENDING, `1` = FAIL.

The gate passes only if **all** of these hold (thresholds are stored in the result):

| Criterion | Rule |
|-----------|------|
| `min_test_examples` | at least `GATE_MIN_TEST_EXAMPLES` (30) paired examples; unpaired or failed examples are excluded, not counted |
| `win_rate` | (wins + ½·ties) / n `>= GATE_MIN_WIN_RATE` (0.55). Ties count half so that a mostly-tied but clearly favourable comparison (30 wins / 10 losses / 60 ties = 0.60) is not mistaken for a 30% win rate, and "all ties" is exactly 0.5 |
| `ci_excludes_null` | the bootstrap CI's lower bound is above 0.50 |
| `score_shift_significance` | one-sided Wilcoxon signed-rank on the per-example judge-score differences, p < `1 - GATE_CI_LEVEL`; catches "wins by a hair, loses big" |
| `refusal_rate_shift` | \|challenger − champion refusal rate\| `<= GATE_MAX_GUARDRAIL_REGRESSION_PCT` |
| `output_length_shift` | KS test on response lengths, p > 0.01 |
| `token_growth` | mean output tokens grew by at most 30% |
| `format_compliance` | only when a format check is configured |
| `regression_suite_pass_rate` | the suite ran and 100% of its cases passed; a case whose model call failed counts as failed |

A guardrail that could not be measured shows `n/a`; a *required* input that is missing
(too few examples, no regression suite) fails the gate. Use `GateCriteria(...)` to change
the other thresholds (`max_token_growth_pct`, `length_shift_alpha`, ...) *before* looking at results.

The gate result file is written once and never edited. It includes the human spot-check
items: a sample cycling through losses, wins and ties (seeded by the challenger hash,
so reproducible) with the user query, both responses, both scores and the judge's rationale.
The registry stores only a slim verdict, without those texts.

### `gate-review`
```bash
python -m prompt_opt.cli.main gate-review [--result-file PATH]
```
Prints the criteria results and the spot-check items of the latest pending gate result
(or a specific file) in full, so the human review is a review of actual responses.

### `gate-approve`
```bash
python -m prompt_opt.cli.main gate-approve --reviewer NAME [--result-file PATH]
                                            [--notes TEXT] [--reject] [--promote]
```
Records the human decision on the most recent pending gate result (or a specific file)
and updates the verdict on the registered version. The gate result itself is not modified:
the decision is written beside it as `<result>.review.json`, and a result can be reviewed
once. Approval does not change the champion unless `--promote` is given. A warning is
printed if the champion changed after the gate ran.

### `promote`
```bash
python -m prompt_opt.cli.main promote --hash HASH [--approver NAME] [--force]
```
Promotes a registered version to champion. Requires a recorded gate PASS for that
version unless `--force` is given.

### `shadow`
```bash
python -m prompt_opt.cli.main shadow --challenger-hash HASH [--n 5] [--split dev]
```
Replays N conversations (train or dev; the test split is reserved for the gate) through both
prompts, each run exactly as it was evaluated: system prompt **and** few-shot demos, with the
conversation's logged assistant reply left out. Champion response is what would be served;
both are logged to `data/shadow_logs/shadow_log.jsonl`. A failed call yields an empty
response plus a recorded error (never an `[ERROR: ...]` pseudo-answer), and the summary
counts errors per prompt and averages latency over successful calls only.

### `canary`
```bash
python -m prompt_opt.cli.main canary ACTION [options]
```

| Action | Extra options | Description |
|--------|---------------|-------------|
| `start` | `--challenger-hash`, `--champion-hash` (default: registry champion), `--force` | Begin ramp at 1% traffic (default schedule: 1% -> 5% -> 25% -> 100%). Requires a gate PASS on the challenger unless `--force`; refuses if another canary is active |
| `status` | – | Print current stage, traffic %, soak elapsed / remaining, breaches |
| `advance` | – | Move to the next stage once the current stage's soak period has elapsed (the next stage's soak starts then, so no stage can be skipped by waiting); completing the last stage promotes the challenger to champion |
| `breach` | – | Record a guardrail breach; too many halts the rollout |
| `rollback` | `--reason` | Halt and mark rolled back |

State persists in `data/canary_state.json` (written atomically) and survives process
restarts (`CanaryController.load()`). In code, `CanaryController.should_use_challenger(request_key=...)`
routes deterministically by a stable key (user / session / conversation id): the same key always
gets the same arm and stays on the challenger as traffic grows; without a key each request is
an independent draw. Request counters are persisted every 50 requests, not on every request.

### `monitor`
```bash
python -m prompt_opt.cli.main monitor [--error-rate F] [--latency-current MS]
    [--latency-baseline MS] [--refusal-rate F] [--refusal-baseline F]
    [--format-compliance F] [--regression-pass-rate F] [--canary-state PATH]
```
Default trigger thresholds: error rate > 5%, latency regression > 30%, refusal delta >
15pp, format compliance < 90%, regression pass rate < 100%. Any breach exits `1`: if a
canary is active it is rolled back and the champion is left alone; otherwise the
registry champion is reverted to the previously promoted version (never to a version
that was itself already rolled back). Rollbacks do not cascade: for one hour after a registry
rollback that has not been followed by a promotion, a persisting breach only alerts (and still
exits `1`) instead of rolling back another version; a new `promote` re-arms it. If there is no
earlier version to return to, that is reported instead of "rolled back to None".

### `score-shadow`
```bash
python -m prompt_opt.cli.main score-shadow [--log-file data/shadow_logs/shadow_log.jsonl]
    [--which champion|challenger] [--output data/recent_scores.json] [--limit N]
```
Judge-scores the responses recorded by `shadow` (with the calibrated judge) and writes a JSON
array of scores, the input `drift-check` and the loop's drift job need. Calls that failed are
skipped, never scored. It prints the refusal rate and mean latency too, which are the other
baseline/observation values `drift-check` accepts.

### `drift-check`
```bash
python -m prompt_opt.cli.main drift-check --scores-file F [--baseline-file F]
    [--baseline-state data/drift_baseline.json] [--save-baseline]
    [--baseline-refusal-rate F] [--baseline-latency-ms MS]
    [--refusal-rate F] [--format-compliance F] [--latency-ms MS]
```
Both score files hold a JSON array of floats. A score shift counts as drift only if a
two-sample KS test says it is real (p < 0.01) **and** the smoothed KL divergence exceeds
`DRIFT_KL_THRESHOLD`. (A bare KL estimate is biased upwards on small samples: identical
distributions exceeded the old 0.1 threshold on every comparison at n=100.) Refusal rate and
latency are compared as EMAs against the baseline; format compliance against a 90% floor.

Exit codes: `0` no drift, `1` drift, `2` **no verdict**: fewer than 30 scores on either side,
no baseline, or an observation (`--refusal-rate`, `--latency-ms`) that needs a baseline value it
was not given. "No verdict" is never reported as "no drift". `--save-baseline` keeps the
baseline (scores plus the refusal/latency baselines) for later runs and for the loop.

### `loop`
```bash
python -m prompt_opt.cli.main loop [--once] [--judge-version V] [--allow-uncalibrated]
```
Starts the APScheduler background loop using `REINGEST_CRON`, `REOPTIMIZE_CRON` and `DRIFT_CRON`.
Each cycle ingests `data/raw_production_logs.jsonl` (drop new exports there),
refreshes splits, and re-runs optimization from the current champion when data allows. The
drift job compares `data/recent_scores.json` with the saved baseline `data/drift_baseline.json`
(both created with `score-shadow` / `drift-check --save-baseline`) and logs the verdict; until
both exist it only logs why it did nothing.
Like `optimize` it needs a provider key and a calibrated judge, and registers a candidate
only if it beats the champion on dev.


## Data Formats
### Raw conversation logs (`--input-file`, `raw_production_logs.jsonl`)
One JSON object per line:
```json
{
  "conversation_id": "conv_001",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "category": "coding",
  "weak_signal": "positive",
  "source_system": "helpdesk",
  "created_at": "2026-01-15T10:30:00Z"
}
```

Only `conversation_id` and `messages` are required. Malformed lines and unusable records are
skipped, counted and listed in the ingestion report. Messages may also carry `tool_calls`
(`{"call_id", "name", "arguments", "result"}` or OpenAI's `{"id", "function": {...}}`).

### Regression suite (`data/regression_suite.jsonl`)
One case per line:

```json
{
  "conversation_id": "reg_001",
  "expected_behavior": "refuse",
  "expected_keywords": ["not able", "can't"],
  "description": "Prompt injection must be refused"
}
```

Supported `expected_behavior` values: `refuse`, `redirect_to_crisis`, `deny_pii`,
`answer_correctly`, `follow_instruction` (uses `expected_content`), `valid_json`,
`recall_name`. Extra fields (`messages`, `category`, ...) are preserved for live replay
during gating.

Matching rules: keywords match **whole words**, case-insensitively and regardless of
apostrophe style (`can't` matches `can’t`), so `4` does not match `1994` and `help` does not
match `helpful`. Pick keywords that a *compliant* reply would not contain (`help`, `harm`,
`game` are traps). A case that cannot be checked fails rather than passes: an unknown
`expected_behavior`, a keyword behaviour with no `expected_keywords`, `follow_instruction` with no
`expected_content`, or a case whose model call failed.

### Persisted splits
`data/splits/{train,dev,test}.jsonl` — one serialized `ConversationRecord` per line,
round-trippable via `DatasetStore.load_split("train")`.

## Storage Layout
| Path | Contents |
|------|----------|
| `data/prompt_opt.db` | SQLite: `prompt_versions` (immutable registry), `champion_events` (promote/rollback audit), eval cache tables, gold labels |
| `data/splits/` | Curated train/dev/test JSONL |
| `data/judge_registry.json` | Frozen judge versions (model + rubric + calibration report) |
| `data/gate_results/` | Write-once gate verdict JSON files (with spot-check items), plus `*.review.json` human decisions beside them |
| `data/shadow_logs/` | Shadow-mode request logs |
| `data/canary_state.json` | Resumable canary rollout state machine |
| `data/raw_production_logs.jsonl` | Drop point for fresh logs consumed by `loop` |

## Python API Usage
```python
from pathlib import Path

from prompt_opt.deployment.canary import CanaryController
from prompt_opt.deployment.rollback import RollbackMonitor
from prompt_opt.eval.harness import EvalHarness, PromptCandidate
from prompt_opt.gate.champion_gate import ChampionGate, GateCriteria
from prompt_opt.ingestion.ingestor import Ingestor
from prompt_opt.judge.registry import JudgeRegistry
from prompt_opt.optimizer.budget import BudgetController
from prompt_opt.optimizer.miprov2 import MIPROv2
from prompt_opt.registry.prompt_registry import PromptRegistry
from prompt_opt.storage.dataset_store import DatasetStore

# Ingest and persist
store = DatasetStore("./data/splits")
result = Ingestor(use_ner=False).ingest_file(Path("logs.jsonl"))
store.save(result)

# Optimize from the current champion
registry = PromptRegistry(db_url="sqlite:///./data/prompt_opt.db")
champ = registry.get_champion()   # None until a first version has been promoted
initial = (
    PromptCandidate(champ.system_prompt, champ.few_shot_demos)
    if champ
    else PromptCandidate("You are a helpful assistant.")
)

# Score with the frozen, calibrated judge (its model + rubric), not a hand-built LLMJudge()
judge, frozen = JudgeRegistry().load_judge()
harness = EvalHarness(judge, judge_version=frozen.version_id)
budget = BudgetController(max_usd_spend=10.0)
optimizer = MIPROv2(n_trials=20, minibatch_size=10)
# All candidates are scored on the full dev split and the initial prompt is always included
best, best_eval = optimizer.optimize(
    initial, store.load_split("train"), store.load_split("dev"), harness, budget
)[0]
if best.content_hash == initial.content_hash:
    raise SystemExit("No candidate beat the current prompt")

version_id = registry.register(
    system_prompt=best.system_prompt,
    few_shot_demos=best.few_shot_demos,
    optimizer_run_id="run_001",
    eval_scores=best_eval.to_dict(),
)

# Gate it
gate = ChampionGate(harness, criteria=GateCriteria(require_human_spot_check=True))
verdict = gate.evaluate(best, initial, store.load_split("test"))
if verdict.decision == "PENDING_HUMAN":
    verdict.record_human_review(reviewer="alice", approved=True)   # -> PASS
registry.attach_gate_result(version_id, verdict.to_dict())
if verdict.decision == "PASS":
    # Ideally shadow/canary first (see the CLI); promote once staging is complete.
    registry.promote(version_id)

# Watch it live
event = RollbackMonitor(registry).check(error_rate=0.07)           # auto-rollbacks
ctrl = CanaryController.load("./data/canary_state.json")           # resume rollout
```

## Running Without API Keys
Everything except `optimize`, `gate`, `shadow`, and `calibrate-judge` (with labeled
data present) works fully offline: `ingest`, `add-gold`, `studio`, `canary`, `monitor`,
`drift-check`, `loop`. LLM-bound commands fail fast with a clear message telling you
which env var to set.

For CI-style verification of the LLM stages without keys, see
`tests/integration/` — they mock `litellm.completion` the same way.

## Testing
```bash
pytest tests/ -q             # full suite (383 tests)
pytest tests/unit -q         # fast unit layer
pytest tests/integration -q  # harness + optimizer smoke tests
ruff check prompt_opt tests  # lint (currently clean)
```

## Project Structure
```
prompt-optimization/
├── data/                          # runtime artifacts (mostly gitignored)
│   ├── sample_conversations.jsonl # example raw logs
│   ├── regression_suite.jsonl     # 10 safety/correctness cases
│   ├── splits/                    # persisted train/dev/test (generated)
│   └── ...
├── prompt_opt/
│   ├── cli/main.py                # Typer CLI (15 commands)
│   ├── config.py                  # pydantic-settings, .env driven
│   ├── llm.py                     # central litellm accessor
│   ├── ingestion/                 # scrub, dedup, stratify, schema
│   ├── labeling/                  # FastAPI studio, rubric, gold store
│   ├── judge/                     # LLM judge, calibrator, judge registry
│   ├── eval/                      # harness, cache, guardrails, metrics
│   ├── optimizer/                 # MIPROv2, GEPA, budget controller
│   ├── gate/                      # champion gate + statistics
│   ├── registry/                  # content-addressed prompt registry
│   ├── deployment/                # shadow runner, canary, rollback monitor
│   ├── feedback/                  # drift detector, cron loop
│   └── storage/                   # dataset split persistence
├── tests/unit + tests/integration
└── ui/templates + ui/static       # studio Jinja templates
```

## Troubleshooting
| Symptom | Cause / Fix |
|---------|-------------|
| `No LLM provider API key found` | Set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` in your shell or in a `.env` file in the directory you run from. Placeholder values like `sk-...` are ignored. |
| `has no gate PASS on record` | `promote` / `canary start` only accept versions whose latest gate verdict is PASS. Run `gate` (and `gate-approve` if pending), or pass `--force`. |
| Existing test split kept frozen | Intended: re-ingest never changes `data/splits/test.jsonl`. Use `ingest --refresh-test` to rebuild it deliberately. |
| `No persisted train/dev splits found` | Run `ingest` before `optimize`. |
| `hash not found in registry` | `gate` needs the *content hash* printed by `optimize`, not the `prompt_vN` id. |
| `Gate is not pending human review` / `already reviewed` | That gate result was already approved/rejected; the decision is in the `.review.json` next to it in `data/gate_results/`. |
| `Only N paired test examples (minimum 30)` | The frozen test split is too small for a meaningful verdict. Ingest more data (then `ingest --refresh-test` if you accept losing comparability with earlier gate results), or lower `GATE_MIN_TEST_EXAMPLES` deliberately. |
| `Regression suite did not run` | The gate requires the regression suite (`data/regression_suite.jsonl`). Provide it, or pass `--allow-no-regression-suite` knowing that known-bad behaviour is then unchecked. |
| `Score improvement not significant` although the win rate looks fine | The judge-score differences do not support an improvement (e.g. small wins, large losses). Inspect the per-example results in the gate file. |
| Calibration always fails | Need >= 5 usable labeled items (items without a conversation in the train/dev splits, or whose judge call failed, are skipped and counted in the message) and a mix of A / B / tie labels (kappa is undefined otherwise). Check the reported position/length bias. |
| `No calibrated judge has been frozen yet` | Run `calibrate-judge` (it must PASS), or pass `--allow-uncalibrated` to `optimize` / `gate` / `loop`. |
| `The rubric ... changed after judge_vN was calibrated` | The rubric definition no longer matches the frozen judge; re-run `calibrate-judge` to freeze a new version. |
| `Optimization failed: N/M examples failed ...` | Model or judge calls kept failing (rate limit, bad key, unsupported params). Nothing was scored or cached; fix the cause and re-run. |
| `No candidate beat the current prompt` | Working as intended: nothing was registered. Try more `--n-trials`, or check whether the guardrails (refusal / length / +30% tokens vs. the current prompt) are rejecting longer answers. |
| Ingest says `Near-duplicate detection used the offline lexical fallback` | The embedding model was unreachable (no key / offline). Exact duplicates and near-identical text are still removed, but paraphrases are not; set a key for semantic dedup. |
| `drift-check` exits `2` / "No verdict" | Fewer than 30 scores on a side, no saved baseline, or `--refusal-rate` / `--latency-ms` given without the matching `--baseline-*` value. |
| `skipped: unknown message role ...` in the ingestion report | The record's messages use a role the pipeline does not know (known aliases such as `human`, `ai`, `function` are mapped). Fix the export or map the role. |
| UnicodeEncodeError in old terminals | Use Windows Terminal / PowerShell 7, or set `$env:PYTHONUTF8 = "1"`. |
| Stale eval results after changing prompts/judge | The eval cache is keyed by `(prompt_hash, conv_id, judge_version, model)`, where the judge version is the frozen id (`judge_vN`); freezing a new judge (re-calibrating) starts a fresh cache namespace. |
