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
| 0. Ingestion | `prompt_opt/ingestion` | Raw production logs are PII-scrubbed (regex + optional spaCy NER), near-deduplicated via embeddings + FAISS, enriched with category/difficulty labels, and stratified into **train / dev / test** splits. The test split stays frozen for gate evaluation only. |
| 1. Labeling & Calibration | `prompt_opt/labeling`, `prompt_opt/judge` | Humans annotate A/B preferences in the web Studio. The LLM judge is then calibrated against these gold labels (Cohen's kappa >= 0.6, Spearman >= 0.7, plus length/position bias checks). Passing judges are frozen into versioned registry entries. |
| 2. Eval Harness | `prompt_opt/eval` | Replays conversations through any candidate prompt, scores responses with the judge, computes guardrail metrics (refusal shift, KS length test, token growth, regression suite), and caches every result in SQLite so repeated evaluations are free. |
| 3. Optimizers | `prompt_opt/optimizer` | **MIPROv2**: bootstraps few-shot demos, asks a proposer LLM for instruction variants, then runs Optuna TPE Bayesian search over instruction x demo combinations. **GEPA**: genetic search with Pareto fronts and reflective mutation driven by judge rationales. Both respect hard budget caps (calls / wall-time / USD). |
| 4. Champion Gate | `prompt_opt/gate` | The only component allowed to touch the frozen test split. Runs paired bootstrap CIs (10,000 resamples) + Wilcoxon signed-rank tests on paired win/loss/tie outcomes, enforces win-rate thresholds, guardrail non-regression, and a 100% regression-suite pass rate. Produces immutable `PASS / FAIL / PENDING_HUMAN` verdicts. |
| 5. Registry & Deployment | `prompt_opt/registry`, `prompt_opt/deployment` | Content-addressed, append-only prompt version store with an audit log of champion transitions. Rollback reverts to the previous promoted version. Shadow mode serves the champion while logging challenger responses; canary mode ramps traffic 1% -> 5% -> 25% -> 100% with soak periods and breach halts. |
| 6. Feedback Loop | `prompt_opt/feedback` | Cron-driven scheduler that re-ingests fresh logs and triggers re-optimization, plus KL-divergence drift detection over production score distributions. |

## Requirements

- **Python 3.11+**
- At least one LLM provider API key (**OpenAI**, **Anthropic**, or **Google**) for
  optimization / evaluation stages. Ingestion, canary, monitor, drift-check, and the
  labeling studio work offline.
- ~2 GB disk for dependencies (spaCy, sentence-transformers, FAISS, torch are pulled in).

## Installation

```bash
# 1. Clone and enter the project
git clone <repo-url>
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
| `MAX_LLM_CALLS_PER_RUN` | `500` | Hard cap per optimizer run |
| `MAX_WALL_SECONDS_PER_RUN` | `3600` | Hard cap per optimizer run |
| `MAX_USD_SPEND_PER_RUN` | `10.0` | Estimated-cost cap per run |
| `GATE_MIN_WIN_RATE` | `0.55` | Min paired win rate to promote |
| `GATE_CI_LEVEL` | `0.95` | Bootstrap CI level |
| `GATE_MAX_GUARDRAIL_REGRESSION_PCT` | `0.05` | Max allowed guardrail regression |
| `DEDUP_COSINE_THRESHOLD` | `0.95` | Near-duplicate similarity cutoff |
| `CALIBRATION_MIN_KAPPA` | `0.6` | Min human-judge agreement |
| `CALIBRATION_MIN_CORRELATION` | `0.7` | Min Spearman correlation |
| `DRIFT_KL_THRESHOLD` | `0.1` | KL divergence alert threshold |
| `REINGEST_CRON` | `0 2 * * *` | Feedback-loop ingest schedule |
| `REOPTIMIZE_CRON` | `0 3 * * 1` | Feedback-loop optimize schedule |
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
# Aim for >= 20 labeled items before calibrating.

# Step 3: Calibrate the LLM judge against human labels
python -m prompt_opt.cli.main calibrate-judge
# PASS  -> judge frozen into data/judge_registry.json
# FAIL  -> failure reasons printed (kappa too low, position bias, etc.)

# Step 4: Run an optimizer over the persisted splits
python -m prompt_opt.cli.main optimize --algorithm miprov2 --n-trials 20 --max-usd 10.0
# or the genetic optimizer:
python -m prompt_opt.cli.main optimize --algorithm gepa --n-trials 10
# Registers the best candidate and prints its content hash, e.g.:
#   Best Candidate Hash: 3d1fdd7c836a3303

# Step 5: Run the champion/challenger gate
python -m prompt_opt.cli.main gate \
    --challenger-hash 3d1fdd7c836a3303 \
    --skip-human-review          # omit this flag to require human sign-off
# Prints criteria table (win rate, CI, guardrails, regression suite),
# saves an immutable result to ./data/gate_results/, and on PASS promotes
# the challenger automatically.

# With human review (default):
python -m prompt_opt.cli.main gate --challenger-hash 3d1fdd7c836a3303
# decision=PENDING_HUMAN -> inspect the printed spot-check items, then:
python -m prompt_opt.cli.main gate-approve --reviewer alice            # approve
python -m prompt_opt.cli.main gate-approve --reviewer bob --reject     # or reject

# Step 6: Shadow-mode the new prompt before exposing traffic
python -m prompt_opt.cli.main shadow --challenger-hash 3d1fdd7c836a3303 --n 20
# Serves the champion; logs challenger responses side-by-side to
# ./data/shadow_logs/shadow_log.jsonl for offline scoring.

# Step 7: Staged canary rollout
python -m prompt_opt.cli.main canary start \
    --challenger-hash 3d1fdd7c836a3303 --champion-hash <current-champion-hash>
python -m prompt_opt.cli.main canary status       # stage, traffic %, soak timer
python -m prompt_opt.cli.main canary advance      # move to next ramp stage when soaked
python -m prompt_opt.cli.main canary breach       # record a live guardrail breach
python -m prompt_opt.cli.main canary rollback --reason "error spike"

# Step 8: Live rollback monitoring
# Wire these flags to your real metrics; any breach auto-reverts the registry
# champion to the previously promoted version.
python -m prompt_opt.cli.main monitor \
    --error-rate 0.02 \
    --latency-current 250 --latency-baseline 200 \
    --refusal-rate 0.08 --refusal-baseline 0.05

# Step 9: Drift detection on production scores
python -m prompt_opt.cli.main drift-check \
    --baseline-file baseline_scores.json \
    --scores-file recent_scores.json

# Step 10: Continuous loop
python -m prompt_opt.cli.main loop           # scheduled by REINGEST_CRON / REOPTIMIZE_CRON
python -m prompt_opt.cli.main loop --once    # single cycle, then exit
```

Typical end-state after one loop iteration:
```
data/
├── splits/{train,dev,test}.jsonl     # curated dataset
├── prompt_opt.db                     # registry versions + eval cache
├── judge_registry.json               # frozen judge configs
├── gate_results/gate_*.json          # immutable gate verdicts
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
| `--use-ner` | off | Enable spaCy NER PII scrubbing (slower, needs `en_core_web_sm`) |

Pipeline: parse -> regex PII scrub (+ NER) -> heuristic category/difficulty enrichment ->
FAISS embedding dedup -> stratified split -> persist to `data/splits/`.

### `add-gold`
```bash
python -m prompt_opt.cli.main add-gold --conversation-id ID --response-a TEXT --response-b TEXT
```
Adds an unlabeled pair to the annotation queue.

### `studio`
```bash
python -m prompt_opt.cli.main studio [--port 8000] [--host 127.0.0.1]
```
FastAPI labeling UI. Routes: `/` (stats), `/queue`, `/annotate/{id}`, `/calibration`.

### `calibrate-judge`
Runs the LLM judge over every labeled gold item and reports Cohen's kappa,
Spearman/Pearson correlation, agreement rate, confusion matrix, and length/position
bias diagnostics. On success the judge is frozen as a new version in
`data/judge_registry.json`.

### `optimize`
```bash
python -m prompt_opt.cli.main optimize [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--algorithm` | `miprov2` | `miprov2` (Bayesian TPE) or `gepa` (genetic) |
| `--n-trials` | `10` | Optuna trials (MIPROv2) / generations (GEPA) |
| `--max-usd` | `10.0` | Spend cap for this run |
| `--judge-version` | `v0` | Judge tag used for eval-cache keying |

Loads `data/splits/{train,dev}.jsonl` (errors out if ingestion hasn't run), seeds the
search from the current registry champion, registers the winner with its eval scores,
and prints the content hash needed by `gate`.

### `gate`
```bash
python -m prompt_opt.cli.main gate --challenger-hash HASH [--champion-hash HASH] [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--challenger-hash` | *(required)* | Candidate content hash from `optimize` |
| `--champion-hash` | registry champion | Compare against a specific champion |
| `--skip-human-review` | off | Produce direct PASS/FAIL without spot-check |
| `--auto-promote / --no-auto-promote` | on | Promote automatically on PASS |
| `--judge-version` | `v0` | Cache-keying judge tag |

Evaluation order: champion baseline first (paired responses captured), challenger
replayed against those responses, bootstrap CI + Wilcoxon tests, guardrail checks,
regression suite replayed live through the challenger prompt (from
`data/regression_suite.jsonl`). Exit codes: `0` = PASS/PENDING, `1` = FAIL.

### `gate-approve`
```bash
python -m prompt_opt.cli.main gate-approve --reviewer NAME [--result-file PATH]
                                            [--notes TEXT] [--reject]
```
Records the human decision on the most recent pending gate result (or a specific file)
and promotes the challenger on approval.

### `shadow`
```bash
python -m prompt_opt.cli.main shadow --challenger-hash HASH [--n 5] [--split dev]
```
Replays N conversations through both prompts. Champion response is what would be
served; both are logged to `data/shadow_logs/shadow_log.jsonl`.

### `canary`
```bash
python -m prompt_opt.cli.main canary ACTION [options]
```

| Action | Extra options | Description |
|--------|---------------|-------------|
| `start` | `--challenger-hash`, `--champion-hash` | Begin ramp at 1% traffic (default schedule: 1% -> 5% -> 25% -> 100%) |
| `status` | – | Print current stage, traffic %, soak timer, breaches |
| `advance` | – | Move to next stage once its soak period elapsed cleanly |
| `breach` | – | Record a guardrail breach; too many halts the rollout |
| `rollback` | `--reason` | Halt and mark rolled back |

State persists in `data/canary_state.json` and survives process restarts
(`CanaryController.load()`).

### `monitor`
```bash
python -m prompt_opt.cli.main monitor [--error-rate F] [--latency-current MS]
    [--latency-baseline MS] [--refusal-rate F] [--refusal-baseline F]
    [--format-compliance F] [--regression-pass-rate F]
```
Default trigger thresholds: error rate > 5%, latency regression > 30%, refusal delta >
15pp, format compliance < 90%, regression pass rate < 100%. Any breach rolls the
registry champion back and exits `1`.

### `drift-check`
```bash
python -m prompt_opt.cli.main drift-check --baseline-file F --scores-file F
                                          [--refusal-rate F] [--format-compliance F]
```
Both files contain a JSON array of floats. Flags critical/warning KL-divergence drift;
exits `1` when events fire.

### `loop`
```bash
python -m prompt_opt.cli.main loop [--once]
```
Starts the APScheduler background loop using `REINGEST_CRON` and `REOPTIMIZE_CRON`.
Each cycle ingests `data/raw_production_logs.jsonl` (drop new exports there),
refreshes splits, and re-runs optimization from the current champion when data allows.


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

Only `conversation_id` and `messages` are required. Malformed lines are skipped with a
warning.

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

### Persisted splits
`data/splits/{train,dev,test}.jsonl` — one serialized `ConversationRecord` per line,
round-trippable via `DatasetStore.load_split("train")`.

## Storage Layout
| Path | Contents |
|------|----------|
| `data/prompt_opt.db` | SQLite: `prompt_versions` (immutable registry), `champion_events` (promote/rollback audit), eval cache tables, gold labels |
| `data/splits/` | Curated train/dev/test JSONL |
| `data/judge_registry.json` | Frozen judge versions (model + rubric + calibration report) |
| `data/gate_results/` | Immutable gate verdict JSON files |
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
from prompt_opt.judge.judge import LLMJudge
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
champ = registry.get_champion()
initial = PromptCandidate(champ.system_prompt, champ.few_shot_demos)

harness = EvalHarness(LLMJudge(), judge_version="v0")
budget = BudgetController(max_usd_spend=10.0)
optimizer = MIPROv2(n_trials=20, minibatch_size=10)
best, best_eval = optimizer.optimize(
    initial, store.load_split("train"), store.load_split("dev"), harness, budget
)[0]

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
pytest tests/ -q             # full suite (31 tests)
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
│   ├── cli/main.py                # Typer CLI (14 commands)
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
| `No LLM provider API key found` | Set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` in `.env`. |
| `No persisted train/dev splits found` | Run `ingest` before `optimize`. |
| `hash not found in registry` | `gate` needs the *content hash* printed by `optimize`, not the `prompt_vN` id. |
| `Gate is not pending human review` | That gate result was already approved/rejected; check `data/gate_results/`. |
| Calibration always fails | Need >= 5 labeled items; watch for reported length/position bias — rebalance your gold set. |
| Dedup warns `using random vector (test mode)` | Embedding model unreachable (no key/offline); dedup degrades gracefully but works better with a key. |
| UnicodeEncodeError in old terminals | Use Windows Terminal / PowerShell 7, or set `$env:PYTHONUTF8 = "1"`. |
| Stale eval results after changing prompts/judge | The eval cache is keyed by `(prompt_hash, conv_id, judge_version, model)`; bump `--judge-version` to invalidate. |
