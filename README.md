# Self-Improving Prompt Optimization Pipeline

A closed-loop system that automatically evolves an LLM's system prompt (and few-shot examples) using production conversation logs, calibrated LLM-judges, and Bayesian/Genetic optimizers.

## Architecture Overview

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

## Features

1. **Phase 0 — Ingestion & Curation**:
   - PII Scrubbing: Multi-stage regex + spaCy NER redaction.
   - Near-duplicate detection: Embedding similarity search via FAISS.
   - Stratification: Balances splits by length, tool use, category, and difficulty.
2. **Phase 1 — Calibration & Studio**:
   - Human Labeling Studio: FastAPI side-by-side preference annotation.
   - Bias diagnostics: Detects length bias and position bias.
   - Judge versioning: Freezes prompt + model + calibration stats.
3. **Phase 2 — Eval Harness**:
   - High-throughput parallel replaying of conversation logs.
   - SQLite cache to bypass redundant LLM invocations.
   - Guardrail metrics: Kolmogorov-Smirnov length checks, refusal shifts.
4. **Phase 3 — Optimizers**:
   - **MIPROv2**: Bayesian optimization over instructions and demonstration exemplars.
   - **GEPA**: Genetic algorithms with reflective mutation (incorporates LLM judge rationale text).
5. **Phase 4 — Champion/Challenger Gate**:
   - Paired bootstrap confidence intervals (10,000 resamples).
   - Wilcoxon signed-rank significance testing.
6. **Phase 5 — Registry & Deployment**:
   - Immutable version registry.
   - Shadow-mode execution & staged Canary rollouts.
7. **Phase 6 — Continuous Feedback**:
   - Score distribution drift detection using KL divergence.

---

## Installation & Setup

1. Install project and dependencies in editable mode:
   ```bash
   pip install -e .[dev]
   ```
2. Copy environment template and fill in API keys:
   ```bash
   cp .env.example .env
   ```

## CLI Usage

The full closed loop, from raw logs to deployed prompt:

```bash
# 1. Ingest raw logs → PII-scrub → dedup → stratify → persist splits to ./data/splits
python -m prompt_opt.cli.main ingest --input-file data/sample_conversations.jsonl

# 2. Launch Human Labeling Studio (annotate pairs for judge calibration)
python -m prompt_opt.cli.main studio --port 8000
python -m prompt_opt.cli.main add-gold --conversation-id c1 --response-a "..." --response-b "..."

# 3. Calibrate judge against the gold annotated set (freezes judge if passed)
python -m prompt_opt.cli.main calibrate-judge

# 4. Run optimizer over persisted train/dev splits; registers best candidate
python -m prompt_opt.cli.main optimize --algorithm miprov2 --n-trials 5   # or: gepa

# 5. Evaluate challenger vs champion on the frozen test split (+ regression suite)
#    PASS → auto-promote; PENDING_HUMAN → review spot-check items, then approve
python -m prompt_opt.cli.main gate --challenger-hash <hash> [--skip-human-review]
python -m prompt_opt.cli.main gate-approve --reviewer alice [--reject]

# 6. Shadow-mode: serve champion, log challenger responses side-by-side
python -m prompt_opt.cli.main shadow --challenger-hash <hash> --n 5

# 7. Staged canary rollout with guardrail halts
python -m prompt_opt.cli.main canary start --challenger-hash <hash> --champion-hash <hash>
python -m prompt_opt.cli.main canary status | advance | breach | rollback

# 8. Live-metric rollback monitor (auto-reverts registry champion on breach)
python -m prompt_opt.cli.main monitor --error-rate 0.06

# 9. Drift detection over production score distributions
python -m prompt_opt.cli.main drift-check --baseline-file scores.json --scores-file recent.json

# 10. Continuous feedback loop (cron-driven ingest + re-optimization)
python -m prompt_opt.cli.main loop          # scheduled via reingest_cron / reoptimize_cron
python -m prompt_opt.cli.main loop --once   # single cycle
```

---

## Testing

Run unit and integration test suites:
```bash
pytest tests/
```
