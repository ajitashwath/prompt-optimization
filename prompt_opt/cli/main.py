"""
Typer-based CLI for the self-improving prompt optimization pipeline.

Commands:
  - `ingest`: Parse, scrub, deduplicate, stratify raw logs, persist splits.
  - `add-gold`: Manually add candidate pairs to the human labeling studio.
  - `calibrate-judge`: Evaluate agreement metrics between LLM and humans.
  - `optimize`: Evolve prompt using MIPROv2 or GEPA over the persisted dev split.
  - `gate`: Evaluate challenger vs champion on the frozen test split.
  - `gate-review`: Show the spot-check items of a gate result awaiting human review.
  - `gate-approve`: Record human spot-check decision on a pending gate result.
  - `promote`: Promote a gate-passed version to champion.
  - `shadow`: Replay conversations through champion + challenger (log-only).
  - `canary`: Manage staged traffic rollout (start/status/advance/breach/rollback).
  - `monitor`: Run rollback-trigger checks against live metrics.
  - `score-shadow`: Judge-score the responses in a shadow log (input for `drift-check`).
  - `drift-check`: Detect judge-score / guardrail distribution drift.
  - `loop`: Start the scheduled continuous feedback loop.
  - `studio`: Spin up the FastAPI human labeling server.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from prompt_opt.config import PROVIDER_KEY_VARS, has_provider_key, settings

app = typer.Typer(help="Self-Improving Prompt Optimization Pipeline CLI")
console = Console()

GATE_RESULTS_DIR = Path("./data/gate_results")
REGRESSION_SUITE_PATH = Path("./data/regression_suite.jsonl")
DRIFT_STATE_PATH = Path("./data/drift_baseline.json")
RECENT_SCORES_PATH = Path("./data/recent_scores.json")
SHADOW_LOG_PATH = Path("./data/shadow_logs/shadow_log.jsonl")


# ─── Shared helpers ──────────────────────────────────────────────────────────


def _registry():
    from prompt_opt.registry.prompt_registry import PromptRegistry

    # PromptRegistry uses the sync SQLAlchemy engine; normalize async URLs.
    db_url = settings.database_url.replace("sqlite+aiosqlite", "sqlite")
    return PromptRegistry(db_url=db_url)


def _require_api_key(command: str) -> None:
    """
    Pre-flight check: LLM-bound commands need at least one provider key.

    Keys may come from the shell environment or from a `.env` file in the
    current directory (exported by `prompt_opt.config` at import). Placeholder
    values such as the `sk-...` in `.env.example` do not count.

    Fails fast with actionable guidance instead of letting litellm retry
    and crash with authentication tracebacks deep inside the pipeline.
    """
    if has_provider_key():
        return
    console.print(
        f"[bold red]No LLM provider API key found - "
        f"`{command}` requires live model calls.[/bold red]\n"
        f"Set one of: {', '.join(PROVIDER_KEY_VARS)}\n"
        f"  in your shell, or in a .env file in the current directory:\n"
        f"  cp .env.example .env   # then replace the placeholder with your key"
    )
    raise typer.Exit(1)


def _dataset_store():
    from prompt_opt.storage.dataset_store import DatasetStore

    return DatasetStore()


def _build_harness(judge_version: str = "", allow_uncalibrated: bool = False):
    """
    Build the eval harness around the frozen, calibrated judge.

    The judge (model + rubric) comes from the judge registry, so scores always come from
    the yardstick that `calibrate-judge` validated; its version id keys the eval cache.
    `judge_version` picks a specific frozen version (default: the latest). Without any
    frozen judge, evaluation refuses to run unless `allow_uncalibrated` is set.
    """
    from prompt_opt.eval.harness import EvalHarness
    from prompt_opt.judge.judge import LLMJudge
    from prompt_opt.judge.registry import JudgeMismatchError, JudgeNotFoundError, JudgeRegistry

    try:
        judge, frozen = JudgeRegistry().load_judge(judge_version or None)
    except JudgeMismatchError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(1) from exc
    except JudgeNotFoundError as exc:
        if judge_version or not allow_uncalibrated:
            console.print(
                f"[bold red]{exc}.[/bold red] Run `prompt-opt calibrate-judge` first "
                f"(or pass --allow-uncalibrated to evaluate with an unvalidated judge)."
            )
            raise typer.Exit(1) from exc
        console.print(
            "[bold yellow]WARNING: no calibrated judge; scores come from an unvalidated "
            f"judge ({settings.judge_model}).[/bold yellow]"
        )
        return EvalHarness(LLMJudge(), judge_version=f"uncalibrated:{settings.judge_model}")

    console.print(f"Using calibrated judge [cyan]{frozen.version_id}[/cyan] ({frozen.model})")
    return EvalHarness(judge, judge_version=frozen.version_id)


def _load_regression_cases() -> list:
    """The regression suite from data/regression_suite.jsonl, or [] if there is none."""
    from prompt_opt.eval.guardrails import load_regression_suite

    if REGRESSION_SUITE_PATH.exists():
        return load_regression_suite(REGRESSION_SUITE_PATH)
    return []


def _resolve_candidate(registry, hash_str: str, role: str):
    """Resolve a registered PromptVersion by content hash into a PromptCandidate."""
    from prompt_opt.eval.harness import PromptCandidate

    version = registry.get_by_content_hash(hash_str)
    if version is None:
        console.print(
            f"[bold red]{role} hash {hash_str!r} not found in registry.[/bold red] "
            f"Run `prompt-opt optimize` first, or check `version_id` vs `content_hash`."
        )
        raise typer.Exit(1)
    return PromptCandidate(
        system_prompt=version.system_prompt,
        few_shot_demos=version.few_shot_demos,
    )


def _gate_passed(version) -> bool:
    """True if the registry holds a PASS gate verdict for this version."""
    return (version.gate_result or {}).get("decision") == "PASS"


def _review_path(result_path: Path) -> Path:
    """Where the human review of a gate result is recorded (the result itself is never edited)."""
    return result_path.with_name(result_path.stem + ".review.json")


def _registry_verdict(data: dict) -> dict:
    """The verdict as stored in the registry: without the spot-check texts (test-split content)."""
    return {k: v for k, v in data.items() if k != "spot_check_items"}


def _load_gate_result(result_file: Path | None) -> tuple[Path, dict]:
    """An explicit result file, or the latest gate result still awaiting human review."""
    if result_file is None:
        pending = []
        if GATE_RESULTS_DIR.exists():
            for path in sorted(GATE_RESULTS_DIR.glob("gate_*.json")):
                if path.name.endswith(".review.json") or _review_path(path).exists():
                    continue
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                if data.get("decision") == "PENDING_HUMAN":
                    pending.append((path, data))
        if not pending:
            console.print("[bold red]No pending gate results found.[/bold red]")
            raise typer.Exit(1)
        return pending[-1]
    if not result_file.exists():
        console.print(f"[bold red]Gate result file not found: {result_file}[/bold red]")
        raise typer.Exit(1)
    with open(result_file, encoding="utf-8") as fh:
        return result_file, json.load(fh)


def _snippet(text: str | None, limit: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _print_spot_checks(items: list[dict], *, full: bool) -> None:
    """Show the human spot-check items: a compact table, or the full texts to review."""
    if not items:
        console.print("[yellow]No spot-check items recorded for this result.[/yellow]")
        return
    if not full:
        table = Table(title="Human Spot-Check Items (full texts: `prompt-opt gate-review`)")
        for col in ("Example", "Outcome", "Chal/Champ", "Query", "Challenger", "Champion"):
            table.add_column(col)
        for item in items:
            champ = item.get("champion_score")
            table.add_row(
                escape(str(item["example_id"])),
                str(item.get("outcome")),
                f"{item['challenger_score']:.2f}/{'-' if champ is None else f'{champ:.2f}'}",
                escape(_snippet(item.get("user_query"), 50)),
                escape(_snippet(item.get("challenger_response"), 50)),
                escape(_snippet(item.get("champion_response"), 50)),
            )
        console.print(table)
        return
    for i, item in enumerate(items, 1):
        champ = item.get("champion_score")
        console.rule(
            f"[{i}/{len(items)}] {escape(str(item['example_id']))}  "
            f"category={escape(str(item.get('category')))}  outcome={item.get('outcome')}  "
            f"scores challenger={item['challenger_score']:.2f} "
            f"champion={'-' if champ is None else f'{champ:.2f}'}"
        )
        for label, key, default in (
            ("User", "user_query", "(not available)"),
            ("Challenger response", "challenger_response", ""),
            ("Champion response", "champion_response", ""),
            ("Judge rationale", "judge_rationale", ""),
        ):
            console.print(f"[bold]{label}:[/bold]", escape(item.get(key) or default))


def _print_ingest_report(report) -> None:
    """Show what ingestion did to the input: nothing is dropped without being reported."""
    if report is None:
        return
    table = Table(title="Ingestion Report")
    table.add_column("Item")
    table.add_column("Count", style="magenta")
    if report.n_lines:
        table.add_row("input lines", str(report.n_lines))
    if report.n_unreadable_lines:
        table.add_row("[yellow]unreadable lines (skipped)[/yellow]", str(report.n_unreadable_lines))
    for reason, n in sorted(report.skipped.items()):
        table.add_row(f"[yellow]skipped: {escape(reason)}[/yellow]", str(n))
    table.add_row("duplicates (exact text)", str(report.n_duplicates_exact))
    method = report.dedup_method or "n/a"
    table.add_row(f"duplicates (near, {method})", str(report.n_duplicates_near))
    if report.n_duplicates_of_reference:
        table.add_row(
            "  of which copies of frozen test items", str(report.n_duplicates_of_reference)
        )
    if report.redactions:
        detail = ", ".join(f"{label}={n}" for label, n in sorted(report.redactions.items()))
        table.add_row("PII redactions", f"{sum(report.redactions.values())} ({detail})")
    table.add_row("conversations kept", str(report.n_kept))
    console.print(table)
    if report.dedup_method == "lexical":
        console.print(
            "[yellow]Near-duplicate detection used the offline lexical fallback (embeddings were "
            "unavailable): it only catches near-identical text, not paraphrases.[/yellow]"
        )


def _print_staging_hint(challenger_hash: str) -> None:
    console.print(
        "The champion is unchanged. Stage the rollout, then it is promoted when the "
        "canary completes:\n"
        f"  prompt-opt shadow --challenger-hash {challenger_hash}\n"
        f"  prompt-opt canary start --challenger-hash {challenger_hash}\n"
        f"Or promote directly:  prompt-opt promote --hash {challenger_hash}"
    )


# ─── Pipeline commands ───────────────────────────────────────────────────────


@app.command()
def ingest(
    input_file: Path = typer.Option(..., exists=True, help="Path to raw JSONL production logs"),
    train_ratio: float = 0.60,
    dev_ratio: float = 0.20,
    test_ratio: float = 0.20,
    use_ner: bool = typer.Option(
        False, help="Enable spaCy NER scrubbing (slower; also on when PII_USE_NER=true)"
    ),
    refresh_test: bool = typer.Option(
        False,
        "--refresh-test",
        help="Rebuild the frozen test split (archives the old one; past gate results "
        "become non-comparable)",
    ),
):
    """PII-scrub, deduplicate, and split raw logs into persisted train/dev/test sets."""
    from prompt_opt.ingestion.ingestor import Ingestor
    from prompt_opt.labeling.rubric import DEFAULT_RUBRIC  # noqa: F401  (env sanity)

    if min(train_ratio, dev_ratio, test_ratio) < 0 or abs(
        train_ratio + dev_ratio + test_ratio - 1.0
    ) > 1e-6:
        console.print(
            f"[bold red]Split ratios must be non-negative and sum to 1 "
            f"(got {train_ratio} + {dev_ratio} + {test_ratio}).[/bold red]"
        )
        raise typer.Exit(1)

    store = _dataset_store()
    test_kept = store.exists("test") and not refresh_test
    # New conversations that duplicate a frozen test item must not leak into train/dev
    reference = store.load_split("test") if test_kept else None

    console.print(f"[bold blue]Starting ingestion for {input_file}...[/bold blue]")
    ingestor = Ingestor(use_ner=use_ner or settings.pii_use_ner)
    result = ingestor.ingest_file(
        input_file,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        reference=reference,
    )
    _print_ingest_report(ingestor.last_report)

    counts = store.save(result, refresh_test=refresh_test)
    if test_kept:
        console.print(
            f"[yellow]Existing test split kept frozen ({counts['test']} records); "
            f"its conversations were removed from train/dev. "
            f"Use --refresh-test to rebuild it.[/yellow]"
        )
    elif refresh_test:
        console.print(
            "[yellow]Test split rebuilt; the previous one was archived. "
            "Earlier gate results are not comparable with new ones.[/yellow]"
        )

    table = Table(title="Dataset Split Statistics")
    table.add_column("Split", style="cyan")
    table.add_column("Count", style="magenta")
    for name in ("train", "dev", "test"):
        table.add_row(name, str(counts[name]))
    console.print(table)
    console.print(f"[bold green]Splits persisted to {store.base_path}[/bold green]")


@app.command()
def add_gold(
    conversation_id: str = typer.Option(..., help="Source conversation ID"),
    response_a: str = typer.Option(..., help="Content of Response A"),
    response_b: str = typer.Option(..., help="Content of Response B"),
):
    """Add a new response comparison pair to the annotation queue."""
    from prompt_opt.labeling.gold_store import GoldStore
    from prompt_opt.labeling.rubric import DEFAULT_RUBRIC

    if not _dataset_store().find_conversations({conversation_id}):
        console.print(
            f"[yellow]Conversation {conversation_id!r} was not found in the train/dev "
            f"splits: annotators cannot see its context and calibration will skip this "
            f"item. Run `ingest` first, or check the id.[/yellow]"
        )

    gold_store = GoldStore()
    item_id = gold_store.add_item(
        conversation_id=conversation_id,
        response_a=response_a,
        response_b=response_b,
        rubric_version=DEFAULT_RUBRIC.version,
    )
    console.print(
        f"[bold green]Successfully added item {item_id} to "
        f"annotation queue.[/bold green]"
    )


@app.command()
def calibrate_judge():
    """Run agreement calibration diagnostics over human-labeled gold set."""
    from prompt_opt.judge.calibrator import JudgeCalibrator
    from prompt_opt.judge.judge import LLMJudge
    from prompt_opt.judge.registry import JudgeRegistry
    from prompt_opt.labeling.gold_store import GoldStore
    from prompt_opt.labeling.rubric import DEFAULT_RUBRIC

    gold_store = GoldStore()
    if not gold_store.get_labeled(rubric_version=DEFAULT_RUBRIC.version):
        console.print(
            "[bold yellow]No human-labeled items available. "
            "Please annotate items in the Studio first.[/bold yellow]"
        )
        return
    _require_api_key("calibrate-judge")

    # The judge must see the same conversation context at calibration time as at eval time.
    gold_ids = {
        g.conversation_id for g in gold_store.get_labeled(rubric_version=DEFAULT_RUBRIC.version)
    }
    conversations = _dataset_store().find_conversations(gold_ids)
    if not conversations:
        console.print(
            "[bold red]None of the labeled items have their conversation in the train/dev "
            "splits.[/bold red] Calibration needs the conversation context: run "
            "`prompt-opt ingest` on the logs the gold items came from."
        )
        raise typer.Exit(1)
    missing = gold_ids - conversations.keys()
    if missing:
        console.print(
            f"[yellow]{len(missing)} conversation(s) not found in train/dev; their gold "
            f"items are skipped.[/yellow]"
        )

    judge = LLMJudge()
    calibrator = JudgeCalibrator(judge, gold_store)

    console.print("[bold blue]Starting Judge Calibration...[/bold blue]")
    report = calibrator.calibrate(
        rubric_version=DEFAULT_RUBRIC.version, conversations=conversations
    )

    if report.n_items == 0:
        console.print(
            "[bold yellow]No human-labeled items available. "
            "Please annotate items in the Studio first.[/bold yellow]"
        )
        return

    console.print(f"Items Evaluated: {report.n_items}")
    console.print(
        f"Cohen's Kappa (Agreement): [bold magenta]{report.kappa:.3f}[/bold magenta] "
        f"({report.kappa_interpretation})"
    )
    console.print(f"Spearman Correlation: [bold magenta]{report.spearman_rho:.3f}[/bold magenta]")
    if report.n_judge_failures or report.n_skipped_no_context:
        console.print(
            f"Unusable items: {report.n_judge_failures} judge failures, "
            f"{report.n_skipped_no_context} without conversation context"
        )
    if report.bias is not None and report.bias.order_inconsistency_rate is not None:
        console.print(
            f"Judge order inconsistency: {report.bias.order_inconsistency_rate:.0%} of items "
            f"(verdict changed when A/B were swapped)"
        )

    if report.passed:
        console.print("[bold green]Calibration Passed! LLM Judge is calibrated.[/bold green]")
        frozen = JudgeRegistry().freeze(
            settings.judge_model, DEFAULT_RUBRIC.version, report, rubric=DEFAULT_RUBRIC
        )
        console.print(
            f"Frozen as [cyan]{frozen.version_id}[/cyan]: `optimize` and `gate` now score "
            f"with this judge."
        )
    else:
        console.print("[bold red]Calibration Failed![/bold red]")
        for reason in report.failure_reasons:
            console.print(f" - {reason}", style="red")


@app.command()
def optimize(
    algorithm: str = typer.Option("miprov2", help="Optimization algorithm: 'miprov2' or 'gepa'"),
    max_usd: float = typer.Option(10.0, help="Max USD budget cap for LLM calls"),
    n_trials: int = typer.Option(10, help="Number of trials / generations"),
    minibatch_size: int = typer.Option(
        0, "--minibatch-size", help="Dev examples per search step (0 = auto: min(dev size, 16))"
    ),
    judge_version: str = typer.Option(
        "", help="Frozen judge version to score with (default: latest calibrated judge)"
    ),
    allow_uncalibrated: bool = typer.Option(
        False, "--allow-uncalibrated", help="Evaluate with an unvalidated judge if none is frozen"
    ),
):
    """
    Run prompt search over the persisted train/dev splits and register the winner.

    Every candidate finalist and the current prompt are scored on the full dev split under
    the same conditions; a candidate is registered only if it beats the current prompt.
    """
    _require_api_key("optimize")
    from prompt_opt.eval.harness import EvaluationError
    from prompt_opt.optimizer.base import OptimizationError, default_minibatch_size
    from prompt_opt.optimizer.budget import BudgetController
    from prompt_opt.optimizer.gepa import GEPA
    from prompt_opt.optimizer.miprov2 import MIPROv2

    if algorithm.lower() not in ("miprov2", "gepa"):
        console.print(f"[bold red]Unknown algorithm: {algorithm}[/bold red]")
        raise typer.Exit(1)

    store = _dataset_store()
    if not store.exists("train") or not store.exists("dev"):
        console.print(
            "[bold red]No persisted train/dev splits found. "
            "Run `prompt-opt ingest --input-file ...` first.[/bold red]"
        )
        raise typer.Exit(1)

    train_convs = store.load_split("train")
    dev_convs = store.load_split("dev")
    if not dev_convs:
        console.print(
            "[bold red]The dev split is empty; there is nothing to score candidates on. "
            "Ingest more data.[/bold red]"
        )
        raise typer.Exit(1)

    console.print(f"[bold blue]Initializing {algorithm} optimization run "
                  f"(train={len(train_convs)}, dev={len(dev_convs)})...[/bold blue]")

    # Start from the current champion if one exists
    from prompt_opt.eval.harness import PromptCandidate

    registry = _registry()
    champ = registry.get_champion()
    if champ:
        console.print(f"Seeding search from current champion {champ.version_id}")
        initial_prompt = PromptCandidate(
            system_prompt=champ.system_prompt,
            few_shot_demos=champ.few_shot_demos,
        )
    else:
        initial_prompt = PromptCandidate(system_prompt="You are a helpful assistant.")

    harness = _build_harness(judge_version, allow_uncalibrated)
    budget = BudgetController(max_usd_spend=max_usd)

    mb = min(minibatch_size or default_minibatch_size(len(dev_convs)), len(dev_convs))
    regression_cases = _load_regression_cases()
    if algorithm.lower() == "miprov2":
        opt = MIPROv2(n_trials=n_trials, minibatch_size=mb, regression_cases=regression_cases)
    else:
        opt = GEPA(max_generations=n_trials, minibatch_size=mb, regression_cases=regression_cases)

    try:
        results = opt.optimize(initial_prompt, train_convs, dev_convs, harness, budget)
    except (EvaluationError, OptimizationError) as exc:
        console.print(f"[bold red]Optimization failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"Budget used: {budget.summary()}")
    best_cand, summary = results[0]
    baseline = next(
        (s for c, s in results if c.content_hash == initial_prompt.content_hash), None
    )

    if baseline is None or baseline.n_examples == 0:
        console.print(
            "[bold red]The budget ran out before the current prompt could be evaluated on "
            "the dev split; nothing to compare against. Raise --max-usd or "
            "MAX_LLM_CALLS_PER_RUN.[/bold red]"
        )
        raise typer.Exit(1)

    if best_cand.content_hash == initial_prompt.content_hash:
        console.print(
            f"[bold yellow]No candidate beat the current prompt on the dev split "
            f"(score {baseline.mean_score:.4f}); nothing registered.[/bold yellow]"
        )
        return

    optimizer_run_id = f"{algorithm}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    version_id = registry.register(
        system_prompt=best_cand.system_prompt,
        few_shot_demos=best_cand.few_shot_demos,
        optimizer_run_id=optimizer_run_id,
        eval_scores=summary.to_dict(),
    )

    console.print("[bold green]Optimization complete![/bold green]")
    console.print(f"Registered Version: [cyan]{version_id}[/cyan]")
    console.print(f"Best Candidate Hash: [cyan]{best_cand.content_hash}[/cyan]")
    console.print(
        f"Dev score: [magenta]{summary.mean_score:.4f}[/magenta] "
        f"(current prompt: {baseline.mean_score:.4f}, n={summary.n_examples})"
    )
    if not summary.guardrail_passed:
        console.print(
            "[yellow]Note: this candidate failed a guardrail against the current prompt, "
            "and no finalist passed while scoring higher.[/yellow]"
        )
    console.print("\nNext step: promote via the gate:")
    console.print(
        f"  prompt-opt gate --challenger-hash {best_cand.content_hash} "
        f"--champion-hash {initial_prompt.content_hash}"
    )


@app.command()
def gate(
    challenger_hash: str = typer.Option(..., help="Content hash of challenger prompt candidate"),
    champion_hash: str = typer.Option(
        "", help="Content hash of current champion (default: registry champion)"
    ),
    skip_human_review: bool = typer.Option(
        False, "--skip-human-review", help="Auto-PASS without human spot-check"
    ),
    auto_promote: bool = typer.Option(
        False,
        "--auto-promote/--no-auto-promote",
        help="Promote to champion immediately on PASS, skipping shadow/canary staging",
    ),
    allow_no_regression_suite: bool = typer.Option(
        False,
        "--allow-no-regression-suite",
        help="Do not fail the gate when no regression suite ran (default: it must run and pass)",
    ),
    judge_version: str = typer.Option(
        "", help="Frozen judge version to score with (default: latest calibrated judge)"
    ),
    allow_uncalibrated: bool = typer.Option(
        False, "--allow-uncalibrated", help="Evaluate with an unvalidated judge if none is frozen"
    ),
):
    """
    Evaluate challenger vs champion on the frozen test split.

    A PASS records the verdict on the registered version but does NOT change the
    champion: stage it with `shadow` / `canary`, which promotes on completion, or
    run `promote` yourself. `--auto-promote` skips staging.
    """
    _require_api_key("gate")
    from prompt_opt.eval.harness import EvaluationError
    from prompt_opt.gate.champion_gate import ChampionGate, GateCriteria

    registry = _registry()
    store = _dataset_store()
    if not store.exists("test"):
        console.print(
            "[bold red]No persisted test split. Run `prompt-opt ingest` first.[/bold red]"
        )
        raise typer.Exit(1)

    challenger = _resolve_candidate(registry, challenger_hash, "Challenger")

    if champion_hash:
        champion = _resolve_candidate(registry, champion_hash, "Champion")
    else:
        champ_row = registry.get_champion()
        if champ_row is None:
            console.print(
            "[bold red]No champion in registry and no --champion-hash given.[/bold red]"
        )
            raise typer.Exit(1)
        champion = _resolve_candidate(registry, champ_row.content_hash, "Champion")

    test_convs = store.load_split("test")
    console.print(f"[bold blue]Running gate: {len(test_convs)} frozen test examples...[/bold blue]")

    regression_cases = _load_regression_cases()

    criteria = GateCriteria(
        require_human_spot_check=not skip_human_review,
        require_regression_suite=not allow_no_regression_suite,
    )
    harness = _build_harness(judge_version, allow_uncalibrated)
    cg = ChampionGate(harness, criteria=criteria, regression_cases=regression_cases)

    try:
        # Replayed through the challenger exactly as evaluated (system prompt + demos)
        regression_pairs = (
            harness.run_regression(challenger, regression_cases) if regression_cases else None
        )
        result = cg.evaluate(
            challenger, champion, test_convs, regression_responses=regression_pairs
        )
    except EvaluationError as exc:
        console.print(f"[bold red]Gate could not be evaluated:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    # Persist the gate result. It is written once and never edited: a human review is
    # recorded next to it (`<result>.review.json`).
    GATE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    result_path = GATE_RESULTS_DIR / f"gate_{stamp}_{result.challenger_hash}.json"
    payload = result.to_dict()
    champ_row = registry.get_champion()
    payload["champion_version_id"] = champ_row.version_id if champ_row else None
    payload["result_path"] = str(result_path)
    with open(result_path, "x", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    # Attach the verdict to the registry entry (whatever the decision) so that
    # `promote` / `canary start` can verify the gate was passed.
    challenger_version = registry.get_by_content_hash(challenger_hash)
    if challenger_version is not None:
        registry.attach_gate_result(challenger_version.version_id, _registry_verdict(payload))

    # ── Report ───────────────────────────────────────────────────────────────
    checks_table = Table(title=f"Gate Criteria Checks: {result.decision}")
    checks_table.add_column("Criterion")
    checks_table.add_column("Value")
    checks_table.add_column("Threshold")
    checks_table.add_column("Passed")
    for c in result.criteria_checks:
        checks_table.add_row(
            str(c["criterion"]), str(c["value"]), str(c["threshold"]), str(c["passed"])
        )
    console.print(checks_table)

    _print_spot_checks(result.spot_check_items, full=False)

    console.print(f"Gate result saved to {result_path}")

    if result.decision == "FAIL":
        for r in result.failure_reasons:
            console.print(f" - Reason: {r}", style="red")
        raise typer.Exit(1)

    if result.decision == "PENDING_HUMAN":
        console.print(
            f"[bold yellow]PENDING HUMAN REVIEW.[/bold yellow] Read the spot-check items:\n"
            f"  prompt-opt gate-review --result-file {result_path}\n"
            f"then record your decision:\n"
            f"  prompt-opt gate-approve --result-file {result_path} --reviewer <name>\n"
            f"(add --reject to reject instead)"
        )
        return

    console.print("[bold green]Gate PASSED![/bold green]")
    if auto_promote and challenger_version is not None:
        registry.promote(challenger_version.version_id, approver="gate-cli")
        console.print(
            f"[bold green]Promoted {challenger_version.version_id} to champion "
            f"(--auto-promote; no shadow/canary staging).[/bold green]"
        )
    else:
        _print_staging_hint(challenger_hash)


@app.command("gate-review")
def gate_review(
    result_file: Path = typer.Option(
        None, help="Path to saved gate result JSON (default: latest pending)"
    ),
):
    """Show the spot-check items of a gate result (full texts) so a human can review them."""
    result_file, data = _load_gate_result(result_file)
    console.print(
        f"Gate result {result_file.name}: challenger {data.get('challenger_hash')} vs champion "
        f"{data.get('champion_hash')} -> {data.get('decision')} "
        f"({data.get('n_paired', '?')} paired test examples)"
    )
    for check in data.get("criteria_checks", []):
        mark = "[green]ok[/green]" if check.get("passed") else "[red]FAIL[/red]"
        console.print(
            f"  {mark} {escape(str(check['criterion']))}: {escape(str(check['value']))} "
            f"(threshold: {escape(str(check['threshold']))})"
        )
    _print_spot_checks(data.get("spot_check_items") or [], full=True)
    console.print(
        f"\nDecide with: prompt-opt gate-approve --result-file {result_file} --reviewer <name> "
        f"[--reject]"
    )


@app.command("gate-approve")
def gate_approve(
    result_file: Path = typer.Option(
        None, help="Path to saved gate result JSON (default: latest pending)"
    ),
    reviewer: str = typer.Option(..., help="Name of the human reviewer"),
    notes: str = typer.Option("", help="Review notes"),
    reject: bool = typer.Option(
        False, "--reject", help="Reject the challenger instead of approving"
    ),
    promote_now: bool = typer.Option(
        False,
        "--promote",
        help="On approval, promote to champion immediately, skipping shadow/canary staging",
    ),
):
    """
    Record the human spot-check decision for a pending gate result.

    The gate result file is never modified: the decision is written next to it as
    `<result>.review.json` (one review per result). Approval records a PASS verdict on
    the registered version but does not change the champion; stage the rollout
    (`shadow` / `canary`) or pass `--promote` to promote right away. Read the items with
    `gate-review` first.
    """
    from prompt_opt.gate.champion_gate import GateResult

    result_file, data = _load_gate_result(result_file)
    review_path = _review_path(result_file)

    if review_path.exists():
        console.print(
            f"[bold red]This gate result was already reviewed ({review_path.name}).[/bold red]"
        )
        raise typer.Exit(1)
    if data.get("decision") != "PENDING_HUMAN":
        console.print(
            f"[bold red]Gate is not pending human review "
            f"(status={data['decision']}).[/bold red]"
        )
        raise typer.Exit(1)

    gate_result = GateResult(
        decision="PENDING_HUMAN",
        challenger_hash=data["challenger_hash"],
        champion_hash=data["champion_hash"],
        timestamp=data.get("timestamp", ""),
    )
    gate_result.record_human_review(
        reviewer=reviewer, notes=notes, approved=not reject
    )

    registry = _registry()
    champ_row = registry.get_champion()
    if champ_row is not None and champ_row.content_hash != data["champion_hash"]:
        console.print(
            f"[yellow]Note: the champion has changed since this gate ran (it compared against "
            f"{data['champion_hash']}, the current champion is {champ_row.content_hash}). "
            f"Consider re-running the gate.[/yellow]"
        )

    reviewed_at = datetime.utcnow().isoformat()
    review = {
        "result_file": result_file.name,
        "challenger_hash": data["challenger_hash"],
        "champion_hash": data["champion_hash"],
        "reviewer": reviewer,
        "notes": notes,
        "approved": not reject,
        "decision": gate_result.decision,
        "reviewed_at": reviewed_at,
    }
    with open(review_path, "x", encoding="utf-8") as fh:
        json.dump(review, fh, indent=2)

    verdict = {
        **data,
        "decision": gate_result.decision,
        "failure_reasons": [*data.get("failure_reasons", []), *gate_result.failure_reasons],
        "human_approved": gate_result.human_approved,
        "human_reviewer": reviewer,
        "human_notes": notes,
        "reviewed_at": reviewed_at,
        "review_path": str(review_path),
    }
    version = registry.get_by_content_hash(data["challenger_hash"])
    if version is not None:
        registry.attach_gate_result(version.version_id, _registry_verdict(verdict))

    if reject:
        console.print(f"[bold red]Gate rejected by {reviewer}.[/bold red]")
        return

    if version is None:
        console.print(
            f"[bold red]Challenger {data['challenger_hash']} is not registered; "
            f"cannot record approval.[/bold red]"
        )
        raise typer.Exit(1)

    if promote_now:
        registry.promote(version.version_id, approver=reviewer)
        console.print(
            f"[bold green]Approved by {reviewer}. Promoted {version.version_id} to "
            f"champion (--promote; no shadow/canary staging).[/bold green]"
        )
    else:
        console.print(f"[bold green]Approved by {reviewer}.[/bold green]")
        _print_staging_hint(data["challenger_hash"])


@app.command()
def promote(
    version_hash: str = typer.Option(..., "--hash", help="Content hash of the version to promote"),
    approver: str = typer.Option("cli", help="Name recorded in the champion audit log"),
    force: bool = typer.Option(
        False, "--force", help="Promote even without a recorded gate PASS (e.g. bootstrapping)"
    ),
):
    """Promote a registered version to champion (requires a gate PASS unless --force)."""
    registry = _registry()
    version = registry.get_by_content_hash(version_hash)
    if version is None:
        console.print(f"[bold red]Hash {version_hash!r} not found in registry.[/bold red]")
        raise typer.Exit(1)
    if version.is_champion:
        console.print(f"[yellow]{version.version_id} is already the champion.[/yellow]")
        return
    if not force and not _gate_passed(version):
        decision = (version.gate_result or {}).get("decision", "not gated")
        console.print(
            f"[bold red]{version.version_id} has no gate PASS on record "
            f"(status: {decision}).[/bold red] Run `prompt-opt gate` first, "
            f"or pass --force."
        )
        raise typer.Exit(1)
    registry.promote(version.version_id, approver=approver)
    console.print(f"[bold green]Promoted {version.version_id} to champion.[/bold green]")


# ─── Deployment commands ─────────────────────────────────────────────────────


@app.command()
def shadow(
    challenger_hash: str = typer.Option(..., help="Content hash of challenger prompt"),
    n: int = typer.Option(5, help="Number of conversations to shadow"),
    split: str = typer.Option("dev", help="Which persisted split to replay (train or dev)"),
):
    """Run challenger alongside champion on sample traffic; serve champion, log both."""
    _require_api_key("shadow")
    from prompt_opt.deployment.shadow import ShadowRunner
    from prompt_opt.eval.harness import replay_turns

    if split not in ("train", "dev"):
        console.print(
            f"[bold red]Cannot shadow the {split!r} split: choose 'train' or 'dev' "
            f"(the test split is reserved for the gate).[/bold red]"
        )
        raise typer.Exit(1)

    registry = _registry()
    challenger = _resolve_candidate(registry, challenger_hash, "Challenger")
    champ_row = registry.get_champion()
    if champ_row is None:
        console.print("[bold red]No champion in registry.[/bold red]")
        raise typer.Exit(1)
    if champ_row.content_hash == challenger.content_hash:
        console.print(
            "[bold red]Challenger is already the champion; nothing to shadow.[/bold red]"
        )
        raise typer.Exit(1)
    champion = _resolve_candidate(registry, champ_row.content_hash, "Champion")

    convs = _dataset_store().load_split(split)[:n]
    if not convs:
        console.print(f"[bold red]No conversations available in {split!r} split.[/bold red]")
        raise typer.Exit(1)

    runner = ShadowRunner(challenger=challenger, champion=champion)
    console.print(f"[bold blue]Shadowing {len(convs)} conversations "
                  f"(serving={champ_row.version_id}, "
                  f"logging={challenger.content_hash})...[/bold blue]")

    lat_champ: list[float] = []
    lat_chal: list[float] = []
    champ_errors = chal_errors = 0
    for conv in convs:
        # Same turns the eval harness replays: no system/tool messages, and without the
        # logged assistant reply (which is what the prompts are being asked to produce)
        res = runner.run_request(replay_turns(conv))
        if res.champion_error:
            champ_errors += 1
        else:
            lat_champ.append(res.latency_champion_ms)
        if res.challenger_error:
            chal_errors += 1
        else:
            lat_chal.append(res.latency_challenger_ms)

    def mean_ms(values: list[float]) -> str:
        return f"{sum(values) / len(values):.0f}" if values else "n/a"

    table = Table(title="Shadow Run Summary (latency over successful calls)")
    table.add_column("Metric")
    table.add_column("Champion")
    table.add_column("Challenger")
    table.add_row("mean latency (ms)", mean_ms(lat_champ), mean_ms(lat_chal))
    table.add_row("errors", str(champ_errors), str(chal_errors))
    console.print(table)
    console.print(f"[green]Shadow log written to {runner._log_file}[/green]")
    if champ_errors or chal_errors:
        console.print(
            "[yellow]Some calls failed; the log records each error and an empty response "
            "for it.[/yellow]"
        )


@app.command()
def canary(
    action: str = typer.Argument(..., help="start | status | advance | breach | rollback"),
    challenger_hash: str = typer.Option("", help="[start] Challenger content hash"),
    champion_hash: str = typer.Option(
        "", help="[start] Champion content hash (default: registry champion)"
    ),
    reason: str = typer.Option("", help="[rollback] Reason string"),
    state_path: str = typer.Option("./data/canary_state.json", help="Canary state file"),
    force: bool = typer.Option(
        False, "--force", help="[start] Start even without a recorded gate PASS"
    ),
):
    """
    Manage staged canary rollout of a challenger prompt.

    The registry champion is untouched while the canary runs; when the final
    stage completes (`advance`), the challenger is promoted to champion.
    """
    from prompt_opt.deployment.canary import CanaryController

    if action == "start":
        if not challenger_hash:
            console.print("[bold red]--challenger-hash is required for start.[/bold red]")
            raise typer.Exit(1)

        registry = _registry()
        challenger_version = registry.get_by_content_hash(challenger_hash)
        if challenger_version is None:
            console.print(
                f"[bold red]Challenger hash {challenger_hash!r} not found in registry."
                f"[/bold red]"
            )
            raise typer.Exit(1)
        if not champion_hash:
            champ_row = registry.get_champion()
            if champ_row is None:
                console.print(
                    "[bold red]No champion in registry and no --champion-hash given."
                    "[/bold red]"
                )
                raise typer.Exit(1)
            champion_hash = champ_row.content_hash
        elif registry.get_by_content_hash(champion_hash) is None:
            console.print(
                f"[bold red]Champion hash {champion_hash!r} not found in registry.[/bold red]"
            )
            raise typer.Exit(1)
        if challenger_hash == champion_hash:
            console.print("[bold red]Challenger and champion are the same prompt.[/bold red]")
            raise typer.Exit(1)
        if not force and not _gate_passed(challenger_version):
            console.print(
                f"[bold red]{challenger_version.version_id} has no gate PASS on record."
                f"[/bold red] Run `prompt-opt gate` first, or pass --force."
            )
            raise typer.Exit(1)
        try:
            existing = CanaryController.load(state_path)
        except FileNotFoundError:
            existing = None
        if existing is not None and existing.state.status == "active":
            console.print(
                f"[bold red]A canary for {existing.state.challenger_hash} is already "
                f"active.[/bold red] Run `prompt-opt canary rollback` before starting "
                f"another."
            )
            raise typer.Exit(1)

        ctrl = CanaryController(
            challenger_hash=challenger_hash,
            champion_hash=champion_hash,
            state_path=state_path,
        )
        console.print(
            f"[bold green]Canary started at {ctrl.current_stage.traffic_pct}% traffic."
            f"[/bold green]"
        )
        return

    try:
        ctrl = CanaryController.load(state_path)
    except FileNotFoundError:
        console.print(
            f"[bold red]No canary state at {state_path}. Run `canary start` first.[/bold red]"
        )
        raise typer.Exit(1)

    if action == "status":
        table = Table(title=f"Canary Status ({ctrl.state.status})")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("champion", ctrl.state.champion_hash)
        table.add_row("challenger", ctrl.state.challenger_hash)
        table.add_row("stage", f"{ctrl.state.stage_idx + 1}/{len(ctrl.state.stages)}")
        table.add_row("traffic_pct", str(ctrl.current_stage.traffic_pct))
        table.add_row("soak_elapsed_s", f"{ctrl.current_stage.elapsed_seconds:.0f}")
        table.add_row("soak_remaining_s", f"{ctrl.current_stage.remaining_seconds:.0f}")
        table.add_row("guardrail_breaches", str(ctrl.current_stage.guardrail_breaches))
        table.add_row("halt_reason", ctrl.state.halt_reason or "-")
        console.print(table)
    elif action == "advance":
        advanced = ctrl.maybe_advance()
        if advanced and ctrl.state.status == "complete":
            registry = _registry()
            version = registry.get_by_content_hash(ctrl.state.challenger_hash)
            if version is None:
                console.print(
                    f"[bold red]Canary complete, but challenger {ctrl.state.challenger_hash} "
                    f"is not registered; cannot promote.[/bold red]"
                )
                raise typer.Exit(1)
            registry.promote(version.version_id, approver="canary")
            console.print(
                f"[bold green]Canary complete. Promoted {version.version_id} to "
                f"champion.[/bold green]"
            )
            return
        pct = ctrl.current_stage.traffic_pct if ctrl.state.status == "active" else 0
        if advanced:
            console.print(f"[bold green]Advanced. Current stage traffic: {pct}%[/bold green]")
        else:
            console.print(
                f"[yellow]Not advanced. Status={ctrl.state.status}, pct={pct}%, "
                f"soak remaining: {ctrl.current_stage.remaining_seconds:.0f}s[/yellow]"
            )
    elif action == "breach":
        ctrl.record_guardrail_breach()
        console.print(
            f"[yellow]Breach recorded. Stage breaches: "
            f"{ctrl.current_stage.guardrail_breaches}. Status: {ctrl.state.status}[/yellow]"
        )
    elif action == "rollback":
        ctrl.rollback(reason or "Manual rollback via CLI")
        console.print(f"[bold red]Canary rolled back: {reason or 'manual'}[/bold red]")
    else:
        console.print(
            f"[bold red]Unknown action {action!r}. "
            f"Use start|status|advance|breach|rollback.[/bold red]"
        )
        raise typer.Exit(1)


@app.command()
def monitor(
    error_rate: float = typer.Option(None, help="Current error rate (0-1)"),
    latency_current: float = typer.Option(None, help="Current p50/p95 latency (ms)"),
    latency_baseline: float = typer.Option(None, help="Baseline latency (ms)"),
    refusal_rate: float = typer.Option(None, help="Current refusal rate (0-1)"),
    refusal_baseline: float = typer.Option(None, help="Baseline refusal rate (0-1)"),
    format_compliance: float = typer.Option(None, help="Format compliance rate (0-1)"),
    regression_pass_rate: float = typer.Option(None, help="Regression suite pass rate (0-1)"),
    canary_state: str = typer.Option(
        "./data/canary_state.json", help="Canary state file (an active canary is rolled back "
        "instead of the champion)"
    ),
):
    """Check live metrics against automatic-rollback thresholds."""
    from prompt_opt.deployment.canary import CanaryController
    from prompt_opt.deployment.rollback import RollbackMonitor

    canary_ctrl = None
    try:
        loaded = CanaryController.load(canary_state)
        if loaded.state.status == "active":
            canary_ctrl = loaded
    except FileNotFoundError:
        pass

    mon = RollbackMonitor(registry=_registry(), canary=canary_ctrl)
    event = mon.check(
        error_rate=error_rate,
        latency_ms_current=latency_current,
        latency_ms_baseline=latency_baseline,
        refusal_rate=refusal_rate,
        refusal_rate_baseline=refusal_baseline,
        format_compliance_rate=format_compliance,
        regression_pass_rate=regression_pass_rate,
    )
    if event is None:
        console.print("[bold green]All metrics within bounds - no rollback needed.[/bold green]")
        return
    console.print(f"[bold red]{event.reason}[/bold red]")
    if event.scope == "canary":
        console.print(
            f"Canary rolled back; champion [cyan]{event.rolled_back_to}[/cyan] is unchanged."
        )
    elif event.scope in ("suppressed", "no_previous"):
        console.print(
            "[yellow]No automatic rollback was performed; human attention needed.[/yellow]"
        )
    else:
        console.print(f"Rolled back to: [cyan]{event.rolled_back_to}[/cyan]")
    raise typer.Exit(1)


@app.command("score-shadow")
def score_shadow(
    log_file: Path = typer.Option(SHADOW_LOG_PATH, help="Shadow log written by `shadow`"),
    which: str = typer.Option("champion", help="Whose responses to score: champion | challenger"),
    output: Path = typer.Option(RECENT_SCORES_PATH, help="Where to write the JSON array of scores"),
    limit: int = typer.Option(0, help="Score at most this many responses (0 = all)"),
    judge_version: str = typer.Option(
        "", help="Frozen judge version to score with (default: latest calibrated judge)"
    ),
    allow_uncalibrated: bool = typer.Option(
        False, "--allow-uncalibrated", help="Score with an unvalidated judge if none is frozen"
    ),
):
    """
    Judge-score the responses in a shadow log, producing the input for `drift-check`.

    Writes a JSON array of scores (one per successfully answered request) and prints the
    refusal rate and mean latency, which are the other baseline/observation values
    `drift-check` accepts. Calls with an error are skipped, never scored.
    """
    _require_api_key("score-shadow")
    from prompt_opt.eval.guardrails import is_refusal
    from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole

    if which not in ("champion", "challenger"):
        console.print("[bold red]--which must be 'champion' or 'challenger'.[/bold red]")
        raise typer.Exit(1)
    if not log_file.exists():
        console.print(f"[bold red]Shadow log not found: {log_file}. Run `shadow` first.[/bold red]")
        raise typer.Exit(1)

    judge = _build_harness(judge_version, allow_uncalibrated).judge
    scores: list[float] = []
    refusals: list[bool] = []
    latencies: list[float] = []
    skipped = failed = 0
    with open(log_file, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            response = record.get(f"{which}_response")
            if record.get(f"{which}_error") or not response:
                skipped += 1
                continue
            turns = [
                Message(role=MessageRole(m["role"]), content=m.get("content"))
                for m in record.get("messages", [])
                if m.get("role") in ("user", "assistant")
            ]
            if not turns:
                skipped += 1
                continue
            conv = ConversationRecord(conversation_id=str(record.get("request_id")), messages=turns)
            try:
                scores.append(judge.score_absolute(conv, response).aggregate_score)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                rid = record.get("request_id")
                console.print(f"[yellow]Judge failed for {rid}: {exc}[/yellow]")
                continue
            refusals.append(is_refusal(response))
            if record.get(f"latency_{which}_ms") is not None:
                latencies.append(float(record[f"latency_{which}_ms"]))
            if limit and len(scores) >= limit:
                break

    if not scores:
        console.print(
            f"[bold red]No {which} responses could be scored "
            f"({skipped} skipped, {failed} judge failures).[/bold red]"
        )
        raise typer.Exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(scores, fh)

    refusal_rate = sum(refusals) / len(refusals)
    console.print(
        f"Scored {len(scores)} {which} responses ({skipped} skipped, {failed} judge failures) "
        f"-> {output}"
    )
    console.print(f"  mean score:   {sum(scores) / len(scores):.4f}")
    console.print(f"  refusal rate: {refusal_rate:.4f}")
    if latencies:
        console.print(f"  mean latency: {sum(latencies) / len(latencies):.0f} ms")
    console.print(
        "Use as a baseline:  prompt-opt drift-check --baseline-file "
        f"{output} --scores-file <recent.json> --baseline-refusal-rate {refusal_rate:.4f} "
        "--save-baseline"
    )


@app.command("drift-check")
def drift_check(
    scores_file: Path = typer.Option(..., exists=True, help="JSON array of recent scores"),
    baseline_file: Path = typer.Option(
        None, help="JSON array of baseline scores (default: the saved baseline)"
    ),
    baseline_state: Path = typer.Option(DRIFT_STATE_PATH, help="Saved baseline state file"),
    save_baseline: bool = typer.Option(
        False, "--save-baseline", help="Save the baseline built from --baseline-file for reuse"
    ),
    baseline_refusal_rate: float = typer.Option(None, help="Baseline refusal rate (0-1)"),
    baseline_latency_ms: float = typer.Option(None, help="Baseline latency (ms)"),
    refusal_rate: float = typer.Option(None, help="Current refusal rate observation"),
    format_compliance: float = typer.Option(None, help="Current format compliance observation"),
    latency_ms: float = typer.Option(None, help="Current latency observation (ms)"),
):
    """
    Detect drift between baseline and recent production judge scores.

    Exit codes: 0 = no drift, 1 = drift detected, 2 = no verdict (insufficient data, missing
    baseline, or a comparison that needs a baseline value that was not given).
    """
    from prompt_opt.feedback.drift_detector import DriftDetector

    def fail(message: str) -> None:
        console.print(f"[bold red]{message}[/bold red]")
        raise typer.Exit(2)

    with open(scores_file, encoding="utf-8") as fh:
        new_scores = [float(s) for s in json.load(fh)]

    if baseline_file is not None:
        if not baseline_file.exists():
            fail(f"Baseline file not found: {baseline_file}")
        with open(baseline_file, encoding="utf-8") as fh:
            baseline_scores = [float(s) for s in json.load(fh)]
        detector = DriftDetector()
        detector.update_baseline(
            judge_scores=baseline_scores,
            refusal_rate=baseline_refusal_rate,
            latency_ms=baseline_latency_ms,
        )
        if save_baseline:
            detector.save(baseline_state)
            console.print(f"Baseline saved to {baseline_state}")
    else:
        if save_baseline:
            fail("--save-baseline needs --baseline-file")
        try:
            detector = DriftDetector.load(baseline_state)
        except FileNotFoundError:
            fail(
                f"No baseline: pass --baseline-file, or create {baseline_state} with "
                f"`drift-check --baseline-file F --scores-file F --save-baseline`."
            )
        detector.set_baseline_metrics(
            refusal_rate=baseline_refusal_rate, latency_ms=baseline_latency_ms
        )

    if refusal_rate is not None and detector.baseline_refusal_rate is None:
        fail("--refusal-rate needs a baseline: pass --baseline-refusal-rate")
    if latency_ms is not None and detector.baseline_latency_ms is None:
        fail("--latency-ms needs a baseline: pass --baseline-latency-ms")

    report = detector.assess(
        new_scores=new_scores,
        refusal_rate=refusal_rate,
        format_compliance=format_compliance,
        latency_ms=latency_ms,
    )

    if report.kl is not None:
        console.print(
            f"Scores: baseline n={report.n_baseline}, recent n={report.n_new}, "
            f"KL={report.kl:.4f}, KS p={report.ks_pvalue:.4g}"
        )
    for note in report.notes:
        console.print(f"[yellow]{escape(note)}[/yellow]")

    if report.verdict == "insufficient_data":
        console.print(
            "[bold yellow]No verdict: not enough data to tell whether the scores drifted."
            "[/bold yellow]"
        )
        raise typer.Exit(2)
    if report.verdict == "no_drift":
        console.print("[bold green]No drift detected.[/bold green]")
        return

    table = Table(title="Drift Events")
    table.add_column("Type")
    table.add_column("Severity")
    table.add_column("Detail")
    for ev in report.events:
        table.add_row(ev.drift_type.value, ev.severity, escape(ev.description))
    console.print(table)
    raise typer.Exit(1)


@app.command()
def loop(
    once: bool = typer.Option(False, "--once", help="Run one cycle of all jobs and exit"),
    judge_version: str = typer.Option(
        "", help="Frozen judge version to score with (default: latest calibrated judge)"
    ),
    allow_uncalibrated: bool = typer.Option(
        False, "--allow-uncalibrated", help="Evaluate with an unvalidated judge if none is frozen"
    ),
):
    """Start the scheduled continuous feedback loop (ingest -> optimize -> drift)."""
    _require_api_key("loop")
    from prompt_opt.feedback.drift_detector import DriftDetector
    from prompt_opt.feedback.loop import FeedbackLoopManager, start_scheduler
    from prompt_opt.ingestion.ingestor import Ingestor

    manager = FeedbackLoopManager(
        registry=_registry(),
        ingestor=Ingestor(use_ner=settings.pii_use_ner),
        harness=_build_harness(judge_version, allow_uncalibrated),
        drift_detector=DriftDetector(),
    )

    if once:
        manager.re_ingest()
        manager.re_optimize()
        manager.drift_check_job()
        console.print("[bold green]One feedback-loop cycle complete.[/bold green]")
        return

    scheduler = start_scheduler(manager)
    if scheduler is None:
        console.print("[bold red]apscheduler is not installed; cannot schedule.[/bold red]")
        raise typer.Exit(1)
    console.print(
        f"[bold blue]Feedback loop started. re-ingest cron={settings.reingest_cron!r}, "
        f"re-optimize cron={settings.reoptimize_cron!r}, drift-check cron="
        f"{settings.drift_cron!r}. Press Ctrl+C to stop.[/bold blue]"
    )
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.shutdown()
        console.print("[yellow]Feedback loop stopped.[/yellow]")


@app.command()
def studio(
    port: int = typer.Option(8000, help="Port to run labeling studio server on"),
    host: str = typer.Option("127.0.0.1", help="Host address"),
):
    """Start the FastAPI human annotation studio web server."""
    import uvicorn

    console.print(f"[bold blue]Starting Labeling Studio on http://{host}:{port}...[/bold blue]")
    uvicorn.run("prompt_opt.labeling.server:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    app()
