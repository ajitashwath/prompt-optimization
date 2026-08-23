"""
Typer-based CLI for the self-improving prompt optimization pipeline.

Commands:
  - `ingest`: Parse, scrub, deduplicate, stratify raw logs, persist splits.
  - `add-gold`: Manually add candidate pairs to the human labeling studio.
  - `calibrate-judge`: Evaluate agreement metrics between LLM and humans.
  - `optimize`: Evolve prompt using MIPROv2 or GEPA over the persisted dev split.
  - `gate`: Evaluate challenger vs champion on the frozen test split.
  - `gate-approve`: Record human spot-check decision on a pending gate result.
  - `shadow`: Replay conversations through champion + challenger (log-only).
  - `canary`: Manage staged traffic rollout (start/status/advance/breach/rollback).
  - `monitor`: Run rollback-trigger checks against live metrics.
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
from rich.table import Table

from prompt_opt.config import settings

app = typer.Typer(help="Self-Improving Prompt Optimization Pipeline CLI")
console = Console()

GATE_RESULTS_DIR = Path("./data/gate_results")
REGRESSION_SUITE_PATH = Path("./data/regression_suite.jsonl")


# ─── Shared helpers ──────────────────────────────────────────────────────────


def _registry():
    from prompt_opt.registry.prompt_registry import PromptRegistry

    # PromptRegistry uses the sync SQLAlchemy engine; normalize async URLs.
    db_url = settings.database_url.replace("sqlite+aiosqlite", "sqlite")
    return PromptRegistry(db_url=db_url)


_PROVIDER_KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")


def _require_api_key(command: str) -> None:
    """
    Pre-flight check: LLM-bound commands need at least one provider key.

    Fails fast with actionable guidance instead of letting litellm retry
    and crash with authentication tracebacks deep inside the pipeline.
    """
    import os

    if any(os.environ.get(var) for var in _PROVIDER_KEY_VARS):
        return
    console.print(
        f"[bold red]No LLM provider API key found - "
        f"`{command}` requires live model calls.[/bold red]\n"
        f"Set one of: {', '.join(_PROVIDER_KEY_VARS)}\n"
        f"  cp .env.example .env   # then fill in your key"
    )
    raise typer.Exit(1)


def _dataset_store():
    from prompt_opt.storage.dataset_store import DatasetStore

    return DatasetStore()


def _build_harness(judge_version: str = "v0"):
    from prompt_opt.eval.harness import EvalHarness
    from prompt_opt.judge.judge import LLMJudge

    judge = LLMJudge()
    return EvalHarness(judge, judge_version=judge_version)


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


def _run_regression_suite(candidate, cases, model: str | None = None):
    """Replay each regression case's messages through the candidate prompt."""
    from prompt_opt.llm import get_litellm

    litellm = get_litellm()
    model = model or settings.optimizee_model
    pairs = []
    for case in cases:
        try:
            resp = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": candidate.system_prompt},
                    *[
                        {"role": m.get("role", "user"), "content": m.get("content") or ""}
                        for m in case.messages
                    ],
                ],
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            console.print(
                f"[yellow]Regression call failed for {case.conversation_id}: {exc}[/yellow]"
            )
            text = f"[ERROR: {exc}]"
        pairs.append((case, text))
    return pairs


# ─── Pipeline commands ───────────────────────────────────────────────────────


@app.command()
def ingest(
    input_file: Path = typer.Option(..., exists=True, help="Path to raw JSONL production logs"),
    train_ratio: float = 0.60,
    dev_ratio: float = 0.20,
    test_ratio: float = 0.20,
    use_ner: bool = typer.Option(False, help="Enable spaCy NER scrubbing (slower)"),
):
    """PII-scrub, deduplicate, and split raw logs into persisted train/dev/test sets."""
    from prompt_opt.ingestion.ingestor import Ingestor
    from prompt_opt.labeling.rubric import DEFAULT_RUBRIC  # noqa: F401  (env sanity)

    console.print(f"[bold blue]Starting ingestion for {input_file}...[/bold blue]")
    ingestor = Ingestor(use_ner=use_ner)
    result = ingestor.ingest_file(
        input_file,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
    )

    store = _dataset_store()
    counts = store.save(result)

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

    judge = LLMJudge()
    calibrator = JudgeCalibrator(judge, gold_store)

    console.print("[bold blue]Starting Judge Calibration...[/bold blue]")
    report = calibrator.calibrate(rubric_version=DEFAULT_RUBRIC.version)

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

    if report.passed:
        console.print("[bold green]Calibration Passed! LLM Judge is calibrated.[/bold green]")
        registry = JudgeRegistry()
        registry.freeze(settings.judge_model, DEFAULT_RUBRIC.version, report)
    else:
        console.print("[bold red]Calibration Failed![/bold red]")
        for reason in report.failure_reasons:
            console.print(f" - {reason}", style="red")


@app.command()
def optimize(
    algorithm: str = typer.Option("miprov2", help="Optimization algorithm: 'miprov2' or 'gepa'"),
    max_usd: float = typer.Option(10.0, help="Max USD budget cap for LLM calls"),
    n_trials: int = typer.Option(10, help="Number of trials / generations"),
    judge_version: str = typer.Option("v0", help="Judge version tag used for cache keying"),
):
    """Run prompt search over the persisted train/dev splits and register the winner."""
    _require_api_key("optimize")
    from prompt_opt.optimizer.budget import BudgetController
    from prompt_opt.optimizer.gepa import GEPA
    from prompt_opt.optimizer.miprov2 import MIPROv2

    store = _dataset_store()
    if not store.exists("train") or not store.exists("dev"):
        console.print(
            "[bold red]No persisted train/dev splits found. "
            "Run `prompt-opt ingest --input-file ...` first.[/bold red]"
        )
        raise typer.Exit(1)

    train_convs = store.load_split("train")
    dev_convs = store.load_split("dev")

    console.print(f"[bold blue]Initializing {algorithm} optimization run "
                  f"(train={len(train_convs)}, dev={len(dev_convs)})...[/bold blue]")

    # Start from the current champion if one exists
    registry = _registry()
    champ = registry.get_champion()
    if champ:
        console.print(f"Seeding search from current champion {champ.version_id}")
        from prompt_opt.eval.harness import PromptCandidate

        initial_prompt = PromptCandidate(
            system_prompt=champ.system_prompt,
            few_shot_demos=champ.few_shot_demos,
        )
    else:
        from prompt_opt.eval.harness import PromptCandidate

        initial_prompt = PromptCandidate(system_prompt="You are a helpful assistant.")

    harness = _build_harness(judge_version)
    budget = BudgetController(max_usd_spend=max_usd)

    if algorithm.lower() == "miprov2":
        opt = MIPROv2(n_trials=n_trials, minibatch_size=min(2, len(dev_convs)))
    elif algorithm.lower() == "gepa":
        opt = GEPA(max_generations=n_trials, minibatch_size=min(2, len(dev_convs)))
    else:
        console.print(f"[bold red]Unknown algorithm: {algorithm}[/bold red]")
        raise typer.Exit(1)

    results = opt.optimize(initial_prompt, train_convs, dev_convs, harness, budget)
    best_cand, summary = results[0]

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
    console.print(f"Mean Score: [magenta]{summary.mean_score:.4f}[/magenta]")
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
    auto_promote: bool = typer.Option(True, help="Promote challenger automatically on PASS"),
    judge_version: str = typer.Option("v0", help="Judge version tag used for cache keying"),
):
    """Evaluate challenger vs champion on the frozen test split and decide promotion."""
    _require_api_key("gate")
    from prompt_opt.eval.guardrails import load_regression_suite
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

    regression_cases = []
    if REGRESSION_SUITE_PATH.exists():
        regression_cases = load_regression_suite(REGRESSION_SUITE_PATH)

    criteria = GateCriteria(require_human_spot_check=not skip_human_review)
    cg = ChampionGate(
        _build_harness(judge_version), criteria=criteria, regression_cases=regression_cases
    )

    regression_pairs = (
        _run_regression_suite(challenger, regression_cases) if regression_cases else None
    )
    result = cg.evaluate(challenger, champion, test_convs, regression_responses=regression_pairs)

    # Persist the immutable gate result
    GATE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    result_path = GATE_RESULTS_DIR / f"gate_{stamp}_{result.challenger_hash}.json"
    payload = result.to_dict()
    champ_row = registry.get_champion()
    payload["champion_version_id"] = champ_row.version_id if champ_row else None
    payload["result_path"] = str(result_path)
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

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

    if result.spot_check_items:
        spot = Table(title="Human Spot-Check Items")
        spot.add_column("Example")
        spot.add_column("Category")
        spot.add_column("Challenger")
        spot.add_column("Champion")
        spot.add_column("Outcome")
        for s in result.spot_check_items:
            spot.add_row(
                s["example_id"], s["category"],
                f"{s['challenger_score']:.2f}",
                f"{s['champion_score']}" if s["champion_score"] is not None else "-",
                str(s["outcome"]),
            )
        console.print(spot)

    console.print(f"Gate result saved to {result_path}")

    if result.decision == "FAIL":
        for r in result.failure_reasons:
            console.print(f" - Reason: {r}", style="red")
        raise typer.Exit(1)

    if result.decision == "PENDING_HUMAN":
        console.print(
            f"[bold yellow]PENDING HUMAN REVIEW.[/bold yellow] "
            f"After inspecting the spot-check items, run:\n"
            f"  prompt-opt gate-approve --result-file {result_path} --reviewer <name>\n"
            f"(add --reject to reject instead)"
        )
        return

    console.print("[bold green]Gate PASSED![/bold green]")
    if auto_promote:
        version_id = registry.register(
            system_prompt=challenger.system_prompt,
            few_shot_demos=challenger.few_shot_demos,
            eval_scores=result.challenger_eval.to_dict() if result.challenger_eval else {},
            gate_result_dict=payload,
        )
        registry.promote(version_id, approver="gate-cli")
        console.print(f"[bold green]Promoted {version_id} to champion.[/bold green]")


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
):
    """Record the human spot-check decision for a pending gate result."""
    from prompt_opt.gate.champion_gate import GateResult

    if result_file is None:
        candidates = (
            sorted(GATE_RESULTS_DIR.glob("gate_*.json"))
            if GATE_RESULTS_DIR.exists() else []
        )
        pending = []
        for p in candidates:
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("decision") == "PENDING_HUMAN":
                pending.append((p, data))
        if not pending:
            console.print("[bold red]No pending gate results found.[/bold red]")
            raise typer.Exit(1)
        result_file, data = pending[-1]
    else:
        if not result_file.exists():
            console.print(f"[bold red]Gate result file not found: {result_file}[/bold red]")
            raise typer.Exit(1)
        with open(result_file, encoding="utf-8") as fh:
            data = json.load(fh)

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

    data["decision"] = gate_result.decision
    data["human_approved"] = gate_result.human_approved
    data["human_reviewer"] = reviewer
    data["human_notes"] = notes
    with open(result_file, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    if reject:
        console.print(f"[bold red]Gate rejected by {reviewer}.[/bold red]")
        return

    registry = _registry()
    version = registry.get_by_content_hash(data["challenger_hash"])
    if version is None:
        console.print(
            f"[bold red]Challenger {data['challenger_hash']} is not registered; "
            f"cannot promote.[/bold red]"
        )
        raise typer.Exit(1)
    registry.promote(version.version_id, approver=reviewer)
    console.print(
        f"[bold green]Approved by {reviewer}. Promoted {version.version_id} to "
        f"champion.[/bold green]"
    )


# ─── Deployment commands ─────────────────────────────────────────────────────


@app.command()
def shadow(
    challenger_hash: str = typer.Option(..., help="Content hash of challenger prompt"),
    n: int = typer.Option(5, help="Number of conversations to shadow"),
    split: str = typer.Option("dev", help="Which persisted split to replay"),
):
    """Run challenger alongside champion on sample traffic; serve champion, log both."""
    _require_api_key("shadow")
    from prompt_opt.deployment.shadow import ShadowRunner

    registry = _registry()
    challenger = _resolve_candidate(registry, challenger_hash, "Challenger")
    champ_row = registry.get_champion()
    if champ_row is None:
        console.print("[bold red]No champion in registry.[/bold red]")
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
    errors = 0
    for conv in convs:
        messages = [
            {"role": m.role.value, "content": m.content or ""}
            for m in conv.without_system_prompt()
        ]
        res = runner.run_request(messages)
        lat_champ.append(res.latency_champion_ms)
        lat_chal.append(res.latency_challenger_ms)
        if res.error:
            errors += 1

    table = Table(title="Shadow Run Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Champion (ms)")
    table.add_column("Challenger (ms)")
    table.add_row(
        "mean latency", "",
        f"{sum(lat_champ) / len(lat_champ):.0f}",
        f"{sum(lat_chal) / len(lat_chal):.0f}",
    )
    table.add_row("errors", str(errors), "-", "-")
    console.print(table)
    console.print(f"[green]Shadow log written to {runner._log_file}[/green]")


@app.command()
def canary(
    action: str = typer.Argument(..., help="start | status | advance | breach | rollback"),
    challenger_hash: str = typer.Option("", help="[start] Challenger content hash"),
    champion_hash: str = typer.Option("", help="[start] Champion content hash"),
    reason: str = typer.Option("", help="[rollback] Reason string"),
    state_path: str = typer.Option("./data/canary_state.json", help="Canary state file"),
):
    """Manage staged canary rollout of a challenger prompt."""
    from prompt_opt.deployment.canary import CanaryController

    if action == "start":
        if not challenger_hash or not champion_hash:
            console.print(
                "[bold red]--challenger-hash and --champion-hash are required "
                "for start.[/bold red]"
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
        table.add_row("guardrail_breaches", str(ctrl.current_stage.guardrail_breaches))
        table.add_row("halt_reason", ctrl.state.halt_reason or "-")
        console.print(table)
    elif action == "advance":
        advanced = ctrl.maybe_advance()
        pct = ctrl.current_stage.traffic_pct if ctrl.state.status == "active" else 0
        if advanced:
            console.print(f"[bold green]Advanced. Current stage traffic: {pct}%[/bold green]")
        else:
            console.print(
                f"[yellow]Not advanced yet (soak incomplete or finished). "
                f"Status={ctrl.state.status}, pct={pct}%[/yellow]"
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
):
    """Check live metrics against automatic-rollback thresholds."""
    from prompt_opt.deployment.rollback import RollbackMonitor

    mon = RollbackMonitor(registry=_registry())
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
    console.print(f"Rolled back to: [cyan]{event.rolled_back_to}[/cyan]")
    raise typer.Exit(1)


@app.command("drift-check")
def drift_check(
    baseline_file: Path = typer.Option(..., exists=True, help="JSON array of baseline scores"),
    scores_file: Path = typer.Option(..., exists=True, help="JSON array of recent scores"),
    refusal_rate: float = typer.Option(None, help="Current EMA refusal rate observation"),
    format_compliance: float = typer.Option(None, help="Current format compliance observation"),
):
    """Detect distribution drift between baseline and recent production judge scores."""
    from prompt_opt.feedback.drift_detector import DriftDetector

    with open(baseline_file, encoding="utf-8") as fh:
        baseline_scores = [float(s) for s in json.load(fh)]
    with open(scores_file, encoding="utf-8") as fh:
        new_scores = [float(s) for s in json.load(fh)]

    detector = DriftDetector()
    detector.update_baseline(judge_scores=baseline_scores)
    events = detector.check(
        new_scores=new_scores,
        refusal_rate=refusal_rate,
        format_compliance=format_compliance,
    )

    if not events:
        console.print("[bold green]No drift detected.[/bold green]")
        return

    table = Table(title="Drift Events")
    table.add_column("Type")
    table.add_column("Severity")
    table.add_column("Detail")
    for ev in events:
        table.add_row(ev.drift_type.value, ev.severity, ev.description)
    console.print(table)
    raise typer.Exit(1)


@app.command()
def loop(
    once: bool = typer.Option(False, "--once", help="Run one cycle of all jobs and exit"),
):
    """Start the scheduled continuous feedback loop (ingest -> optimize -> drift)."""
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    from prompt_opt.feedback.drift_detector import DriftDetector
    from prompt_opt.feedback.loop import FeedbackLoopManager
    from prompt_opt.ingestion.ingestor import Ingestor

    manager = FeedbackLoopManager(
        registry=_registry(),
        ingestor=Ingestor(use_ner=False),
        harness=_build_harness(),
        drift_detector=DriftDetector(),
    )

    if once:
        manager.re_ingest()
        manager.re_optimize()
        console.print("[bold green]One feedback-loop cycle complete.[/bold green]")
        return

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        manager.re_ingest,
        CronTrigger.from_crontab(settings.reingest_cron),
        id="re_ingest_job",
    )
    scheduler.add_job(
        manager.re_optimize,
        CronTrigger.from_crontab(settings.reoptimize_cron),
        id="re_optimize_job",
    )
    scheduler.start()
    console.print(
        f"[bold blue]Feedback loop started. re-ingest cron={settings.reingest_cron!r}, "
        f"re-optimize cron={settings.reoptimize_cron!r}. Press Ctrl+C to stop.[/bold blue]"
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
