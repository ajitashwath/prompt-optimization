"""
Typer-based CLI for the self-improving prompt optimization pipeline.

Commands:
  - `ingest`: Parse and stratify raw production logs.
  - `add-gold`: Manually add candidate pairs to the human labeling studio.
  - `calibrate-judge`: Evaluate agreement metrics between LLM and humans.
  - `optimize`: Evolve prompt using MIPROv2 or GEPA.
  - `gate`: Evaluate candidate on test split and promote/rollback.
  - `studio`: Spin up the FastAPI human labeling server.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from prompt_opt.config import settings
from prompt_opt.eval.cache import EvalCache
from prompt_opt.eval.harness import EvalHarness, PromptCandidate
from prompt_opt.feedback.drift_detector import DriftDetector
from prompt_opt.gate.champion_gate import ChampionGate, GateCriteria
from prompt_opt.ingestion.ingestor import Ingestor
from prompt_opt.ingestion.schema import ConversationRecord, RawConversation
from prompt_opt.judge.calibrator import JudgeCalibrator
from prompt_opt.judge.judge import LLMJudge
from prompt_opt.judge.registry import JudgeRegistry
from prompt_opt.labeling.gold_store import GoldStore
from prompt_opt.labeling.rubric import DEFAULT_RUBRIC
from prompt_opt.optimizer.budget import BudgetController
from prompt_opt.optimizer.gepa import GEPA
from prompt_opt.optimizer.miprov2 import MIPROv2
from prompt_opt.registry.prompt_registry import PromptRegistry

app = typer.Typer(help="Self-Improving Prompt Optimization Pipeline CLI")
console = Console()


@app.command()
def ingest(
    input_file: Path = typer.Option(..., exists=True, help="Path to raw JSONL production logs"),
    train_ratio: float = 0.60,
    dev_ratio: float = 0.20,
    test_ratio: float = 0.20,
):
    """PII-scrub, deduplicate, and split raw logs into train/dev/test sets."""
    console.print(f"[bold blue]Starting ingestion for {input_file}...[/bold blue]")
    ingestor = Ingestor(use_ner=False)  # default spaCy NER disabled for speed/env sanity in CLI unless requested
    result = ingestor.ingest_file(
        input_file,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
    )

    table = Table(title="Dataset Split Statistics")
    table.add_column("Split", style="cyan")
    table.add_column("Count", style="magenta")
    for name, size in result.sizes.items():
        table.add_row(name, str(size))
    console.print(table)
    console.print("[bold green]Ingestion complete![/bold green]")


@app.command()
def add_gold(
    conversation_id: str = typer.Option(..., help="Source conversation ID"),
    response_a: str = typer.Option(..., help="Content of Response A"),
    response_b: str = typer.Option(..., help="Content of Response B"),
):
    """Add a new response comparison pair to the annotation queue."""
    gold_store = GoldStore()
    item_id = gold_store.add_item(
        conversation_id=conversation_id,
        response_a=response_a,
        response_b=response_b,
        rubric_version=DEFAULT_RUBRIC.version,
    )
    console.print(f"[bold green]Successfully added item {item_id} to annotation queue.[/bold green]")


@app.command()
def calibrate_judge():
    """Run agreement calibration diagnostics over human-labeled gold set."""
    gold_store = GoldStore()
    judge = LLMJudge()
    calibrator = JudgeCalibrator(judge, gold_store)

    console.print("[bold blue]Starting Judge Calibration...[/bold blue]")
    report = calibrator.calibrate(rubric_version=DEFAULT_RUBRIC.version)

    if report.n_items == 0:
        console.print("[bold yellow]No human-labeled items available. Please annotate items in the Studio first.[/bold yellow]")
        return

    console.print(f"Items Evaluated: {report.n_items}")
    console.print(f"Cohen's Kappa (Agreement): [bold magenta]{report.kappa:.3f}[/bold magenta] ({report.kappa_interpretation})")
    console.print(f"Spearman Correlation: [bold magenta]{report.spearman_rho:.3f}[/bold magenta]")

    if report.passed:
        console.print("[bold green]Calibration Passed! LLM Judge is calibrated.[/bold green]")
        # Freeze judge in registry
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
):
    """Run prompt search to propose and evaluate candidate system prompts on dev split."""
    console.print(f"[bold blue]Initializing {algorithm} optimization run...[/bold blue]")

    # Create dummy records for optimization demonstration
    from prompt_opt.ingestion.schema import Message, MessageRole
    dummy_convs = [
        ConversationRecord(
            conversation_id="stub_01",
            messages=[
                Message(role=MessageRole.USER, content="Hello"),
                Message(role=MessageRole.ASSISTANT, content="Hi, how can I help you?"),
            ],
            category="general",
            split="train",
        ),
        ConversationRecord(
            conversation_id="stub_02",
            messages=[
                Message(role=MessageRole.USER, content="Write a python function to add two numbers"),
                Message(role=MessageRole.ASSISTANT, content="def add(a, b): return a + b"),
            ],
            category="coding",
            split="dev",
        ),
    ]

    train_convs = [c for c in dummy_convs if c.split == "train"]
    dev_convs = [c for c in dummy_convs if c.split == "dev"]

    initial_prompt = PromptCandidate(
        system_prompt="You are a helpful assistant.",
        few_shot_demos=None,
    )

    judge = LLMJudge()
    harness = EvalHarness(judge, judge_version="stub_v1")
    budget = BudgetController(max_usd_spend=max_usd)

    if algorithm.lower() == "miprov2":
        opt = MIPROv2(n_trials=n_trials, minibatch_size=2)
    elif algorithm.lower() == "gepa":
        opt = GEPA(max_generations=n_trials, minibatch_size=2)
    else:
        console.print(f"[bold red]Unknown algorithm: {algorithm}[/bold red]")
        raise typer.Exit(1)

    results = opt.optimize(initial_prompt, train_convs, dev_convs, harness, budget)
    best_cand, summary = results[0]

    console.print("[bold green]Optimization complete![/bold green]")
    console.print(f"Best Candidate Hash: [cyan]{best_cand.content_hash}[/cyan]")
    console.print(f"Mean Score: [magenta]{summary.mean_score:.4f}[/magenta]")


@app.command()
def gate(
    challenger_hash: str = typer.Option(..., help="Hash of challenger prompt candidate"),
    champion_hash: str = typer.Option(..., help="Hash of current champion prompt"),
):
    """Evaluate challenger on frozen test split and decide on promotion."""
    console.print(f"[bold blue]Running Champion/Challenger Gate...[/bold blue]")

    # Stubs for demonstration
    from prompt_opt.ingestion.schema import Message, MessageRole
    test_convs = [
        ConversationRecord(
            conversation_id="test_01",
            messages=[
                Message(role=MessageRole.USER, content="Tell me a joke"),
                Message(role=MessageRole.ASSISTANT, content="Why did the chicken cross the road?"),
            ],
            category="general",
            split="test",
        )
    ]

    challenger = PromptCandidate(
        system_prompt="You are a witty, helpful AI companion.",
    )
    champion = PromptCandidate(
        system_prompt="You are a helpful assistant.",
    )

    judge = LLMJudge()
    harness = EvalHarness(judge, judge_version="stub_v1")
    cg = ChampionGate(harness)

    result = cg.evaluate(challenger, champion, test_convs)

    console.print(f"Gate Decision: [bold]{result.decision}[/bold]")
    if result.decision == "FAIL":
        for r in result.failure_reasons:
            console.print(f" - Reason: {r}", style="red")
    else:
        console.print("[bold green]Challenger promoted to Champion candidate status![/bold green]")


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
