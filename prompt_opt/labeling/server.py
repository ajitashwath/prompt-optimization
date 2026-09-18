"""
FastAPI human labeling studio and judge calibration interface.

Enables labelers to:
  1. Browse and annotate pairwise comparisons (A vs B).
  2. View live inter-annotator agreement statistics (kappa, correlation).
  3. Adjudicate disagreements.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from prompt_opt.judge.calibrator import JudgeCalibrator
from prompt_opt.judge.judge import LLMJudge
from prompt_opt.labeling.gold_store import GoldStore
from prompt_opt.labeling.rubric import DEFAULT_RUBRIC
from prompt_opt.storage.dataset_store import DatasetStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Human Labeling Studio")

# Setup template and static folders relative to the workspace root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "ui" / "templates"))

# Mount static files if directory exists
static_path = BASE_DIR / "ui" / "static"
static_path.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Shared stores (default sqlite)
gold_store = GoldStore()
judge = LLMJudge()
calibrator = JudgeCalibrator(judge, gold_store)


def _conversations_for(conversation_ids: set[str]) -> dict:
    """Conversation context (train/dev splits) for the given gold-item conversation ids."""
    return DatasetStore().find_conversations(conversation_ids)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Studio home page showing labeling stats."""
    total = gold_store.count_total()
    labeled = gold_store.count_labeled()
    unlabeled = total - labeled

    return templates.TemplateResponse(
        "base.html",
        {
            "request": request,
            "title": "Labeling Studio",
            "active_tab": "home",
            "stats": {"total": total, "labeled": labeled, "unlabeled": unlabeled},
        },
    )


@app.get("/queue", response_class=HTMLResponse)
async def view_queue(request: Request):
    """View active queue of items to label."""
    unlabeled_items = gold_store.get_unlabeled(limit=20)
    return templates.TemplateResponse(
        "queue.html",
        {
            "request": request,
            "title": "Annotation Queue",
            "active_tab": "queue",
            "items": unlabeled_items,
        },
    )


@app.get("/annotate/{item_id}", response_class=HTMLResponse)
async def annotate_item_view(request: Request, item_id: str):
    """Side-by-side pairwise comparison view for annotation."""
    item = gold_store.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    # Annotators must judge the responses in the context the judge will see
    conversation = _conversations_for({item.conversation_id}).get(item.conversation_id)

    return templates.TemplateResponse(
        "annotate.html",
        {
            "request": request,
            "title": f"Annotating Item {item_id[:8]}",
            "active_tab": "queue",
            "item": item,
            "conversation": conversation,
            "rubric": DEFAULT_RUBRIC,
        },
    )


@app.post("/annotate/{item_id}")
async def submit_annotation(
    item_id: str,
    preferred: Literal["A", "B", "tie"] = Form(...),
    rationale: str = Form(...),
    labeler_id: str = Form("human_labeler_1"),
):
    """Record preferred choice and rationale."""
    try:
        gold_store.submit_label(
            item_id=item_id,
            preferred=preferred,
            rationale=rationale,
            labeler_id=labeler_id,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Item not found")

    return RedirectResponse(url="/queue", status_code=303)


@app.get("/calibration", response_class=HTMLResponse)
def view_calibration(request: Request):
    """
    Run calibration dynamically and view the bias diagnostics report.

    A plain `def` so FastAPI runs it in a worker thread: it makes blocking LLM calls
    (two per labeled item) and must not stall the event loop.
    """
    labeled = gold_store.get_labeled(rubric_version=DEFAULT_RUBRIC.version)
    conversations = _conversations_for({item.conversation_id for item in labeled})
    report = calibrator.calibrate(
        rubric_version=DEFAULT_RUBRIC.version, conversations=conversations
    )
    return templates.TemplateResponse(
        "calibration.html",
        {
            "request": request,
            "title": "Judge Calibration",
            "active_tab": "calibration",
            "report": report,
        },
    )
