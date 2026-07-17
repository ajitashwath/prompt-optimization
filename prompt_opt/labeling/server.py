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
from pydantic import BaseModel

from prompt_opt.config import settings
from prompt_opt.judge.calibrator import JudgeCalibrator
from prompt_opt.judge.judge import LLMJudge
from prompt_opt.labeling.gold_store import GoldStore
from prompt_opt.labeling.rubric import DEFAULT_RUBRIC

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
    # Find item in the store
    labeled_items = gold_store.get_labeled()
    unlabeled_items = gold_store.get_unlabeled(limit=100)
    all_items = labeled_items + unlabeled_items
    item = next((i for i in all_items if i.id == item_id), None)

    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    return templates.TemplateResponse(
        "annotate.html",
        {
            "request": request,
            "title": f"Annotating Item {item_id[:8]}",
            "active_tab": "queue",
            "item": item,
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
async def view_calibration(request: Request):
    """Run calibration dynamically and view the bias diagnostics report."""
    report = calibrator.calibrate(rubric_version=DEFAULT_RUBRIC.version)
    return templates.TemplateResponse(
        "calibration.html",
        {
            "request": request,
            "title": "Judge Calibration",
            "active_tab": "calibration",
            "report": report,
        },
    )
