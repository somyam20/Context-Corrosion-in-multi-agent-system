"""
FastAPI backend for the Context Corrosion multi-agent LLM experiment.

Endpoints:
  POST /run-experiment   – Execute one or two experiment runs and return metrics + plots.
  GET  /experiment/{id}  – Retrieve full logged JSON for a past experiment.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import EXPERIMENTS_DIR, TASKS
from app.experiment import run_single_experiment
from app.metrics.analyzer import compare_runs
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Context Corrosion API",
    description=(
        "Demonstrates structural dominance bias in multi-agent LLM deliberation. "
        "A dominant agent (higher token budget, speaks first) measurably biases peer agents "
        "toward its framing — even when that framing is incorrect."
    ),
    version="1.0.0",
)

# Serve generated plots as static files
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
app.mount("/static/experiments", StaticFiles(directory=EXPERIMENTS_DIR), name="experiments")


# ── Request / Response Models ─────────────────────────────────────────────────

class ExperimentRequest(BaseModel):
    """
    Parameters for a single Context Corrosion experiment run.

    - task_id         : 0-4 (see /tasks for descriptions)
    - temperature     : shared across all agents (0.0–1.0)
    - rounds          : number of discussion rounds (recommended: 3)
    - biased_dominant : inject biased framing into dominant agent (Run A) or not (Run B)
    - run_counterfactual : if True, automatically run both biased=True and biased=False
                           and return a structured comparison
    """

    task_id: int = Field(default=0, ge=0, le=4, description="Task index (0–4)")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    rounds: int = Field(default=3, ge=1, le=5)
    biased_dominant: bool = Field(default=True)
    run_counterfactual: bool = Field(
        default=False,
        description="Run both biased and unbiased variants and return comparison",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_experiment(experiment_id: str) -> dict[str, Any]:
    path = os.path.join(EXPERIMENTS_DIR, experiment_id, "experiment.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id!r} not found.")
    with open(path) as f:
        return json.load(f)


def _summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return a concise metrics summary (omits large matrices for top-level response)."""
    return {
        "avg_peer_convergence_final_round": metrics.get("avg_peer_convergence_final_round"),
        "avg_peer_belief_shift": metrics.get("avg_peer_belief_shift"),
        "dominant_belief_shift": metrics.get("dominant_belief_shift"),
        "entropy_decay": metrics.get("entropy_decay"),
        "entropy_per_round": metrics.get("entropy_per_round"),
        "belief_shift_per_agent": metrics.get("belief_shift"),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/run-experiment")
async def run_experiment(req: ExperimentRequest) -> dict[str, Any]:
    """
    Run a Context Corrosion experiment.

    If `run_counterfactual=True`, both biased (Run A) and unbiased (Run B)
    variants are executed and a structured comparison is returned.
    """
    try:
        if req.run_counterfactual:
            # Run A – biased dominant
            result_a = run_single_experiment(
                task_id=req.task_id,
                temperature=req.temperature,
                rounds=req.rounds,
                biased_dominant=True,
            )
            # Run B – unbiased dominant
            result_b = run_single_experiment(
                task_id=req.task_id,
                temperature=req.temperature,
                rounds=req.rounds,
                biased_dominant=False,
            )

            comparison = compare_runs(result_a["metrics"], result_b["metrics"])

            return {
                "mode": "counterfactual",
                "run_A": {
                    "experiment_id": result_a["experiment_id"],
                    "biased_dominant": True,
                    "metrics_summary": _summarize_metrics(result_a["metrics"]),
                    "plots": result_a["plots"],
                },
                "run_B": {
                    "experiment_id": result_b["experiment_id"],
                    "biased_dominant": False,
                    "metrics_summary": _summarize_metrics(result_b["metrics"]),
                    "plots": result_b["plots"],
                },
                "counterfactual_comparison": comparison,
            }

        else:
            result = run_single_experiment(
                task_id=req.task_id,
                temperature=req.temperature,
                rounds=req.rounds,
                biased_dominant=req.biased_dominant,
            )
            return {
                "mode": "single",
                "experiment_id": result["experiment_id"],
                "task_title": result["task_title"],
                "biased_dominant": req.biased_dominant,
                "metrics_summary": _summarize_metrics(result["metrics"]),
                "plots": result["plots"],
            }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Experiment failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Experiment failed: {e}")


@app.get("/experiment/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict[str, Any]:
    """Retrieve the full logged JSON result for a completed experiment."""
    return _load_experiment(experiment_id)


@app.get("/tasks")
async def list_tasks() -> dict[str, Any]:
    """List all available ambiguous reasoning tasks."""
    return {
        str(tid): {
            "task_id": tid,
            "title": t["title"],
            "description": t["description"],
        }
        for tid, t in TASKS.items()
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
