"""
FastAPI backend for the Context Corrosion multi-agent LLM experiment.

Endpoints
---------
POST /run-user-query        – User-supplied question; SSE streaming live transcript.
POST /run-experiment        – Predefined task, single or counterfactual.
POST /run-monte-carlo       – N-trial statistical validation (predefined task).
POST /run-user-monte-carlo  – N-trial statistical validation (user query).
GET  /experiment/{id}       – Full logged JSON for any completed experiment.
GET  /tasks                 – Available predefined tasks.
GET  /health
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import EXPERIMENTS_DIR, TASKS
from app.experiment import (
    _build_task_description,
    run_experiment_core,
    run_monte_carlo,
    run_single_experiment,
    run_user_query_monte_carlo,
    stream_experiment_core,
    validate_query_ambiguity,
)
from app.metrics.analyzer import (
    compare_runs,
    compute_corruption_index,
    corruption_index_label,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Context Corrosion API",
    description=(
        "Demonstrates structural dominance bias in multi-agent LLM deliberation. "
        "Supports both predefined tasks and live user queries with SSE streaming."
    ),
    version="3.0.0",
)

os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
app.mount(
    "/static/experiments",
    StaticFiles(directory=EXPERIMENTS_DIR),
    name="experiments",
)


# ── JSON body sanitizer ───────────────────────────────────────────────────────

def _sanitize_body(raw: bytes) -> dict[str, Any]:
    """
    Parse a JSON request body that may contain literal unescaped control
    characters (newlines, tabs, carriage returns) inside string values.

    The standard JSON spec forbids literal control characters (U+0000–U+001F)
    inside strings. When users paste multi-line text into curl or Swagger UI,
    these appear verbatim. Strategy:

    1. Try normal json.loads — fast path for well-formed bodies.
    2. On failure, apply a character-level scan that replaces bare control
       characters inside JSON string literals with their escape sequences,
       then retry parsing.

    This is intentionally conservative: the scan only touches characters
    inside double-quoted string literals, leaving JSON structural characters
    (braces, colons, commas) untouched.
    """
    text = raw.decode("utf-8", errors="replace")

    # Fast path
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Repair path: escape raw control chars inside string literals
    _CTRL_ESC = {
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\b": "\\b",
        "\f": "\\f",
    }

    out: list[str] = []
    inside_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if inside_string:
            if ch == "\\":
                # Pass escape sequences through verbatim (consume next char too)
                out.append(ch)
                i += 1
                if i < len(text):
                    out.append(text[i])
            elif ch == '"':
                inside_string = False
                out.append(ch)
            elif ch in _CTRL_ESC:
                out.append(_CTRL_ESC[ch])
            else:
                out.append(ch)
        else:
            if ch == '"':
                inside_string = True
                out.append(ch)
            else:
                out.append(ch)
        i += 1

    repaired = "".join(out)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Request body is not valid JSON even after control-character repair: {exc}",
        )




class UserQueryRequest(BaseModel):
    """
    Run a context corrosion experiment on a user-supplied question or paragraph.

    The query is validated for ambiguity before the experiment starts.
    The response is a Server-Sent Events stream so the frontend can render
    each agent statement as it arrives (chat-style timeline).

    stream : bool
        True  (default) → SSE streaming response, events arrive agent-by-agent.
        False           → block until completion and return full JSON in one shot.
    """
    query: str = Field(..., min_length=15, max_length=2000,
                       description="Ambiguous question or policy dilemma (15–2000 chars)")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    rounds: int = Field(default=3, ge=1, le=5)
    biased_dominant: bool = Field(
        default=True,
        description="Inject structural dominance bias into agent D",
    )
    stream: bool = Field(
        default=True,
        description="SSE streaming (True) or single blocking JSON response (False)",
    )
    run_counterfactual: bool = Field(
        default=False,
        description="Run both biased and unbiased variants and return a comparison (forces blocking mode)",
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class ExperimentRequest(BaseModel):
    """Run a predefined task experiment (single or counterfactual)."""
    task_id: int = Field(default=0, ge=0, le=4)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    rounds: int = Field(default=3, ge=1, le=5)
    biased_dominant: bool = Field(default=True)
    run_counterfactual: bool = Field(
        default=False,
        description="Auto-run both biased and unbiased variants and compare",
    )


class MonteCarloRequest(BaseModel):
    """N-trial Monte Carlo over a predefined task."""
    task_id: int = Field(default=0, ge=0, le=4)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    rounds: int = Field(default=3, ge=1, le=5)
    n_trials: int = Field(default=5, ge=2, le=20)


class UserMonteCarloRequest(BaseModel):
    """N-trial Monte Carlo over a user-supplied query."""
    query: str = Field(..., min_length=15, max_length=2000)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    rounds: int = Field(default=3, ge=1, le=5)
    biased_dominant: bool = Field(default=True)
    n_trials: int = Field(default=5, ge=2, le=20)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        return v.strip()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_experiment(experiment_id: str) -> dict[str, Any]:
    path = os.path.join(EXPERIMENTS_DIR, experiment_id, "experiment.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id!r} not found.")
    with open(path) as f:
        return json.load(f)


def _summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Concise metrics summary for API top-level responses."""
    dc = metrics.get("directional_convergence", {})
    lc = metrics.get("lifecycle_entropy", {})
    ci = compute_corruption_index(metrics)
    return {
        "corruption_index": ci,
        "corruption_severity": corruption_index_label(ci),
        "avg_peer_convergence_final_round": metrics.get("avg_peer_convergence_final_round"),
        "avg_peer_to_peer_convergence": metrics.get("avg_peer_to_peer_convergence"),
        "avg_peer_directional_delta": metrics.get("avg_peer_directional_delta"),
        "directional_delta_per_peer": dc.get("directional_delta"),
        "dominant_self_drift": metrics.get("dominant_self_drift"),
        "entropy_decay_phase1_to_final": metrics.get("entropy_decay_phase1_to_final"),
        "lifecycle_entropy": lc,
        "raw_belief_shift_per_agent": metrics.get("raw_belief_shift"),
    }


async def _sse_generator(
    gen: AsyncGenerator[dict[str, Any], None],
) -> AsyncGenerator[str, None]:
    """
    Wrap an async experiment event generator into SSE wire format.
    Each event: "data: {json}\\n\\n"
    Heartbeat every 15 s keeps the connection alive during long LLM calls.
    """
    heartbeat_interval = 15.0
    last_event_time = asyncio.get_event_loop().time()

    try:
        async for event in gen:
            # Enrich metrics event with corruption index label
            if event.get("event") == "metrics" and "metrics" in event:
                ci = event.get("corruption_index", 0.0)
                event["corruption_severity"] = corruption_index_label(ci)

            payload = json.dumps(event, default=str)
            yield f"data: {payload}\n\n"
            last_event_time = asyncio.get_event_loop().time()

    except asyncio.CancelledError:
        yield "data: " + json.dumps({"event": "cancelled"}) + "\n\n"
    except Exception as exc:
        logger.exception("Stream error: %s", exc)
        yield "data: " + json.dumps({"event": "error", "message": str(exc)}) + "\n\n"
    finally:
        yield "data: " + json.dumps({"event": "stream_end"}) + "\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/run-user-query")
async def run_user_query(request: Request, _schema: UserQueryRequest = None):
    """
    Run a context corrosion experiment on a user-supplied question.

    Accepts JSON body with fields:
      query           : str  — ambiguous question or policy paragraph (15–2000 chars)
      temperature     : float (default 0.7)
      rounds          : int   (default 3, max 5)
      biased_dominant : bool  (default true)
      stream          : bool  (default true)

    Multi-line queries pasted directly into Swagger UI or curl are handled
    correctly — the body parser repairs unescaped newlines before JSON parsing.

    When stream=True (default):
      Returns a Server-Sent Events stream. Each agent's response is yielded
      as it completes, allowing the frontend to render a live chat-style
      transcript showing how agents interact and influence each other.

      Event types (in order):
        start       – experiment initialised
        phase_start – phase boundary marker
        phase1      – each agent's independent answer (5 events)
        discussion  – each agent's statement per round (5 × rounds events)
        final_vote  – each agent's final position (5 events)
        metrics     – full metrics object + corruption_index
        plots       – file paths to generated charts
        complete    – experiment ID for later retrieval via GET /experiment/{id}

    When stream=False:
      Blocks until all phases complete and returns a single JSON response.
    """
    # ── Parse + sanitize body (handles multi-line queries from curl/Swagger) ──
    raw_body = await request.body()
    try:
        data = _sanitize_body(raw_body)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Body parse error: {exc}")

    try:
        req = UserQueryRequest(**data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())

    # ── Query validation ──────────────────────────────────────────────────────
    valid, reason = validate_query_ambiguity(req.query)
    if not valid:
        raise HTTPException(status_code=422, detail=reason)

    task_description = _build_task_description(req.query)
    task_title = req.query[:80] + ("…" if len(req.query) > 80 else "")

    # ── Counterfactual mode ──────────────────────────────────────────────────
    if req.run_counterfactual:
        try:
            loop = asyncio.get_event_loop()
            result_a = await loop.run_in_executor(
                None,
                lambda: run_experiment_core(
                    task_description=task_description,
                    temperature=req.temperature,
                    rounds=req.rounds,
                    biased_dominant=True,
                    task_title=task_title,
                    source="user_query",
                ),
            )
            result_b = await loop.run_in_executor(
                None,
                lambda: run_experiment_core(
                    task_description=task_description,
                    temperature=req.temperature,
                    rounds=req.rounds,
                    biased_dominant=False,
                    task_title=task_title,
                    source="user_query",
                ),
            )
            comparison = compare_runs(result_a["metrics"], result_b["metrics"])
            return {
                "mode": "user_query_counterfactual",
                "query": req.query,
                "task_title": task_title,
                "run_A": {
                    "experiment_id": result_a["experiment_id"],
                    "biased_dominant": True,
                    "corruption_index": compute_corruption_index(result_a["metrics"]),
                    "corruption_severity": corruption_index_label(compute_corruption_index(result_a["metrics"])),
                    "metrics_summary": _summarize_metrics(result_a["metrics"]),
                    "plots": result_a["plots"],
                },
                "run_B": {
                    "experiment_id": result_b["experiment_id"],
                    "biased_dominant": False,
                    "corruption_index": compute_corruption_index(result_b["metrics"]),
                    "corruption_severity": corruption_index_label(compute_corruption_index(result_b["metrics"])),
                    "metrics_summary": _summarize_metrics(result_b["metrics"]),
                    "plots": result_b["plots"],
                },
                "counterfactual_comparison": comparison,
            }
        except Exception as exc:
            logger.exception("User query counterfactual failed")
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Streaming mode ────────────────────────────────────────────────────────
    if req.stream:
        gen = stream_experiment_core(
            task_description=task_description,
            temperature=req.temperature,
            rounds=req.rounds,
            biased_dominant=req.biased_dominant,
            task_title=task_title,
        )
        return StreamingResponse(
            _sse_generator(gen),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── Blocking mode ─────────────────────────────────────────────────────────
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_experiment_core(
                task_description=task_description,
                temperature=req.temperature,
                rounds=req.rounds,
                biased_dominant=req.biased_dominant,
                task_title=task_title,
                source="user_query",
            ),
        )
        return {
            "mode": "user_query_blocking",
            "experiment_id": result["experiment_id"],
            "query": req.query,
            "task_title": task_title,
            "biased_dominant": req.biased_dominant,
            "corruption_index": compute_corruption_index(result["metrics"]),
            "corruption_severity": corruption_index_label(
                compute_corruption_index(result["metrics"])
            ),
            "transcript": _build_transcript(result),
            "metrics_summary": _summarize_metrics(result["metrics"]),
            "plots": result["plots"],
        }
    except Exception as exc:
        logger.exception("User query experiment failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/run-experiment")
async def run_experiment_endpoint(req: ExperimentRequest) -> dict[str, Any]:
    """Run a predefined task experiment (single or counterfactual)."""
    try:
        if req.run_counterfactual:
            result_a = await asyncio.get_event_loop().run_in_executor(
                None, run_single_experiment, req.task_id, req.temperature, req.rounds, True
            )
            result_b = await asyncio.get_event_loop().run_in_executor(
                None, run_single_experiment, req.task_id, req.temperature, req.rounds, False
            )
            comparison = compare_runs(result_a["metrics"], result_b["metrics"])
            return {
                "mode": "counterfactual",
                "run_A": {
                    "experiment_id": result_a["experiment_id"],
                    "biased_dominant": True,
                    "corruption_index": compute_corruption_index(result_a["metrics"]),
                    "metrics_summary": _summarize_metrics(result_a["metrics"]),
                    "plots": result_a["plots"],
                },
                "run_B": {
                    "experiment_id": result_b["experiment_id"],
                    "biased_dominant": False,
                    "corruption_index": compute_corruption_index(result_b["metrics"]),
                    "metrics_summary": _summarize_metrics(result_b["metrics"]),
                    "plots": result_b["plots"],
                },
                "counterfactual_comparison": comparison,
            }

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            run_single_experiment,
            req.task_id,
            req.temperature,
            req.rounds,
            req.biased_dominant,
        )
        ci = compute_corruption_index(result["metrics"])
        return {
            "mode": "single",
            "experiment_id": result["experiment_id"],
            "task_title": result["task_title"],
            "biased_dominant": req.biased_dominant,
            "corruption_index": ci,
            "corruption_severity": corruption_index_label(ci),
            "metrics_summary": _summarize_metrics(result["metrics"]),
            "plots": result["plots"],
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Experiment failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-monte-carlo")
async def run_monte_carlo_endpoint(req: MonteCarloRequest) -> dict[str, Any]:
    """
    N-trial Monte Carlo over a predefined task.
    Runs n_trials of biased=True AND n_trials of biased=False,
    then returns Welch's t-test and Cohen's d across all key metrics.
    """
    try:
        loop = asyncio.get_event_loop()
        mc_a = await loop.run_in_executor(
            None, run_monte_carlo, req.task_id, req.temperature, req.rounds, True, req.n_trials
        )
        mc_b = await loop.run_in_executor(
            None, run_monte_carlo, req.task_id, req.temperature, req.rounds, False, req.n_trials
        )
        comparison = compare_runs(mc_a["all_metrics"], mc_b["all_metrics"])
        return {
            "mode": "monte_carlo",
            "n_trials": req.n_trials,
            "task_id": req.task_id,
            "run_A_aggregate": mc_a["aggregate_metrics"],
            "run_B_aggregate": mc_b["aggregate_metrics"],
            "run_A_trial_ids": mc_a["trial_ids"],
            "run_B_trial_ids": mc_b["trial_ids"],
            "statistical_comparison": comparison,
        }
    except Exception as e:
        logger.exception("Monte Carlo failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-user-monte-carlo")
async def run_user_monte_carlo_endpoint(request: Request) -> dict[str, Any]:
    """
    N-trial Monte Carlo over a user-supplied query (Research Mode).
    Multi-line query bodies are accepted (unescaped newlines repaired automatically).
    """
    raw_body = await request.body()
    try:
        data = _sanitize_body(raw_body)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Body parse error: {exc}")

    try:
        req = UserMonteCarloRequest(**data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())

    valid, reason = validate_query_ambiguity(req.query)
    if not valid:
        raise HTTPException(status_code=422, detail=reason)

    try:
        loop = asyncio.get_event_loop()
        mc_a = await loop.run_in_executor(
            None,
            run_user_query_monte_carlo,
            req.query, req.temperature, req.rounds, True, req.n_trials,
        )
        mc_b = await loop.run_in_executor(
            None,
            run_user_query_monte_carlo,
            req.query, req.temperature, req.rounds, False, req.n_trials,
        )
        comparison = compare_runs(mc_a["all_metrics"], mc_b["all_metrics"])
        return {
            "mode": "user_query_monte_carlo",
            "query": req.query,
            "n_trials": req.n_trials,
            "run_A_aggregate": mc_a["aggregate_metrics"],
            "run_B_aggregate": mc_b["aggregate_metrics"],
            "run_A_trial_ids": mc_a["trial_ids"],
            "run_B_trial_ids": mc_b["trial_ids"],
            "statistical_comparison": comparison,
        }
    except Exception as e:
        logger.exception("User Monte Carlo failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiment/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict[str, Any]:
    """Retrieve full structured JSON log for a completed experiment."""
    return _load_experiment(experiment_id)


@app.get("/tasks")
async def list_tasks() -> dict[str, Any]:
    """List all available predefined task scenarios."""
    return {
        str(tid): {"task_id": tid, "title": t["title"], "description": t["description"]}
        for tid, t in TASKS.items()
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ── Transcript builder (blocking mode helper) ─────────────────────────────────

def _build_transcript(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build a chronological, human-readable transcript from a completed experiment result.
    Suitable for rendering as a chat-style timeline in blocking mode responses.
    """
    transcript: list[dict[str, Any]] = []

    # Phase 1
    for agent_id, parsed in result["phase1"]["parsed"].items():
        transcript.append({
            "phase": "phase1",
            "agent_id": agent_id,
            "role": "dominant" if agent_id == "D" else "peer",
            "label": "Independent Thought",
            "text": result["phase1"]["raw"].get(agent_id, ""),
            "parsed": parsed,
        })

    # Phase 2
    for entry in result["phase2"]["shared_memory"]:
        transcript.append({
            "phase": "discussion",
            "round": entry["round"],
            "agent_id": entry["agent_id"],
            "role": "dominant" if entry["agent_id"] == "D" else "peer",
            "label": f"Discussion Round {entry['round']}",
            "text": entry["text"],
        })

    # Phase 3
    for agent_id, parsed in result["phase3"]["parsed"].items():
        transcript.append({
            "phase": "final_vote",
            "agent_id": agent_id,
            "role": "dominant" if agent_id == "D" else "peer",
            "label": "Final Vote",
            "text": result["phase3"]["raw"].get(agent_id, ""),
            "parsed": parsed,
        })

    return transcript