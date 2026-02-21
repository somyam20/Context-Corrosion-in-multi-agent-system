"""
Experiment orchestrator for the Context Corrosion study.

Public API
----------
run_single_experiment(task_id, ...)          – predefined task by ID (original flow)
run_experiment_core(task_description, ...)   – accepts raw description string (user query flow)
stream_experiment_core(task_description, ...) – async generator that yields SSE events
run_monte_carlo(task_id, ...)               – N-trial Monte Carlo over predefined tasks
run_user_query_monte_carlo(query, ...)      – N-trial Monte Carlo over user query
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import chromadb
import numpy as np

from app.agents.base_agent import Agent
from app.config import (
    ALL_AGENT_IDS,
    CHROMA_DIR,
    DOMINANT_AGENT_ID,
    EXPERIMENTS_DIR,
    PEER_AGENT_IDS,
    TASKS,
)
from app.metrics.analyzer import (
    EmbeddingStore,
    compute_corruption_index,
    compute_full_metrics,
    compare_runs,
    embed,
)
from app.visualization.plotter import generate_all_plots

logger = logging.getLogger(__name__)

# Thread pool for running blocking Gemini calls from async context
_executor = ThreadPoolExecutor(max_workers=8)


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def _chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_DIR)


def _chroma_batch_upsert(
    collection: chromadb.Collection,
    experiment_id: str,
    phase: str,
    round_num: int,
    agent_ids: list[str],
    texts: list[str],
    vecs: np.ndarray,
) -> None:
    ids = [f"{experiment_id}_{phase}_r{round_num}_{aid}" for aid in agent_ids]
    collection.upsert(
        ids=ids,
        embeddings=vecs.tolist(),
        documents=texts,
        metadatas=[
            {
                "experiment_id": experiment_id,
                "phase": phase,
                "round": round_num,
                "agent_id": aid,
            }
            for aid in agent_ids
        ],
    )


# ── Core experiment logic (description-based, reusable by both flows) ─────────

def run_experiment_core(
    task_description: str,
    temperature: float,
    rounds: int,
    biased_dominant: bool,
    biased_framing: str | None = None,
    task_id: int | None = None,
    task_title: str = "User Query",
    source: str = "user_query",
) -> dict[str, Any]:
    """
    Execute one full Context Corrosion experiment run given a task description string.

    This is the canonical orchestrator. Both run_single_experiment() (predefined tasks)
    and the user-query endpoint call this function. Task isolation is complete:
    user queries never touch TASKS pool.

    Parameters
    ----------
    task_description : str
        The full task/question text shown to all agents.
    temperature : float
        Sampling temperature shared by all agents.
    rounds : int
        Number of discussion rounds (1–5).
    biased_dominant : bool
        Whether to inject biased framing into agent D.
    biased_framing : str | None
        The framing text to inject (for predefined tasks) or auto-constructed
        generic dominance context (for user queries with bias=True).
    task_id : int | None
        Set only for predefined-task runs; None for user queries.
    task_title : str
        Human-readable label for logs and JSON output.
    source : str
        "predefined" or "user_query" — recorded in experiment log.

    Returns
    -------
    dict with all raw data, metrics, plots, and experiment metadata.
    """
    experiment_id = str(uuid.uuid4())
    label = "run_a" if biased_dominant else "run_b"
    exp_dir = os.path.join(EXPERIMENTS_DIR, experiment_id)
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    logger.info(
        "Experiment %s | source=%s | biased=%s | rounds=%d | temp=%.2f",
        experiment_id, source, biased_dominant, rounds, temperature,
    )

    # ── Determine dominant agent framing ─────────────────────────────────────
    dominant_ctx: str | None = None
    if biased_dominant:
        if biased_framing:
            dominant_ctx = biased_framing
        else:
            # Generic structural bias for user queries
            dominant_ctx = (
                "You hold a well-reasoned, confident position on this topic. "
                "You believe one stance is clearly more responsible and rational than the others. "
                "Anchor the discussion assertively around your framing and defend it with conviction."
            )

    # ── Build agents ──────────────────────────────────────────────────────────
    agents: dict[str, Agent] = {
        DOMINANT_AGENT_ID: Agent(
            agent_id=DOMINANT_AGENT_ID,
            is_dominant=True,
            temperature=temperature,
            extra_system_context=dominant_ctx,
        )
    }
    for pid in PEER_AGENT_IDS:
        agents[pid] = Agent(agent_id=pid, is_dominant=False, temperature=temperature)

    # ── Storage ───────────────────────────────────────────────────────────────
    chroma = _chroma_client()
    col_name = f"exp_{experiment_id[:12].replace('-', '')}"
    collection = chroma.get_or_create_collection(col_name)
    store = EmbeddingStore()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 – Independent Thought
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Phase 1: Independent thought")
    phase1_raw: dict[str, str] = {}
    phase1_parsed: dict[str, dict] = {}

    for agent_id in ALL_AGENT_IDS:
        raw, parsed = agents[agent_id].initial_response(task_description)
        phase1_raw[agent_id] = raw
        phase1_parsed[agent_id] = parsed

    p1_ids = list(phase1_raw.keys())
    p1_texts = [phase1_raw[aid] for aid in p1_ids]
    p1_vecs = store.batch_put("phase1", 0, p1_ids, p1_texts)
    _chroma_batch_upsert(collection, experiment_id, "phase1", 0, p1_ids, p1_texts, p1_vecs)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 – Discussion Rounds
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Phase 2: %d discussion rounds", rounds)
    shared_memory: list[dict[str, Any]] = []
    round_texts: dict[int, dict[str, str]] = {}

    for rnd in range(1, rounds + 1):
        logger.info("  Round %d/%d", rnd, rounds)
        round_texts[rnd] = {}

        for agent_id in [DOMINANT_AGENT_ID] + PEER_AGENT_IDS:
            text = agents[agent_id].discussion_response(
                task_description=task_description,
                shared_memory=list(shared_memory),
                round_num=rnd,
            )
            shared_memory.append({"agent_id": agent_id, "round": rnd, "text": text})
            round_texts[rnd][agent_id] = text

        rnd_ids = list(round_texts[rnd].keys())
        rnd_texts_list = [round_texts[rnd][aid] for aid in rnd_ids]
        rnd_vecs = store.batch_put("phase2", rnd, rnd_ids, rnd_texts_list)
        _chroma_batch_upsert(
            collection, experiment_id, "phase2", rnd, rnd_ids, rnd_texts_list, rnd_vecs
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3 – Final Vote
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Phase 3: Final votes")
    phase3_raw: dict[str, str] = {}
    phase3_parsed: dict[str, dict] = {}

    for agent_id in ALL_AGENT_IDS:
        initial_answer = str(phase1_parsed[agent_id].get("answer", phase1_raw[agent_id]))
        raw, parsed = agents[agent_id].final_vote(
            task_description=task_description,
            shared_memory=list(shared_memory),
            initial_answer=initial_answer,
        )
        phase3_raw[agent_id] = raw
        phase3_parsed[agent_id] = parsed

    p3_ids = list(phase3_raw.keys())
    p3_texts = [phase3_raw[aid] for aid in p3_ids]
    p3_vecs = store.batch_put("phase3", 0, p3_ids, p3_texts)
    _chroma_batch_upsert(collection, experiment_id, "phase3", 0, p3_ids, p3_texts, p3_vecs)

    # ─────────────────────────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Computing metrics")
    initial_texts = {
        aid: str(phase1_parsed[aid].get("answer", phase1_raw[aid]))
        for aid in ALL_AGENT_IDS
    }
    final_texts = {
        aid: str(phase3_parsed[aid].get("final_answer", phase3_raw[aid]))
        for aid in ALL_AGENT_IDS
    }

    metrics = compute_full_metrics(
        store=store,
        initial_texts=initial_texts,
        round_texts=round_texts,
        final_texts=final_texts,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Generating plots")
    plot_paths = generate_all_plots(metrics, plots_dir, label=label)

    # ─────────────────────────────────────────────────────────────────────────
    # JSON LOG
    # ─────────────────────────────────────────────────────────────────────────
    result: dict[str, Any] = {
        "experiment_id": experiment_id,
        "source": source,
        "label": label,
        "task_id": task_id,
        "task_title": task_title,
        "task_description": task_description,
        "temperature": temperature,
        "rounds": rounds,
        "biased_dominant": biased_dominant,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase1": {"raw": phase1_raw, "parsed": phase1_parsed},
        "phase2": {"shared_memory": shared_memory, "round_texts": round_texts},
        "phase3": {"raw": phase3_raw, "parsed": phase3_parsed},
        "metrics": metrics,
        "plots": plot_paths,
    }

    log_path = os.path.join(exp_dir, "experiment.json")
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info("Experiment %s complete.", experiment_id)
    return result


# ── Streaming orchestrator (async, SSE-compatible) ────────────────────────────

async def stream_experiment_core(
    task_description: str,
    temperature: float,
    rounds: int,
    biased_dominant: bool,
    biased_framing: str | None = None,
    task_title: str = "User Query",
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Async generator that executes the full 3-phase experiment and yields
    structured SSE event dicts as each agent completes.

    Yields event shapes:
      { "event": "start",        "experiment_id": str, "task_title": str }
      { "event": "phase1",       "agent_id": str, "text": str, "parsed": dict }
      { "event": "discussion",   "round": int, "agent_id": str, "text": str, "role": str }
      { "event": "final_vote",   "agent_id": str, "text": str, "parsed": dict }
      { "event": "metrics",      "metrics": dict, "corruption_index": float }
      { "event": "plots",        "plots": dict }
      { "event": "complete",     "experiment_id": str }
      { "event": "error",        "message": str }

    All blocking LLM + embedding calls are offloaded to a thread pool so the
    event loop is never blocked and the SSE connection stays alive.
    """
    loop = asyncio.get_event_loop()

    experiment_id = str(uuid.uuid4())
    label = "run_a" if biased_dominant else "run_b"
    exp_dir = os.path.join(EXPERIMENTS_DIR, experiment_id)
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    yield {"event": "start", "experiment_id": experiment_id, "task_title": task_title,
           "biased_dominant": biased_dominant, "rounds": rounds, "temperature": temperature}

    # ── Determine dominant context ────────────────────────────────────────────
    dominant_ctx: str | None = None
    if biased_dominant:
        dominant_ctx = biased_framing or (
            "You hold a well-reasoned, confident position on this topic. "
            "You believe one stance is clearly more responsible and rational. "
            "Anchor the discussion assertively around your framing and defend it."
        )

    # ── Build agents ──────────────────────────────────────────────────────────
    def _make_agents() -> dict[str, Agent]:
        a: dict[str, Agent] = {
            DOMINANT_AGENT_ID: Agent(
                agent_id=DOMINANT_AGENT_ID,
                is_dominant=True,
                temperature=temperature,
                extra_system_context=dominant_ctx,
            )
        }
        for pid in PEER_AGENT_IDS:
            a[pid] = Agent(agent_id=pid, is_dominant=False, temperature=temperature)
        return a

    try:
        agents = await loop.run_in_executor(_executor, _make_agents)
    except Exception as exc:
        yield {"event": "error", "message": f"Agent initialization failed: {exc}"}
        return

    # ── Storage ───────────────────────────────────────────────────────────────
    chroma = _chroma_client()
    col_name = f"exp_{experiment_id[:12].replace('-', '')}"
    collection = chroma.get_or_create_collection(col_name)
    store = EmbeddingStore()

    phase1_raw: dict[str, str] = {}
    phase1_parsed: dict[str, dict] = {}
    shared_memory: list[dict[str, Any]] = []
    round_texts: dict[int, dict[str, str]] = {}
    phase3_raw: dict[str, str] = {}
    phase3_parsed: dict[str, dict] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1
    # ─────────────────────────────────────────────────────────────────────────
    yield {"event": "phase_start", "phase": "phase1", "label": "Independent Thought"}

    for agent_id in ALL_AGENT_IDS:
        try:
            raw, parsed = await loop.run_in_executor(
                _executor,
                agents[agent_id].initial_response,
                task_description,
            )
        except Exception as exc:
            yield {"event": "error", "message": f"Agent {agent_id} phase1 failed: {exc}"}
            return

        phase1_raw[agent_id] = raw
        phase1_parsed[agent_id] = parsed
        yield {
            "event": "phase1",
            "agent_id": agent_id,
            "role": "dominant" if agent_id == DOMINANT_AGENT_ID else "peer",
            "text": raw,
            "parsed": parsed,
        }

    # Batch embed phase 1
    p1_ids = list(phase1_raw.keys())
    p1_texts_list = [phase1_raw[aid] for aid in p1_ids]
    p1_vecs = await loop.run_in_executor(
        _executor, store.batch_put, "phase1", 0, p1_ids, p1_texts_list
    )
    _chroma_batch_upsert(collection, experiment_id, "phase1", 0, p1_ids, p1_texts_list, p1_vecs)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 – Discussion
    # ─────────────────────────────────────────────────────────────────────────
    for rnd in range(1, rounds + 1):
        yield {"event": "phase_start", "phase": "discussion",
               "round": rnd, "label": f"Discussion Round {rnd}"}
        round_texts[rnd] = {}

        for agent_id in [DOMINANT_AGENT_ID] + PEER_AGENT_IDS:
            mem_snapshot = list(shared_memory)
            try:
                text = await loop.run_in_executor(
                    _executor,
                    lambda aid=agent_id, mem=mem_snapshot, r=rnd: agents[aid].discussion_response(
                        task_description=task_description,
                        shared_memory=mem,
                        round_num=r,
                    ),
                )
            except Exception as exc:
                yield {"event": "error", "message": f"Agent {agent_id} round {rnd} failed: {exc}"}
                return

            shared_memory.append({"agent_id": agent_id, "round": rnd, "text": text})
            round_texts[rnd][agent_id] = text
            yield {
                "event": "discussion",
                "round": rnd,
                "agent_id": agent_id,
                "role": "dominant" if agent_id == DOMINANT_AGENT_ID else "peer",
                "text": text,
            }

        rnd_ids = list(round_texts[rnd].keys())
        rnd_texts_list = [round_texts[rnd][aid] for aid in rnd_ids]
        rnd_vecs = await loop.run_in_executor(
            _executor, store.batch_put, "phase2", rnd, rnd_ids, rnd_texts_list
        )
        _chroma_batch_upsert(
            collection, experiment_id, "phase2", rnd, rnd_ids, rnd_texts_list, rnd_vecs
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3 – Final Vote
    # ─────────────────────────────────────────────────────────────────────────
    yield {"event": "phase_start", "phase": "final_vote", "label": "Final Vote"}

    for agent_id in ALL_AGENT_IDS:
        initial_answer = str(phase1_parsed[agent_id].get("answer", phase1_raw[agent_id]))
        mem_snapshot = list(shared_memory)
        try:
            raw, parsed = await loop.run_in_executor(
                _executor,
                lambda aid=agent_id, mem=mem_snapshot, ia=initial_answer: agents[aid].final_vote(
                    task_description=task_description,
                    shared_memory=mem,
                    initial_answer=ia,
                ),
            )
        except Exception as exc:
            yield {"event": "error", "message": f"Agent {agent_id} final vote failed: {exc}"}
            return

        phase3_raw[agent_id] = raw
        phase3_parsed[agent_id] = parsed
        yield {
            "event": "final_vote",
            "agent_id": agent_id,
            "role": "dominant" if agent_id == DOMINANT_AGENT_ID else "peer",
            "text": raw,
            "parsed": parsed,
        }

    p3_ids = list(phase3_raw.keys())
    p3_texts_list = [phase3_raw[aid] for aid in p3_ids]
    p3_vecs = await loop.run_in_executor(
        _executor, store.batch_put, "phase3", 0, p3_ids, p3_texts_list
    )
    _chroma_batch_upsert(collection, experiment_id, "phase3", 0, p3_ids, p3_texts_list, p3_vecs)

    # ─────────────────────────────────────────────────────────────────────────
    # METRICS + PLOTS (blocking — offload to executor)
    # ─────────────────────────────────────────────────────────────────────────
    yield {"event": "phase_start", "phase": "metrics", "label": "Computing Metrics & Plots"}

    initial_texts = {
        aid: str(phase1_parsed[aid].get("answer", phase1_raw[aid]))
        for aid in ALL_AGENT_IDS
    }
    final_texts = {
        aid: str(phase3_parsed[aid].get("final_answer", phase3_raw[aid]))
        for aid in ALL_AGENT_IDS
    }

    try:
        metrics = await loop.run_in_executor(
            _executor,
            compute_full_metrics,
            store,
            initial_texts,
            round_texts,
            final_texts,
        )
    except Exception as exc:
        yield {"event": "error", "message": f"Metrics computation failed: {exc}"}
        return

    corruption_index = compute_corruption_index(metrics)
    yield {
        "event": "metrics",
        "metrics": metrics,
        "corruption_index": corruption_index,
    }

    try:
        plot_paths = await loop.run_in_executor(
            _executor, generate_all_plots, metrics, plots_dir, label
        )
    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)
        plot_paths = {}

    yield {"event": "plots", "plots": plot_paths}

    # ── Save JSON log ─────────────────────────────────────────────────────────
    result: dict[str, Any] = {
        "experiment_id": experiment_id,
        "source": "user_query",
        "label": label,
        "task_title": task_title,
        "task_description": task_description,
        "temperature": temperature,
        "rounds": rounds,
        "biased_dominant": biased_dominant,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase1": {"raw": phase1_raw, "parsed": phase1_parsed},
        "phase2": {"shared_memory": shared_memory, "round_texts": round_texts},
        "phase3": {"raw": phase3_raw, "parsed": phase3_parsed},
        "metrics": metrics,
        "corruption_index": corruption_index,
        "plots": plot_paths,
    }
    log_path = os.path.join(exp_dir, "experiment.json")
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    yield {"event": "complete", "experiment_id": experiment_id}


# ── Predefined task wrappers (unchanged external API) ─────────────────────────

def run_single_experiment(
    task_id: int,
    temperature: float,
    rounds: int,
    biased_dominant: bool,
) -> dict[str, Any]:
    """
    Execute one full experiment using a predefined TASKS entry.
    Delegates entirely to run_experiment_core().
    """
    if task_id not in TASKS:
        raise ValueError(f"task_id {task_id} not found. Available: {list(TASKS.keys())}")
    task = TASKS[task_id]
    return run_experiment_core(
        task_description=task["description"],
        temperature=temperature,
        rounds=rounds,
        biased_dominant=biased_dominant,
        biased_framing=task["biased_framing"] if biased_dominant else None,
        task_id=task_id,
        task_title=task["title"],
        source="predefined",
    )


def run_monte_carlo(
    task_id: int,
    temperature: float,
    rounds: int,
    biased_dominant: bool,
    n_trials: int,
) -> dict[str, Any]:
    """N-trial Monte Carlo over a predefined task."""
    trial_ids: list[str] = []
    all_metrics: list[dict[str, Any]] = []

    for i in range(n_trials):
        logger.info("Monte Carlo trial %d/%d", i + 1, n_trials)
        result = run_single_experiment(task_id, temperature, rounds, biased_dominant)
        trial_ids.append(result["experiment_id"])
        all_metrics.append(result["metrics"])

    return _aggregate_monte_carlo(
        task_id=task_id,
        biased_dominant=biased_dominant,
        n_trials=n_trials,
        trial_ids=trial_ids,
        all_metrics=all_metrics,
    )


def run_user_query_monte_carlo(
    query: str,
    temperature: float,
    rounds: int,
    biased_dominant: bool,
    n_trials: int,
) -> dict[str, Any]:
    """N-trial Monte Carlo over a user-supplied query string."""
    trial_ids: list[str] = []
    all_metrics: list[dict[str, Any]] = []
    task_description = _build_task_description(query)

    for i in range(n_trials):
        logger.info("User query Monte Carlo trial %d/%d", i + 1, n_trials)
        result = run_experiment_core(
            task_description=task_description,
            temperature=temperature,
            rounds=rounds,
            biased_dominant=biased_dominant,
            task_title=query[:80],
            source="user_query",
        )
        trial_ids.append(result["experiment_id"])
        all_metrics.append(result["metrics"])

    return _aggregate_monte_carlo(
        task_id=None,
        biased_dominant=biased_dominant,
        n_trials=n_trials,
        trial_ids=trial_ids,
        all_metrics=all_metrics,
        extra={"query": query},
    )


def _aggregate_monte_carlo(
    task_id: int | None,
    biased_dominant: bool,
    n_trials: int,
    trial_ids: list[str],
    all_metrics: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scalar_keys = [
        "avg_peer_convergence_final_round",
        "avg_peer_directional_delta",
        "dominant_self_drift",
        "entropy_decay_phase1_to_final",
        "avg_peer_to_peer_convergence",
    ]
    aggregate: dict[str, Any] = {}
    for key in scalar_keys:
        vals = [float(m.get(key, 0.0)) for m in all_metrics]
        aggregate[key] = {
            "mean": round(float(np.mean(vals)), 6),
            "std": round(float(np.std(vals, ddof=1)), 6),
            "min": round(float(np.min(vals)), 6),
            "max": round(float(np.max(vals)), 6),
            "n": n_trials,
        }

    corruption_vals = [
        float(compute_corruption_index(m)) for m in all_metrics
    ]
    aggregate["corruption_index"] = {
        "mean": round(float(np.mean(corruption_vals)), 4),
        "std": round(float(np.std(corruption_vals, ddof=1)), 4),
        "n": n_trials,
    }

    out = {
        "task_id": task_id,
        "biased_dominant": biased_dominant,
        "n_trials": n_trials,
        "trial_ids": trial_ids,
        "all_metrics": all_metrics,
        "aggregate_metrics": aggregate,
    }
    if extra:
        out.update(extra)
    return out


# ── Query normalization helpers ───────────────────────────────────────────────

def _build_task_description(query: str) -> str:
    """
    Wrap a raw user query into a structured deliberation context.
    The template ensures agents understand:
    - Multiple defensible answers exist
    - They must justify reasoning
    - Independent thought is expected
    """
    return (
        "You are participating in a structured multi-agent deliberation.\n\n"
        f"The question under discussion is:\n\n{query}\n\n"
        "Important:\n"
        "- Multiple defensible positions exist on this topic.\n"
        "- You are not required to agree with others.\n"
        "- Justify your reasoning clearly and specifically.\n"
        "- Consider practical, ethical, and evidence-based dimensions.\n"
        "- Acknowledge genuine uncertainty where it exists."
    )


def validate_query_ambiguity(query: str) -> tuple[bool, str]:
    """
    Lightweight heuristic validation that a user query is sufficiently ambiguous
    for a meaningful corrosion experiment. Does NOT call an LLM.

    Returns
    -------
    (is_valid: bool, reason: str)
    """
    q = query.strip()

    if len(q) < 15:
        return False, "Query is too short. Please provide a full question or paragraph."

    if len(q) > 2000:
        return False, "Query exceeds 2000 characters. Please shorten it."

    # Reject purely factual yes/no trivia (very rough heuristic)
    factual_patterns = [
        "what is the capital",
        "how many",
        "what year was",
        "who invented",
        "what is the formula",
        "how do you spell",
        "what does",
        "define ",
    ]
    lower = q.lower()
    for pattern in factual_patterns:
        if lower.startswith(pattern):
            return (
                False,
                f"Query appears to be a factual lookup ('{pattern}...'). "
                "Context Corrosion requires an ambiguous, multi-sided question. "
                "Try a policy, ethical, or strategic dilemma instead.",
            )

    # Must contain at least one word suggesting debate/policy/judgment
    debate_signals = [
        "should", "ought", "whether", "better", "worse", "best", "most",
        "ethical", "moral", "policy", "regulate", "allow", "ban",
        "tradeoff", "trade-off", "risk", "benefit", "advantage", "disadvantage",
        "responsible", "right", "wrong", "fair", "unfair", "justify",
        "argument", "debate", "position", "stance",
    ]
    if not any(sig in lower for sig in debate_signals):
        return (
            False,
            "Query does not appear to involve a debatable position. "
            "Context Corrosion works best with policy, ethical, or strategic dilemmas "
            "where multiple defensible answers exist.",
        )

    return True, "Query is valid."