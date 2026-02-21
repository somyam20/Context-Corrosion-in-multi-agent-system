# """
# Experiment orchestrator for the Context Corrosion study.

# Phases:
#   1. Independent Thought  – each agent answers independently
#   2. Discussion           – N rounds, dominant speaks first each round
#   3. Final Vote           – each agent casts a final structured position

# Also handles ChromaDB embedding storage and JSON experiment logging.
# """

# from __future__ import annotations

# import json
# import logging
# import os
# import uuid
# from datetime import datetime, timezone
# from typing import Any

# import chromadb

# from app.agents.base_agent import Agent
# from app.config import (
#     ALL_AGENT_IDS,
#     CHROMA_DIR,
#     DOMINANT_AGENT_ID,
#     EXPERIMENTS_DIR,
#     PEER_AGENT_IDS,
#     TASKS,
# )
# from app.metrics.analyzer import compute_full_metrics
# from app.visualization.plotter import generate_all_plots

# logger = logging.getLogger(__name__)


# def _chroma_client() -> chromadb.PersistentClient:
#     return chromadb.PersistentClient(path=CHROMA_DIR)


# def _store_embeddings(
#     collection: chromadb.Collection,
#     experiment_id: str,
#     phase: str,
#     round_num: int,
#     agent_id: str,
#     text: str,
#     embedding: list[float],
# ) -> None:
#     """Upsert a single embedding into ChromaDB."""
#     doc_id = f"{experiment_id}_{phase}_r{round_num}_{agent_id}"
#     collection.upsert(
#         ids=[doc_id],
#         embeddings=[embedding],
#         documents=[text],
#         metadatas=[{
#             "experiment_id": experiment_id,
#             "phase": phase,
#             "round": round_num,
#             "agent_id": agent_id,
#         }],
#     )


# def run_single_experiment(
#     task_id: int,
#     temperature: float,
#     rounds: int,
#     biased_dominant: bool,
# ) -> dict[str, Any]:
#     """
#     Execute one full Context Corrosion experiment run.

#     Parameters
#     ----------
#     task_id : int
#         Index into TASKS dict.
#     temperature : float
#         Sampling temperature for all agents.
#     rounds : int
#         Number of discussion rounds.
#     biased_dominant : bool
#         If True, inject biased framing into the dominant agent.

#     Returns
#     -------
#     dict containing all raw data, metrics, and plot paths.
#     """
#     if task_id not in TASKS:
#         raise ValueError(f"task_id {task_id} not found. Available: {list(TASKS.keys())}")

#     task = TASKS[task_id]
#     experiment_id = str(uuid.uuid4())
#     label = "run_a" if biased_dominant else "run_b"
#     exp_dir = os.path.join(EXPERIMENTS_DIR, experiment_id)
#     plots_dir = os.path.join(exp_dir, "plots")
#     os.makedirs(plots_dir, exist_ok=True)

#     logger.info(
#         "Starting experiment %s | task=%d | biased=%s | rounds=%d | temp=%.2f",
#         experiment_id, task_id, biased_dominant, rounds, temperature,
#     )

#     # ── Build agents ──────────────────────────────────────────────────────────
#     dominant_context = task["biased_framing"] if biased_dominant else None
#     agents: dict[str, Agent] = {}

#     agents[DOMINANT_AGENT_ID] = Agent(
#         agent_id=DOMINANT_AGENT_ID,
#         is_dominant=True,
#         temperature=temperature,
#         extra_system_context=dominant_context,
#     )
#     for pid in PEER_AGENT_IDS:
#         agents[pid] = Agent(
#             agent_id=pid,
#             is_dominant=False,
#             temperature=temperature,
#         )

#     # ── ChromaDB collection ───────────────────────────────────────────────────
#     chroma = _chroma_client()
#     collection_name = f"exp_{experiment_id[:12].replace('-', '')}"
#     collection = chroma.get_or_create_collection(collection_name)

#     # ── Import embedder for storage ───────────────────────────────────────────
#     from app.metrics.analyzer import embed

#     task_desc = task["description"]

#     # ─────────────────────────────────────────────────────────────────────────
#     # PHASE 1 – Independent Thought
#     # ─────────────────────────────────────────────────────────────────────────
#     logger.info("Phase 1: Independent thought")
#     phase1_raw: dict[str, str] = {}
#     phase1_parsed: dict[str, dict] = {}

#     # Dominant speaks in phase 1 as well (independent)
#     for agent_id in ALL_AGENT_IDS:
#         raw, parsed = agents[agent_id].initial_response(task_desc)
#         phase1_raw[agent_id] = raw
#         phase1_parsed[agent_id] = parsed

#         # Store embedding
#         emb = embed([raw])[0].tolist()
#         _store_embeddings(collection, experiment_id, "phase1", 0, agent_id, raw, emb)
#         logger.debug("Agent %s | phase1 raw: %.80s…", agent_id, raw)

#     # ─────────────────────────────────────────────────────────────────────────
#     # PHASE 2 – Discussion Rounds
#     # ─────────────────────────────────────────────────────────────────────────
#     logger.info("Phase 2: %d discussion rounds", rounds)
#     shared_memory: list[dict[str, Any]] = []
#     round_texts: dict[int, dict[str, str]] = {}

#     for rnd in range(1, rounds + 1):
#         logger.info("  Round %d/%d", rnd, rounds)
#         round_texts[rnd] = {}

#         # Dominant speaks FIRST in every round
#         for agent_id in [DOMINANT_AGENT_ID] + PEER_AGENT_IDS:
#             text = agents[agent_id].discussion_response(
#                 task_description=task_desc,
#                 shared_memory=list(shared_memory),  # snapshot before this agent adds
#                 round_num=rnd,
#             )
#             shared_memory.append({
#                 "agent_id": agent_id,
#                 "round": rnd,
#                 "text": text,
#             })
#             round_texts[rnd][agent_id] = text

#             emb = embed([text])[0].tolist()
#             _store_embeddings(collection, experiment_id, "phase2", rnd, agent_id, text, emb)

#     # ─────────────────────────────────────────────────────────────────────────
#     # PHASE 3 – Final Vote
#     # ─────────────────────────────────────────────────────────────────────────
#     logger.info("Phase 3: Final votes")
#     phase3_raw: dict[str, str] = {}
#     phase3_parsed: dict[str, dict] = {}

#     for agent_id in ALL_AGENT_IDS:
#         initial_answer = phase1_parsed[agent_id].get("answer", phase1_raw[agent_id])
#         raw, parsed = agents[agent_id].final_vote(
#             task_description=task_desc,
#             shared_memory=list(shared_memory),
#             initial_answer=str(initial_answer),
#         )
#         phase3_raw[agent_id] = raw
#         phase3_parsed[agent_id] = parsed

#         emb = embed([raw])[0].tolist()
#         _store_embeddings(collection, experiment_id, "phase3", 0, agent_id, raw, emb)

#     # ─────────────────────────────────────────────────────────────────────────
#     # METRICS
#     # ─────────────────────────────────────────────────────────────────────────
#     logger.info("Computing metrics")

#     initial_texts = {
#         aid: phase1_parsed[aid].get("answer", phase1_raw[aid])
#         for aid in ALL_AGENT_IDS
#     }
#     final_texts = {
#         aid: phase3_parsed[aid].get("final_answer", phase3_raw[aid])
#         for aid in ALL_AGENT_IDS
#     }

#     metrics = compute_full_metrics(
#         initial_texts=initial_texts,
#         round_texts=round_texts,
#         final_texts=final_texts,
#     )

#     # ─────────────────────────────────────────────────────────────────────────
#     # PLOTS
#     # ─────────────────────────────────────────────────────────────────────────
#     logger.info("Generating plots")
#     plot_paths = generate_all_plots(metrics, plots_dir, label=label)

#     # ─────────────────────────────────────────────────────────────────────────
#     # LOG JSON
#     # ─────────────────────────────────────────────────────────────────────────
#     result: dict[str, Any] = {
#         "experiment_id": experiment_id,
#         "label": label,
#         "task_id": task_id,
#         "task_title": task["title"],
#         "temperature": temperature,
#         "rounds": rounds,
#         "biased_dominant": biased_dominant,
#         "timestamp": datetime.now(timezone.utc).isoformat(),
#         "phase1": {"raw": phase1_raw, "parsed": phase1_parsed},
#         "phase2": {"shared_memory": shared_memory, "round_texts": round_texts},
#         "phase3": {"raw": phase3_raw, "parsed": phase3_parsed},
#         "metrics": metrics,
#         "plots": plot_paths,
#     }

#     log_path = os.path.join(exp_dir, "experiment.json")
#     with open(log_path, "w") as f:
#         json.dump(result, f, indent=2, default=str)

#     logger.info("Experiment %s complete. Log: %s", experiment_id, log_path)
#     return result
"""
Experiment orchestrator for the Context Corrosion study.

Phases:
  1. Independent Thought  – each agent answers independently
  2. Discussion           – N rounds, dominant speaks first each round
  3. Final Vote           – each agent casts a final structured position

Changes vs v1:
  - EmbeddingStore is passed through to metrics (Issue 7: no re-encoding).
  - Batch embed per round during storage (Issue 1).
  - ChromaDB batches upserts per round (not per agent).
  - Monte Carlo helper run_monte_carlo() added (Issue 6).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

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
from app.metrics.analyzer import EmbeddingStore, compute_full_metrics, compare_runs, embed
from app.visualization.plotter import generate_all_plots

logger = logging.getLogger(__name__)


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
    """Batch upsert embeddings for one round into ChromaDB (Issue 7)."""
    ids = [f"{experiment_id}_{phase}_r{round_num}_{aid}" for aid in agent_ids]
    collection.upsert(
        ids=ids,
        embeddings=vecs.tolist(),
        documents=texts,
        metadatas=[
            {"experiment_id": experiment_id, "phase": phase, "round": round_num, "agent_id": aid}
            for aid in agent_ids
        ],
    )


def run_single_experiment(
    task_id: int,
    temperature: float,
    rounds: int,
    biased_dominant: bool,
) -> dict[str, Any]:
    """
    Execute one full Context Corrosion experiment run.

    Parameters
    ----------
    task_id          : Index into TASKS dict (0-4).
    temperature      : Sampling temperature shared by all agents.
    rounds           : Number of discussion rounds (1-5).
    biased_dominant  : If True, inject biased framing into the dominant agent.

    Returns
    -------
    dict with all raw data, metrics, and plot paths.
    """
    if task_id not in TASKS:
        raise ValueError(f"task_id {task_id} not found. Available: {list(TASKS.keys())}")

    task = TASKS[task_id]
    experiment_id = str(uuid.uuid4())
    label = "run_a" if biased_dominant else "run_b"
    exp_dir = os.path.join(EXPERIMENTS_DIR, experiment_id)
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    logger.info(
        "Experiment %s | task=%d | biased=%s | rounds=%d | temp=%.2f",
        experiment_id, task_id, biased_dominant, rounds, temperature,
    )

    # ── Agents ────────────────────────────────────────────────────────────────
    dominant_ctx = task["biased_framing"] if biased_dominant else None
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

    # ── ChromaDB + EmbeddingStore ─────────────────────────────────────────────
    chroma = _chroma_client()
    col_name = f"exp_{experiment_id[:12].replace('-', '')}"
    collection = chroma.get_or_create_collection(col_name)
    store = EmbeddingStore()

    task_desc = task["description"]

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 – Independent Thought
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Phase 1: Independent thought")
    phase1_raw: dict[str, str] = {}
    phase1_parsed: dict[str, dict] = {}

    for agent_id in ALL_AGENT_IDS:
        raw, parsed = agents[agent_id].initial_response(task_desc)
        phase1_raw[agent_id] = raw
        phase1_parsed[agent_id] = parsed

    # Batch embed all phase1 responses at once (Issue 1)
    p1_ids = list(phase1_raw.keys())
    p1_texts = [phase1_raw[aid] for aid in p1_ids]
    p1_vecs = store.batch_put("phase1", 0, p1_ids, p1_texts)
    _chroma_batch_upsert(collection, experiment_id, "phase1", 0, p1_ids, p1_texts, p1_vecs)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 – Discussion Rounds (dominant speaks first every round)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Phase 2: %d discussion rounds", rounds)
    shared_memory: list[dict[str, Any]] = []
    round_texts: dict[int, dict[str, str]] = {}

    for rnd in range(1, rounds + 1):
        logger.info("  Round %d/%d", rnd, rounds)
        round_texts[rnd] = {}

        for agent_id in [DOMINANT_AGENT_ID] + PEER_AGENT_IDS:
            text = agents[agent_id].discussion_response(
                task_description=task_desc,
                shared_memory=list(shared_memory),
                round_num=rnd,
            )
            shared_memory.append({"agent_id": agent_id, "round": rnd, "text": text})
            round_texts[rnd][agent_id] = text

        # Batch embed the entire round in one call (Issue 1)
        rnd_ids = list(round_texts[rnd].keys())
        rnd_texts_list = [round_texts[rnd][aid] for aid in rnd_ids]
        rnd_vecs = store.batch_put("phase2", rnd, rnd_ids, rnd_texts_list)
        _chroma_batch_upsert(collection, experiment_id, "phase2", rnd, rnd_ids, rnd_texts_list, rnd_vecs)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3 – Final Vote
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Phase 3: Final votes")
    phase3_raw: dict[str, str] = {}
    phase3_parsed: dict[str, dict] = {}

    for agent_id in ALL_AGENT_IDS:
        initial_answer = str(phase1_parsed[agent_id].get("answer", phase1_raw[agent_id]))
        raw, parsed = agents[agent_id].final_vote(
            task_description=task_desc,
            shared_memory=list(shared_memory),
            initial_answer=initial_answer,
        )
        phase3_raw[agent_id] = raw
        phase3_parsed[agent_id] = parsed

    # Batch embed all final votes (Issue 1)
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
        "label": label,
        "task_id": task_id,
        "task_title": task["title"],
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


def run_monte_carlo(
    task_id: int,
    temperature: float,
    rounds: int,
    biased_dominant: bool,
    n_trials: int,
) -> dict[str, Any]:
    """
    Run n_trials independent experiment instances and return aggregated results
    plus a list of per-trial metrics for statistical comparison (Issue 6).

    Parameters
    ----------
    n_trials : int
        Number of independent replications (recommended: 5-20).

    Returns
    -------
    dict with keys: trial_ids, all_metrics (list), aggregate_metrics
    """
    trial_ids: list[str] = []
    all_metrics: list[dict[str, Any]] = []

    for i in range(n_trials):
        logger.info("Monte Carlo trial %d/%d", i + 1, n_trials)
        result = run_single_experiment(task_id, temperature, rounds, biased_dominant)
        trial_ids.append(result["experiment_id"])
        all_metrics.append(result["metrics"])

    # Aggregate scalar keys
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

    return {
        "task_id": task_id,
        "biased_dominant": biased_dominant,
        "n_trials": n_trials,
        "trial_ids": trial_ids,
        "all_metrics": all_metrics,
        "aggregate_metrics": aggregate,
    }