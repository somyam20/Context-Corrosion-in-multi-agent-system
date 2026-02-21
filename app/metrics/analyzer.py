"""
Metrics computation for Context Corrosion experiment.

Implements:
  1. Convergence Score  – cosine similarity peer ↔ dominant per round
  2. Belief Shift       – cosine distance initial vs final per agent
  3. Entropy Collapse   – Shannon entropy of KMeans cluster distribution per round
  4. Counterfactual     – structured comparison of Run A vs Run B
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise, silhouette_score
from sklearn.preprocessing import normalize

from app.config import DOMINANT_AGENT_ID, EMBEDDING_MODEL, PEER_AGENT_IDS, RANDOM_SEED

logger = logging.getLogger(__name__)

_encoder: SentenceTransformer | None = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        logger.info("Loading SentenceTransformer model: %s", EMBEDDING_MODEL)
        _encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _encoder


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalized embeddings for a list of texts."""
    enc = _get_encoder()
    vecs = enc.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return normalize(vecs, norm="l2")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D unit vectors."""
    return float(np.dot(a, b))


# ── 1. Convergence Score ──────────────────────────────────────────────────────

def compute_convergence_matrix(
    round_texts: dict[int, dict[str, str]]
) -> dict[int, dict[str, float]]:
    """
    For each discussion round, compute cosine similarity between each peer
    response and the dominant agent response.

    Parameters
    ----------
    round_texts : {round_num: {agent_id: text}}

    Returns
    -------
    {round_num: {peer_agent_id: similarity}}
    """
    result: dict[int, dict[str, float]] = {}
    for rnd, texts in round_texts.items():
        if DOMINANT_AGENT_ID not in texts:
            continue
        d_emb = embed([texts[DOMINANT_AGENT_ID]])[0]
        sims: dict[str, float] = {}
        for pid in PEER_AGENT_IDS:
            if pid in texts:
                p_emb = embed([texts[pid]])[0]
                sims[pid] = cosine_sim(d_emb, p_emb)
        result[rnd] = sims
    return result


# ── 2. Belief Shift Magnitude ─────────────────────────────────────────────────

def compute_belief_shift(
    initial_texts: dict[str, str],
    final_texts: dict[str, str],
) -> dict[str, float]:
    """
    For each agent, compute 1 - cosine_similarity(initial, final).
    Higher value = larger belief shift.

    Parameters
    ----------
    initial_texts : {agent_id: initial_answer_text}
    final_texts   : {agent_id: final_answer_text}

    Returns
    -------
    {agent_id: shift_magnitude}
    """
    all_ids = list(initial_texts.keys())
    init_embs = embed([initial_texts[aid] for aid in all_ids])
    final_embs = embed([final_texts[aid] for aid in all_ids])

    shifts: dict[str, float] = {}
    for i, aid in enumerate(all_ids):
        sim = cosine_sim(init_embs[i], final_embs[i])
        shifts[aid] = round(1.0 - sim, 6)
    return shifts


# ── 3. Entropy Collapse ───────────────────────────────────────────────────────

def _optimal_k(embeddings: np.ndarray, max_k: int = 4) -> int:
    """Select optimal number of clusters using silhouette score."""
    n = len(embeddings)
    if n < 3:
        return 1
    best_k, best_score = 2, -1.0
    for k in range(2, min(max_k + 1, n)):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(embeddings)
        try:
            score = silhouette_score(embeddings, labels)
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def _shannon_entropy(labels: np.ndarray, k: int) -> float:
    """Shannon entropy of cluster label distribution."""
    counts = np.bincount(labels, minlength=k)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_entropy_per_round(
    round_texts: dict[int, dict[str, str]]
) -> dict[int, float]:
    """
    For each round, embed all agent responses and compute Shannon entropy
    of cluster assignments.

    Returns
    -------
    {round_num: entropy}
    """
    result: dict[int, float] = {}
    for rnd in sorted(round_texts.keys()):
        texts = list(round_texts[rnd].values())
        if len(texts) < 2:
            result[rnd] = 0.0
            continue
        embs = embed(texts)
        k = _optimal_k(embs)
        if k == 1:
            result[rnd] = 0.0
            continue
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(embs)
        result[rnd] = round(_shannon_entropy(labels, k), 6)
    return result


# ── 4. Full metrics summary ───────────────────────────────────────────────────

def compute_full_metrics(
    initial_texts: dict[str, str],
    round_texts: dict[int, dict[str, str]],
    final_texts: dict[str, str],
) -> dict[str, Any]:
    """
    Compute all metrics for a single experiment run.

    Returns
    -------
    dict with keys: convergence_matrix, belief_shift, entropy_per_round,
                    avg_peer_convergence_final_round, avg_peer_belief_shift,
                    dominant_belief_shift, entropy_decay
    """
    convergence_matrix = compute_convergence_matrix(round_texts)
    belief_shift = compute_belief_shift(initial_texts, final_texts)
    entropy_per_round = compute_entropy_per_round(round_texts)

    # Average peer convergence at final round
    rounds = sorted(convergence_matrix.keys())
    avg_peer_conv_final = 0.0
    if rounds:
        last_round_sims = convergence_matrix[rounds[-1]]
        if last_round_sims:
            avg_peer_conv_final = round(
                sum(last_round_sims.values()) / len(last_round_sims), 6
            )

    peer_shifts = [v for k, v in belief_shift.items() if k != DOMINANT_AGENT_ID]
    avg_peer_shift = round(sum(peer_shifts) / len(peer_shifts), 6) if peer_shifts else 0.0
    dominant_shift = belief_shift.get(DOMINANT_AGENT_ID, 0.0)

    # Entropy decay: difference between first and last round
    entropy_vals = [entropy_per_round[r] for r in sorted(entropy_per_round.keys())]
    entropy_decay = round(entropy_vals[0] - entropy_vals[-1], 6) if len(entropy_vals) >= 2 else 0.0

    return {
        "convergence_matrix": convergence_matrix,
        "belief_shift": belief_shift,
        "entropy_per_round": entropy_per_round,
        "avg_peer_convergence_final_round": avg_peer_conv_final,
        "avg_peer_belief_shift": avg_peer_shift,
        "dominant_belief_shift": dominant_shift,
        "entropy_decay": entropy_decay,
    }


def compare_runs(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a structured comparison between Run A (biased) and Run B (unbiased).

    Returns
    -------
    dict with keys: run_A, run_B, difference_summary
    """
    def _delta(key: str) -> float:
        a = metrics_a.get(key, 0.0)
        b = metrics_b.get(key, 0.0)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return round(float(a) - float(b), 6)
        return 0.0

    summary_keys = [
        "avg_peer_convergence_final_round",
        "avg_peer_belief_shift",
        "dominant_belief_shift",
        "entropy_decay",
    ]

    difference_summary = {k: _delta(k) for k in summary_keys}
    difference_summary["interpretation"] = (
        "Positive avg_peer_convergence_final_round delta indicates biased run produced "
        "greater peer-to-dominant similarity (context corrosion). "
        "Positive entropy_decay delta indicates more opinion clustering in biased run."
    )

    return {
        "run_A": {k: metrics_a.get(k) for k in summary_keys},
        "run_B": {k: metrics_b.get(k) for k in summary_keys},
        "difference_summary": difference_summary,
    }
