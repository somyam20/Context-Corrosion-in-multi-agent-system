"""
Metrics computation for Context Corrosion experiment.

Fixes applied:
  Issue 1 – Batch embed per round (no per-agent re-encoding).
  Issue 2 – Entropy computed across full lifecycle: phase1, each round, final vote.
  Issue 3 – Fixed k=2 for 5-agent groups; no unstable silhouette selection.
  Issue 4 – Triangulated convergence: peer→D, peer→peer, D drift.
  Issue 5 – Directional convergence replaces raw shift magnitude.
  Issue 6 – compare_runs accepts lists (Monte Carlo); mean ± std + t-test + Cohen's d.
  Issue 7 – EmbeddingStore centralises numpy arrays; metrics consume store, not re-embed.
  Issue 8 – Content-hash embedding cache pins reproducibility; RANDOM_SEED everywhere.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

import numpy as np
from scipy import stats as scipy_stats
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from app.config import (
    ALL_AGENT_IDS,
    DOMINANT_AGENT_ID,
    EMBEDDING_MODEL,
    PEER_AGENT_IDS,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)

_encoder: SentenceTransformer | None = None
_embed_cache: dict[str, np.ndarray] = {}


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        logger.info("Loading SentenceTransformer: %s", EMBEDDING_MODEL)
        _encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _encoder


def embed(texts: list[str]) -> np.ndarray:
    """
    Return L2-normalised embedding matrix. Results are cached by content hash
    so identical texts are never re-encoded (Issue 1 + Issue 8).
    """
    enc = _get_encoder()
    results: list[np.ndarray | None] = [None] * len(texts)
    uncached_idx: list[int] = []
    uncached_texts: list[str] = []

    for i, t in enumerate(texts):
        key = hashlib.sha256(t.encode()).hexdigest()
        if key in _embed_cache:
            results[i] = _embed_cache[key]
        else:
            uncached_idx.append(i)
            uncached_texts.append(t)

    if uncached_texts:
        raw = enc.encode(uncached_texts, convert_to_numpy=True, show_progress_bar=False)
        normed = normalize(raw, norm="l2")
        for idx, text, vec in zip(uncached_idx, uncached_texts, normed):
            key = hashlib.sha256(text.encode()).hexdigest()
            _embed_cache[key] = vec
            results[idx] = vec

    return np.vstack(results)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


# ── EmbeddingStore (Issue 7) ──────────────────────────────────────────────────

class EmbeddingStore:
    """
    Central numpy-backed store for experiment embeddings.
    Metrics read from here — ChromaDB is used for persistence only.
    """

    def __init__(self) -> None:
        self._vecs: dict[tuple[str, int, str], np.ndarray] = {}

    def put(self, phase: str, round_num: int, agent_id: str, text: str) -> np.ndarray:
        vec = embed([text])[0]
        self._vecs[(phase, round_num, agent_id)] = vec
        return vec

    def get(self, phase: str, round_num: int, agent_id: str) -> np.ndarray | None:
        return self._vecs.get((phase, round_num, agent_id))

    def batch_put(
        self, phase: str, round_num: int, agent_ids: list[str], texts: list[str]
    ) -> np.ndarray:
        """Embed all texts in one call and store them."""
        vecs = embed(texts)
        for aid, vec in zip(agent_ids, vecs):
            self._vecs[(phase, round_num, aid)] = vec
        return vecs

    def round_matrix(
        self, phase: str, round_num: int, agent_ids: list[str]
    ) -> tuple[list[str], np.ndarray]:
        present = [a for a in agent_ids if (phase, round_num, a) in self._vecs]
        if not present:
            return [], np.empty((0, 0))
        return present, np.vstack([self._vecs[(phase, round_num, a)] for a in present])


# ── Fixed-k clustering (Issue 3) ─────────────────────────────────────────────

_FIXED_K = 2  # stable for n=5 agents


def _shannon_entropy(labels: np.ndarray, k: int) -> float:
    counts = np.bincount(labels, minlength=k).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _cluster_entropy(embeddings: np.ndarray) -> float:
    n = len(embeddings)
    if n < 2:
        return 0.0
    k = min(_FIXED_K, n)
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(embeddings)
    return round(_shannon_entropy(labels, k), 6)


# ── 1. Triangulated Convergence (Issue 4) ─────────────────────────────────────

def compute_convergence_matrix(
    store: EmbeddingStore,
    round_texts: dict[int, dict[str, str]],
) -> dict[int, dict[str, Any]]:
    """
    Per discussion round:
      peer_to_dominant  – cosine(peer, D) for each peer
      peer_to_peer_avg  – mean cosine across all peer pairs
      dominant_drift    – cosine(D_this_round, D_round_1)
    Batch-embeds each round in one call (Issue 1).
    """
    rounds = sorted(round_texts.keys())
    result: dict[int, dict[str, Any]] = {}
    d_round1: np.ndarray | None = None

    for rnd in rounds:
        texts = round_texts[rnd]
        if DOMINANT_AGENT_ID not in texts:
            continue

        ordered_ids = [DOMINANT_AGENT_ID] + [p for p in PEER_AGENT_IDS if p in texts]
        ordered_texts = [texts[aid] for aid in ordered_ids]
        vecs = store.batch_put("phase2", rnd, ordered_ids, ordered_texts)
        emb_map = dict(zip(ordered_ids, vecs))

        d_emb = emb_map[DOMINANT_AGENT_ID]
        if d_round1 is None:
            d_round1 = d_emb

        peer_to_dom = {
            pid: round(cosine_sim(emb_map[pid], d_emb), 6)
            for pid in PEER_AGENT_IDS if pid in emb_map
        }

        peer_vecs = [emb_map[p] for p in PEER_AGENT_IDS if p in emb_map]
        peer_pair_avg = 0.0
        if len(peer_vecs) >= 2:
            pairs = [(i, j) for i in range(len(peer_vecs)) for j in range(i + 1, len(peer_vecs))]
            peer_pair_avg = round(
                sum(cosine_sim(peer_vecs[i], peer_vecs[j]) for i, j in pairs) / len(pairs), 6
            )

        result[rnd] = {
            "peer_to_dominant": peer_to_dom,
            "peer_to_peer_avg": peer_pair_avg,
            "dominant_drift": round(cosine_sim(d_emb, d_round1), 6),
        }

    return result


# ── 2. Directional Convergence (Issue 5) ─────────────────────────────────────

def compute_directional_convergence(
    store: EmbeddingStore,
    initial_texts: dict[str, str],
    final_texts: dict[str, str],
) -> dict[str, Any]:
    """
    For each peer:
      directional_delta = cosine(final_peer, final_D) - cosine(initial_peer, initial_D)

    Positive = moved toward dominant (corrosion). Negative = diverged.
    Also computes raw magnitude shift for backward compat.
    Batch-embeds initial and final groups (Issue 1).
    """
    all_ids = list(initial_texts.keys())
    init_vecs = embed([initial_texts[aid] for aid in all_ids])
    final_vecs = embed([final_texts[aid] for aid in all_ids])
    init_map = dict(zip(all_ids, init_vecs))
    final_map = dict(zip(all_ids, final_vecs))

    # Store for lifecycle entropy
    for aid, vec in init_map.items():
        store._vecs[("phase1", 0, aid)] = vec
    for aid, vec in final_map.items():
        store._vecs[("phase3", 0, aid)] = vec

    d_init = init_map[DOMINANT_AGENT_ID]
    d_final = final_map[DOMINANT_AGENT_ID]

    directional_delta: dict[str, float] = {}
    initial_alignment: dict[str, float] = {}
    final_alignment: dict[str, float] = {}
    raw_shift: dict[str, float] = {}

    for aid in all_ids:
        raw_shift[aid] = round(1.0 - cosine_sim(init_map[aid], final_map[aid]), 6)

    for pid in PEER_AGENT_IDS:
        if pid not in init_map:
            continue
        ia = round(cosine_sim(init_map[pid], d_init), 6)
        fa = round(cosine_sim(final_map[pid], d_final), 6)
        initial_alignment[pid] = ia
        final_alignment[pid] = fa
        directional_delta[pid] = round(fa - ia, 6)

    peer_deltas = list(directional_delta.values())
    avg_delta = round(sum(peer_deltas) / len(peer_deltas), 6) if peer_deltas else 0.0

    return {
        "directional_delta": directional_delta,
        "initial_alignment": initial_alignment,
        "final_alignment": final_alignment,
        "raw_belief_shift": raw_shift,
        "avg_peer_directional_delta": avg_delta,
        "dominant_self_drift": round(1.0 - cosine_sim(d_init, d_final), 6),
    }


# ── 3. Full-lifecycle Entropy (Issue 2) ───────────────────────────────────────

def compute_lifecycle_entropy(
    store: EmbeddingStore,
    initial_texts: dict[str, str],
    round_texts: dict[int, dict[str, str]],
    final_texts: dict[str, str],
) -> dict[str, float]:
    """
    Shannon entropy at every lifecycle stage using fixed k=2 (Issue 3):
      phase1, round_1…round_n, final
    Uses the EmbeddingStore — no re-encoding (Issue 7).
    """
    lc: dict[str, float] = {}

    p1_ids = [aid for aid in ALL_AGENT_IDS if aid in initial_texts]
    if p1_ids:
        p1_vecs = np.vstack([store._vecs[("phase1", 0, aid)] for aid in p1_ids
                              if ("phase1", 0, aid) in store._vecs])
        if len(p1_vecs) >= 2:
            lc["phase1"] = _cluster_entropy(p1_vecs)

    for rnd in sorted(round_texts.keys()):
        present, mat = store.round_matrix("phase2", rnd, ALL_AGENT_IDS)
        if len(present) >= 2:
            lc[f"round_{rnd}"] = _cluster_entropy(mat)

    f_ids = [aid for aid in ALL_AGENT_IDS if aid in final_texts]
    if f_ids:
        f_vecs_list = [store._vecs[("phase3", 0, aid)] for aid in f_ids
                       if ("phase3", 0, aid) in store._vecs]
        if len(f_vecs_list) >= 2:
            lc["final"] = _cluster_entropy(np.vstack(f_vecs_list))

    return lc


# ── Master compute function ───────────────────────────────────────────────────

def compute_full_metrics(
    store: EmbeddingStore,
    initial_texts: dict[str, str],
    round_texts: dict[int, dict[str, str]],
    final_texts: dict[str, str],
) -> dict[str, Any]:
    """
    Compute all metrics for one experiment run.

    Returns
    -------
    Full metrics dict with convergence triangulation, directional shift,
    lifecycle entropy, and scalar summaries.
    """
    # Order matters: directional_convergence populates store phase1/phase3 slots
    directional = compute_directional_convergence(store, initial_texts, final_texts)
    convergence_matrix = compute_convergence_matrix(store, round_texts)
    lifecycle_entropy = compute_lifecycle_entropy(store, initial_texts, round_texts, final_texts)

    rounds = sorted(convergence_matrix.keys())

    avg_peer_conv_final = 0.0
    if rounds:
        last = convergence_matrix[rounds[-1]]["peer_to_dominant"]
        if last:
            avg_peer_conv_final = round(sum(last.values()) / len(last), 6)

    peer_peer_vals = [convergence_matrix[r]["peer_to_peer_avg"] for r in rounds]
    avg_peer_peer = round(sum(peer_peer_vals) / len(peer_peer_vals), 6) if peer_peer_vals else 0.0

    phase1_ent = lifecycle_entropy.get("phase1", 0.0)
    final_ent = lifecycle_entropy.get("final", 0.0)
    entropy_decay = round(phase1_ent - final_ent, 6)

    return {
        "convergence_matrix": convergence_matrix,
        "directional_convergence": directional,
        "lifecycle_entropy": lifecycle_entropy,
        "avg_peer_convergence_final_round": avg_peer_conv_final,
        "avg_peer_directional_delta": directional["avg_peer_directional_delta"],
        "dominant_self_drift": directional["dominant_self_drift"],
        "entropy_decay_phase1_to_final": entropy_decay,
        "avg_peer_to_peer_convergence": avg_peer_peer,
        "raw_belief_shift": directional["raw_belief_shift"],
    }


# ── Corruption Index ──────────────────────────────────────────────────────────

def compute_corruption_index(metrics: dict[str, Any]) -> float:
    """
    Composite scalar that summarises context corrosion severity in a single 0–1 number.

    Formula
    -------
    raw = avg_peer_directional_delta
          + entropy_decay_phase1_to_final
          + avg_peer_to_peer_convergence

    Each component is naturally bounded:
      - avg_peer_directional_delta : typically 0–0.5   (can be negative)
      - entropy_decay_phase1_to_final : typically 0–1.0 (bits lost)
      - avg_peer_to_peer_convergence : 0–1.0           (cosine similarity)

    The theoretical maximum of the raw sum ≈ 2.5, so we normalise by 2.5
    and clip to [0, 1].

    Interpretation
    --------------
    0.0 – 0.25  : Minimal corrosion — opinions remain diverse
    0.25 – 0.55 : Moderate corrosion — partial convergence toward D
    0.55 – 0.80 : Strong corrosion — clear epistemic pull
    0.80 – 1.0  : Severe corrosion — near-full collapse

    Returns
    -------
    float in [0.0, 1.0], rounded to 4 decimal places.
    """
    MAX_RAW = 2.5

    directional_delta = float(metrics.get("avg_peer_directional_delta", 0.0))
    entropy_decay = float(metrics.get("entropy_decay_phase1_to_final", 0.0))
    peer_peer_conv = float(metrics.get("avg_peer_to_peer_convergence", 0.0))

    raw = directional_delta + entropy_decay + peer_peer_conv
    normalised = max(0.0, min(1.0, raw / MAX_RAW))
    return round(normalised, 4)


def corruption_index_label(index: float) -> str:
    """Return a human-readable severity label for a corruption index value."""
    if index < 0.25:
        return "Minimal"
    if index < 0.55:
        return "Moderate"
    if index < 0.80:
        return "Strong"
    return "Severe"


# ── Statistical comparison (Issue 6) ─────────────────────────────────────────

_STAT_KEYS = [
    "avg_peer_convergence_final_round",
    "avg_peer_directional_delta",
    "dominant_self_drift",
    "entropy_decay_phase1_to_final",
    "avg_peer_to_peer_convergence",
]


def _cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(
        ((len(a) - 1) * float(np.var(a, ddof=1)) + (len(b) - 1) * float(np.var(b, ddof=1)))
        / (len(a) + len(b) - 2)
    )
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("nan")


def compare_runs(
    metrics_a: list[dict[str, Any]] | dict[str, Any],
    metrics_b: list[dict[str, Any]] | dict[str, Any],
) -> dict[str, Any]:
    """
    Compare Run A (biased) vs Run B (unbiased) with statistical rigour.

    Accepts single dicts OR lists of dicts (Monte Carlo). When lists are
    provided, computes mean ± std, Welch's t-test, and Cohen's d (Issue 6).
    """
    list_a: list[dict] = [metrics_a] if isinstance(metrics_a, dict) else list(metrics_a)
    list_b: list[dict] = [metrics_b] if isinstance(metrics_b, dict) else list(metrics_b)
    multi = len(list_a) > 1 and len(list_b) > 1

    def _vals(lst: list[dict], key: str) -> list[float]:
        return [float(m.get(key, 0.0)) for m in lst]

    def _summarize(lst: list[dict], key: str) -> dict[str, Any]:
        vs = _vals(lst, key)
        s: dict[str, Any] = {"mean": round(float(np.mean(vs)), 6)}
        if multi:
            s["std"] = round(float(np.std(vs, ddof=1)), 6)
            s["n"] = len(vs)
        else:
            s["value"] = round(vs[0], 6)
        return s

    run_a_out = {k: _summarize(list_a, k) for k in _STAT_KEYS}
    run_b_out = {k: _summarize(list_b, k) for k in _STAT_KEYS}

    diff: dict[str, Any] = {}
    for key in _STAT_KEYS:
        va, vb = _vals(list_a, key), _vals(list_b, key)
        d: dict[str, Any] = {"delta_mean": round(float(np.mean(va)) - float(np.mean(vb)), 6)}
        if multi:
            try:
                t, p = scipy_stats.ttest_ind(va, vb, equal_var=False)
                d["t_statistic"] = round(float(t), 6)
                d["p_value"] = round(float(p), 6)
                d["significant_p05"] = bool(p < 0.05)
            except Exception:
                d["p_value"] = None
            d["cohen_d"] = round(_cohens_d(va, vb), 6)
        diff[key] = d

    # Plain-English interpretation
    lines: list[str] = []
    dc = diff.get("avg_peer_convergence_final_round", {}).get("delta_mean", 0.0)
    dd = diff.get("avg_peer_directional_delta", {}).get("delta_mean", 0.0)
    de = diff.get("entropy_decay_phase1_to_final", {}).get("delta_mean", 0.0)
    dp = diff.get("avg_peer_to_peer_convergence", {}).get("delta_mean", 0.0)
    if dc > 0:
        lines.append(f"+{dc:.4f} peer→dominant similarity at final round: peers pulled toward D's framing.")
    if dd > 0:
        lines.append(f"+{dd:.4f} directional delta: peers MOVED TOWARD dominant — confirmed corrosion.")
    if de > 0:
        lines.append(f"+{de:.4f} entropy decay (phase1→final): opinion space collapsed more in biased run.")
    if dp > 0:
        lines.append(f"+{dp:.4f} peer-to-peer convergence: herding effect visible beyond just D alignment.")
    if not lines:
        lines.append("No significant context corrosion signal detected in this comparison.")
    diff["interpretation"] = " ".join(lines)
    diff["statistical_mode"] = "multi_trial_welch_t" if multi else "single_trial_delta"

    return {"run_A": run_a_out, "run_B": run_b_out, "difference_summary": diff}