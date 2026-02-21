# """
# Matplotlib-based visualization for Context Corrosion experiment results.

# Generates:
#   1. Similarity heatmap         – peer × round cosine similarity to dominant
#   2. Similarity progression     – line graph per peer agent
#   3. Entropy decay graph        – entropy across rounds
#   4. Belief shift comparison    – bar chart dominant vs peers
# """

# from __future__ import annotations

# import logging
# import os
# from typing import Any

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import numpy as np
# from dotenv import load_dotenv
# load_dotenv()
# logger = logging.getLogger(__name__)


# def _ensure_dir(path: str) -> None:
#     os.makedirs(path, exist_ok=True)


# def plot_similarity_heatmap(
#     convergence_matrix: dict[int, dict[str, float]],
#     output_path: str,
#     title: str = "Peer-to-Dominant Cosine Similarity Heatmap",
# ) -> str:
#     """
#     Render a heatmap of peer cosine similarity to dominant agent per round.

#     Returns
#     -------
#     str : path to saved figure
#     """
#     rounds = sorted(convergence_matrix.keys())
#     if not rounds:
#         logger.warning("Empty convergence matrix; skipping heatmap.")
#         return ""

#     peers = sorted(
#         set(p for r in rounds for p in convergence_matrix[r].keys())
#     )
#     data = np.array(
#         [[convergence_matrix[r].get(p, 0.0) for r in rounds] for p in peers]
#     )

#     fig, ax = plt.subplots(figsize=(max(6, len(rounds) * 1.5), max(4, len(peers) * 1.2)))
#     im = ax.imshow(data, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")

#     ax.set_xticks(range(len(rounds)))
#     ax.set_xticklabels([f"Round {r}" for r in rounds])
#     ax.set_yticks(range(len(peers)))
#     ax.set_yticklabels(peers)
#     ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
#     ax.set_xlabel("Discussion Round")
#     ax.set_ylabel("Peer Agent")

#     plt.colorbar(im, ax=ax, label="Cosine Similarity")

#     for i in range(len(peers)):
#         for j in range(len(rounds)):
#             ax.text(
#                 j, i, f"{data[i, j]:.2f}",
#                 ha="center", va="center",
#                 color="black" if data[i, j] < 0.7 else "white",
#                 fontsize=9,
#             )

#     plt.tight_layout()
#     _ensure_dir(os.path.dirname(output_path))
#     fig.savefig(output_path, dpi=120)
#     plt.close(fig)
#     logger.info("Saved heatmap → %s", output_path)
#     return output_path


# def plot_similarity_progression(
#     convergence_matrix: dict[int, dict[str, float]],
#     output_path: str,
#     title: str = "Peer-to-Dominant Similarity Progression",
# ) -> str:
#     """
#     Line graph showing how each peer's similarity to dominant changes per round.
#     """
#     rounds = sorted(convergence_matrix.keys())
#     if not rounds:
#         return ""

#     peers = sorted(
#         set(p for r in rounds for p in convergence_matrix[r].keys())
#     )

#     fig, ax = plt.subplots(figsize=(8, 5))
#     colors = plt.cm.tab10(np.linspace(0, 0.5, len(peers)))

#     for peer, color in zip(peers, colors):
#         sims = [convergence_matrix[r].get(peer, 0.0) for r in rounds]
#         ax.plot(rounds, sims, marker="o", label=peer, color=color, linewidth=2)
#         for r, s in zip(rounds, sims):
#             ax.annotate(f"{s:.2f}", (r, s), textcoords="offset points",
#                         xytext=(0, 6), ha="center", fontsize=8)

#     ax.set_xticks(rounds)
#     ax.set_xticklabels([f"Round {r}" for r in rounds])
#     ax.set_ylim(0, 1.05)
#     ax.set_xlabel("Discussion Round")
#     ax.set_ylabel("Cosine Similarity to Dominant")
#     ax.set_title(title, fontsize=13, fontweight="bold")
#     ax.legend(title="Peer Agent")
#     ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
#     ax.grid(True, linestyle="--", alpha=0.4)

#     plt.tight_layout()
#     _ensure_dir(os.path.dirname(output_path))
#     fig.savefig(output_path, dpi=120)
#     plt.close(fig)
#     logger.info("Saved progression plot → %s", output_path)
#     return output_path


# def plot_entropy_decay(
#     entropy_per_round: dict[int, float],
#     output_path: str,
#     title: str = "Opinion Entropy Across Discussion Rounds",
# ) -> str:
#     """
#     Bar + line chart of Shannon entropy per round, illustrating opinion collapse.
#     """
#     rounds = sorted(entropy_per_round.keys())
#     entropies = [entropy_per_round[r] for r in rounds]

#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.bar(rounds, entropies, color="steelblue", alpha=0.6, label="Entropy")
#     ax.plot(rounds, entropies, marker="D", color="navy", linewidth=2, label="Trend")

#     for r, e in zip(rounds, entropies):
#         ax.text(r, e + 0.01, f"{e:.3f}", ha="center", va="bottom", fontsize=9)

#     ax.set_xticks(rounds)
#     ax.set_xticklabels([f"Round {r}" for r in rounds])
#     ax.set_xlabel("Discussion Round")
#     ax.set_ylabel("Shannon Entropy (bits)")
#     ax.set_title(title, fontsize=13, fontweight="bold")
#     ax.legend()
#     ax.grid(True, axis="y", linestyle="--", alpha=0.4)

#     plt.tight_layout()
#     _ensure_dir(os.path.dirname(output_path))
#     fig.savefig(output_path, dpi=120)
#     plt.close(fig)
#     logger.info("Saved entropy plot → %s", output_path)
#     return output_path


# def plot_belief_shift(
#     belief_shift: dict[str, float],
#     output_path: str,
#     title: str = "Belief Shift Magnitude per Agent",
# ) -> str:
#     """
#     Horizontal bar chart comparing belief shift magnitude per agent.
#     Dominant agent is highlighted in a different color.
#     """
#     agents = sorted(belief_shift.keys())
#     shifts = [belief_shift[a] for a in agents]
#     colors = ["crimson" if a == "D" else "steelblue" for a in agents]

#     fig, ax = plt.subplots(figsize=(8, max(4, len(agents) * 0.8)))
#     bars = ax.barh(agents, shifts, color=colors, alpha=0.8)

#     for bar, val in zip(bars, shifts):
#         ax.text(
#             bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
#             f"{val:.4f}", va="center", ha="left", fontsize=9,
#         )

#     ax.set_xlabel("Belief Shift (1 – cosine_similarity(initial, final))")
#     ax.set_title(title, fontsize=13, fontweight="bold")
#     ax.set_xlim(0, max(shifts) * 1.2 + 0.01 if shifts else 1.0)

#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor="crimson", alpha=0.8, label="Dominant (D)"),
#         Patch(facecolor="steelblue", alpha=0.8, label="Peer Agents"),
#     ]
#     ax.legend(handles=legend_elements, loc="lower right")
#     ax.grid(True, axis="x", linestyle="--", alpha=0.4)

#     plt.tight_layout()
#     _ensure_dir(os.path.dirname(output_path))
#     fig.savefig(output_path, dpi=120)
#     plt.close(fig)
#     logger.info("Saved belief-shift plot → %s", output_path)
#     return output_path


# def generate_all_plots(
#     metrics: dict[str, Any],
#     plots_dir: str,
#     label: str = "",
# ) -> dict[str, str]:
#     """
#     Generate all four plots and return a dict of {plot_name: file_path}.

#     Parameters
#     ----------
#     metrics : dict
#         Output of compute_full_metrics.
#     plots_dir : str
#         Directory to save plots.
#     label : str
#         Optional suffix for filenames (e.g., "run_a").
#     """
#     suffix = f"_{label}" if label else ""
#     paths: dict[str, str] = {}

#     paths["similarity_heatmap"] = plot_similarity_heatmap(
#         metrics["convergence_matrix"],
#         os.path.join(plots_dir, f"similarity_heatmap{suffix}.png"),
#     )
#     paths["similarity_progression"] = plot_similarity_progression(
#         metrics["convergence_matrix"],
#         os.path.join(plots_dir, f"similarity_progression{suffix}.png"),
#     )
#     paths["entropy_plot"] = plot_entropy_decay(
#         metrics["entropy_per_round"],
#         os.path.join(plots_dir, f"entropy_decay{suffix}.png"),
#     )
#     paths["shift_plot"] = plot_belief_shift(
#         metrics["belief_shift"],
#         os.path.join(plots_dir, f"belief_shift{suffix}.png"),
#     )

#     return paths
"""
Matplotlib visualizations for Context Corrosion experiment.

Updated to consume new metric shapes:
  - convergence_matrix  : {round: {peer_to_dominant, peer_to_peer_avg, dominant_drift}}
  - lifecycle_entropy   : {phase1, round_1…round_n, final}
  - directional_convergence : {directional_delta, initial_alignment, final_alignment, raw_belief_shift}
"""

from __future__ import annotations

import logging
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ── 1. Triangulated similarity heatmap ───────────────────────────────────────

def plot_similarity_heatmap(
    convergence_matrix: dict[int, dict[str, Any]],
    output_path: str,
    title: str = "Peer-to-Dominant Cosine Similarity Heatmap",
) -> str:
    """
    Heatmap of peer→dominant cosine similarity per round.
    Now reads from convergence_matrix[rnd]['peer_to_dominant'] (Issue 4).
    """
    rounds = sorted(convergence_matrix.keys())
    if not rounds:
        return ""

    peers = sorted(
        set(p for r in rounds for p in convergence_matrix[r].get("peer_to_dominant", {}).keys())
    )
    data = np.array(
        [[convergence_matrix[r]["peer_to_dominant"].get(p, 0.0) for r in rounds] for p in peers]
    )

    fig, ax = plt.subplots(figsize=(max(6, len(rounds) * 1.8), max(4, len(peers) * 1.2)))
    im = ax.imshow(data, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels([f"Round {r}" for r in rounds])
    ax.set_yticks(range(len(peers)))
    ax.set_yticklabels(peers)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Discussion Round")
    ax.set_ylabel("Peer Agent")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    for i in range(len(peers)):
        for j in range(len(rounds)):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.7 else "black", fontsize=9)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved heatmap → %s", output_path)
    return output_path


# ── 2. Triangulated similarity progression ────────────────────────────────────

def plot_similarity_progression(
    convergence_matrix: dict[int, dict[str, Any]],
    output_path: str,
    title: str = "Convergence Triangulation by Round",
) -> str:
    """
    Three-panel line graph:
      Top    – peer→dominant similarity per peer
      Middle – peer-to-peer average similarity
      Bottom – dominant agent drift (similarity to its own round-1 response)
    """
    rounds = sorted(convergence_matrix.keys())
    if not rounds:
        return ""

    peers = sorted(
        set(p for r in rounds for p in convergence_matrix[r].get("peer_to_dominant", {}).keys())
    )
    peer_peer_avg = [convergence_matrix[r].get("peer_to_peer_avg", 0.0) for r in rounds]
    dom_drift = [convergence_matrix[r].get("dominant_drift", 1.0) for r in rounds]

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    colors = plt.cm.tab10(np.linspace(0, 0.5, max(len(peers), 1)))

    # Panel 1: peer → dominant
    ax = axes[0]
    for peer, color in zip(peers, colors):
        sims = [convergence_matrix[r]["peer_to_dominant"].get(peer, 0.0) for r in rounds]
        ax.plot(rounds, sims, marker="o", label=peer, color=color, linewidth=2)
    ax.set_ylabel("Peer → Dominant Sim")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Peer", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("Peer-to-Dominant Alignment", fontsize=10)

    # Panel 2: peer ↔ peer
    ax = axes[1]
    ax.plot(rounds, peer_peer_avg, marker="s", color="darkorange", linewidth=2, label="P↔P avg")
    ax.fill_between(rounds, peer_peer_avg, alpha=0.15, color="darkorange")
    ax.set_ylabel("Peer ↔ Peer Sim")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("Peer-to-Peer Herding", fontsize=10)

    # Panel 3: dominant drift
    ax = axes[2]
    ax.plot(rounds, dom_drift, marker="D", color="crimson", linewidth=2, label="D drift")
    ax.set_ylabel("D Drift (sim to D_round1)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Discussion Round")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("Dominant Agent Self-Consistency", fontsize=10)
    ax.set_xticks(rounds)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved progression → %s", output_path)
    return output_path


# ── 3. Full lifecycle entropy ─────────────────────────────────────────────────

def plot_entropy_decay(
    lifecycle_entropy: dict[str, float],
    output_path: str,
    title: str = "Opinion Entropy — Full Experiment Lifecycle",
) -> str:
    """
    Bar + line chart across full lifecycle: phase1, round_1…round_n, final.
    Replaces round-only entropy view (Issue 2).
    """
    order = ["phase1"] + sorted(
        [k for k in lifecycle_entropy if k.startswith("round_")]
    ) + ["final"]
    stages = [s for s in order if s in lifecycle_entropy]
    values = [lifecycle_entropy[s] for s in stages]
    labels = [s.replace("_", " ").title() for s in stages]

    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(max(8, len(stages) * 1.4), 5))
    bars = ax.bar(x, values, color="steelblue", alpha=0.65)
    ax.plot(x, values, marker="D", color="navy", linewidth=2, label="Entropy trend")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)

    # Shade phase boundaries
    if "phase1" in stages:
        ax.axvspan(-0.5, 0.5, color="green", alpha=0.06, label="Phase 1")
    if "final" in stages:
        ax.axvspan(len(stages) - 1.5, len(stages) - 0.5, color="red", alpha=0.06, label="Phase 3")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Shannon Entropy (bits, k=2 clusters)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved entropy lifecycle plot → %s", output_path)
    return output_path


# ── 4. Directional convergence bar chart ─────────────────────────────────────

def plot_belief_shift(
    directional_convergence: dict[str, Any],
    output_path: str,
    title: str = "Directional Convergence Toward Dominant Agent",
) -> str:
    """
    Three-section horizontal bar chart per peer agent (Issue 5):
      - Initial alignment to dominant
      - Final alignment to dominant
      - Directional delta (Δ = final - initial; positive = corrosion)

    Also shows dominant's own drift (raw belief shift) for reference.
    """
    dd = directional_convergence.get("directional_delta", {})
    ia = directional_convergence.get("initial_alignment", {})
    fa = directional_convergence.get("final_alignment", {})
    raw = directional_convergence.get("raw_belief_shift", {})
    dom_drift = directional_convergence.get("dominant_self_drift", 0.0)

    peers = sorted(dd.keys())
    if not peers:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(peers) * 0.9 + 2)))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Left: alignment before/after
    ax = axes[0]
    y = np.arange(len(peers))
    width = 0.35
    bars_init = ax.barh(y - width / 2, [ia.get(p, 0) for p in peers],
                        width, label="Initial Alignment", color="skyblue", alpha=0.8)
    bars_final = ax.barh(y + width / 2, [fa.get(p, 0) for p in peers],
                         width, label="Final Alignment", color="steelblue", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(peers)
    ax.set_xlabel("Cosine Similarity to Dominant")
    ax.set_title("Alignment Before vs After Discussion", fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.legend()
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    # Right: directional delta
    ax2 = axes[1]
    delta_vals = [dd.get(p, 0) for p in peers]
    colors = ["crimson" if v > 0 else "royalblue" for v in delta_vals]
    ax2.barh(peers, delta_vals, color=colors, alpha=0.8)
    ax2.axvline(0, color="black", linewidth=0.8)
    for i, (p, v) in enumerate(zip(peers, delta_vals)):
        ax2.text(v + (0.003 if v >= 0 else -0.003), i, f"{v:+.4f}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax2.set_xlabel("Δ Alignment (final − initial); positive = corrosion")
    ax2.set_title(
        f"Directional Δ per Peer\n(D self-drift: {dom_drift:.4f})", fontsize=10
    )
    ax2.grid(True, axis="x", linestyle="--", alpha=0.4)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="crimson", alpha=0.8, label="Moved toward D (corrosion)"),
        Patch(facecolor="royalblue", alpha=0.8, label="Moved away from D"),
    ]
    ax2.legend(handles=legend_els, fontsize=8)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved directional convergence plot → %s", output_path)
    return output_path


# ── Master generate function ──────────────────────────────────────────────────

def generate_all_plots(
    metrics: dict[str, Any],
    plots_dir: str,
    label: str = "",
) -> dict[str, str]:
    """
    Generate all four plots for one experiment run.

    Parameters
    ----------
    metrics   : output of compute_full_metrics
    plots_dir : directory to save .png files
    label     : optional filename suffix (e.g. "run_a")
    """
    suffix = f"_{label}" if label else ""
    paths: dict[str, str] = {}

    paths["similarity_heatmap"] = plot_similarity_heatmap(
        metrics["convergence_matrix"],
        os.path.join(plots_dir, f"similarity_heatmap{suffix}.png"),
    )
    paths["similarity_progression"] = plot_similarity_progression(
        metrics["convergence_matrix"],
        os.path.join(plots_dir, f"similarity_progression{suffix}.png"),
    )
    paths["entropy_plot"] = plot_entropy_decay(
        metrics["lifecycle_entropy"],
        os.path.join(plots_dir, f"entropy_lifecycle{suffix}.png"),
    )
    paths["shift_plot"] = plot_belief_shift(
        metrics["directional_convergence"],
        os.path.join(plots_dir, f"directional_convergence{suffix}.png"),
    )

    return paths