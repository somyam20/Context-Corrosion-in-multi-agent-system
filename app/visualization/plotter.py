"""
Matplotlib-based visualization for Context Corrosion experiment results.

Generates:
  1. Similarity heatmap         – peer × round cosine similarity to dominant
  2. Similarity progression     – line graph per peer agent
  3. Entropy decay graph        – entropy across rounds
  4. Belief shift comparison    – bar chart dominant vs peers
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
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_similarity_heatmap(
    convergence_matrix: dict[int, dict[str, float]],
    output_path: str,
    title: str = "Peer-to-Dominant Cosine Similarity Heatmap",
) -> str:
    """
    Render a heatmap of peer cosine similarity to dominant agent per round.

    Returns
    -------
    str : path to saved figure
    """
    rounds = sorted(convergence_matrix.keys())
    if not rounds:
        logger.warning("Empty convergence matrix; skipping heatmap.")
        return ""

    peers = sorted(
        set(p for r in rounds for p in convergence_matrix[r].keys())
    )
    data = np.array(
        [[convergence_matrix[r].get(p, 0.0) for r in rounds] for p in peers]
    )

    fig, ax = plt.subplots(figsize=(max(6, len(rounds) * 1.5), max(4, len(peers) * 1.2)))
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
            ax.text(
                j, i, f"{data[i, j]:.2f}",
                ha="center", va="center",
                color="black" if data[i, j] < 0.7 else "white",
                fontsize=9,
            )

    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved heatmap → %s", output_path)
    return output_path


def plot_similarity_progression(
    convergence_matrix: dict[int, dict[str, float]],
    output_path: str,
    title: str = "Peer-to-Dominant Similarity Progression",
) -> str:
    """
    Line graph showing how each peer's similarity to dominant changes per round.
    """
    rounds = sorted(convergence_matrix.keys())
    if not rounds:
        return ""

    peers = sorted(
        set(p for r in rounds for p in convergence_matrix[r].keys())
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(peers)))

    for peer, color in zip(peers, colors):
        sims = [convergence_matrix[r].get(peer, 0.0) for r in rounds]
        ax.plot(rounds, sims, marker="o", label=peer, color=color, linewidth=2)
        for r, s in zip(rounds, sims):
            ax.annotate(f"{s:.2f}", (r, s), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8)

    ax.set_xticks(rounds)
    ax.set_xticklabels([f"Round {r}" for r in rounds])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Discussion Round")
    ax.set_ylabel("Cosine Similarity to Dominant")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(title="Peer Agent")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved progression plot → %s", output_path)
    return output_path


def plot_entropy_decay(
    entropy_per_round: dict[int, float],
    output_path: str,
    title: str = "Opinion Entropy Across Discussion Rounds",
) -> str:
    """
    Bar + line chart of Shannon entropy per round, illustrating opinion collapse.
    """
    rounds = sorted(entropy_per_round.keys())
    entropies = [entropy_per_round[r] for r in rounds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rounds, entropies, color="steelblue", alpha=0.6, label="Entropy")
    ax.plot(rounds, entropies, marker="D", color="navy", linewidth=2, label="Trend")

    for r, e in zip(rounds, entropies):
        ax.text(r, e + 0.01, f"{e:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(rounds)
    ax.set_xticklabels([f"Round {r}" for r in rounds])
    ax.set_xlabel("Discussion Round")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved entropy plot → %s", output_path)
    return output_path


def plot_belief_shift(
    belief_shift: dict[str, float],
    output_path: str,
    title: str = "Belief Shift Magnitude per Agent",
) -> str:
    """
    Horizontal bar chart comparing belief shift magnitude per agent.
    Dominant agent is highlighted in a different color.
    """
    agents = sorted(belief_shift.keys())
    shifts = [belief_shift[a] for a in agents]
    colors = ["crimson" if a == "D" else "steelblue" for a in agents]

    fig, ax = plt.subplots(figsize=(8, max(4, len(agents) * 0.8)))
    bars = ax.barh(agents, shifts, color=colors, alpha=0.8)

    for bar, val in zip(bars, shifts):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Belief Shift (1 – cosine_similarity(initial, final))")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(shifts) * 1.2 + 0.01 if shifts else 1.0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="crimson", alpha=0.8, label="Dominant (D)"),
        Patch(facecolor="steelblue", alpha=0.8, label="Peer Agents"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved belief-shift plot → %s", output_path)
    return output_path


def generate_all_plots(
    metrics: dict[str, Any],
    plots_dir: str,
    label: str = "",
) -> dict[str, str]:
    """
    Generate all four plots and return a dict of {plot_name: file_path}.

    Parameters
    ----------
    metrics : dict
        Output of compute_full_metrics.
    plots_dir : str
        Directory to save plots.
    label : str
        Optional suffix for filenames (e.g., "run_a").
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
        metrics["entropy_per_round"],
        os.path.join(plots_dir, f"entropy_decay{suffix}.png"),
    )
    paths["shift_plot"] = plot_belief_shift(
        metrics["belief_shift"],
        os.path.join(plots_dir, f"belief_shift{suffix}.png"),
    )

    return paths
