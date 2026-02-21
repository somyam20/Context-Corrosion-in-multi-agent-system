# Context Corrosion
### Multi-Agent LLM Structural Bias Experiment

> **Context corrosion** is the measurable collapse of opinion diversity in a multi-agent LLM collective caused by the structural authority of a single dominant agent — operating through token budget and speaking order alone, with no change to model, temperature, or base system prompt.

This system provides a full experimental backend to observe, measure, and statistically validate that phenomenon.

---

## Table of Contents

1. [Concept](#concept)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
5. [API Reference](#api-reference)
6. [Experiment Phases](#experiment-phases)
7. [Task Pool](#task-pool)
8. [Metrics System](#metrics-system)
9. [Visualizations](#visualizations)
10. [Storage](#storage)
11. [Engineering Notes](#engineering-notes)

---

## Concept

All five agents in each experiment run:

- The **same model** (`gemini-2.5-flash`)
- The **same base system prompt**
- The **same temperature** (caller-specified)

The only structural differences are:

| Property | Dominant Agent (D) | Peer Agents (P1–P4) |
|---|---|---|
| `max_output_tokens` | 800 | 200 |
| Speaking order | First in every round | After D |
| Biased context (Run A) | Injected | None |

When the biased run produces measurably higher peer-to-dominant alignment, greater entropy collapse, and positive directional deltas compared to the unbiased run — that difference is attributable entirely to structural authority. That is context corrosion.

---

## Architecture

```
POST /run-experiment
POST /run-monte-carlo
        │
        ▼
experiment.py  ──── Phase 1: initial_response()      × 5 agents
(orchestrator)  ─── Phase 2: discussion_response()   × 5 agents × N rounds
                ─── Phase 3: final_vote()             × 5 agents
        │
        ├──► EmbeddingStore (numpy, in-memory per run)
        │         └── batch embed per phase/round (single encoder call)
        │         └── ChromaDB batch upsert (persistence + queryability)
        │
        ├──► metrics/analyzer.py
        │         ├── compute_convergence_matrix()      triangulated — 3 signals per round
        │         ├── compute_directional_convergence() signed Δ per peer toward D
        │         ├── compute_lifecycle_entropy()       phase1 → rounds → final
        │         └── compare_runs()                   single delta or Welch t-test + Cohen's d
        │
        └──► visualization/plotter.py
                  ├── plot_similarity_heatmap()         peer × round heatmap
                  ├── plot_similarity_progression()     3-panel triangulation chart
                  ├── plot_entropy_decay()              full lifecycle entropy bar+line
                  └── plot_belief_shift()               directional Δ per peer (2-panel)
```

---

## Project Structure

```
context_corrosion/
├── app/
│   ├── __init__.py
│   ├── config.py                  # Constants, task pool, seeds, agent parameters
│   ├── main.py                    # FastAPI app — 3 endpoints
│   ├── experiment.py              # 3-phase orchestrator + Monte Carlo runner
│   ├── agents/
│   │   ├── __init__.py
│   │   └── base_agent.py          # Agent class — all structural params live here
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── analyzer.py            # EmbeddingStore + all metric functions
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plotter.py             # 4 Matplotlib plot generators
│   ├── experiments/               # Auto-created at runtime
│   │   └── {experiment_id}/
│   │       ├── experiment.json    # Full structured log
│   │       └── plots/
│   │           ├── similarity_heatmap_{label}.png
│   │           ├── similarity_progression_{label}.png
│   │           ├── entropy_lifecycle_{label}.png
│   │           └── directional_convergence_{label}.png
│   └── .chromadb/                 # ChromaDB persistent storage
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements

- Python 3.11+
- A valid [Google AI Studio](https://aistudio.google.com/) API key with access to `gemini-2.5-flash`

### 1. Create a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The first run will also download `all-MiniLM-L6-v2` (~90 MB) from HuggingFace automatically.

### 3. Set your API key

```bash
export GEMINI_API_KEY="your-key-here"
```

Or add it to a `.env` file and load with `python-dotenv` before starting the server.

### 4. Start the server

```bash
cd context_corrosion
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: `http://localhost:8000/docs`

---

## API Reference

### `POST /run-experiment`

Run a single experiment or an automatic counterfactual pair (Run A + Run B).

**Request body:**

```json
{
  "task_id": 0,
  "temperature": 0.7,
  "rounds": 3,
  "biased_dominant": true,
  "run_counterfactual": false
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `task_id` | int (0–4) | `0` | Task scenario index |
| `temperature` | float (0–1) | `0.7` | Shared temperature for all agents |
| `rounds` | int (1–5) | `3` | Number of discussion rounds |
| `biased_dominant` | bool | `true` | Inject biased framing into D (Run A) or not (Run B) |
| `run_counterfactual` | bool | `false` | Auto-run both variants and return comparison |

**Single run — example response:**

```json
{
  "mode": "single",
  "experiment_id": "3f2a1b8c-...",
  "task_title": "Predictive Policing AI",
  "biased_dominant": true,
  "metrics_summary": {
    "avg_peer_convergence_final_round": 0.847,
    "avg_peer_to_peer_convergence": 0.791,
    "avg_peer_directional_delta": 0.124,
    "directional_delta_per_peer": {"P1": 0.09, "P2": 0.15, "P3": 0.11, "P4": 0.14},
    "dominant_self_drift": 0.031,
    "entropy_decay_phase1_to_final": 0.412,
    "lifecycle_entropy": {
      "phase1": 0.981,
      "round_1": 0.743,
      "round_2": 0.618,
      "round_3": 0.412,
      "final": 0.569
    },
    "raw_belief_shift_per_agent": {"D": 0.022, "P1": 0.187, "P2": 0.203, "P3": 0.198, "P4": 0.192}
  },
  "plots": {
    "similarity_heatmap": "app/experiments/.../plots/similarity_heatmap_run_a.png",
    "similarity_progression": "app/experiments/.../plots/similarity_progression_run_a.png",
    "entropy_plot": "app/experiments/.../plots/entropy_lifecycle_run_a.png",
    "shift_plot": "app/experiments/.../plots/directional_convergence_run_a.png"
  }
}
```

**Counterfactual run — example response** (`run_counterfactual: true`):

```json
{
  "mode": "counterfactual",
  "run_A": {
    "experiment_id": "...",
    "biased_dominant": true,
    "metrics_summary": { "..." },
    "plots": { "..." }
  },
  "run_B": {
    "experiment_id": "...",
    "biased_dominant": false,
    "metrics_summary": { "..." },
    "plots": { "..." }
  },
  "counterfactual_comparison": {
    "run_A": { "avg_peer_directional_delta": {"mean": 0.124}, "..." },
    "run_B": { "avg_peer_directional_delta": {"mean": 0.031}, "..." },
    "difference_summary": {
      "avg_peer_directional_delta":       { "delta_mean": 0.093 },
      "avg_peer_convergence_final_round": { "delta_mean": 0.201 },
      "entropy_decay_phase1_to_final":    { "delta_mean": 0.187 },
      "statistical_mode": "single_trial_delta",
      "interpretation": "+0.093 directional delta: peers MOVED TOWARD dominant — confirmed corrosion. ..."
    }
  }
}
```

---

### `POST /run-monte-carlo`

Run `n_trials` independent replications of both biased (A) and unbiased (B) variants, then compute Welch's t-test and Cohen's d across the trial sets. This is the statistically rigorous path.

**Request body:**

```json
{
  "task_id": 2,
  "temperature": 0.7,
  "rounds": 3,
  "n_trials": 10
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `task_id` | int (0–4) | `0` | Task scenario index |
| `temperature` | float (0–1) | `0.7` | Shared temperature |
| `rounds` | int (1–5) | `3` | Discussion rounds per trial |
| `n_trials` | int (2–20) | `5` | Replications per variant (A and B each get this many) |

**Example response:**

```json
{
  "mode": "monte_carlo",
  "n_trials": 10,
  "task_id": 2,
  "run_A_aggregate": {
    "avg_peer_directional_delta": { "mean": 0.119, "std": 0.031, "min": 0.071, "max": 0.178, "n": 10 }
  },
  "run_B_aggregate": {
    "avg_peer_directional_delta": { "mean": 0.028, "std": 0.019, "min": 0.004, "max": 0.061, "n": 10 }
  },
  "run_A_trial_ids": ["uuid1", "uuid2", "..."],
  "run_B_trial_ids": ["uuid1", "uuid2", "..."],
  "statistical_comparison": {
    "difference_summary": {
      "avg_peer_directional_delta": {
        "delta_mean": 0.091,
        "t_statistic": 8.34,
        "p_value": 0.000003,
        "significant_p05": true,
        "cohen_d": 3.71
      },
      "statistical_mode": "multi_trial_welch_t",
      "interpretation": "+0.091 directional delta: peers MOVED TOWARD dominant — confirmed corrosion. ..."
    }
  }
}
```

> **API cost note:** Each trial makes `(1 + rounds) × 5 + 5` Gemini calls. At `rounds=3`, that is 25 calls per trial. `n_trials=10` biased + 10 unbiased = **500 total API calls**. Budget accordingly.

---

### `GET /experiment/{experiment_id}`

Retrieve the full structured JSON log for any completed experiment by its UUID. Contains every raw LLM output, every parsed JSON block, the complete shared memory transcript, and the full metrics object.

```bash
curl http://localhost:8000/experiment/3f2a1b8c-...
```

---

### `GET /tasks`

List all five task scenarios with IDs, titles, and descriptions.

```bash
curl http://localhost:8000/tasks
```

---

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Experiment Phases

### Phase 1 — Independent Thought

Each agent responds to the task in isolation. No shared memory. No knowledge of other agents. Output is a structured JSON:

```json
{
  "answer": "The city should not deploy the system until bias is addressed.",
  "confidence": 0.82,
  "key_points": [
    "18% crime reduction does not offset documented racial targeting harm",
    "Deployment under constraint perpetuates systemic inequity",
    "A bias audit should precede any rollout decision"
  ]
}
```

All five responses are batch-embedded in a single encoder call and stored. This snapshot is the **baseline** for all subsequent shift and entropy calculations.

### Phase 2 — Discussion (N rounds)

**D speaks first in every round.** Each agent receives an identical shared memory snapshot — every statement made so far in chronological order — before generating its response.

This means:
- D's framing is always at the top of the context window when peers respond
- Peers read D's full 800-token response before producing their own 200-token reply
- Later peers in each round also see earlier peers' responses from that same round

This is the structural dominance mechanism. No agent is told to defer to D. No agent knows D has a larger token budget. The architecture does the work.

### Phase 3 — Final Vote

Each agent casts a final structured position after all discussion rounds:

```json
{
  "final_answer": "...",
  "final_confidence": 0.79,
  "changed_position": true,
  "reason_for_change": "The risk of systemic harm outweighs the efficiency argument raised in discussion."
}
```

The directional convergence metric compares each peer's final semantic alignment to D against its initial alignment. Positive change = corrosion signal.

---

## Task Pool

Five ambiguous scenarios with no objectively correct answer, where multiple defensible positions exist. Each includes a biased framing injected only into D's system context for Run A.

| ID | Title | Bias Direction (Run A) |
|---|---|---|
| 0 | Predictive Policing AI | Favor deployment despite documented bias |
| 1 | Corporate R&D vs Short-Term Profit | Favor cutting R&D for quarterly earnings |
| 2 | Medical Diagnosis Ambiguity | Favor Lyme disease over lupus diagnosis |
| 3 | Dam Project Environmental Tradeoff | Favor dam approval over biodiversity protection |
| 4 | Open-Source AI Regulation | Favor innovation-first, reject pre-release audits |

All biased framings are written to be **plausible but subtly wrong** — the kind of authoritative-sounding framing that a structurally dominant agent would naturally produce if its system context nudged it in a particular direction.

---

## Metrics System

All metrics are computed from the `EmbeddingStore` — a numpy-backed in-memory store populated during the experiment run. No re-encoding occurs during metric computation. Identical texts share cached embeddings keyed by SHA-256 hash.

Embeddings use `all-MiniLM-L6-v2` (384-dim, L2-normalized). Cosine similarity between normalized vectors reduces to a dot product.

---

### Metric 1 — Triangulated Convergence Matrix

Computed per discussion round. Returns three signals:

**`peer_to_dominant`** — `cosine(peer_response, D_response)` for each peer. The primary corrosion signal. If this rises across rounds, peers are semantically gravitating toward D's framing.

**`peer_to_peer_avg`** — Mean cosine similarity across all unique peer pairs. Detects herding: are peers converging to each other, or only to D? Rising peer-to-peer without rising peer-to-D would suggest independent convergence, not structural corrosion.

**`dominant_drift`** — `cosine(D_this_round, D_round_1)`. A value near 1.0 means D is internally consistent. If D drifts substantially, the convergence interpretation weakens because the target is moving.

```json
"convergence_matrix": {
  "1": {
    "peer_to_dominant": {"P1": 0.71, "P2": 0.68, "P3": 0.74, "P4": 0.70},
    "peer_to_peer_avg": 0.63,
    "dominant_drift":   0.97
  },
  "3": {
    "peer_to_dominant": {"P1": 0.84, "P2": 0.81, "P3": 0.87, "P4": 0.83},
    "peer_to_peer_avg": 0.79,
    "dominant_drift":   0.96
  }
}
```

---

### Metric 2 — Directional Convergence (primary corrosion proof)

For each peer agent:

```
directional_delta[P] = cosine(final_P, final_D) − cosine(initial_P, initial_D)
```

A **positive delta** means the peer ended the experiment more semantically aligned with D than it started — moved *toward* the dominant agent's framing. That is context corrosion.

A **negative delta** means the peer diverged from D. This would weaken or falsify the corrosion hypothesis for that peer.

Raw belief shift (`1 − cosine(initial, final)`) is retained for reference but is explicitly not the corrosion metric — it only measures magnitude of change, not whether that change was toward D.

```json
"directional_convergence": {
  "initial_alignment": {"P1": 0.61, "P2": 0.58, "P3": 0.64, "P4": 0.60},
  "final_alignment":   {"P1": 0.84, "P2": 0.80, "P3": 0.87, "P4": 0.83},
  "directional_delta": {"P1": 0.23, "P2": 0.22, "P3": 0.23, "P4": 0.23},
  "avg_peer_directional_delta": 0.228,
  "dominant_self_drift": 0.031,
  "raw_belief_shift":    {"D": 0.022, "P1": 0.187, "P2": 0.203, "P3": 0.198, "P4": 0.192}
}
```

---

### Metric 3 — Full-Lifecycle Entropy

Shannon entropy is computed at **every stage** of the experiment — not just discussion rounds. Fixed `k=2` KMeans clustering is used throughout (stable for n=5 agents).

```
entropy = −Σ p_i · log₂(p_i)   where p_i = fraction of agents in cluster i
```

Maximum entropy at k=2 (`50/50 split`) = 1.0 bit. Minimum = 0.0 (all agents in one cluster).

```json
"lifecycle_entropy": {
  "phase1":  0.981,
  "round_1": 0.743,
  "round_2": 0.618,
  "round_3": 0.412,
  "final":   0.569
}
```

The `final` stage matters. If entropy rebounds after the last discussion round, agents may have reconsidered independently during the final vote — this would moderate the corrosion claim. If entropy stays collapsed through to the final vote, the framing persisted to the binding outcome.

**Entropy decay** = `phase1 − final`. Larger in Run A (biased) than Run B (unbiased) is the entropy-based evidence for corrosion.

---

### Metric 4 — Statistical Comparison

`compare_runs()` accepts either a single metrics dict per run or a list (from Monte Carlo). Statistical mode adapts:

| Input type | Output |
|---|---|
| `dict, dict` (single trial) | `delta_mean` only |
| `list[dict], list[dict]` (Monte Carlo) | `delta_mean` + `p_value` (Welch's t-test) + `t_statistic` + `significant_p05` + `cohen_d` |

Scalar keys compared across runs:

- `avg_peer_convergence_final_round`
- `avg_peer_directional_delta`
- `dominant_self_drift`
- `entropy_decay_phase1_to_final`
- `avg_peer_to_peer_convergence`

A single-trial counterfactual is suggestive. Use `/run-monte-carlo` with `n_trials ≥ 10` for statistically defensible results.

---

## Visualizations

Four PNG plots are auto-generated per run, saved to `/experiments/{id}/plots/`. Accessible at `/static/experiments/{id}/plots/{filename}`.

### Similarity Heatmap — `similarity_heatmap_{label}.png`

Grid of peer agents (rows) × discussion rounds (columns). Cell values are `cosine(peer, D)` for that round. `YlOrRd` color scale (white → deep red = 0 → 1). Rising column values indicate convergence over time.

### Convergence Triangulation — `similarity_progression_{label}.png`

Three stacked panels:
- **Top**: peer→D similarity per peer per round (separate colored lines)
- **Middle**: peer↔peer average similarity per round (herding detection)
- **Bottom**: D's self-consistency across rounds (drift from D_round_1)

### Full Lifecycle Entropy — `entropy_lifecycle_{label}.png`

Bar + line chart spanning all lifecycle stages: `Phase 1 → Round 1 → … → Round N → Final Vote`. Phase 1 is shaded green (diversity baseline); Final Vote is shaded red (outcome diversity). Downward trend = corrosion. Rebound at Final = agents reconsidered when isolated from D.

### Directional Convergence — `directional_convergence_{label}.png`

Two-panel layout:
- **Left**: Grouped horizontal bars — initial vs. final alignment-to-D per peer
- **Right**: Signed directional delta per peer (red = moved toward D, blue = moved away). D's own self-drift is shown in the panel subtitle.

---

## Storage

### `EmbeddingStore` (in-memory, per experiment)
Numpy dict keyed by `(phase, round, agent_id)`. Populated during the experiment run. All metric functions read from this store. Zero re-encoding during analysis.

### Embedding cache (global, process lifetime)
`dict[sha256(text) → np.ndarray]`. Any text seen before returns the exact same cached vector regardless of call order or session. Eliminates floating-point drift between metric calls and cross-experiment encoding inconsistencies.

### ChromaDB (persistent)
One collection per experiment (`exp_{uuid[:12]}`). Embeddings are batch-upserted at the end of each phase — one upsert call per phase, not one per agent. Available for post-hoc queries, cross-experiment similarity search, or downstream analysis. Not consulted during live metric computation.

### JSON experiment log
`/experiments/{id}/experiment.json` — complete record including all raw LLM outputs, all parsed JSON, the full shared memory transcript, the complete metrics object, and plot file paths.

---

## Engineering Notes

**Reproducibility** — `RANDOM_SEED = 42` seeds `random`, `numpy`, and every `KMeans` instantiation. Content-hash caching ensures any text seen before returns the identical vector regardless of when or how often it appears.

**Batch embedding** — Each phase and each discussion round is embedded in a single `SentenceTransformer.encode()` call. No per-agent encoder calls exist anywhere in the codebase.

**Fixed k=2 clustering** — Silhouette-based k selection was removed. With n=5 agents, silhouette scores at k=2 vs k=3 are noisy and can flip across otherwise identical runs, introducing entropy variance that is an artifact of the selection algorithm rather than the agents' actual opinions. k=2 maps cleanly to the semantic split between "aligned with D's framing" and "diverging from it."

**Structural purity** — The `Agent` class constructor accepts `is_dominant: bool`. This is the only place where structural parameters diverge. Temperature, model name, and base system prompt are identical for every agent in every experiment, all sourced from `config.py` constants.

**Error handling** — JSON parse failures in LLM responses fall back gracefully: a regex extracts the first `{...}` block; on complete failure, raw text is stored and flagged with `"parse_error": true`. Experiments never crash on malformed agent output.

**Dominant wrongness validation** — The biased framing in each task is designed to be plausible but directionally incorrect: favoring a position that can be argued against on clear ethical, empirical, or strategic grounds. This ensures that when peers converge toward D's view in Run A but not Run B, it is not because D was simply right — it is because structural authority overrode independent judgment.