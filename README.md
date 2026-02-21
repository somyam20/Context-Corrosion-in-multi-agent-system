# Context Corrosion — Multi-Agent LLM Bias Experiment

Demonstrates **context corrosion**: a structurally dominant LLM agent measurably
biases peer agents toward its framing through token budget and speaking-order
advantages alone — without any change to model, temperature, or base prompt.

---

## Project Structure

```
context_corrosion/
├── app/
│   ├── __init__.py
│   ├── config.py             # All constants, task pool, seeds
│   ├── main.py               # FastAPI application
│   ├── experiment.py         # Experiment orchestrator (3 phases)
│   ├── agents/
│   │   ├── __init__.py
│   │   └── base_agent.py     # Agent class (Gemini via google-generativeai)
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── analyzer.py       # Convergence, belief shift, entropy, comparison
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plotter.py        # Matplotlib plot generators
│   └── experiments/          # Auto-created; stores JSON logs + plots
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Python environment (Python 3.11+)

```bash
python3.11 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Gemini API key

```bash
export GEMINI_API_KEY="your-key-here"
```

Or create a `.env` file and load it with `python-dotenv`.

### 3. Start the server

```bash
cd context_corrosion
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API Usage

### List available tasks

```bash
curl http://localhost:8000/tasks
```

### Run a single experiment (biased dominant)

```bash
curl -X POST http://localhost:8000/run-experiment \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": 0,
    "temperature": 0.7,
    "rounds": 3,
    "biased_dominant": true,
    "run_counterfactual": false
  }'
```

### Run counterfactual (Run A + Run B automatically)

```bash
curl -X POST http://localhost:8000/run-experiment \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": 2,
    "temperature": 0.7,
    "rounds": 3,
    "biased_dominant": true,
    "run_counterfactual": true
  }'
```

Returns structured comparison:
```json
{
  "mode": "counterfactual",
  "run_A": { "experiment_id": "...", "metrics_summary": {...}, "plots": {...} },
  "run_B": { "experiment_id": "...", "metrics_summary": {...}, "plots": {...} },
  "counterfactual_comparison": {
    "run_A": { "avg_peer_convergence_final_round": 0.87, ... },
    "run_B": { "avg_peer_convergence_final_round": 0.63, ... },
    "difference_summary": {
      "avg_peer_convergence_final_round": 0.24,
      "interpretation": "..."
    }
  }
}
```

### Retrieve past experiment

```bash
curl http://localhost:8000/experiment/{experiment_id}
```

---

## How Metrics Prove Context Corrosion

| Metric | What it measures | Evidence of corrosion |
|--------|-----------------|----------------------|
| **Convergence Score** | Cosine similarity between each peer response and dominant agent per round | Score rises across rounds in Run A (biased), stays lower in Run B |
| **Belief Shift Magnitude** | `1 − cosine_sim(initial_answer, final_answer)` per agent | Peers shift more than dominant in biased run; peers stay diverse in unbiased run |
| **Entropy Collapse** | Shannon entropy of KMeans clusters on all agent embeddings per round | Entropy drops faster in Run A as opinions converge toward dominant framing |
| **Counterfactual Sensitivity** | Δ between Run A and Run B on all above metrics | Positive delta on convergence + entropy decay isolates structural bias as the cause |

The structural dominance mechanisms — **token budget** (800 vs 200 tokens) and
**speaking order** (dominant always first) — are the only variables that differ
between agents. When a biased framing produces higher convergence and entropy
collapse vs. the same run without bias, that is direct experimental evidence of
context corrosion driven by structural authority.

---

## Swagger UI

Visit `http://localhost:8000/docs` for interactive API documentation.
