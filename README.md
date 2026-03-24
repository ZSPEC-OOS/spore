# SPORE — System for Progressive Online Research & Evolution

A self-growing language-learning system that crawls the web, stores memory traces, and visualizes transformer internals through a unified geometric interpretability dashboard.

## Project layout

```
spore/                  Python package (core engine)
  __init__.py           Public API exports
  models.py             LearningPhase, Neuron, MemoryNode dataclasses
  config.py             AIModelConfig (env-driven, key-masking)
  ai_client.py          ExternalAIClient (OpenAI-compatible)
  search.py             SearchProvider (mock) + DuckDuckGoSearchProvider
  crawler.py            DuckDuckGoCrawler
  visualizer.py         GeometricActivationVisualizer contract + migration brief
  engine.py             LanguageLearningEngine (orchestrator)
  cli.py                Interactive CLI + spore-cli entry point
server.py               FastAPI REST server (config & diagnostics)
streamlit_app.py        Unified geometric visualizer (single supported UI)
tests/                  pytest test suite
data/                   Runtime data (config.json, excluded from git)
pyproject.toml          Package metadata & dependencies
```

## Quick start

### 1) Install

```bash
pip install -e ".[dev]"
```

### 2) Run the interactive CLI

```bash
spore-cli
# or: python -m spore.cli
```

Commands inside the CLI:

| Command | Description |
|---|---|
| `learn general` | Start absorbing general language knowledge |
| `learn <topic>` | Focus learning on a specific topic |
| `stop` | Pause the learning cycle |
| `ask <question>` | Query accumulated knowledge |
| `visualize` | Show geometric dashboard launch/readiness info |
| `status` | Show current phase, memory count, tracked concepts |
| `ai show` | Show current AI model config |
| `ai config` | Interactively update AI model settings |
| `ai test` | Test connectivity to configured AI model |
| `exit` | Quit |

### 3) Run the FastAPI server

```bash
uvicorn server:app --reload
```

### 4) Run the unified geometric visualizer (single main visualizer)

```bash
streamlit run streamlit_app.py
```

The dashboard is intentionally the only supported visualizer and follows a mechanistic-interpretability framing:
- layer-wise UMAP/PCA of residual stream activations,
- SAE feature analysis and feature-space UMAP,
- prompt trajectories through layer space,
- attention rollout + logit lens.

## Unified Streamlit interpretability dashboard

Pages:
- **Page 1 — Latent Space Projections**: UMAP/PCA projection explorer with layer animation.
- **Page 2 — SAE Feature Explorer**: feature histogram, top examples, and SAE logit effects.
- **Page 3 — SAE Feature UMAP**: semantic organization of SAE features (cluster + activation statistics).
- **Page 4 — Prompt Trajectory**: prompt token trajectories projected with pre-fit PCA/UMAP reducers when available.
- **Page 5 — Attention & Logit Lens**: per-head/layer attention heatmaps + rollout and per-layer logit-lens top-k predictions.

Sidebar controls include:
- model selection
- layer choice
- dataset subset size
- refresh button (clears Streamlit caches)
- checkpoint comparison dropdown

### “Mind empty but ready” state

If no activation/projection artifacts exist yet, the dashboard still loads and reports missing data paths. Generate artifacts in this order:

```bash
python collect_activations.py
python reduce_activations.py
python build_sae_dataset.py
python train_sae.py
```

### Checkpoint comparison layout

For epoch/checkpoint switching, point **Artifacts root** to a directory like:

```text
artifacts/
  epoch_0001/
    projections/
    activations/
    sae_checkpoints/latest/
    sae_data/
  epoch_0002/
    projections/
    activations/
    sae_checkpoints/latest/
    sae_data/
```

Selecting an epoch and pressing **Apply checkpoint** updates default paths used across pages.

Trajectory reuse note:
- `reduce_activations.py` saves fitted projection models in `projections/models/`.
- Prompt Trajectory can load pre-fit reducers (`layer_XX_pca.pkl` / `layer_XX_umap.pkl`) to avoid retraining reducers during inference.

## Running tests

```bash
pytest
```
