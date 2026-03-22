# SPORE — System for Progressive Online Research & Evolution

A self-growing neural language learning system that crawls the web, absorbs knowledge into a concept-memory graph, and visualizes its own intelligence in real time.

## Project layout

```
spore/                  Python package (core engine)
  __init__.py           Public API exports
  models.py             LearningPhase, Neuron, MemoryNode dataclasses
  config.py             AIModelConfig (env-driven, key-masking)
  ai_client.py          ExternalAIClient (OpenAI-compatible)
  search.py             SearchProvider (mock) + DuckDuckGoSearchProvider
  crawler.py            DuckDuckGoCrawler
  visualizer.py         NeuralNetworkVisualizer (SVG output)
  engine.py             LanguageLearningEngine (orchestrator)
  cli.py                Interactive CLI + spore-cli entry point
server.py               FastAPI REST server (config & diagnostics)
web/
  visualizer.html       Browser neural visualizer (File System Access API)
  index.html            Landing page
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
| `pause` / `resume` | Pause or resume the learning cycle |
| `ask <question>` | Query accumulated knowledge |
| `visualize` | Print an SVG snapshot of the current graph |
| `status` | Show current phase, neuron count, connections |
| `ai show` | Show current AI model config |
| `ai config` | Interactively update AI model settings |
| `ai test` | Test connectivity to the configured AI model |
| `exit` | Quit |

### 3) Run the FastAPI server

```bash
uvicorn server:app --reload
```

Endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/config` | Return current AI config (key masked) |
| `POST` | `/api/config` | Update AI config (persisted to `data/config.json`) |
| `GET` | `/api/test/search` | Search provider health check |
| `GET` | `/api/test/ai` | AI model connectivity test |

### 4) Open the browser visualizer

Open `web/visualizer.html` directly in Chrome/Edge (requires File System Access API).

Features:
- **Brain storage**: choose a local folder; handle persists across sessions via IndexedDB
- **Per-node files**: each concept saves to `nodes/{id}.json`; graph index in `spore-index.json`
- **Smart connections**: synapses grow brighter/stronger as knowledge flows between nodes
- **Dead nodes**: deleting a node file turns that node red and fades it over ~3 minutes
- **AI curriculum**: 5 levels (Greetings → Personal → Concepts → Reasoning → Abstract); the attached AI model tests SPORE's knowledge before advancing levels
- **Rate limiting**: max 15 DDG searches/min; max 20 AI requests/min

## AI model configuration

Set via environment variables or interactively with `ai config` in the CLI:

```bash
export SPORE_AI_MODEL_NAME="My Model"
export SPORE_AI_MODEL_ID="gpt-4o-mini"
export SPORE_AI_BASE_URL="https://api.openai.com/v1"
export SPORE_AI_API_KEY="sk-..."
```

## Running tests

```bash
pytest
```

## Optional dependencies

```bash
pip install duckduckgo-search   # real DDG search (falls back to mock)
pip install openai               # real AI model calls
```
