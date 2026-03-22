# SPORE — Neural Linguistic Web-Learning System (NLWS)

This repository now contains an assembled, runnable **NLWS prototype** based on your concept:

- Multi-phase learning engine (`initialization` → `general_language` → `topic_specialization` → `expert_mode` → `testing`)
- Knowledge ingestion/crawling abstraction with pluggable web search providers
- Concept-memory graph growth with neural-style activation
- Real-time network visualization output (SVG + browser canvas)
- Interactive CLI for learning/test cycles

## Project layout

- `nlws/system.py` — Python orchestration engine, data model, and interactive CLI
- `web/visualizer.html` — Standalone real-time neural growth visual demo

## Quick start

### 1) Run the Python CLI prototype

```bash
python3 nlws/system.py
```

Commands:

- `learn general`
- `learn <topic>`
- `stop`
- `ask <question>`
- `visualize`
- `status`
- `ai show`
- `ai config`
- `ai test`
- `exit`

> Note: The crawler prefers DuckDuckGo when `duckduckgo-search` is installed, and falls back to the built-in mock provider when unavailable.

### Optional AI model configuration for crawl/query enhancement

The AI model is only used to enrich crawl/search query generation and connectivity validation (not chat responses).

Configure via environment variables or interactively in CLI with `ai config`.

```bash
export SPORE_AI_MODEL_NAME="Crawler Model"
export SPORE_AI_MODEL_ID="gpt-4o-mini"
export SPORE_AI_BASE_URL="https://api.openai.com/v1"   # optional
export SPORE_AI_API_KEY="sk-..."
```

Optional dependencies:

```bash
pip install duckduckgo-search openai
```

### 2) Open the site

Open `index.html` in a browser for the main landing page, then launch the visualizer from there (or open `web/visualizer.html` directly).

## Future integration points for SPORE

The main extension points are intentionally explicit:

- `SearchProvider.search()` to connect your real search/crawl stack
- `DuckDuckGoCrawler._extract_content()` to bind extraction/scraping pipeline
- `LanguageLearningEngine._synthesize_response()` to route to a model/runtime
- persistence hooks for memory and graph state snapshots
