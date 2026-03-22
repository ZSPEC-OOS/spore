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
- `exit`

> Note: The crawler is implemented with a **mock provider** by default so it runs locally without network dependencies.

### 2) Open the visualizer

Open `web/visualizer.html` in any browser.

## Future integration points for SPORE

The main extension points are intentionally explicit:

- `SearchProvider.search()` to connect your real search/crawl stack
- `DuckDuckGoCrawler._extract_content()` to bind extraction/scraping pipeline
- `LanguageLearningEngine._synthesize_response()` to route to a model/runtime
- persistence hooks for memory and graph state snapshots

