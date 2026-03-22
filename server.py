"""SPORE HTTP Server.

Provides REST API endpoints for:
- Diagnostic tests (AI model connectivity, DuckDuckGo search health)
- AI model configuration persistence (local JSON + optional Firebase)

Run with:
    uvicorn server:app --host 127.0.0.1 --port 8000 --reload
  or:
    python server.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
except ImportError as exc:
    raise RuntimeError("Run: pip install fastapi uvicorn") from exc

from spore import (
    AIModelConfig,
    ExternalAIClient,
    DuckDuckGoSearchProvider,
    SearchProvider,
)

# ---------------------------------------------------------------------------
# Config persistence (local JSON file — Firebase sync happens in the browser)
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "data" / "config.json"
_WEB_DIR     = Path(__file__).parent / "web"


def _load_config() -> AIModelConfig:
    """Load AI config from local JSON, falling back to environment variables."""
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text())
            return AIModelConfig(
                name=data.get("name", ""),
                model_id=data.get("model_id", ""),
                base_url=data.get("base_url", ""),
                api_key=data.get("api_key", ""),
            )
        except Exception:
            pass
    return AIModelConfig.from_env()


def _save_config(config: AIModelConfig) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(
        json.dumps(
            {
                "name": config.name,
                "model_id": config.model_id,
                "base_url": config.base_url,
                "api_key": config.api_key,
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="SPORE NLWS API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class AIConfigRequest(BaseModel):
    name: Optional[str] = ""
    model_id: Optional[str] = ""
    base_url: Optional[str] = ""
    api_key: Optional[str] = ""


class ProbeRequest(BaseModel):
    model_id: str
    base_url: str
    api_key: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/config")
async def get_config():
    """Return the current AI config (API key masked)."""
    config = _load_config()
    return config.as_display_dict()


@app.post("/api/config")
async def save_config(req: AIConfigRequest):
    """Persist AI config to local JSON file."""
    config = AIModelConfig(
        name=(req.name or "").strip() or "Default Search Model",
        model_id=(req.model_id or "").strip(),
        base_url=(req.base_url or "").strip(),
        api_key=(req.api_key or "").strip(),
    )
    _save_config(config)
    return {"status": "saved", "config": config.as_display_dict()}


@app.post("/api/test/ai/probe")
async def probe_ai(req: ProbeRequest):
    """Test a specific base_url + model_id without touching saved config."""
    config = AIModelConfig(
        name="probe",
        model_id=req.model_id.strip(),
        base_url=req.base_url.strip(),
        api_key=req.api_key.strip(),
    )
    client = ExternalAIClient(config)
    ok, message = await client.test_connection()
    return {"ok": ok, "message": message, "baseUrl": req.base_url, "modelId": req.model_id}


@app.get("/api/test/ai")
async def test_ai_model():
    """Test connectivity to the configured AI model."""
    config = _load_config()
    client = ExternalAIClient(config)
    ok, message = await client.test_connection()
    return {
        "ok": ok,
        "message": message,
        "model": config.model_id or "(not set)",
        "name": config.name,
    }


@app.get("/api/test/search")
async def test_search():
    """Test DuckDuckGo search provider health."""
    try:
        provider: SearchProvider = DuckDuckGoSearchProvider(max_results=3)
        provider_name = "DuckDuckGoSearchProvider"
    except RuntimeError:
        provider = SearchProvider()
        provider_name = "MockSearchProvider (install duckduckgo-search for live results)"

    ok, message = await provider.health_check()
    return {
        "ok": ok,
        "message": message,
        "provider": provider_name,
    }


# Serve static assets from web/ (must be mounted before the catch-all GET /)
if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the visualizer as the homepage."""
    return FileResponse(_WEB_DIR / "visualizer.html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
