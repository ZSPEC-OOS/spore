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
from spore.nlf import NLFFramework

# ---------------------------------------------------------------------------
# Config persistence (local JSON file — Firebase sync happens in the browser)
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "data" / "config.json"
_NLF_STATE_PATH = Path(__file__).parent / "data" / "nlf_state.json"
_NLF_STAGES_DIR = Path(__file__).parent / "data" / "nlf" / "stages"
_WEB_DIR     = Path(__file__).parent / "web"

# Singleton NLF framework (loaded lazily)
_nlf_framework: NLFFramework | None = None


def _get_nlf() -> NLFFramework:
    global _nlf_framework
    if _nlf_framework is None:
        _nlf_framework = NLFFramework.load(_NLF_STATE_PATH)
        # Auto-load any stage JSON files present in data/nlf/stages/
        if _NLF_STAGES_DIR.exists():
            for stage_file in sorted(_NLF_STAGES_DIR.glob("stage_*.json")):
                try:
                    _nlf_framework.load_stage_from_file(stage_file)
                except Exception:
                    pass
    return _nlf_framework


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


class NLFStageRequest(BaseModel):
    spec: dict


class NLFSimpleImportRequest(BaseModel):
    spec: dict


class NLFCycleRequest(BaseModel):
    category_id: str
    gold_map: Optional[dict] = None
    max_instances: int = 100


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


# ---------------------------------------------------------------------------
# NLF Framework routes
# ---------------------------------------------------------------------------


@app.post("/api/nlf/stages")
async def nlf_load_stage(req: NLFStageRequest):
    """Load a stage spec dict into the NLF framework."""
    fw = _get_nlf()
    stage = fw.load_stage_from_dict(req.spec)
    return {"status": "loaded", "stage": stage.number, "name": stage.name}


@app.get("/api/nlf/stages")
async def nlf_list_stages():
    """List all loaded stages and their unlock/mastery status."""
    fw = _get_nlf()
    return fw.mastery_report()


@app.get("/api/nlf/stages/{stage_number}")
async def nlf_get_stage(stage_number: int):
    """Return mastery details for a single stage."""
    fw = _get_nlf()
    stage = fw.get_stage(stage_number)
    if stage is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Stage {stage_number} not loaded.")
    report = fw.mastery_report()
    return report.get(f"stage_{stage_number}", {})


@app.post("/api/nlf/cycle")
async def nlf_run_cycle(req: NLFCycleRequest):
    """Run a training cycle for the specified category."""
    fw = _get_nlf()
    gold_map: dict | None = None
    if req.gold_map:
        gold_map = {k: int(v) for k, v in req.gold_map.items()}
    try:
        cycle, result = fw.run_cycle(
            req.category_id,
            gold_map=gold_map,
            max_instances=req.max_instances,
        )
    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(exc))

    fw.save(_NLF_STATE_PATH)
    return {
        "cycle_id": cycle.cycle_id,
        "category_id": cycle.category_id,
        "stage": cycle.stage_number,
        "instances": len(cycle.instances),
        "accuracy": round(result.accuracy, 4),
        "correct": result.correct,
        "incorrect": result.incorrect,
        "preferred": result.preferred,
    }


@app.post("/api/nlf/stages/import")
async def nlf_import_stage(req: NLFSimpleImportRequest):
    """
    Load a stage from the simplified human-readable spec format.

    Accepts stage_number, stage_name, description, and categories with
    questions (each having a ``question`` string and a ``responses`` list of
    up to 10 seed slot templates) plus optional cycles_min / cycles_max.
    """
    fw = _get_nlf()
    try:
        stage = fw.load_stage_from_simple_spec(req.spec)
    except (KeyError, ValueError) as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=str(exc))
    fw.save(_NLF_STATE_PATH)
    return {
        "status": "imported",
        "stage": stage.number,
        "name": stage.name,
        "categories": len(stage.categories),
        "questions": sum(len(c.questions) for c in stage.categories),
    }


@app.get("/api/nlf/stages/{stage_number}/detail")
async def nlf_stage_detail(stage_number: int):
    """
    Return full detail for a stage: categories, questions, and all 10 slots.

    Used by the training UI to render the question / response-slot browser.
    """
    from fastapi import HTTPException
    fw = _get_nlf()
    stage = fw.get_stage(stage_number)
    if stage is None:
        raise HTTPException(status_code=404, detail=f"Stage {stage_number} not loaded.")

    categories = []
    for cat in stage.categories:
        questions = []
        for q in cat.questions:
            slots = [
                {
                    "slot_number": s.slot_number,
                    "text":        s.text,
                    "function":    s.function.value,
                    "register":    s.register.value,
                    "use_case":    s.use_case,
                }
                for s in sorted(q.slots, key=lambda s: s.slot_number)
            ]
            questions.append({
                "id":             q.id,
                "canonical_form": q.canonical_form,
                "slots":          slots,
                "variant_count":  len(q.variants),
            })
        categories.append({
            "id":              cat.id,
            "name":            cat.name,
            "description":     cat.description,
            "cycles_min":      getattr(cat, "cycles_min", None),
            "cycles_max":      getattr(cat, "cycles_max", None),
            "accuracy":        round(cat.accuracy, 3),
            "generalization":  round(cat.generalization, 3),
            "retention":       round(cat.retention, 3),
            "cycles_completed": cat.cycles_completed,
            "mastered":        cat.is_mastered,
            "questions":       questions,
        })
    return {
        "stage":      stage.number,
        "name":       stage.name,
        "description": stage.description,
        "unlocked":   stage.unlocked,
        "complete":   stage.is_complete,
        "categories": categories,
    }


@app.get("/api/nlf/mastery")
async def nlf_mastery():
    """Return full mastery report across all loaded stages."""
    fw = _get_nlf()
    return fw.mastery_report()


# Serve static assets from web/ (must be mounted before the catch-all GET /)
if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the visualizer as the homepage."""
    return FileResponse(_WEB_DIR / "visualizer.html")


@app.get("/training.html")
async def training_page():
    """Serve the training section page."""
    return FileResponse(_WEB_DIR / "training.html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
