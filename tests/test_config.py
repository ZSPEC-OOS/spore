"""Tests for spore.config — AIModelConfig."""

from spore.config import AIModelConfig


def test_defaults():
    cfg = AIModelConfig()
    assert cfg.name     == "Default Search Model"
    assert cfg.model_id == ""
    assert cfg.base_url == ""
    assert cfg.api_key  == ""


def test_from_env_picks_up_vars(monkeypatch):
    monkeypatch.setenv("SPORE_AI_MODEL_ID",   "gpt-4o")
    monkeypatch.setenv("SPORE_AI_BASE_URL",   "https://api.openai.com/v1")
    monkeypatch.setenv("SPORE_AI_API_KEY",    "sk-test")
    monkeypatch.setenv("SPORE_AI_MODEL_NAME", "Test Model")
    cfg = AIModelConfig.from_env()
    assert cfg.model_id == "gpt-4o"
    assert cfg.base_url == "https://api.openai.com/v1"
    assert cfg.api_key  == "sk-test"
    assert cfg.name     == "Test Model"


def test_display_dict_masks_api_key():
    cfg = AIModelConfig(api_key="super-secret")
    d   = cfg.as_display_dict()
    assert d["api_key"] == "••••••••"
    assert "super-secret" not in str(d)


def test_display_dict_no_key_shows_not_set():
    cfg = AIModelConfig()
    d   = cfg.as_display_dict()
    assert d["api_key"]  == "(not set)"
    assert d["model_id"] == "(not set)"
