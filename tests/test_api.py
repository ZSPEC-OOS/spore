"""Tests for the FastAPI server endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from server import app


@pytest.mark.asyncio
async def test_config_get_returns_display_dict():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert "model_id" in body
    assert "api_key"  in body
    # API key must never be returned in plaintext
    assert body["api_key"] in {"(not set)", "••••••••"}


@pytest.mark.asyncio
async def test_search_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/test/search")
    assert r.status_code == 200
    body = r.json()
    assert "ok"       in body
    assert "provider" in body


@pytest.mark.asyncio
async def test_ai_test_endpoint_unconfigured():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/test/ai")
    assert r.status_code == 200
    body = r.json()
    assert "ok"      in body
    assert "message" in body
    # Without a real key configured the test must report not-ok, not crash
    assert body["ok"] is False
