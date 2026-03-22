"""Tests for spore.search — SearchProvider (mock) and health check."""

import pytest

from spore.search import SearchProvider


@pytest.mark.asyncio
async def test_mock_search_returns_three_results():
    provider = SearchProvider()
    results  = await provider.search("hello world")
    assert len(results) == 3
    for r in results:
        assert "url"     in r
        assert "title"   in r
        assert "snippet" in r


@pytest.mark.asyncio
async def test_mock_search_url_contains_slug():
    provider = SearchProvider()
    results  = await provider.search("machine learning")
    assert all("machine-learning" in r["url"] for r in results)


@pytest.mark.asyncio
async def test_health_check_mock_always_ok():
    provider = SearchProvider()
    ok, msg  = await provider.health_check()
    assert ok  is True
    assert "Mock" in msg
