"""Tests for spore.crawler — DuckDuckGoCrawler with mock provider."""

import pytest

from spore.crawler import DuckDuckGoCrawler
from spore.models import MemoryNode
from spore.search import SearchProvider


@pytest.mark.asyncio
async def test_search_and_learn_returns_memory_nodes():
    crawler = DuckDuckGoCrawler(search_provider=SearchProvider())
    nodes   = await crawler.search_and_learn("greetings")
    assert len(nodes) > 0
    assert all(isinstance(n, MemoryNode) for n in nodes)


@pytest.mark.asyncio
async def test_deduplication_prevents_revisiting():
    crawler = DuckDuckGoCrawler(search_provider=SearchProvider())
    nodes1  = await crawler.search_and_learn("hello")
    nodes2  = await crawler.search_and_learn("hello")  # same query → same URLs
    assert len(nodes2) == 0, "Already-visited URLs must not produce new MemoryNodes"


@pytest.mark.asyncio
async def test_confidence_above_zero():
    crawler = DuckDuckGoCrawler(search_provider=SearchProvider())
    nodes   = await crawler.search_and_learn("language learning")
    assert all(n.confidence > 0 for n in nodes)


@pytest.mark.asyncio
async def test_category_is_known_value():
    crawler    = DuckDuckGoCrawler(search_provider=SearchProvider())
    nodes      = await crawler.search_and_learn("how to learn")
    categories = {"definition", "tutorial", "research", "discussion", "general"}
    assert all(n.category in categories for n in nodes)
