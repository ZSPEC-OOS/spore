"""Search provider abstraction and DuckDuckGo implementation.

SearchProvider         — abstract base with a mock offline implementation.
DuckDuckGoSearchProvider — live DDG search (requires duckduckgo-search).

Replace or extend SearchProvider to plug in any real search/crawl stack.
"""

from __future__ import annotations

import asyncio
import re
from typing import Dict, List, Tuple


class SearchProvider:
    """Search provider abstraction.

    Replace this class with SPORE-integrated search/crawling adapters.
    The default implementation returns mock results to keep the prototype
    runnable offline.
    """

    async def search(self, query: str) -> List[Dict[str, str]]:
        slug = re.sub(r"\s+", "-", query.strip().lower())
        return [
            {
                "title":   f"Overview: {query}",
                "url":     f"https://example.org/{slug}/overview",
                "snippet": f"{query} refers to a practical framework with definitions, examples, and tradeoffs.",
            },
            {
                "title":   f"Guide: {query}",
                "url":     f"https://example.org/{slug}/guide",
                "snippet": f"How to apply {query} with step-by-step methods and evaluation criteria.",
            },
            {
                "title":   f"Research: {query}",
                "url":     f"https://example.org/{slug}/research",
                "snippet": f"A study analysing {query}, benchmarks, and limitations.",
            },
        ]

    async def health_check(self) -> Tuple[bool, str]:
        return True, "Mock provider ready"


class DuckDuckGoSearchProvider(SearchProvider):
    """DuckDuckGo-backed search provider.

    Requires: pip install duckduckgo-search
    Raises RuntimeError at construction time if the package is absent.
    """

    def __init__(self, max_results: int = 5) -> None:
        self.max_results = max_results
        try:
            from ddgs import DDGS  # noqa: F401
        except Exception as exc:
            raise RuntimeError("Install `duckduckgo-search` to enable live search.") from exc

    async def search(self, query: str) -> List[Dict[str, str]]:
        return await asyncio.to_thread(self._sync_search, query)

    def _sync_search(self, query: str) -> List[Dict[str, str]]:
        from ddgs import DDGS

        out: List[Dict[str, str]] = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=self.max_results):
                out.append(
                    {
                        "title":   item.get("title", f"Result for {query}"),
                        "url":     item.get("href", ""),
                        "snippet": item.get("body", ""),
                    }
                )
        return [r for r in out if r["url"]]

    async def health_check(self) -> Tuple[bool, str]:
        try:
            results = await self.search("site:example.com language learning")
            return True, f"DDG reachable ({len(results)} sample results)"
        except Exception as exc:
            return False, f"DDG health check failed: {exc}"
