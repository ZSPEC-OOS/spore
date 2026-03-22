"""Web crawler and knowledge ingestion pipeline for SPORE.

DuckDuckGoCrawler orchestrates:
  1. Query expansion via the optional AI client
  2. Search via the pluggable SearchProvider
  3. Content extraction and deduplication
  4. MemoryNode construction with confidence scoring and categorisation

Extension point: override _extract_content() to bind a real scraper.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Set

from .ai_client import ExternalAIClient
from .models import MemoryNode
from .search import DuckDuckGoSearchProvider, SearchProvider


class DuckDuckGoCrawler:
    """Crawler pipeline with pluggable search provider."""

    def __init__(
        self,
        search_provider: Optional[SearchProvider] = None,
        ai_client: Optional[ExternalAIClient] = None,
    ) -> None:
        self.provider    = search_provider or self._build_default_provider()
        self.ai_client   = ai_client
        self.visited_urls: Set[str]         = set()
        self.knowledge_base: List[MemoryNode] = []

    async def search_and_learn(self, query: str, context: str = "") -> List[MemoryNode]:
        queries = [query]
        if self.ai_client:
            queries = await self.ai_client.suggest_search_queries(query, context=context)

        search_results: List[Dict[str, str]] = []
        for variant in queries:
            search_results.extend(await self.provider.search(variant))

        # Deduplicate by URL before processing
        deduped: Dict[str, Dict[str, str]] = {}
        for result in search_results:
            url = result.get("url", "")
            if url and url not in deduped:
                deduped[url] = result

        learned_nodes: List[MemoryNode] = []
        for result in deduped.values():
            if result["url"] in self.visited_urls:
                continue

            content = await self._extract_content(result)
            node = MemoryNode(
                content=content,
                source=result["url"],
                timestamp=datetime.utcnow(),
                confidence=self._calculate_confidence(content, context),
                category=self._categorize(content),
                related_concepts=self._extract_concepts(content),
            )
            self.visited_urls.add(result["url"])
            self.knowledge_base.append(node)
            learned_nodes.append(node)

        return learned_nodes

    # ── Extension points ──────────────────────────────────────────────────────

    async def _extract_content(self, result: Dict[str, str]) -> str:
        """Override to bind a real scraping/extraction pipeline."""
        return f"{result['title']}. {result['snippet']}"

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_default_provider(self) -> SearchProvider:
        try:
            return DuckDuckGoSearchProvider()
        except Exception:
            return SearchProvider()

    def _calculate_confidence(self, content: str, context: str) -> float:
        score = 0.5
        if context and context.lower() in content.lower():
            score += 0.3
        if any(marker in content.lower() for marker in ["research", "study", "paper", "analysis"]):
            score += 0.2
        return min(1.0, score)

    def _categorize(self, content: str) -> str:
        categories = {
            "definition": ["defined", "refers to", "means"],
            "tutorial":   ["how to", "guide", "steps", "tutorial"],
            "research":   ["study", "paper", "research", "analysis"],
            "discussion": ["debate", "opinion", "argue"],
        }
        lowered = content.lower()
        for category, markers in categories.items():
            if any(marker in lowered for marker in markers):
                return category
        return "general"

    def _extract_concepts(self, content: str) -> List[str]:
        words = re.findall(r"\b[A-Z][a-z]{3,}\b", content)
        return list(dict.fromkeys(words))[:10]
