"""External AI model client for SPORE.

Used ONLY for search/crawl enrichment (generating alternate search queries
and validating API connectivity).  Never used to synthesise SPORE's answers.
"""

from __future__ import annotations

from typing import List, Tuple

from .config import AIModelConfig


class ExternalAIClient:
    """Optional OpenAI-compatible client used ONLY for search/crawl enrichment."""

    def __init__(self, config: AIModelConfig) -> None:
        self.config  = config
        self._client = None

    def is_configured(self) -> bool:
        return bool(self.config.model_id and self.config.api_key)

    async def test_connection(self) -> Tuple[bool, str]:
        if not self.is_configured():
            return False, "AI model is not configured (model_id/api_key missing)."

        try:
            client = self._get_client()
        except Exception as exc:
            return False, f"AI client unavailable: {exc}"

        try:
            response = await client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {"role": "system", "content": "You validate API connectivity."},
                    {"role": "user",   "content": "Reply with: OK"},
                ],
                temperature=0,
                max_tokens=5,
            )
            text = response.choices[0].message.content if response.choices else ""
            return True, f"Connected (response: {text or 'empty'})"
        except Exception as exc:
            return False, f"Connection failed: {exc}"

    async def suggest_search_queries(self, query: str, context: str = "") -> List[str]:
        if not self.is_configured():
            return [query]

        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate up to 2 alternate web search queries for crawling. "
                            "Return one query per line, no bullets, no explanation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\nContext: {context or 'none'}",
                    },
                ],
                temperature=0.2,
                max_tokens=120,
            )
            text = response.choices[0].message.content if response.choices else ""
            variants = [line.strip("-• \t") for line in text.splitlines() if line.strip()]
            unique = [query]
            for candidate in variants:
                if candidate.lower() != query.lower() and candidate not in unique:
                    unique.append(candidate)
                if len(unique) >= 3:
                    break
            return unique
        except Exception:
            return [query]

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except Exception as exc:
            raise RuntimeError("Install `openai` package to use external AI model.") from exc

        kwargs = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = AsyncOpenAI(**kwargs)
        return self._client
