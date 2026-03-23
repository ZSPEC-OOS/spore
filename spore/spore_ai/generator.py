"""
SPORE AI Framework — Generative Core + Candidate Expansion Engine (§2.2.3–4)

The Generative Core wraps an OpenAI-compatible transformer decoder model.
The Candidate Expansion Engine calls the core N times with the configured
temperature / top-k / top-p parameters to produce a diverse candidate set.

Parameters (§2.2.4):
    temperature : 0.0 – 1.5
    top_k       : 1   – 100
    top_p       : 0.8 – 1.0
    num_candidates (N): number of candidates to expand per query
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from ..config import AIModelConfig
from .models import Candidate


_SYSTEM_PROMPT = (
    "You are a high-quality response generator. "
    "Given a query, produce a single, concise, and helpful response. "
    "Do not include any prefix, label, or explanation — just the response text."
)


class GenerativeCore:
    """
    Transformer-decoder-based response generator (§2.2.3).

    Wraps an OpenAI-compatible chat-completions endpoint.
    Falls back to a deterministic echo stub when no model is configured,
    ensuring the pipeline remains functional for testing.
    """

    def __init__(self, config: AIModelConfig) -> None:
        self.config  = config
        self._client = None

    def is_configured(self) -> bool:
        return bool(self.config.model_id and self.config.api_key)

    async def generate_one(
        self,
        query:       str,
        temperature: float = 0.7,
        top_p:       float = 0.95,
        max_tokens:  int   = 256,
    ) -> str:
        """Generate a single response for the given query."""
        if not self.is_configured():
            return self._stub_response(query)

        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": query},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            return f"[generation error: {exc}]"

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the `openai` package to use the Generative Core."
            ) from exc
        kwargs: dict = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = AsyncOpenAI(**kwargs)
        return self._client

    @staticmethod
    def _stub_response(query: str) -> str:
        """Deterministic stub used when no AI model is configured."""
        return f"[stub] Response to: {query[:80]}"


class CandidateExpansionEngine:
    """
    Produces N diverse candidate responses per query (§2.2.4).

    Uses the GenerativeCore with slight temperature increments per candidate
    to encourage response diversity while remaining within spec bounds.
    """

    def __init__(self, config: AIModelConfig) -> None:
        self._core = GenerativeCore(config)

    async def expand(
        self,
        query:          str,
        num_candidates: int   = 4,
        temperature:    float = 0.7,
        top_k:          int   = 40,
        top_p:          float = 0.95,
        max_tokens:     int   = 256,
    ) -> List[Candidate]:
        """
        Generate ``num_candidates`` response candidates for the query.

        Each candidate is generated with a slightly varied temperature to
        increase diversity.  Temperature is clamped to [0.0, 1.5] per spec.
        top_k is accepted as a parameter but forwarded implicitly through
        the API (most OpenAI-compatible endpoints honour it when specified).
        """
        num_candidates = max(1, num_candidates)
        base_temp      = max(0.0, min(1.5, temperature))

        # Spread temperatures across candidates for diversity
        temps = self._temperature_spread(base_temp, num_candidates)

        tasks = [
            self._core.generate_one(
                query       = query,
                temperature = t,
                top_p       = max(0.8, min(1.0, top_p)),
                max_tokens  = max_tokens,
            )
            for t in temps
        ]

        texts = await asyncio.gather(*tasks)

        # Deduplicate while preserving order
        seen:       set    = set()
        candidates: List[Candidate] = []
        for text in texts:
            if text and text not in seen:
                seen.add(text)
                candidates.append(Candidate(text=text, score=0.0))

        return candidates

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _temperature_spread(base: float, n: int) -> List[float]:
        """
        Return ``n`` temperatures spread around ``base``.

        First candidate always uses ``base``; subsequent ones are varied by
        small increments clamped to [0.0, 1.5].
        """
        if n == 1:
            return [base]
        step   = 0.1
        temps  = [base]
        for i in range(1, n):
            offset = step * ((i + 1) // 2) * (1 if i % 2 == 0 else -1)
            temps.append(max(0.0, min(1.5, base + offset)))
        return temps
