"""
SPORE AI Framework — Evaluation & Ranking Engine (§2.2.5)

Scores each candidate response relative to the query and produces a scalar
quality score.  Two ranking strategies are available:

1. HeuristicRanker   — fast, no model required; uses TF-IDF-like overlap,
                        length normalisation, and coherence signals.
2. ModelRanker       — uses an OpenAI-compatible cross-encoder prompt to
                        obtain a quality score from a language model.

The EvaluationRankingEngine composes both, falling back to heuristics when
no model is configured.

Metrics (§6.2):
    Pairwise accuracy  ≥ 85%
    NDCG               ≥ 0.9
"""

from __future__ import annotations

import math
import re
from typing import List

from ..config import AIModelConfig
from .models import Candidate


# ---------------------------------------------------------------------------
# Heuristic ranker
# ---------------------------------------------------------------------------

class HeuristicRanker:
    """
    Scores candidates via lexical overlap + length + coherence signals.

    Score components:
    - Lexical overlap with query  (weighted 0.5)
    - Length normalisation        (prefers moderate-length responses, 0.3)
    - Coherence proxy             (ends with punctuation, no repetition, 0.2)
    """

    def score(self, query: str, candidate_text: str) -> float:
        if not candidate_text.strip():
            return 0.0

        overlap    = self._lexical_overlap(query, candidate_text)
        length_s   = self._length_score(candidate_text)
        coherence  = self._coherence_score(candidate_text)

        return round(0.5 * overlap + 0.3 * length_s + 0.2 * coherence, 4)

    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _lexical_overlap(self, query: str, response: str) -> float:
        q_words = set(self._tokenise(query))
        r_words = set(self._tokenise(response))
        if not q_words:
            return 0.0
        return len(q_words & r_words) / len(q_words)

    @staticmethod
    def _length_score(text: str) -> float:
        """Prefer responses between 20 and 300 characters."""
        n = len(text.strip())
        if n < 5:
            return 0.0
        if n <= 20:
            return n / 20.0
        if n <= 300:
            return 1.0
        # Penalise very long responses
        return max(0.0, 1.0 - (n - 300) / 1000.0)

    @staticmethod
    def _coherence_score(text: str) -> float:
        score = 0.5
        # Bonus: ends with sentence-ending punctuation
        if re.search(r"[.!?]$", text.strip()):
            score += 0.3
        # Penalty: obvious repetition (same word appears >3 times)
        words = re.findall(r"\w+", text.lower())
        if words:
            freq   = max(words.count(w) for w in set(words))
            repeat = freq / max(len(words), 1)
            if repeat > 0.3:
                score -= 0.2
        return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Model-based ranker (optional — requires configured AI client)
# ---------------------------------------------------------------------------

_RANK_SYSTEM = (
    "You are a response quality judge. "
    "Given a query and a candidate response, rate the response quality on a "
    "scale from 0.0 (completely irrelevant / unhelpful) to 1.0 (perfect). "
    "Reply with a single float and nothing else."
)


class ModelRanker:
    """
    Scores candidates by prompting an OpenAI-compatible model (§2.2.5).

    The model acts as a cross-encoder: it receives the (query, candidate)
    pair and returns a scalar quality score in [0, 1].
    """

    def __init__(self, config: AIModelConfig) -> None:
        self.config  = config
        self._client = None

    def is_configured(self) -> bool:
        return bool(self.config.model_id and self.config.api_key)

    async def score(self, query: str, candidate_text: str) -> float:
        if not self.is_configured():
            return 0.0

        client = self._get_client()
        prompt = f"Query: {query}\n\nCandidate response: {candidate_text}"
        try:
            response = await client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {"role": "system", "content": _RANK_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=8,
            )
            raw = (response.choices[0].message.content or "0").strip()
            return max(0.0, min(1.0, float(raw)))
        except Exception:
            return 0.0

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the `openai` package to use the ModelRanker."
            ) from exc
        kwargs: dict = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = AsyncOpenAI(**kwargs)
        return self._client


# ---------------------------------------------------------------------------
# EvaluationRankingEngine — composes both rankers
# ---------------------------------------------------------------------------

class EvaluationRankingEngine:
    """
    Scores and ranks a list of candidates relative to a query (§2.2.5).

    Strategy:
    - Uses ModelRanker when the AI model is configured.
    - Blends model score (0.7) and heuristic score (0.3) for robustness.
    - Falls back to HeuristicRanker alone when no model is available.
    """

    def __init__(self, config: AIModelConfig) -> None:
        self._heuristic = HeuristicRanker()
        self._model     = ModelRanker(config)

    async def rank(self, query: str, candidates: List[Candidate]) -> List[Candidate]:
        """
        Score each candidate and return them sorted by score descending.

        Mutates scores in place, then returns a new sorted list.
        """
        for candidate in candidates:
            heuristic_score = self._heuristic.score(query, candidate.text)

            if self._model.is_configured():
                model_score     = await self._model.score(query, candidate.text)
                candidate.score = round(0.7 * model_score + 0.3 * heuristic_score, 4)
            else:
                candidate.score = heuristic_score

        return sorted(candidates, key=lambda c: c.score, reverse=True)

    # ------------------------------------------------------------------
    # Offline evaluation helpers (§6.2)
    # ------------------------------------------------------------------

    @staticmethod
    def pairwise_accuracy(ranked: List[Candidate], gold_index: int) -> float:
        """
        Compute pairwise accuracy: fraction of pairs where the gold response
        is ranked above the non-gold response.
        """
        if len(ranked) < 2:
            return 1.0
        gold_score = ranked[gold_index].score if gold_index < len(ranked) else 0.0
        wins = sum(1 for i, c in enumerate(ranked) if i != gold_index and gold_score > c.score)
        total = len(ranked) - 1
        return wins / total if total else 1.0

    @staticmethod
    def ndcg(ranked: List[Candidate], gold_index: int) -> float:
        """
        Normalised Discounted Cumulative Gain (§6.2).

        Gold candidate has relevance 1, all others 0.
        """
        def dcg(relevances: List[float]) -> float:
            return sum(
                rel / math.log2(i + 2)
                for i, rel in enumerate(relevances)
            )

        actual_rels   = [0.0] * len(ranked)
        if gold_index < len(ranked):
            actual_rels[gold_index] = 1.0
        ideal_rels    = sorted(actual_rels, reverse=True)

        actual_dcg = dcg(actual_rels)
        ideal_dcg  = dcg(ideal_rels)
        return actual_dcg / ideal_dcg if ideal_dcg else 0.0
