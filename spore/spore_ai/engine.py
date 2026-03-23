"""
SPORE AI Framework — SporeAIEngine (§2.1)

Orchestrates the full generate → rank → select → feedback pipeline.

Inference workflow (§7.2):
    1. Receive query
    2. Generate N candidates          (CandidateExpansionEngine)
    3. Rank candidates                (EvaluationRankingEngine)
    4. Apply learned feedback weights (FeedbackLoopSystem)
    5. Select best response           (SelectionLayer)
    6. Return output                  (GenerateResponse)

Non-functional targets (§4.3):
    Latency   < 500 ms per query (single-model mode)
    Throughput ≥ 100 QPS (horizontal scaling)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..config import AIModelConfig
from .feedback import FeedbackLoopSystem
from .generator import CandidateExpansionEngine
from .models import (
    Candidate,
    FeedbackMode,
    FeedbackRecord,
    GenerateRequest,
    GenerateResponse,
    SelectionPolicy,
)
from .ranker import EvaluationRankingEngine
from .selector import SelectionLayer
from .tokenizer import SporeTokenizer


class SporeAIEngine:
    """
    SPORE AI top-level engine — Self-Propagating Optimized Response Engine.

    Architecture (§2.1):

        [Data Ingestion]  ──  handled externally via DataIngestionLayer
             ↓
        [Tokenization]    ──  SporeTokenizer (used for logging / future use)
             ↓
        [Generative Core + Candidate Expansion]
             ↓
        [Evaluation & Ranking]
             ↓
        [Feedback re-ranking]
             ↓
        [Selection]
             ↓
        GenerateResponse

    Usage::

        engine = SporeAIEngine.from_env()
        response = await engine.generate(GenerateRequest(query="What is AI?"))
        print(response.best_response)
    """

    def __init__(
        self,
        config:           AIModelConfig,
        selection_policy: SelectionPolicy = SelectionPolicy.ARGMAX,
        feedback_path:    Optional[Path]  = None,
    ) -> None:
        self.config    = config
        self.tokenizer = SporeTokenizer()
        self.generator = CandidateExpansionEngine(config)
        self.ranker    = EvaluationRankingEngine(config)
        self.selector  = SelectionLayer(policy=selection_policy)
        self.feedback  = (
            FeedbackLoopSystem.load(feedback_path)
            if feedback_path and Path(feedback_path).exists()
            else FeedbackLoopSystem()
        )
        self._feedback_path = feedback_path

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, **kwargs) -> "SporeAIEngine":
        """Create an engine loading AI configuration from environment variables."""
        return cls(config=AIModelConfig.from_env(), **kwargs)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Execute the full SPORE AI inference pipeline for a single query.

        Args:
            request: GenerateRequest with query and generation parameters.

        Returns:
            GenerateResponse with the best response and all ranked candidates.
        """
        # Step 1: Candidate expansion
        candidates = await self.generator.expand(
            query          = request.query,
            num_candidates = request.num_candidates,
            temperature    = request.temperature,
            top_k          = request.top_k,
            top_p          = request.top_p,
        )

        if not candidates:
            return GenerateResponse(best_response="", candidates=[])

        # Step 2: Rank candidates
        ranked = await self.ranker.rank(request.query, candidates)

        # Step 3: Apply feedback-learned preferences
        ranked = self.feedback.top_candidates(request.query, ranked)

        # Step 4: Select best response
        effective_policy = request.selection_policy
        self.selector.policy = effective_policy
        best = self.selector.select(ranked)

        return GenerateResponse(
            best_response = best.text,
            candidates    = ranked,
        )

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def record_feedback(
        self,
        query:           str,
        candidates:      List[Candidate],
        selected_index:  int,
        preferred_index: Optional[int] = None,
        mode:            FeedbackMode  = FeedbackMode.SUPERVISED,
    ) -> None:
        """
        Record human or oracle feedback for a previous generate() call.

        Args:
            query:           Original query string.
            candidates:      Candidate list returned by generate().
            selected_index:  Index of the automatically selected candidate.
            preferred_index: Index of the human-preferred candidate (SFT/pref).
            mode:            FeedbackMode.SUPERVISED or PREFERENCE_LEARNING.
        """
        record = FeedbackRecord(
            query           = query,
            candidates      = candidates,
            selected_index  = selected_index,
            preferred_index = preferred_index,
            mode            = mode,
        )
        self.feedback.record(record)
        if self._feedback_path:
            self.feedback.save(self._feedback_path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_feedback(self, path: str | Path) -> None:
        self.feedback.save(path)

    def export_training_data(self, mode: FeedbackMode = FeedbackMode.SUPERVISED) -> list:
        """
        Export accumulated feedback as a training dataset.

        Args:
            mode: SUPERVISED → SFT format; PREFERENCE_LEARNING → pairwise format.

        Returns:
            List of dicts suitable for offline fine-tuning.
        """
        if mode == FeedbackMode.SUPERVISED:
            return self.feedback.export_sft_dataset()
        return self.feedback.export_preference_dataset()
