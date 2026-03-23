"""
SPORE AI Framework — Public API

Self-Propagating Optimized Response Engine v1.0

Components:
    SporeAIEngine            — top-level orchestrator
    GenerateRequest          — query + generation parameters
    GenerateResponse         — best response + ranked candidates
    Candidate                — a single candidate with score
    DataIngestionLayer       — JSONL/CSV dataset ingestion + validation
    SporeTokenizer           — subword BPE tokeniser
    CandidateExpansionEngine — multi-candidate generation
    EvaluationRankingEngine  — scoring and ranking
    SelectionLayer           — argmax / weighted selection
    FeedbackLoopSystem       — supervised and preference feedback
    SelectionPolicy          — ARGMAX | WEIGHTED
    FeedbackMode             — SUPERVISED | PREFERENCE_LEARNING
"""

from .engine    import SporeAIEngine
from .feedback  import FeedbackLoopSystem
from .generator import CandidateExpansionEngine
from .ingestion import DataIngestionLayer
from .models    import (
    Candidate,
    DataRecord,
    FeedbackMode,
    FeedbackRecord,
    GenerateRequest,
    GenerateResponse,
    ScoredDataRecord,
    ScoredResponse,
    SelectionPolicy,
)
from .ranker    import EvaluationRankingEngine, HeuristicRanker
from .selector  import SelectionLayer
from .tokenizer import SporeTokenizer

__all__ = [
    "SporeAIEngine",
    "FeedbackLoopSystem",
    "CandidateExpansionEngine",
    "DataIngestionLayer",
    "Candidate",
    "DataRecord",
    "FeedbackMode",
    "FeedbackRecord",
    "GenerateRequest",
    "GenerateResponse",
    "ScoredDataRecord",
    "ScoredResponse",
    "SelectionPolicy",
    "EvaluationRankingEngine",
    "HeuristicRanker",
    "SelectionLayer",
    "SporeTokenizer",
]
