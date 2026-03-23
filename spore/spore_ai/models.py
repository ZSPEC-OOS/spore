"""
SPORE AI Framework — Data Models (§3.1)

Defines the canonical data structures used across the generate-rank-select
pipeline described in the SPORE AI specification v1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SelectionPolicy(str, Enum):
    """Response selection strategy (§2.2.6)."""
    ARGMAX   = "argmax"           # Always pick the highest-scored candidate
    WEIGHTED = "weighted"         # Sample proportionally to scores


class FeedbackMode(str, Enum):
    """Feedback loop training mode (§2.2.7)."""
    SUPERVISED           = "supervised"           # Supervised fine-tuning
    PREFERENCE_LEARNING  = "preference_learning"  # Preference / RLHF-style


# ---------------------------------------------------------------------------
# Dataset schemas  (§3.1)
# ---------------------------------------------------------------------------

@dataclass
class DataRecord:
    """
    Required schema (§3.1.1).

    ``correct_index`` is 0-based into ``responses``.
    """
    question:      str
    responses:     List[str]
    correct_index: int

    def __post_init__(self) -> None:
        if not 0 <= self.correct_index < len(self.responses):
            raise ValueError(
                f"correct_index {self.correct_index} out of range "
                f"for {len(self.responses)} responses."
            )


@dataclass
class ScoredDataRecord:
    """Extended schema (§3.1.2) — each response carries a float quality score."""
    question:  str
    responses: List[ScoredResponse]


@dataclass
class ScoredResponse:
    """A single response with an associated quality score."""
    text:  str
    score: float


# ---------------------------------------------------------------------------
# Runtime pipeline structures
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """
    A single generated response option produced by the Candidate Expansion
    Engine (§2.2.4) and scored by the Evaluation & Ranking Engine (§2.2.5).
    """
    text:  str
    score: float = 0.0


@dataclass
class GenerateRequest:
    """
    Input protocol (§3.3.1).

    ``generation_params`` is forwarded to the Candidate Expansion Engine
    (temperature, top_k, top_p) and may be omitted to use defaults.
    """
    query:          str
    num_candidates: int = 4
    temperature:    float = 0.7
    top_k:          int  = 40
    top_p:          float = 0.95
    selection_policy: SelectionPolicy = SelectionPolicy.ARGMAX


@dataclass
class GenerateResponse:
    """
    Output protocol (§3.3.2).

    ``best_response`` is the text selected by the Selection Layer.
    ``candidates`` lists all generated candidates with their ranking scores.
    """
    best_response: str
    candidates:    List[Candidate] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Feedback structures (§2.2.7)
# ---------------------------------------------------------------------------

@dataclass
class FeedbackRecord:
    """
    A single piece of human or automated feedback used to update the system.

    For SUPERVISED mode: ``preferred_index`` identifies the gold response.
    For PREFERENCE_LEARNING mode: ``preferred_index`` beats ``selected_index``.
    """
    query:           str
    candidates:      List[Candidate]
    selected_index:  int
    preferred_index: Optional[int] = None
    mode:            FeedbackMode  = FeedbackMode.SUPERVISED
