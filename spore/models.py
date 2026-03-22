"""Core data models for SPORE.

Contains the fundamental data structures used across the system:
- LearningPhase  — enum of engine states
- Neuron         — a node in the conceptual knowledge graph
- MemoryNode     — a learned knowledge fragment with provenance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List


class LearningPhase(Enum):
    INITIALIZATION       = "initialization"
    GENERAL_LANGUAGE     = "general_language"
    TOPIC_SPECIALIZATION = "topic_specialization"
    EXPERT_MODE          = "expert_mode"
    TESTING              = "testing"


@dataclass
class Neuron:
    id:               str
    concept:          str
    weight:           float     = 0.1
    connections:      List[str] = field(default_factory=list)
    activation_count: int       = 0
    layer:            int       = 0
    x:                float     = 0.0
    y:                float     = 0.0

    def activate(self) -> None:
        self.activation_count += 1
        self.weight = min(1.0, self.weight + 0.05)


@dataclass
class MemoryNode:
    content:          str
    source:           str
    timestamp:        datetime
    confidence:       float
    category:         str
    related_concepts: List[str] = field(default_factory=list)
