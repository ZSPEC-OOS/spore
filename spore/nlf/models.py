"""
NLF Framework Data Models
AI Natural Language Fluency Framework v1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SlotFunction(str, Enum):
    """Functional role of each response slot (per framework §4.1)."""
    DIRECT = "direct"               # Slots 1-2: straightforward, neutral
    ELABORATED = "elaborated"       # Slots 3-4: expanded, polite
    COLLOQUIAL = "colloquial"       # Slots 5-6: informal, contemporary
    DEFLECTION = "deflection"       # Slot 7: clarification request / hedge
    META = "meta"                   # Slot 8: reflection on question/process
    PLAYFUL = "playful"             # Slot 9: humor, wordplay
    LIMITATION = "limitation"       # Slot 10: inability / ethical boundary


class Register(str, Enum):
    """Sociolinguistic register levels."""
    FORMAL = "formal"
    NEUTRAL = "neutral"
    CASUAL = "casual"
    PLAYFUL = "playful"


class FeedbackType(str, Enum):
    """Training feedback signal types (per framework §6.1)."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PREFERRED = "preferred"         # Alternative slot would have been better


class VariantTechnique(str, Enum):
    """Surface variation generation techniques (per framework §3.2)."""
    LEXICAL_SUBSTITUTION = "lexical_substitution"
    SYNTACTIC_ALTERNATION = "syntactic_alternation"
    PRAGMATIC_MODULATION = "pragmatic_modulation"
    ELLIPSIS_EXPANSION = "ellipsis_expansion"
    CONTEXTUAL_EMBEDDING = "contextual_embedding"
    NOISE_TOLERANCE = "noise_tolerance"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class ResponseSlot:
    """
    One of the 10 response options for a given question type (§4.1).

    Slot numbers 1-10 follow the framework's functional taxonomy:
      1-2  → Direct/Cooperative
      3-4  → Elaborated/Formal
      5-6  → Colloquial/Casual
      7    → Deflection/Clarification
      8    → Meta-Commentary
      9    → Playful/Creative
      10   → Limitation/Refusal
    """
    slot_number: int                # 1–10
    text: str                       # Surface text of this response
    function: SlotFunction
    register: Register
    use_case: str = ""              # Human-readable context hint

    def __post_init__(self) -> None:
        if not 1 <= self.slot_number <= 10:
            raise ValueError(f"slot_number must be 1-10, got {self.slot_number}")


@dataclass
class SurfaceVariant:
    """
    A single surface-form instantiation of a base question (§3.2).

    Validated against:
    - Semantic equivalence to base question
    - Grammaticality
    - Register consistency
    - Deduplication (cosine similarity < 0.85 to existing variants)
    """
    text: str
    technique: VariantTechnique
    register: Register
    validated: bool = False         # True once validation checks pass
    similarity_score: float = 0.0  # Max cosine sim to existing variants (must be < 0.85)


@dataclass
class BaseQuestion:
    """
    Core semantic template that underlies many surface forms (§2).

    Example: "How are you?" → greeting + status inquiry base question,
    with 100+ surface variants and 10 response slots.
    """
    id: str
    semantic_intent: str            # E.g. "greeting + status inquiry"
    canonical_form: str             # E.g. "How are you?"
    slots: list[ResponseSlot] = field(default_factory=list)
    variants: list[SurfaceVariant] = field(default_factory=list)

    def get_slot(self, number: int) -> Optional[ResponseSlot]:
        return next((s for s in self.slots if s.slot_number == number), None)

    def add_slot(self, slot: ResponseSlot) -> None:
        if any(s.slot_number == slot.slot_number for s in self.slots):
            raise ValueError(f"Slot {slot.slot_number} already exists for question '{self.id}'")
        self.slots.append(slot)
        self.slots.sort(key=lambda s: s.slot_number)

    def add_variant(self, variant: SurfaceVariant) -> None:
        if variant.similarity_score >= 0.85:
            raise ValueError(
                f"Variant '{variant.text}' too similar to existing variants "
                f"(sim={variant.similarity_score:.2f} ≥ 0.85)"
            )
        self.variants.append(variant)


@dataclass
class Category:
    """
    A thematic grouping of base questions within a Stage.

    Each category trains one conversational competency cluster.
    Contains ~5 base questions and requires ≥85% accuracy to advance (§7.1).
    """
    id: str
    name: str
    description: str
    stage_number: int
    questions: list[BaseQuestion] = field(default_factory=list)

    # Mastery tracking
    accuracy: float = 0.0           # Current held-out accuracy (target ≥ 0.85)
    generalization: float = 0.0     # Novel-variant accuracy (target ≥ 0.80)
    retention: float = 0.0          # Previous-material accuracy (target ≥ 0.90)
    cycles_completed: int = 0

    @property
    def is_mastered(self) -> bool:
        return (
            self.accuracy >= 0.85
            and self.generalization >= 0.80
            and self.retention >= 0.90
        )


@dataclass
class Stage:
    """
    Developmental plateau (§7). Must be mastered before the next unlocks.

    The 11 stages (0-10) cover:
      Stage 0  → Foundational social language    (5 categories)
      Stage 1  → Early conceptual language       (3 categories)
      Stage 2  → Intermediate fluency            (3 categories)
      Stage 3  → Advanced fluency                (3 categories)
      Stage 4  → Metacognitive reasoning         (3 categories)
      Stage 5  → Domain expertise                (3 categories)
      Stage 6  → Creative/aesthetic language     (3 categories)
      Stage 7  → Ethical/philosophical reasoning (2 categories)
      Stage 8  → Metacommunication               (2 categories)
      Stage 9  → Specialized adaptation          (2 categories)
      Stage 10 → Meta-learning & system evolution(1 category)
    """
    number: int                     # 0–10
    name: str
    description: str
    categories: list[Category] = field(default_factory=list)
    unlocked: bool = False          # Set True when previous stage is mastered

    # History window per stage (§5.1)
    MAX_HISTORY_TURNS: dict[int, int] = field(default_factory=lambda: {
        0: 2, 1: 2, 2: 5, 3: 5, 4: 8, 5: 8, 6: 8, 7: 10, 8: 10, 9: 10, 10: 10
    })

    @property
    def max_history(self) -> int:
        return self.MAX_HISTORY_TURNS.get(self.number, 10)

    @property
    def is_complete(self) -> bool:
        return all(cat.is_mastered for cat in self.categories)


# ---------------------------------------------------------------------------
# Training runtime structures
# ---------------------------------------------------------------------------

@dataclass
class ContextTurn:
    """A single turn in the conversation history window (§5)."""
    role: str                       # "user" or "assistant"
    text: str
    slot_number: Optional[int] = None   # Set for assistant turns


@dataclass
class TrainingInstance:
    """
    One complete training example (§8.1, ~75 tokens total).

    question + context → selected_slot → feedback
    """
    instance_id: str
    category_id: str
    base_question_id: str
    surface_variant: str            # The specific variant presented
    context: list[ContextTurn] = field(default_factory=list)
    selected_slot: Optional[int] = None
    gold_slot: Optional[int] = None
    feedback: Optional[FeedbackType] = None
    preferred_slot: Optional[int] = None   # For FeedbackType.PREFERRED


@dataclass
class TrainingCycle:
    """
    A complete pass through a category's question-variant space (§2).
    Typically 500-2,000 unique instances.
    """
    cycle_id: str
    category_id: str
    stage_number: int
    instances: list[TrainingInstance] = field(default_factory=list)
    completed: bool = False

    @property
    def accuracy(self) -> float:
        scored = [i for i in self.instances if i.feedback is not None]
        if not scored:
            return 0.0
        correct = sum(1 for i in scored if i.feedback == FeedbackType.CORRECT)
        return correct / len(scored)


@dataclass
class FeedbackSignal:
    """
    Binary or graded feedback for a training instance (§6.1).
    """
    instance_id: str
    feedback: FeedbackType
    preferred_slot: Optional[int] = None
    notes: str = ""
