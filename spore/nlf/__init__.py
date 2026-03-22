"""
spore.nlf – AI Natural Language Fluency Framework v1.0

Public API::

    from spore.nlf import NLFFramework, Stage, Category, BaseQuestion, ResponseSlot
    from spore.nlf.models import SlotFunction, Register, FeedbackType
"""

from .framework import NLFFramework
from .models import (
    BaseQuestion,
    Category,
    ContextTurn,
    FeedbackSignal,
    FeedbackType,
    Register,
    ResponseSlot,
    SlotFunction,
    Stage,
    SurfaceVariant,
    TrainingCycle,
    TrainingInstance,
    VariantTechnique,
)
from .context_assembler import ContextAssembler
from .feedback_comparator import FeedbackComparator, EvaluationResult
from .response_formatter import ResponseFormatter
from .slot_selector import SlotSelector
from .variant_generator import VariantGenerator

__all__ = [
    # Framework
    "NLFFramework",
    # Pipeline components
    "ContextAssembler",
    "FeedbackComparator",
    "EvaluationResult",
    "ResponseFormatter",
    "SlotSelector",
    "VariantGenerator",
    # Models
    "BaseQuestion",
    "Category",
    "ContextTurn",
    "FeedbackSignal",
    "FeedbackType",
    "Register",
    "ResponseSlot",
    "SlotFunction",
    "Stage",
    "SurfaceVariant",
    "TrainingCycle",
    "TrainingInstance",
    "VariantTechnique",
]
