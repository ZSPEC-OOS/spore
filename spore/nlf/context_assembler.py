"""
NLF Context Assembler
Assembles the history window + current question into a training instance (framework §5).

History window size is capped per stage:
  Stages 0-1  → 0-2 turns
  Stages 2-3  → 3-5 turns
  Stages 4-6  → 6-8 turns
  Stages 7-10 → 8-10 turns
"""

from __future__ import annotations

import uuid
from typing import Optional

from .models import ContextTurn, TrainingInstance


# Maximum history turns per stage (§5.1)
_STAGE_MAX_HISTORY: dict[int, int] = {
    0: 2, 1: 2,
    2: 5, 3: 5,
    4: 8, 5: 8, 6: 8,
    7: 10, 8: 10, 9: 10, 10: 10,
}


def _max_history_for_stage(stage: int) -> int:
    return _STAGE_MAX_HISTORY.get(stage, 10)


class ContextAssembler:
    """
    Maintains a rolling conversation history and assembles TrainingInstances.

    Usage::

        assembler = ContextAssembler(stage=2)
        assembler.add_turn("user", "Hello there!")
        assembler.add_turn("assistant", "Hi! How can I help?", slot_number=1)

        instance = assembler.assemble(
            surface_variant="How's it going?",
            category_id="cat_greetings",
            base_question_id="bq_how_are_you",
        )
    """

    def __init__(self, stage: int = 0) -> None:
        self.stage = stage
        self._max_history = _max_history_for_stage(stage)
        self._history: list[ContextTurn] = []

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def add_turn(self, role: str, text: str, *, slot_number: Optional[int] = None) -> None:
        """Append a turn; trims oldest turns to stay within window."""
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{role}'")
        self._history.append(ContextTurn(role=role, text=text, slot_number=slot_number))
        # Keep only the most recent max_history turns
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def clear_history(self) -> None:
        """Reset conversation history (new conversation)."""
        self._history = []

    @property
    def history(self) -> list[ContextTurn]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Instance assembly
    # ------------------------------------------------------------------

    def assemble(
        self,
        surface_variant: str,
        category_id: str,
        base_question_id: str,
        *,
        gold_slot: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> TrainingInstance:
        """
        Build a TrainingInstance from current history + the presented variant.

        The history snapshot is taken at call time (before the AI's response
        is recorded), so context reflects what the AI sees when answering.
        """
        return TrainingInstance(
            instance_id=instance_id or str(uuid.uuid4()),
            category_id=category_id,
            base_question_id=base_question_id,
            surface_variant=surface_variant,
            context=list(self._history),     # snapshot of current window
            gold_slot=gold_slot,
        )

    def record_response(
        self,
        instance: TrainingInstance,
        selected_slot: int,
        slot_text: str,
    ) -> None:
        """
        Record the AI's slot selection back into the history.

        Call this after the AI selects a slot so subsequent instances
        see the prior response in context.
        """
        instance.selected_slot = selected_slot
        self.add_turn("assistant", slot_text, slot_number=selected_slot)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_prompt(self, surface_variant: str) -> str:
        """
        Render the assembled context as a plain-text prompt string.

        Useful for LLM-based implementations (§11.2).
        """
        lines: list[str] = []
        for turn in self._history:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.text}")
        lines.append(f"User: {surface_variant}")
        return "\n".join(lines)
