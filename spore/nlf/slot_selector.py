"""
NLF Slot Selector
Selects the appropriate response slot (1-10) given context (framework §4, §6).

The selector maintains per-question slot preference distributions that are
updated by feedback signals over training cycles.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

from .models import (
    BaseQuestion,
    ContextTurn,
    FeedbackSignal,
    FeedbackType,
    Register,
    ResponseSlot,
    SlotFunction,
)


# ---------------------------------------------------------------------------
# Heuristic context rules (§5.3)
# ---------------------------------------------------------------------------

def _infer_context_register(history: list[ContextTurn]) -> Register:
    """
    Infer appropriate register from conversation history.
    Simple heuristic: look at vocabulary in user turns.
    """
    if not history:
        return Register.NEUTRAL

    user_text = " ".join(t.text for t in history if t.role == "user").lower()

    formal_markers = {"good morning", "good afternoon", "certainly", "indeed", "may i", "pleased"}
    casual_markers = {"hey", "yo", "sup", "gonna", "wanna", "kinda", "lol", "haha"}

    if any(m in user_text for m in formal_markers):
        return Register.FORMAL
    if any(m in user_text for m in casual_markers):
        return Register.CASUAL
    return Register.NEUTRAL


def _context_suggests_slot(history: list[ContextTurn], question_text: str) -> Optional[int]:
    """
    Apply the framework's explicit contextual sensitivity rules (§5.3).

    Returns a slot number override, or None if no rule applies.
    """
    if not history:
        return None

    # "How are you?" after user shared bad news → Slot 10 (limitation)
    bad_news_markers = {
        "died", "death", "cancer", "lost", "fired", "broke up", "accident",
        "hospital", "divorced", "devastated", "terrible", "awful",
    }
    last_user = next(
        (t.text.lower() for t in reversed(history) if t.role == "user"), ""
    )
    if any(m in last_user for m in bad_news_markers):
        return 10

    # Reciprocal: user just asked the same question → Slot 4 (reciprocal)
    q_lower = question_text.lower().strip("?! ")
    if last_user.strip("?! ") in {q_lower, q_lower.replace("'", "")}:
        return 4

    return None


# ---------------------------------------------------------------------------
# Slot Selector
# ---------------------------------------------------------------------------

class SlotSelector:
    """
    Selects one of 10 response slots for a given question + context.

    Maintains a preference distribution per (category_id, base_question_id)
    that is updated via apply_feedback().

    Selection algorithm (simplified):
      1. Check hard context rules → override slot if rule fires.
      2. Filter slots to those matching inferred register (soft constraint).
      3. Sample from preference distribution (exploration vs exploitation).
      4. Update distribution on feedback.
    """

    def __init__(self, exploration_rate: float = 0.15) -> None:
        # exploration_rate: probability of random slot selection (ε-greedy)
        self.exploration_rate = exploration_rate
        # preference[key][slot_number] = cumulative score
        self._preferences: dict[str, dict[int, float]] = defaultdict(
            lambda: {i: 1.0 for i in range(1, 11)}
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        question: BaseQuestion,
        context: list[ContextTurn],
        *,
        category_id: str = "",
    ) -> ResponseSlot:
        """
        Select the most appropriate response slot.

        Returns the chosen ResponseSlot object.
        Raises ValueError if the question has no slots defined.
        """
        if not question.slots:
            raise ValueError(f"Question '{question.id}' has no response slots defined.")

        slot_map = {s.slot_number: s for s in question.slots}

        # 1. Hard context rule override
        override = _context_suggests_slot(context, question.canonical_form)
        if override is not None and override in slot_map:
            return slot_map[override]

        # 2. Infer register preference
        preferred_register = _infer_context_register(context)

        # 3. Filter to register-compatible slots (fallback to all if none match)
        compatible = [
            s for s in question.slots
            if s.register == preferred_register or preferred_register == Register.NEUTRAL
        ]
        if not compatible:
            compatible = question.slots

        # 4. ε-greedy selection over preference distribution
        pref_key = f"{category_id}::{question.id}"
        prefs = self._preferences[pref_key]

        if random.random() < self.exploration_rate:
            chosen = random.choice(compatible)
        else:
            # Weighted sample from compatible slots by preference score
            compatible_nums = [s.slot_number for s in compatible]
            weights = [prefs.get(n, 1.0) for n in compatible_nums]
            total = sum(weights)
            r = random.random() * total
            cumulative = 0.0
            chosen_num = compatible_nums[-1]
            for num, w in zip(compatible_nums, weights):
                cumulative += w
                if r <= cumulative:
                    chosen_num = num
                    break
            chosen = slot_map[chosen_num]

        return chosen

    # ------------------------------------------------------------------
    # Feedback & learning
    # ------------------------------------------------------------------

    def apply_feedback(
        self,
        signal: FeedbackSignal,
        category_id: str,
        base_question_id: str,
    ) -> None:
        """
        Update slot preference distribution based on feedback (§6.2).

        - CORRECT  → reinforce selected slot (+1.0)
        - INCORRECT → penalise selected slot (×0.5), reinforce gold slot if known
        - PREFERRED → mild reinforce preferred slot (+0.5), mild penalise selected (×0.8)
        """
        pref_key = f"{category_id}::{base_question_id}"
        prefs = self._preferences[pref_key]

        # Retrieve selected slot from the instance (stored externally)
        # The feedback signal carries instance_id; caller must provide slot info.
        # We trust the caller to pass the correct selected/preferred slots.
        pass  # No-op here; update via update_from_instance()

    def update_from_instance(
        self,
        category_id: str,
        base_question_id: str,
        selected_slot: int,
        feedback: FeedbackType,
        *,
        gold_slot: Optional[int] = None,
        preferred_slot: Optional[int] = None,
    ) -> None:
        """Update preference scores given explicit slot numbers and feedback."""
        pref_key = f"{category_id}::{base_question_id}"
        prefs = self._preferences[pref_key]

        if feedback == FeedbackType.CORRECT:
            prefs[selected_slot] = prefs.get(selected_slot, 1.0) + 1.0

        elif feedback == FeedbackType.INCORRECT:
            prefs[selected_slot] = prefs.get(selected_slot, 1.0) * 0.5
            if gold_slot is not None:
                prefs[gold_slot] = prefs.get(gold_slot, 1.0) + 1.0

        elif feedback == FeedbackType.PREFERRED:
            prefs[selected_slot] = prefs.get(selected_slot, 1.0) * 0.8
            if preferred_slot is not None:
                prefs[preferred_slot] = prefs.get(preferred_slot, 1.0) + 0.5

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_preferences(self, category_id: str, base_question_id: str) -> dict[int, float]:
        """Return current slot preference scores for a question."""
        pref_key = f"{category_id}::{base_question_id}"
        return dict(self._preferences[pref_key])

    def top_slot(self, category_id: str, base_question_id: str) -> int:
        """Return the slot number with the highest preference score."""
        prefs = self.get_preferences(category_id, base_question_id)
        return max(prefs, key=lambda k: prefs[k])
