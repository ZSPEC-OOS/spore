"""
NLF Response Formatter
Maps a selected ResponseSlot to a final surface string (framework §11.1).

The formatter applies light post-processing to slot text:
- Sentence-case normalisation
- Punctuation normalisation
- Optional slot-internal variation (combinatorial mixing)
"""

from __future__ import annotations

import random
import re

from .models import ResponseSlot


# ---------------------------------------------------------------------------
# Punctuation / casing helpers
# ---------------------------------------------------------------------------

def _ensure_terminal_punctuation(text: str) -> str:
    """Add a period if the text has no terminal punctuation."""
    stripped = text.rstrip()
    if stripped and stripped[-1] not in ".!?":
        return stripped + "."
    return stripped


def _sentence_case(text: str) -> str:
    """Capitalise first character, leave the rest unchanged."""
    if not text:
        return text
    return text[0].upper() + text[1:]


# ---------------------------------------------------------------------------
# Slot-internal variation
# ---------------------------------------------------------------------------

# Small synonym pools for common words; applied probabilistically
_SYNONYM_POOL: dict[str, list[str]] = {
    "good": ["good", "great", "well", "fine"],
    "okay": ["okay", "alright", "fine", "decent"],
    "thanks": ["thanks", "thank you", "cheers"],
    "hello": ["hello", "hi", "hey"],
    "sorry": ["sorry", "apologies", "my apologies"],
}

_VARIATION_PROBABILITY = 0.20  # 20% chance of synonym substitution


def _apply_lexical_variation(text: str, *, vary: bool = True) -> str:
    """Probabilistically substitute words with synonyms for surface variation."""
    if not vary:
        return text
    words = text.split()
    result = []
    for word in words:
        clean = re.sub(r"[^\w']", "", word.lower())
        if clean in _SYNONYM_POOL and random.random() < _VARIATION_PROBABILITY:
            # Preserve trailing punctuation
            suffix = re.sub(r"[\w']+", "", word)
            replacement = random.choice(_SYNONYM_POOL[clean])
            # Match original capitalisation
            if word[0].isupper():
                replacement = replacement.capitalize()
            result.append(replacement + suffix)
        else:
            result.append(word)
    return " ".join(result)


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class ResponseFormatter:
    """
    Converts a ResponseSlot into a final surface string.

    Usage::

        formatter = ResponseFormatter()
        output = formatter.format(slot)
        # "I'm doing quite well, thank you for asking."

        # With variation disabled (for deterministic output / testing):
        output = formatter.format(slot, vary=False)
    """

    def __init__(self, vary: bool = True) -> None:
        self.vary = vary

    def format(self, slot: ResponseSlot, *, vary: Optional[bool] = None) -> str:
        """
        Format a ResponseSlot into a final output string.

        Args:
            slot: The ResponseSlot selected by SlotSelector.
            vary: Override instance-level vary setting if provided.

        Returns:
            A clean, punctuated surface string.
        """
        apply_variation = self.vary if vary is None else vary
        text = slot.text

        # Apply lexical variation for training diversity (§3.1)
        text = _apply_lexical_variation(text, vary=apply_variation)

        # Normalise casing and punctuation
        text = _sentence_case(text)
        text = _ensure_terminal_punctuation(text)

        return text

    def format_with_slot_number(self, slot: ResponseSlot, *, vary: Optional[bool] = None) -> tuple[int, str]:
        """Return (slot_number, formatted_text) tuple."""
        return slot.slot_number, self.format(slot, vary=vary)


# Resolve forward reference from type hint
from typing import Optional  # noqa: E402 (avoid circular at module top)
