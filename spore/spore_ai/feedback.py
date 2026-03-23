"""
SPORE AI Framework — Feedback Loop System (§2.2.7)

Incorporates evaluation results back into the system via two modes:

1. SUPERVISED fine-tuning  — adjusts slot preference weights when a
   gold response is provided.
2. PREFERENCE_LEARNING     — updates pairwise preference scores when a
   human or oracle indicates that one candidate is better than another.

The FeedbackLoopSystem maintains an in-memory preference store and can
export training-ready datasets for offline model fine-tuning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import Candidate, FeedbackMode, FeedbackRecord


class FeedbackLoopSystem:
    """
    Manages feedback signals and updates internal quality estimates (§2.2.7).

    Preference store format:
        {query_hash: {candidate_text: preference_score}}

    preference_score starts at 0.5 (neutral) and moves towards 1.0 (preferred)
    or 0.0 (dispreferred) based on accumulated signals.

    Usage::

        fb = FeedbackLoopSystem()
        fb.record(FeedbackRecord(...))
        dataset = fb.export_sft_dataset()
    """

    _ALPHA = 0.2  # Learning rate for preference score update

    def __init__(self) -> None:
        # {query: {candidate_text: preference_score}}
        self._preferences: Dict[str, Dict[str, float]] = {}
        self._history:     List[FeedbackRecord]        = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, record: FeedbackRecord) -> None:
        """
        Process a feedback record and update preference scores.

        Args:
            record: FeedbackRecord with query, candidates, selected_index,
                    preferred_index, and feedback mode.
        """
        self._history.append(record)

        if record.mode == FeedbackMode.SUPERVISED:
            self._update_supervised(record)
        else:
            self._update_preference(record)

    def score(self, query: str, candidate_text: str) -> float:
        """Return the learned preference score for a (query, candidate) pair."""
        return self._preferences.get(query, {}).get(candidate_text, 0.5)

    def top_candidates(self, query: str, candidates: List[Candidate]) -> List[Candidate]:
        """
        Re-rank candidates using learned preferences (for feedback-loop
        integration into the ranking engine).
        """
        prefs = self._preferences.get(query, {})
        for c in candidates:
            learned = prefs.get(c.text, 0.5)
            # Blend ranking score with learned preference
            c.score = round(0.7 * c.score + 0.3 * learned, 4)
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    def export_sft_dataset(self) -> List[Dict]:
        """
        Export a supervised fine-tuning dataset (§2.2.7 — SFT mode).

        Format:
            [{"query": str, "response": str}, ...]

        Includes only records where a preferred response was identified.
        """
        dataset: List[Dict] = []
        for record in self._history:
            if record.preferred_index is not None:
                idx = record.preferred_index
                if 0 <= idx < len(record.candidates):
                    dataset.append({
                        "query":    record.query,
                        "response": record.candidates[idx].text,
                    })
        return dataset

    def export_preference_dataset(self) -> List[Dict]:
        """
        Export a preference learning dataset (§2.2.7 — preference learning mode).

        Format:
            [{"query": str, "chosen": str, "rejected": str}, ...]
        """
        dataset: List[Dict] = []
        for record in self._history:
            if (
                record.mode == FeedbackMode.PREFERENCE_LEARNING
                and record.preferred_index is not None
            ):
                chosen_idx   = record.preferred_index
                rejected_idx = record.selected_index
                if (
                    chosen_idx != rejected_idx
                    and 0 <= chosen_idx  < len(record.candidates)
                    and 0 <= rejected_idx < len(record.candidates)
                ):
                    dataset.append({
                        "query":    record.query,
                        "chosen":   record.candidates[chosen_idx].text,
                        "rejected": record.candidates[rejected_idx].text,
                    })
        return dataset

    def save(self, path: str | Path) -> None:
        """Persist preference store and history length to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "preferences":   self._preferences,
                    "total_records": len(self._history),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeedbackLoopSystem":
        """Restore a persisted FeedbackLoopSystem."""
        fb   = cls()
        path = Path(path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            fb._preferences = data.get("preferences", {})
        return fb

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_supervised(self, record: FeedbackRecord) -> None:
        """Increase preference score for the gold (preferred) response."""
        if record.preferred_index is None:
            return
        prefs = self._preferences.setdefault(record.query, {})
        for i, candidate in enumerate(record.candidates):
            current = prefs.get(candidate.text, 0.5)
            if i == record.preferred_index:
                prefs[candidate.text] = min(1.0, current + self._ALPHA)
            else:
                prefs[candidate.text] = max(0.0, current - self._ALPHA * 0.5)

    def _update_preference(self, record: FeedbackRecord) -> None:
        """
        Update preference scores using a pairwise signal.

        The preferred candidate gains +ALPHA; the selected (rejected)
        candidate loses -ALPHA.
        """
        prefs = self._preferences.setdefault(record.query, {})
        if record.preferred_index is not None:
            idx = record.preferred_index
            if 0 <= idx < len(record.candidates):
                text    = record.candidates[idx].text
                current = prefs.get(text, 0.5)
                prefs[text] = min(1.0, current + self._ALPHA)

        sel_idx = record.selected_index
        if (
            0 <= sel_idx < len(record.candidates)
            and sel_idx != record.preferred_index
        ):
            text    = record.candidates[sel_idx].text
            current = prefs.get(text, 0.5)
            prefs[text] = max(0.0, current - self._ALPHA)
