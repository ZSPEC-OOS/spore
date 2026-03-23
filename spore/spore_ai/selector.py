"""
SPORE AI Framework — Selection Layer (§2.2.6)

Chooses the optimal response from a ranked list of candidates.

Supported policies (§2.2.6):
    ARGMAX   — always return the highest-scored candidate.
    WEIGHTED — probabilistically sample candidates proportional to their scores
               (temperature-softmax weighting encourages exploration).
"""

from __future__ import annotations

import math
import random
from typing import List

from .models import Candidate, SelectionPolicy


class SelectionLayer:
    """
    Selects the best candidate from a ranked list (§2.2.6).

    Usage::

        layer = SelectionLayer(policy=SelectionPolicy.ARGMAX)
        best  = layer.select(candidates)
    """

    def __init__(
        self,
        policy:             SelectionPolicy = SelectionPolicy.ARGMAX,
        softmax_temperature: float          = 1.0,
    ) -> None:
        self.policy              = policy
        self.softmax_temperature = max(1e-6, softmax_temperature)

    def select(self, candidates: List[Candidate]) -> Candidate:
        """
        Return the selected candidate according to the configured policy.

        Args:
            candidates: Ranked list of Candidates (highest score first).

        Returns:
            The chosen Candidate.

        Raises:
            ValueError: If the candidate list is empty.
        """
        if not candidates:
            raise ValueError("Cannot select from an empty candidate list.")

        if self.policy == SelectionPolicy.ARGMAX:
            return self._argmax(candidates)
        return self._weighted(candidates)

    # ------------------------------------------------------------------
    # Policies
    # ------------------------------------------------------------------

    @staticmethod
    def _argmax(candidates: List[Candidate]) -> Candidate:
        """Return the candidate with the highest score."""
        return max(candidates, key=lambda c: c.score)

    def _weighted(self, candidates: List[Candidate]) -> Candidate:
        """
        Sample a candidate proportionally to softmax(score / temperature).

        Higher temperature → more uniform sampling (exploration).
        Lower temperature  → near-deterministic selection (exploitation).
        """
        scores = [c.score / self.softmax_temperature for c in candidates]
        # Numerical stability: subtract max before exp
        max_s  = max(scores)
        weights = [math.exp(s - max_s) for s in scores]
        total   = sum(weights)
        probs   = [w / total for w in weights]

        r     = random.random()
        cumul = 0.0
        for candidate, prob in zip(candidates, probs):
            cumul += prob
            if r <= cumul:
                return candidate
        return candidates[-1]
