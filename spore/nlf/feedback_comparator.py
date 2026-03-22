"""
NLF Feedback Comparator
Compares selected slot to gold standard and emits a feedback signal (framework §6).

The comparator supports:
- Binary feedback (correct / incorrect)
- Graded feedback with a preferred alternative
- Batch evaluation across a training cycle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .models import FeedbackSignal, FeedbackType, TrainingInstance


@dataclass
class EvaluationResult:
    """Aggregate statistics for a batch of evaluated instances."""
    total: int
    correct: int
    incorrect: int
    preferred: int          # Instances marked as "preferred alternative exists"
    accuracy: float         # correct / (correct + incorrect)


class FeedbackComparator:
    """
    Evaluates a TrainingInstance against a gold standard slot and emits
    a FeedbackSignal.

    Gold standard assignment:
    - If a ``gold_slot`` is pre-assigned on the instance, use it directly.
    - Otherwise an external gold function (human or oracle) must supply it.

    Usage::

        comparator = FeedbackComparator()

        # Single instance
        signal = comparator.evaluate(instance, gold_slot=1)

        # Batch
        results = comparator.evaluate_batch(instances, gold_map={"id1": 1, "id2": 4})
        print(results.accuracy)
    """

    def evaluate(
        self,
        instance: TrainingInstance,
        *,
        gold_slot: Optional[int] = None,
        preferred_slot: Optional[int] = None,
        tolerance: int = 0,
    ) -> FeedbackSignal:
        """
        Compare instance.selected_slot against the gold slot.

        Args:
            instance: The completed TrainingInstance (must have selected_slot set).
            gold_slot: Override gold slot (uses instance.gold_slot if None).
            preferred_slot: Alternative slot that would have been better.
            tolerance: If > 0, slots within ±tolerance of gold are also correct.

        Returns:
            FeedbackSignal with CORRECT, INCORRECT, or PREFERRED feedback.

        Raises:
            ValueError if selected_slot is not set on the instance.
        """
        if instance.selected_slot is None:
            raise ValueError(
                f"Instance '{instance.instance_id}' has no selected_slot; "
                "slot selection must occur before evaluation."
            )

        effective_gold = gold_slot if gold_slot is not None else instance.gold_slot

        # No gold available → cannot evaluate definitively
        if effective_gold is None:
            return FeedbackSignal(
                instance_id=instance.instance_id,
                feedback=FeedbackType.PREFERRED,
                preferred_slot=preferred_slot,
                notes="No gold slot available; marked as PREFERRED for manual review.",
            )

        selected = instance.selected_slot
        is_correct = (
            selected == effective_gold
            or (tolerance > 0 and abs(selected - effective_gold) <= tolerance)
        )

        if is_correct:
            feedback = FeedbackType.CORRECT
        elif preferred_slot is not None:
            feedback = FeedbackType.PREFERRED
        else:
            feedback = FeedbackType.INCORRECT

        return FeedbackSignal(
            instance_id=instance.instance_id,
            feedback=feedback,
            preferred_slot=preferred_slot if not is_correct else None,
        )

    def evaluate_batch(
        self,
        instances: Sequence[TrainingInstance],
        *,
        gold_map: Optional[dict[str, int]] = None,
        tolerance: int = 0,
    ) -> tuple[list[FeedbackSignal], EvaluationResult]:
        """
        Evaluate a batch of instances.

        Args:
            instances: Sequence of completed TrainingInstances.
            gold_map: Mapping of instance_id → gold_slot. Takes priority over
                      instance.gold_slot when provided.
            tolerance: Passed through to evaluate().

        Returns:
            (signals, result) where signals is one FeedbackSignal per instance
            and result contains aggregate statistics.
        """
        gold_map = gold_map or {}
        signals: list[FeedbackSignal] = []
        correct = incorrect = preferred = 0

        for inst in instances:
            gold = gold_map.get(inst.instance_id, inst.gold_slot)
            sig = self.evaluate(inst, gold_slot=gold, tolerance=tolerance)
            signals.append(sig)

            if sig.feedback == FeedbackType.CORRECT:
                correct += 1
            elif sig.feedback == FeedbackType.INCORRECT:
                incorrect += 1
            else:
                preferred += 1

        total = len(instances)
        denominator = correct + incorrect
        accuracy = correct / denominator if denominator > 0 else 0.0

        result = EvaluationResult(
            total=total,
            correct=correct,
            incorrect=incorrect,
            preferred=preferred,
            accuracy=accuracy,
        )
        return signals, result
