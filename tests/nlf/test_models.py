"""Tests for NLF framework data models."""

import pytest

from spore.nlf.models import (
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


def _make_slot(n: int) -> ResponseSlot:
    return ResponseSlot(
        slot_number=n,
        text=f"Response {n}",
        function=SlotFunction.DIRECT,
        register=Register.NEUTRAL,
    )


def _make_question() -> BaseQuestion:
    q = BaseQuestion(
        id="bq_test",
        semantic_intent="test intent",
        canonical_form="How are you?",
    )
    for i in range(1, 11):
        q.add_slot(_make_slot(i))
    return q


# ---------------------------------------------------------------------------
# ResponseSlot
# ---------------------------------------------------------------------------

class TestResponseSlot:
    def test_valid_slot(self):
        slot = _make_slot(5)
        assert slot.slot_number == 5
        assert slot.function == SlotFunction.DIRECT

    def test_invalid_slot_number_low(self):
        with pytest.raises(ValueError, match="slot_number must be 1-10"):
            ResponseSlot(slot_number=0, text="x", function=SlotFunction.DIRECT, register=Register.NEUTRAL)

    def test_invalid_slot_number_high(self):
        with pytest.raises(ValueError, match="slot_number must be 1-10"):
            ResponseSlot(slot_number=11, text="x", function=SlotFunction.DIRECT, register=Register.NEUTRAL)


# ---------------------------------------------------------------------------
# BaseQuestion
# ---------------------------------------------------------------------------

class TestBaseQuestion:
    def test_add_slot(self):
        q = BaseQuestion(id="bq1", semantic_intent="x", canonical_form="Hi?")
        q.add_slot(_make_slot(1))
        assert len(q.slots) == 1

    def test_duplicate_slot_raises(self):
        q = BaseQuestion(id="bq1", semantic_intent="x", canonical_form="Hi?")
        q.add_slot(_make_slot(1))
        with pytest.raises(ValueError, match="Slot 1 already exists"):
            q.add_slot(_make_slot(1))

    def test_slots_sorted(self):
        q = BaseQuestion(id="bq1", semantic_intent="x", canonical_form="Hi?")
        q.add_slot(_make_slot(3))
        q.add_slot(_make_slot(1))
        q.add_slot(_make_slot(2))
        assert [s.slot_number for s in q.slots] == [1, 2, 3]

    def test_get_slot(self):
        q = _make_question()
        assert q.get_slot(5).slot_number == 5
        assert q.get_slot(99) is None

    def test_add_variant_similarity_check(self):
        q = BaseQuestion(id="bq1", semantic_intent="x", canonical_form="Hi?")
        # Variant identical to canonical → similarity ≥ 0.85 → rejected
        bad_variant = SurfaceVariant(
            text="Hi?",
            technique=VariantTechnique.LEXICAL_SUBSTITUTION,
            register=Register.NEUTRAL,
            validated=True,
            similarity_score=1.0,  # identical
        )
        with pytest.raises(ValueError, match="too similar"):
            q.add_variant(bad_variant)


# ---------------------------------------------------------------------------
# Category mastery
# ---------------------------------------------------------------------------

class TestCategoryMastery:
    def test_not_mastered_by_default(self):
        cat = Category(id="c1", name="Test", description="", stage_number=0)
        assert not cat.is_mastered

    def test_mastered_when_thresholds_met(self):
        cat = Category(id="c1", name="Test", description="", stage_number=0)
        cat.accuracy = 0.90
        cat.generalization = 0.82
        cat.retention = 0.95
        assert cat.is_mastered

    def test_not_mastered_when_one_threshold_fails(self):
        cat = Category(id="c1", name="Test", description="", stage_number=0)
        cat.accuracy = 0.90
        cat.generalization = 0.75   # below 0.80
        cat.retention = 0.95
        assert not cat.is_mastered


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class TestStage:
    def test_max_history_by_stage(self):
        for num, expected in [(0, 2), (1, 2), (2, 5), (3, 5), (4, 8), (7, 10), (10, 10)]:
            stage = Stage(number=num, name="", description="")
            assert stage.max_history == expected

    def test_is_complete_when_all_categories_mastered(self):
        stage = Stage(number=0, name="", description="")
        cat = Category(id="c1", name="", description="", stage_number=0)
        cat.accuracy = 0.90
        cat.generalization = 0.82
        cat.retention = 0.95
        stage.categories.append(cat)
        assert stage.is_complete


# ---------------------------------------------------------------------------
# TrainingCycle accuracy
# ---------------------------------------------------------------------------

class TestTrainingCycle:
    def _make_instance(self, iid: str, feedback: FeedbackType) -> TrainingInstance:
        inst = TrainingInstance(
            instance_id=iid,
            category_id="c1",
            base_question_id="bq1",
            surface_variant="Hi?",
            selected_slot=1,
            gold_slot=1,
            feedback=feedback,
        )
        return inst

    def test_accuracy_all_correct(self):
        cycle = TrainingCycle(cycle_id="cyc1", category_id="c1", stage_number=0)
        cycle.instances = [
            self._make_instance("i1", FeedbackType.CORRECT),
            self._make_instance("i2", FeedbackType.CORRECT),
        ]
        assert cycle.accuracy == 1.0

    def test_accuracy_half_correct(self):
        cycle = TrainingCycle(cycle_id="cyc1", category_id="c1", stage_number=0)
        cycle.instances = [
            self._make_instance("i1", FeedbackType.CORRECT),
            self._make_instance("i2", FeedbackType.INCORRECT),
        ]
        assert cycle.accuracy == 0.5

    def test_accuracy_no_feedback(self):
        cycle = TrainingCycle(cycle_id="cyc1", category_id="c1", stage_number=0)
        inst = TrainingInstance(
            instance_id="i1", category_id="c1",
            base_question_id="bq1", surface_variant="Hi?",
        )
        cycle.instances = [inst]
        assert cycle.accuracy == 0.0
