"""Tests for NLF pipeline components: variant generator, context assembler,
slot selector, response formatter, feedback comparator."""

import pytest

from spore.nlf.models import (
    BaseQuestion,
    ContextTurn,
    FeedbackType,
    Register,
    ResponseSlot,
    SlotFunction,
    SurfaceVariant,
    TrainingInstance,
    VariantTechnique,
)
from spore.nlf.variant_generator import VariantGenerator, validate_variant, _cosine_similarity
from spore.nlf.context_assembler import ContextAssembler
from spore.nlf.slot_selector import SlotSelector
from spore.nlf.response_formatter import ResponseFormatter
from spore.nlf.feedback_comparator import FeedbackComparator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_full_question() -> BaseQuestion:
    q = BaseQuestion(id="bq_how_are_you", semantic_intent="greeting", canonical_form="How are you?")
    slot_data = [
        (1, "Good", SlotFunction.DIRECT, Register.NEUTRAL),
        (2, "Fine, thanks", SlotFunction.DIRECT, Register.NEUTRAL),
        (3, "I'm doing quite well, thank you for asking", SlotFunction.ELABORATED, Register.FORMAL),
        (4, "Very well, and yourself?", SlotFunction.ELABORATED, Register.FORMAL),
        (5, "Pretty good", SlotFunction.COLLOQUIAL, Register.CASUAL),
        (6, "Can't complain", SlotFunction.COLLOQUIAL, Register.CASUAL),
        (7, "Do you mean physically or emotionally?", SlotFunction.DEFLECTION, Register.NEUTRAL),
        (8, "That's a question I struggle to answer authentically", SlotFunction.META, Register.NEUTRAL),
        (9, "Living the dream!", SlotFunction.PLAYFUL, Register.PLAYFUL),
        (10, "I don't experience states in that way", SlotFunction.LIMITATION, Register.NEUTRAL),
    ]
    for num, text, func, reg in slot_data:
        q.add_slot(ResponseSlot(slot_number=num, text=text, function=func, register=reg))
    q.variants = [
        SurfaceVariant(text="How are you?", technique=VariantTechnique.SYNTACTIC_ALTERNATION,
                       register=Register.NEUTRAL, validated=True, similarity_score=0.0),
        SurfaceVariant(text="How's it going?", technique=VariantTechnique.SYNTACTIC_ALTERNATION,
                       register=Register.CASUAL, validated=True, similarity_score=0.3),
        SurfaceVariant(text="What's up?", technique=VariantTechnique.PRAGMATIC_MODULATION,
                       register=Register.CASUAL, validated=True, similarity_score=0.2),
    ]
    return q


# ---------------------------------------------------------------------------
# VariantGenerator
# ---------------------------------------------------------------------------

class TestVariantGenerator:
    def test_cosine_similarity_identical(self):
        assert _cosine_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_cosine_similarity_disjoint(self):
        assert _cosine_similarity("hello", "world") == pytest.approx(0.0)

    def test_validate_variant_ok(self):
        v = validate_variant(
            "How's it going?",
            existing_texts=["How are you?"],
            register=Register.CASUAL,
            technique=VariantTechnique.SYNTACTIC_ALTERNATION,
        )
        assert v.validated is True
        assert v.similarity_score < 0.85

    def test_validate_variant_duplicate_raises(self):
        with pytest.raises(ValueError, match="too similar"):
            validate_variant(
                "How are you?",
                existing_texts=["How are you?"],
                register=Register.NEUTRAL,
                technique=VariantTechnique.SYNTACTIC_ALTERNATION,
            )

    def test_from_spec_deduplication(self):
        spec = {
            "base": "How are you?",
            "variants": [
                {"text": "How are you?", "technique": "syntactic_alternation", "register": "neutral"},  # dupe
                {"text": "How's it going?", "technique": "syntactic_alternation", "register": "casual"},
            ]
        }
        variants = VariantGenerator.from_spec(spec)
        texts = [v.text for v in variants]
        # "How are you?" should appear once (as the base)
        assert texts.count("How are you?") == 1
        assert "How's it going?" in texts

    def test_generate_includes_canonical(self):
        gen = VariantGenerator()
        variants = gen.generate(
            base="How are you?",
            templates={
                VariantTechnique.PRAGMATIC_MODULATION: [("What's up?", Register.CASUAL)],
            }
        )
        assert variants[0].text == "How are you?"
        assert any(v.text == "What's up?" for v in variants)


# ---------------------------------------------------------------------------
# ContextAssembler
# ---------------------------------------------------------------------------

class TestContextAssembler:
    def test_history_capped_by_stage(self):
        assembler = ContextAssembler(stage=0)  # max 2 turns
        for i in range(5):
            assembler.add_turn("user", f"Turn {i}")
        assert len(assembler.history) == 2

    def test_assembles_instance(self):
        assembler = ContextAssembler(stage=1)
        assembler.add_turn("user", "Hello")
        instance = assembler.assemble(
            surface_variant="How are you?",
            category_id="cat1",
            base_question_id="bq1",
        )
        assert instance.surface_variant == "How are you?"
        assert len(instance.context) == 1
        assert instance.context[0].text == "Hello"

    def test_record_response_adds_to_history(self):
        assembler = ContextAssembler(stage=2)
        instance = assembler.assemble("Hi?", "cat1", "bq1")
        assembler.record_response(instance, selected_slot=1, slot_text="Good.")
        assert len(assembler.history) == 1
        assert assembler.history[0].role == "assistant"
        assert assembler.history[0].slot_number == 1

    def test_render_prompt(self):
        assembler = ContextAssembler(stage=1)
        assembler.add_turn("user", "Hey")
        prompt = assembler.render_prompt("How are you?")
        assert "User: Hey" in prompt
        assert "User: How are you?" in prompt

    def test_invalid_role_raises(self):
        assembler = ContextAssembler()
        with pytest.raises(ValueError, match="role must be"):
            assembler.add_turn("system", "oops")


# ---------------------------------------------------------------------------
# SlotSelector
# ---------------------------------------------------------------------------

class TestSlotSelector:
    def test_selects_valid_slot(self):
        question = _make_full_question()
        selector = SlotSelector(exploration_rate=0.0)
        context = []
        slot = selector.select(question, context, category_id="cat1")
        assert 1 <= slot.slot_number <= 10

    def test_no_slots_raises(self):
        question = BaseQuestion(id="bq_empty", semantic_intent="x", canonical_form="?")
        selector = SlotSelector()
        with pytest.raises(ValueError, match="no response slots"):
            selector.select(question, [])

    def test_feedback_reinforces_slot(self):
        selector = SlotSelector(exploration_rate=0.0)
        selector.update_from_instance("cat1", "bq1", selected_slot=1, feedback=FeedbackType.CORRECT)
        selector.update_from_instance("cat1", "bq1", selected_slot=1, feedback=FeedbackType.CORRECT)
        prefs = selector.get_preferences("cat1", "bq1")
        assert prefs[1] > prefs[2]  # slot 1 should have higher preference

    def test_feedback_penalises_slot(self):
        selector = SlotSelector(exploration_rate=0.0)
        initial = selector.get_preferences("cat1", "bq1")[1]
        selector.update_from_instance("cat1", "bq1", selected_slot=1, feedback=FeedbackType.INCORRECT)
        assert selector.get_preferences("cat1", "bq1")[1] < initial

    def test_context_override_bad_news(self):
        """After user shares bad news, slot 10 should be selected."""
        question = _make_full_question()
        selector = SlotSelector(exploration_rate=0.0)
        # Prime preferences so slot 10 is not naturally highest
        for _ in range(10):
            selector.update_from_instance("cat1", question.id, selected_slot=1, feedback=FeedbackType.CORRECT)
        context = [ContextTurn(role="user", text="My father just died yesterday.")]
        slot = selector.select(question, context, category_id="cat1")
        assert slot.slot_number == 10


# ---------------------------------------------------------------------------
# ResponseFormatter
# ---------------------------------------------------------------------------

class TestResponseFormatter:
    def _slot(self, text: str) -> ResponseSlot:
        return ResponseSlot(slot_number=1, text=text, function=SlotFunction.DIRECT, register=Register.NEUTRAL)

    def test_sentence_case(self):
        formatter = ResponseFormatter(vary=False)
        result = formatter.format(self._slot("good"))
        assert result[0].isupper()

    def test_adds_period(self):
        formatter = ResponseFormatter(vary=False)
        result = formatter.format(self._slot("Good"))
        assert result.endswith(".")

    def test_preserves_existing_punctuation(self):
        formatter = ResponseFormatter(vary=False)
        result = formatter.format(self._slot("Really?"))
        assert result.endswith("?")
        assert not result.endswith("?.")

    def test_format_with_slot_number(self):
        formatter = ResponseFormatter(vary=False)
        slot = ResponseSlot(slot_number=5, text="Pretty good", function=SlotFunction.COLLOQUIAL, register=Register.CASUAL)
        num, text = formatter.format_with_slot_number(slot)
        assert num == 5
        assert text.startswith("Pretty")


# ---------------------------------------------------------------------------
# FeedbackComparator
# ---------------------------------------------------------------------------

class TestFeedbackComparator:
    def _instance(self, selected: int, gold: int) -> TrainingInstance:
        return TrainingInstance(
            instance_id="i1",
            category_id="c1",
            base_question_id="bq1",
            surface_variant="Hi?",
            selected_slot=selected,
            gold_slot=gold,
        )

    def test_correct_feedback(self):
        comp = FeedbackComparator()
        inst = self._instance(selected=1, gold=1)
        sig = comp.evaluate(inst)
        assert sig.feedback == FeedbackType.CORRECT

    def test_incorrect_feedback(self):
        comp = FeedbackComparator()
        inst = self._instance(selected=2, gold=1)
        sig = comp.evaluate(inst)
        assert sig.feedback == FeedbackType.INCORRECT

    def test_tolerance(self):
        comp = FeedbackComparator()
        inst = self._instance(selected=2, gold=1)
        sig = comp.evaluate(inst, tolerance=1)
        assert sig.feedback == FeedbackType.CORRECT

    def test_no_selected_slot_raises(self):
        comp = FeedbackComparator()
        inst = TrainingInstance(instance_id="i1", category_id="c1",
                                base_question_id="bq1", surface_variant="Hi?")
        with pytest.raises(ValueError, match="no selected_slot"):
            comp.evaluate(inst)

    def test_batch_accuracy(self):
        comp = FeedbackComparator()
        instances = [
            self._instance(1, 1),
            self._instance(1, 1),
            self._instance(2, 1),
        ]
        instances[0].instance_id = "i1"
        instances[1].instance_id = "i2"
        instances[2].instance_id = "i3"
        signals, result = comp.evaluate_batch(instances)
        assert result.correct == 2
        assert result.incorrect == 1
        assert result.accuracy == pytest.approx(2 / 3)
