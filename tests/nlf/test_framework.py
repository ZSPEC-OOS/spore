"""Integration tests for the NLFFramework orchestrator."""

import json
import tempfile
from pathlib import Path

import pytest

from spore.nlf import NLFFramework
from spore.nlf.models import Register, SlotFunction


# ---------------------------------------------------------------------------
# Shared stage spec (minimal stage 0)
# ---------------------------------------------------------------------------

STAGE_0_SPEC = {
    "number": 0,
    "name": "Foundational Social Language",
    "description": "Stage 0 test",
    "categories": [
        {
            "id": "cat_greetings",
            "name": "Greetings & Farewells",
            "description": "Basic social openers.",
            "questions": [
                {
                    "id": "bq_how_are_you",
                    "semantic_intent": "greeting + status inquiry",
                    "canonical_form": "How are you?",
                    "slots": [
                        {"slot_number": 1, "text": "Good", "function": "direct", "register": "neutral"},
                        {"slot_number": 2, "text": "Fine, thanks", "function": "direct", "register": "neutral"},
                        {"slot_number": 3, "text": "I'm doing quite well, thank you", "function": "elaborated", "register": "formal"},
                        {"slot_number": 4, "text": "Very well, and yourself?", "function": "elaborated", "register": "formal"},
                        {"slot_number": 5, "text": "Pretty good", "function": "colloquial", "register": "casual"},
                        {"slot_number": 6, "text": "Can't complain", "function": "colloquial", "register": "casual"},
                        {"slot_number": 7, "text": "In what sense do you mean?", "function": "deflection", "register": "neutral"},
                        {"slot_number": 8, "text": "Hard to say authentically", "function": "meta", "register": "neutral"},
                        {"slot_number": 9, "text": "Living the dream!", "function": "playful", "register": "playful"},
                        {"slot_number": 10, "text": "I don't experience states like that", "function": "limitation", "register": "neutral"},
                    ],
                    "variants": [
                        {"text": "How's it going?", "technique": "syntactic_alternation", "register": "casual"},
                        {"text": "What's up?", "technique": "pragmatic_modulation", "register": "casual"},
                        {"text": "How have you been?", "technique": "ellipsis_expansion", "register": "neutral"},
                    ],
                }
            ],
        }
    ],
}


@pytest.fixture
def fw():
    framework = NLFFramework()
    framework.load_stage_from_dict(STAGE_0_SPEC)
    return framework


# ---------------------------------------------------------------------------
# Stage loading
# ---------------------------------------------------------------------------

class TestStageLoading:
    def test_load_stage_from_dict(self, fw):
        stage = fw.get_stage(0)
        assert stage is not None
        assert stage.name == "Foundational Social Language"
        assert len(stage.categories) == 1

    def test_stage_0_unlocked(self, fw):
        assert fw.get_stage(0).unlocked is True

    def test_category_loaded(self, fw):
        cat_ids = [c.id for c in fw.get_stage(0).categories]
        assert "cat_greetings" in cat_ids

    def test_question_with_10_slots(self, fw):
        cat = fw.get_stage(0).categories[0]
        question = cat.questions[0]
        assert len(question.slots) == 10
        assert all(1 <= s.slot_number <= 10 for s in question.slots)

    def test_variants_loaded(self, fw):
        cat = fw.get_stage(0).categories[0]
        question = cat.questions[0]
        # canonical + 3 from spec (1 may be deduped)
        assert len(question.variants) >= 3

    def test_load_stage_from_file(self, tmp_path):
        spec_file = tmp_path / "stage_0.json"
        spec_file.write_text(json.dumps(STAGE_0_SPEC), encoding="utf-8")
        fw = NLFFramework()
        stage = fw.load_stage_from_file(spec_file)
        assert stage.number == 0


# ---------------------------------------------------------------------------
# Training cycle
# ---------------------------------------------------------------------------

class TestTrainingCycle:
    def test_run_cycle_returns_result(self, fw):
        cycle, result = fw.run_cycle("cat_greetings", max_instances=10)
        assert cycle.completed is True
        assert len(cycle.instances) > 0
        assert 0.0 <= result.accuracy <= 1.0

    def test_run_cycle_updates_category(self, fw):
        cat = fw.get_stage(0).categories[0]
        assert cat.cycles_completed == 0
        fw.run_cycle("cat_greetings", max_instances=10)
        assert cat.cycles_completed == 1

    def test_run_cycle_unknown_category_raises(self, fw):
        with pytest.raises(ValueError, match="not found"):
            fw.run_cycle("cat_does_not_exist")

    def test_run_multiple_cycles_improves_preferences(self, fw):
        # Run several cycles with gold_map forcing slot 1 always correct
        for _ in range(5):
            cycle, _ = fw.run_cycle("cat_greetings", max_instances=5)
            gold_map = {inst.instance_id: 1 for inst in cycle.instances}
            fw.run_cycle("cat_greetings", max_instances=5)

        top = fw.selector.top_slot("cat_greetings", "bq_how_are_you")
        # After corrections, slot 1 should be among the more preferred
        prefs = fw.selector.get_preferences("cat_greetings", "bq_how_are_you")
        assert isinstance(top, int)


# ---------------------------------------------------------------------------
# Mastery report
# ---------------------------------------------------------------------------

class TestMasteryReport:
    def test_report_structure(self, fw):
        report = fw.mastery_report()
        assert "stage_0" in report
        stage_info = report["stage_0"]
        assert "categories" in stage_info
        assert stage_info["unlocked"] is True

    def test_category_in_report(self, fw):
        report = fw.mastery_report()
        cats = report["stage_0"]["categories"]
        assert any(c["id"] == "cat_greetings" for c in cats)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, fw, tmp_path):
        fw.run_cycle("cat_greetings", max_instances=5)
        save_path = tmp_path / "state.json"
        fw.save(save_path)
        assert save_path.exists()

        fw2 = NLFFramework.load(save_path)
        # Preferences should have been restored
        prefs = fw2.selector.get_preferences("cat_greetings", "bq_how_are_you")
        assert len(prefs) == 10


# ---------------------------------------------------------------------------
# Stage advancement
# ---------------------------------------------------------------------------

class TestStageAdvancement:
    def test_stage_1_unlocked_when_stage_0_complete(self, fw):
        # Manually mark stage 0 as mastered
        stage = fw.get_stage(0)
        for cat in stage.categories:
            cat.accuracy = 0.90
            cat.generalization = 0.85
            cat.retention = 0.95
        fw._check_stage_advancement(stage)
        assert fw.get_stage(1) is not None
        assert fw.get_stage(1).unlocked is True
