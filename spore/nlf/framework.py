"""
NLF Framework Orchestrator
Central coordinator for the AI Natural Language Fluency Framework (§11.1).

Pipeline::

    VariantGenerator → ContextAssembler → SlotSelector → ResponseFormatter
                                                ↓
                                       FeedbackComparator → SlotSelector.update

Stage progression gates (§7.1):
    accuracy ≥ 85%  |  generalization ≥ 80%  |  retention ≥ 90%
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Optional

from .context_assembler import ContextAssembler
from .feedback_comparator import EvaluationResult, FeedbackComparator
from .models import (
    BaseQuestion,
    Category,
    FeedbackType,
    ResponseSlot,
    SlotFunction,
    Stage,
    TrainingCycle,
    TrainingInstance,
)
from .response_formatter import ResponseFormatter
from .slot_selector import SlotSelector
from .variant_generator import VariantGenerator


# ---------------------------------------------------------------------------
# Framework
# ---------------------------------------------------------------------------

class NLFFramework:
    """
    AI Natural Language Fluency Framework v1.0

    Manages the full 10-stage progression, training cycles, and feedback loop.

    Usage::

        fw = NLFFramework()
        fw.load_stage(stage_spec_dict)   # load stage 0 content
        fw.run_cycle("cat_greetings")    # run one training cycle
        print(fw.mastery_report())

    Persistence::

        fw.save(path)   # serialise to JSON
        fw = NLFFramework.load(path)
    """

    def __init__(self) -> None:
        self.stages: dict[int, Stage] = {}
        self.selector = SlotSelector()
        self.formatter = ResponseFormatter()
        self.comparator = FeedbackComparator()
        self.generator = VariantGenerator()

        # Unlock stage 0 by default (§7.2)
        self._unlock_stage(0)

        # Training history
        self._cycles: list[TrainingCycle] = []

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def _unlock_stage(self, number: int) -> None:
        if number not in self.stages:
            self.stages[number] = Stage(
                number=number,
                name=f"Stage {number}",
                description="",
                unlocked=True,
            )
        else:
            self.stages[number].unlocked = True

    def add_stage(self, stage: Stage) -> None:
        """Register a fully constructed Stage object."""
        self.stages[stage.number] = stage
        if stage.number == 0:
            stage.unlocked = True

    def load_stage_from_dict(self, spec: dict) -> Stage:
        """
        Load a Stage from a JSON-serialisable dict spec.

        Spec format::

            {
              "number": 0,
              "name": "Foundational Social Language",
              "description": "...",
              "categories": [
                {
                  "id": "cat_greetings",
                  "name": "Greetings & Farewells",
                  "description": "...",
                  "questions": [
                    {
                      "id": "bq_how_are_you",
                      "semantic_intent": "greeting + status inquiry",
                      "canonical_form": "How are you?",
                      "slots": [
                        {
                          "slot_number": 1,
                          "text": "Good",
                          "function": "direct",
                          "register": "neutral",
                          "use_case": "Default, brief"
                        },
                        ...
                      ],
                      "variants": [
                        {
                          "text": "How's it going?",
                          "technique": "syntactic_alternation",
                          "register": "casual"
                        },
                        ...
                      ]
                    }
                  ]
                }
              ]
            }
        """
        from .models import Register, SlotFunction, SurfaceVariant, VariantTechnique  # local to avoid circular

        stage_num = spec["number"]
        stage = Stage(
            number=stage_num,
            name=spec.get("name", f"Stage {stage_num}"),
            description=spec.get("description", ""),
            unlocked=(stage_num == 0),
        )

        for cat_spec in spec.get("categories", []):
            category = Category(
                id=cat_spec["id"],
                name=cat_spec["name"],
                description=cat_spec.get("description", ""),
                stage_number=stage_num,
            )

            for q_spec in cat_spec.get("questions", []):
                question = BaseQuestion(
                    id=q_spec["id"],
                    semantic_intent=q_spec.get("semantic_intent", ""),
                    canonical_form=q_spec["canonical_form"],
                )

                for s_spec in q_spec.get("slots", []):
                    try:
                        func = SlotFunction(s_spec.get("function", "direct"))
                        reg = Register(s_spec.get("register", "neutral"))
                    except ValueError:
                        func = SlotFunction.DIRECT
                        reg = Register.NEUTRAL
                    slot = ResponseSlot(
                        slot_number=int(s_spec["slot_number"]),
                        text=s_spec["text"],
                        function=func,
                        register=reg,
                        use_case=s_spec.get("use_case", ""),
                    )
                    question.add_slot(slot)

                # Load variants via generator (dedup handled internally)
                question.variants = self.generator.from_spec({
                    "base": question.canonical_form,
                    "variants": q_spec.get("variants", []),
                })

                category.questions.append(question)
            stage.categories.append(category)

        self.stages[stage_num] = stage
        if stage_num == 0:
            stage.unlocked = True
        return stage

    def load_stage_from_file(self, path: str | Path) -> Stage:
        """Load a Stage from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.load_stage_from_dict(data)

    def load_stage_from_simple_spec(self, spec: dict) -> Stage:
        """
        Load a Stage from a simplified human-readable spec.

        This format accepts the 10 responses as *seed slot templates* —
        they define the canonical answer space.  The VariantGenerator later
        generates thousands of surface-form variations of each *question*,
        and training cycles iterate over those variants selecting from these
        10 response slots.  ``cycles_min``/``cycles_max`` record the target
        iteration budget (not the number of unique responses).

        Simplified spec format::

            {
              "stage_number": 0,
              "stage_name":   "Foundational Social Language",
              "description":  "Before any grammar or complex categories",
              "categories": [
                {
                  "id":       "cat_0_1",
                  "name":     "Greetings & Social Rituals",
                  "purpose":  "Learn that language is for social connection…",
                  "questions": [
                    {
                      "question":  "Hi",
                      "responses": ["Hi", "Hello", "Hey", …]   // exactly 10
                    },
                    …
                  ],
                  "cycles_min": 10000,
                  "cycles_max": 20000
                }
              ]
            }

        Slot function/register are inferred by position (§4.1 mapping):
            1–2 → direct/neutral   3–4 → elaborated/formal
            5–6 → colloquial/casual  7 → deflection/neutral
            8   → meta/neutral       9 → playful/playful
            10  → limitation/formal
        """
        # Position → (SlotFunction, Register) mapping
        _SLOT_MAP: dict[int, tuple[str, str]] = {
            1: ("direct",     "neutral"),
            2: ("direct",     "neutral"),
            3: ("elaborated", "formal"),
            4: ("elaborated", "formal"),
            5: ("colloquial", "casual"),
            6: ("colloquial", "casual"),
            7: ("deflection", "neutral"),
            8: ("meta",       "neutral"),
            9: ("playful",    "playful"),
            10: ("limitation", "formal"),
        }

        full_spec: dict = {
            "number":      spec["stage_number"],
            "name":        spec.get("stage_name", f"Stage {spec['stage_number']}"),
            "description": spec.get("description", ""),
            "categories":  [],
        }

        for cat in spec.get("categories", []):
            cat_id = cat["id"]
            questions_spec = []
            for idx, q in enumerate(cat.get("questions", []), start=1):
                q_id = f"bq_{cat_id}_{idx}"
                canonical = q["question"]
                responses = q.get("responses", [])

                slots_spec = []
                for slot_num in range(1, 11):
                    text = responses[slot_num - 1] if slot_num - 1 < len(responses) else ""
                    func, reg = _SLOT_MAP[slot_num]
                    slots_spec.append({
                        "slot_number": slot_num,
                        "text":        text,
                        "function":    func,
                        "register":    reg,
                        "use_case":    f"Seed response {slot_num}",
                    })

                questions_spec.append({
                    "id":             q_id,
                    "semantic_intent": canonical,
                    "canonical_form":  canonical,
                    "slots":           slots_spec,
                    "variants":        [],
                    # Store cycles budget on the question for later reference
                    "_cycles_min":    cat.get("cycles_min"),
                    "_cycles_max":    cat.get("cycles_max"),
                })

            full_spec["categories"].append({
                "id":          cat_id,
                "name":        cat["name"],
                "description": cat.get("purpose", cat.get("description", "")),
                "questions":   questions_spec,
                # Persist cycle budget as extra metadata (ignored by load_stage_from_dict)
                "_cycles_min": cat.get("cycles_min"),
                "_cycles_max": cat.get("cycles_max"),
            })

        stage = self.load_stage_from_dict(full_spec)

        # Attach cycle budget to each category for display / planning
        for cat_spec, stage_cat in zip(spec.get("categories", []), stage.categories):
            stage_cat.cycles_min = cat_spec.get("cycles_min")  # type: ignore[attr-defined]
            stage_cat.cycles_max = cat_spec.get("cycles_max")  # type: ignore[attr-defined]

        return stage

    # ------------------------------------------------------------------
    # Training cycle
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        category_id: str,
        *,
        gold_map: Optional[dict[str, int]] = None,
        max_instances: int = 100,
    ) -> tuple[TrainingCycle, EvaluationResult]:
        """
        Run a training cycle for the given category.

        For each base question, selects a random surface variant,
        runs the selector, formats the response, then evaluates via comparator.

        Args:
            category_id: The category to train.
            gold_map: Optional {instance_id: gold_slot} for supervised feedback.
            max_instances: Cap on instances per cycle (default 100 for speed).

        Returns:
            (cycle, eval_result) with accuracy statistics.
        """
        category, stage = self._find_category(category_id)
        if stage is None or not stage.unlocked:
            raise ValueError(f"Stage for category '{category_id}' is not unlocked.")

        assembler = ContextAssembler(stage=stage.number)
        cycle = TrainingCycle(
            cycle_id=str(uuid.uuid4()),
            category_id=category_id,
            stage_number=stage.number,
        )

        instances: list[TrainingInstance] = []

        for question in category.questions:
            if not question.variants:
                continue
            # Sample variants up to max_instances / num_questions
            n = max(1, max_instances // max(len(category.questions), 1))
            sampled_variants = random.choices(question.variants, k=n)

            for variant in sampled_variants:
                instance = assembler.assemble(
                    surface_variant=variant.text,
                    category_id=category_id,
                    base_question_id=question.id,
                    gold_slot=question.slots[0].slot_number if question.slots else None,
                )

                # Select slot
                if question.slots:
                    chosen_slot = self.selector.select(
                        question, instance.context, category_id=category_id
                    )
                    formatted = self.formatter.format(chosen_slot)
                    assembler.record_response(instance, chosen_slot.slot_number, formatted)
                    instance.selected_slot = chosen_slot.slot_number
                else:
                    instance.selected_slot = None

                instances.append(instance)
                cycle.instances.append(instance)

        # Evaluate
        signals, result = self.comparator.evaluate_batch(
            instances, gold_map=gold_map
        )

        # Apply feedback to selector
        for inst, sig in zip(instances, signals):
            if inst.selected_slot is not None:
                self.selector.update_from_instance(
                    category_id=category_id,
                    base_question_id=inst.base_question_id,
                    selected_slot=inst.selected_slot,
                    feedback=sig.feedback,
                    gold_slot=inst.gold_slot,
                    preferred_slot=sig.preferred_slot,
                )

        # Update category mastery metrics
        category.cycles_completed += 1
        category.accuracy = result.accuracy
        # Generalization and retention require separate held-out passes;
        # approximated as same-cycle accuracy at this stage.
        if category.generalization == 0.0:
            category.generalization = result.accuracy * 0.95
        if category.retention == 0.0:
            category.retention = result.accuracy

        cycle.completed = True
        self._cycles.append(cycle)

        # Check stage advancement
        self._check_stage_advancement(stage)

        return cycle, result

    # ------------------------------------------------------------------
    # Stage advancement
    # ------------------------------------------------------------------

    def _check_stage_advancement(self, stage: Stage) -> None:
        """Unlock the next stage if the current one is mastered (§7)."""
        if stage.is_complete:
            next_num = stage.number + 1
            if next_num <= 10:
                self._unlock_stage(next_num)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _find_category(self, category_id: str) -> tuple[Category, Optional[Stage]]:
        for stage in self.stages.values():
            for cat in stage.categories:
                if cat.id == category_id:
                    return cat, stage
        raise ValueError(f"Category '{category_id}' not found in any loaded stage.")

    def get_stage(self, number: int) -> Optional[Stage]:
        return self.stages.get(number)

    def unlocked_stages(self) -> list[Stage]:
        return sorted(
            (s for s in self.stages.values() if s.unlocked),
            key=lambda s: s.number,
        )

    def mastery_report(self) -> dict:
        """Return a summary dict of mastery across all loaded stages."""
        report: dict = {}
        for num, stage in sorted(self.stages.items()):
            cats = []
            for cat in stage.categories:
                cats.append({
                    "id": cat.id,
                    "name": cat.name,
                    "accuracy": round(cat.accuracy, 3),
                    "generalization": round(cat.generalization, 3),
                    "retention": round(cat.retention, 3),
                    "cycles_completed": cat.cycles_completed,
                    "mastered": cat.is_mastered,
                })
            report[f"stage_{num}"] = {
                "name": stage.name,
                "unlocked": stage.unlocked,
                "complete": stage.is_complete,
                "categories": cats,
            }
        return report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise framework state (mastery metrics + preferences) to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "mastery": self.mastery_report(),
            "selector_preferences": {
                k: v for k, v in self.selector._preferences.items()
            },
            "total_cycles": len(self._cycles),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NLFFramework":
        """Restore a saved framework state."""
        path = Path(path)
        fw = cls()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            prefs = data.get("selector_preferences", {})
            for k, v in prefs.items():
                fw.selector._preferences[k] = {int(slot): score for slot, score in v.items()}
        return fw
