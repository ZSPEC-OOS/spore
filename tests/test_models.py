"""Tests for spore.models — LearningPhase, Neuron, MemoryNode."""

from datetime import datetime

import pytest

from spore.models import LearningPhase, MemoryNode, Neuron


def test_learning_phase_values():
    assert LearningPhase.INITIALIZATION.value       == "initialization"
    assert LearningPhase.GENERAL_LANGUAGE.value     == "general_language"
    assert LearningPhase.TOPIC_SPECIALIZATION.value == "topic_specialization"
    assert LearningPhase.EXPERT_MODE.value          == "expert_mode"
    assert LearningPhase.TESTING.value              == "testing"
    assert len(LearningPhase)                       == 5


def test_neuron_defaults():
    n = Neuron(id="n1", concept="hello")
    assert n.weight           == 0.1
    assert n.activation_count == 0
    assert n.connections      == []
    assert n.layer            == 0


def test_neuron_activate_increments_and_clamps():
    n = Neuron(id="n1", concept="hello")
    n.activate()
    assert n.activation_count == 1
    assert n.weight            == pytest.approx(0.15)

    # Drive weight to max
    for _ in range(100):
        n.activate()
    assert n.weight <= 1.0


def test_memory_node_fields():
    m = MemoryNode(
        content="language is communication",
        source="https://example.com",
        timestamp=datetime.utcnow(),
        confidence=0.8,
        category="definition",
        related_concepts=["syntax", "semantics"],
    )
    assert m.confidence         == 0.8
    assert "language" in m.content
    assert len(m.related_concepts) == 2
