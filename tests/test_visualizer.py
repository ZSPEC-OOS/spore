"""Tests for spore.visualizer — NeuralNetworkVisualizer."""

from spore.visualizer import NeuralNetworkVisualizer


def test_add_neuron_returns_neuron_with_correct_concept():
    viz    = NeuralNetworkVisualizer()
    neuron = viz.add_neuron("Language", layer=0)
    assert neuron.concept == "Language"
    assert neuron.layer   == 0
    assert neuron.id in viz.neurons


def test_connect_creates_bidirectional_edge():
    viz = NeuralNetworkVisualizer()
    a   = viz.add_neuron("A")
    b   = viz.add_neuron("B")
    viz.connect(a.id, b.id)
    assert len(viz.connections) == 1
    assert b.id in a.connections
    assert a.id in b.connections


def test_connect_self_is_no_op():
    viz = NeuralNetworkVisualizer()
    a   = viz.add_neuron("A")
    viz.connect(a.id, a.id)
    assert len(viz.connections) == 0


def test_connect_duplicate_is_no_op():
    viz = NeuralNetworkVisualizer()
    a   = viz.add_neuron("A")
    b   = viz.add_neuron("B")
    viz.connect(a.id, b.id)
    viz.connect(a.id, b.id)
    assert len(viz.connections) == 1


def test_generate_svg_contains_neuron_labels():
    viz = NeuralNetworkVisualizer()
    viz.add_neuron("Syntax")
    svg = viz.generate_svg()
    assert "<svg" in svg
    assert "Syntax" in svg
    assert "Neurons: 1" in svg
