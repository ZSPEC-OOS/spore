"""SVG neural network visualiser for SPORE.

NeuralNetworkVisualizer maintains a lightweight in-memory graph of Neurons
and their connections, and renders it as an animated SVG string.

Extension point: generate_svg() can be replaced with a canvas/WebGL
renderer once the Python prototype graduates to a richer front-end.
"""

from __future__ import annotations

import random
from typing import Dict, Set, Tuple

from .models import Neuron


class NeuralNetworkVisualizer:
    """Generates SVG views of the evolving conceptual network."""

    def __init__(self) -> None:
        self.neurons:     Dict[str, Neuron]         = {}
        self.connections: Set[Tuple[str, str]]      = set()

    def add_neuron(self, concept: str, layer: int = 1) -> Neuron:
        neuron_id = f"{concept}_{len(self.neurons)}"
        radius    = 100 + (layer * 80) + random.uniform(-30, 30)
        neuron    = Neuron(
            id      = neuron_id,
            concept = concept,
            layer   = layer,
            x       = 400 + radius * (random.random() - 0.5) * 2,
            y       = 300 + radius * (random.random() - 0.5) * 2,
        )
        self.neurons[neuron_id] = neuron
        return neuron

    def connect(self, n1_id: str, n2_id: str) -> None:
        if n1_id == n2_id:
            return
        if n1_id in self.neurons and n2_id in self.neurons:
            edge = tuple(sorted((n1_id, n2_id)))
            if edge not in self.connections:
                self.connections.add(edge)
                self.neurons[n1_id].connections.append(n2_id)
                self.neurons[n2_id].connections.append(n1_id)

    def generate_svg(self) -> str:
        svg_parts = [
            '<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">',
            "<defs>",
            '  <radialGradient id="nodeGradient" cx="50%" cy="50%">',
            '    <stop offset="0%"   style="stop-color:#00f2ff"/>',
            '    <stop offset="100%" style="stop-color:#0066ff"/>',
            "  </radialGradient>",
            '  <filter id="glow">',
            '    <feGaussianBlur stdDeviation="3" result="coloredBlur"/>',
            "    <feMerge>",
            '      <feMergeNode in="coloredBlur"/>',
            '      <feMergeNode in="SourceGraphic"/>',
            "    </feMerge>",
            "  </filter>",
            "</defs>",
            '<rect width="800" height="600" fill="#0a0a1a"/>',
        ]

        for n1, n2 in self.connections:
            a       = self.neurons[n1]
            b       = self.neurons[n2]
            opacity = min(0.6, (a.weight + b.weight) / 2)
            svg_parts.append(
                f'<line x1="{a.x}" y1="{a.y}" x2="{b.x}" y2="{b.y}" '
                f'stroke="#00f2ff" stroke-width="{opacity * 2}" opacity="{opacity}">'
                '<animate attributeName="stroke-dasharray" '
                'values="0,20;20,0;0,20" dur="3s" repeatCount="indefinite"/>'
                "</line>"
            )

        for neuron in self.neurons.values():
            size           = 8 + (neuron.weight * 12) + (neuron.activation_count * 0.5)
            glow_intensity = neuron.weight
            label          = neuron.concept[:12]
            svg_parts.append(
                f'<circle cx="{neuron.x}" cy="{neuron.y}" r="{size}" '
                f'fill="url(#nodeGradient)" filter="url(#glow)" '
                f'opacity="{0.6 + glow_intensity * 0.4}">'
                f'<animate attributeName="r" values="{size};{size * 1.2};{size}" '
                f'dur="2s" repeatCount="indefinite"/>'
                f"</circle>"
                f'<text x="{neuron.x}" y="{neuron.y + 4}" text-anchor="middle" '
                f'fill="white" font-size="8" font-family="monospace">{label}</text>'
            )

        svg_parts.append(
            f'<text x="20" y="30" fill="#00f2ff" font-family="monospace" font-size="14">'
            f'Neurons: {len(self.neurons)} | Connections: {len(self.connections)}</text>'
        )
        svg_parts.append("</svg>")
        return "\n".join(svg_parts)
