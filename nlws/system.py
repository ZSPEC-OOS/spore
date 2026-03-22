"""Neural Linguistic Web-Learning System (NLWS).

Assembled for SPORE as a runnable prototype with:
- phase-based learning orchestration
- pluggable search provider
- memory graph growth and activation
- SVG visualization output
- interactive CLI loop
"""

from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class LearningPhase(Enum):
    INITIALIZATION = "initialization"
    GENERAL_LANGUAGE = "general_language"
    TOPIC_SPECIALIZATION = "topic_specialization"
    EXPERT_MODE = "expert_mode"
    TESTING = "testing"


@dataclass
class Neuron:
    id: str
    concept: str
    weight: float = 0.1
    connections: List[str] = field(default_factory=list)
    activation_count: int = 0
    layer: int = 0
    x: float = 0.0
    y: float = 0.0

    def activate(self) -> None:
        self.activation_count += 1
        self.weight = min(1.0, self.weight + 0.05)


@dataclass
class MemoryNode:
    content: str
    source: str
    timestamp: datetime
    confidence: float
    category: str
    related_concepts: List[str] = field(default_factory=list)


class NeuralNetworkVisualizer:
    """Generates SVG views of the evolving conceptual network."""

    def __init__(self) -> None:
        self.neurons: Dict[str, Neuron] = {}
        self.connections: Set[Tuple[str, str]] = set()

    def add_neuron(self, concept: str, layer: int = 1) -> Neuron:
        neuron_id = f"{concept}_{len(self.neurons)}"
        radius = 100 + (layer * 80) + random.uniform(-30, 30)
        neuron = Neuron(
            id=neuron_id,
            concept=concept,
            layer=layer,
            x=400 + radius * (random.random() - 0.5) * 2,
            y=300 + radius * (random.random() - 0.5) * 2,
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
            '    <stop offset="0%" style="stop-color:#00f2ff"/>',
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
            a = self.neurons[n1]
            b = self.neurons[n2]
            opacity = min(0.6, (a.weight + b.weight) / 2)
            svg_parts.append(
                f'<line x1="{a.x}" y1="{a.y}" x2="{b.x}" y2="{b.y}" '
                f'stroke="#00f2ff" stroke-width="{opacity * 2}" opacity="{opacity}">'
                '<animate attributeName="stroke-dasharray" values="0,20;20,0;0,20" dur="3s" repeatCount="indefinite"/>'
                "</line>"
            )

        for neuron in self.neurons.values():
            size = 8 + (neuron.weight * 12) + (neuron.activation_count * 0.5)
            glow_intensity = neuron.weight
            label = neuron.concept[:12]
            svg_parts.append(
                f'<circle cx="{neuron.x}" cy="{neuron.y}" r="{size}" '
                f'fill="url(#nodeGradient)" filter="url(#glow)" opacity="{0.6 + glow_intensity * 0.4}">'
                f'<animate attributeName="r" values="{size};{size * 1.2};{size}" dur="2s" repeatCount="indefinite"/>'
                "</circle>"
                f'<text x="{neuron.x}" y="{neuron.y + 4}" text-anchor="middle" '
                f'fill="white" font-size="8" font-family="monospace">{label}</text>'
            )

        svg_parts.append(
            f'<text x="20" y="30" fill="#00f2ff" font-family="monospace" font-size="14">'
            f'Neurons: {len(self.neurons)} | Connections: {len(self.connections)}</text>'
        )
        svg_parts.append("</svg>")
        return "\n".join(svg_parts)


class SearchProvider:
    """Search provider abstraction.

    Replace this class with SPORE-integrated search/crawling adapters.
    """

    async def search(self, query: str) -> List[Dict[str, str]]:
        # Mocked results to keep the prototype runnable offline.
        slug = re.sub(r"\s+", "-", query.strip().lower())
        return [
            {
                "title": f"Overview: {query}",
                "url": f"https://example.org/{slug}/overview",
                "snippet": f"{query} refers to a practical framework with definitions, examples, and tradeoffs.",
            },
            {
                "title": f"Guide: {query}",
                "url": f"https://example.org/{slug}/guide",
                "snippet": f"How to apply {query} with step-by-step methods and evaluation criteria.",
            },
            {
                "title": f"Research: {query}",
                "url": f"https://example.org/{slug}/research",
                "snippet": f"A study analyzing {query}, benchmarks, and limitations.",
            },
        ]


class DuckDuckGoCrawler:
    """Crawler pipeline with pluggable search provider."""

    def __init__(self, search_provider: Optional[SearchProvider] = None):
        self.provider = search_provider or SearchProvider()
        self.visited_urls: Set[str] = set()
        self.knowledge_base: List[MemoryNode] = []

    async def search_and_learn(self, query: str, context: str = "") -> List[MemoryNode]:
        search_results = await self.provider.search(query)
        learned_nodes: List[MemoryNode] = []

        for result in search_results:
            if result["url"] in self.visited_urls:
                continue

            content = await self._extract_content(result)
            node = MemoryNode(
                content=content,
                source=result["url"],
                timestamp=datetime.utcnow(),
                confidence=self._calculate_confidence(content, context),
                category=self._categorize(content),
                related_concepts=self._extract_concepts(content),
            )
            self.visited_urls.add(result["url"])
            self.knowledge_base.append(node)
            learned_nodes.append(node)

        return learned_nodes

    async def _extract_content(self, result: Dict[str, str]) -> str:
        return f"{result['title']}. {result['snippet']}"

    def _calculate_confidence(self, content: str, context: str) -> float:
        score = 0.5
        if context and context.lower() in content.lower():
            score += 0.3
        if any(marker in content.lower() for marker in ["research", "study", "paper", "analysis"]):
            score += 0.2
        return min(1.0, score)

    def _categorize(self, content: str) -> str:
        categories = {
            "definition": ["defined", "refers to", "means"],
            "tutorial": ["how to", "guide", "steps", "tutorial"],
            "research": ["study", "paper", "research", "analysis"],
            "discussion": ["debate", "opinion", "argue"],
        }
        lowered = content.lower()
        for category, markers in categories.items():
            if any(marker in lowered for marker in markers):
                return category
        return "general"

    def _extract_concepts(self, content: str) -> List[str]:
        words = re.findall(r"\b[A-Z][a-z]{3,}\b", content)
        return list(dict.fromkeys(words))[:10]


class LanguageLearningEngine:
    def __init__(self):
        self.phase = LearningPhase.INITIALIZATION
        self.visualizer = NeuralNetworkVisualizer()
        self.crawler = DuckDuckGoCrawler()
        self.memory: List[MemoryNode] = []
        self.language_patterns: Dict[str, int] = {}
        self.topic: Optional[str] = None
        self.is_learning = False
        self._initialize_core_neurons()

    def _initialize_core_neurons(self) -> None:
        root = self.visualizer.add_neuron("Language", layer=0)
        for concept in ["Syntax", "Semantics", "Context", "Intent", "Tone"]:
            neuron = self.visualizer.add_neuron(concept, layer=1)
            self.visualizer.connect(root.id, neuron.id)

    async def start_general_language_learning(self) -> None:
        self.phase = LearningPhase.GENERAL_LANGUAGE
        self.is_learning = True

        queries = [
            "common conversation patterns dialogue examples",
            "how to understand user intent in questions",
            "polite conversation etiquette online",
            "context clues in language understanding",
            "tone detection in text communication",
        ]

        for query in queries:
            if not self.is_learning:
                break
            print(f"🔍 Learning: {query}")
            nodes = await self.crawler.search_and_learn(query)
            for node in nodes:
                await self._integrate_knowledge(node)
                self._grow_neural_network(node)
            await asyncio.sleep(0.1)

        self.phase = LearningPhase.TESTING
        print("✅ General language learning complete. Ready for testing or topic specialization.")

    async def specialize_topic(self, topic: str) -> None:
        self.topic = topic
        self.phase = LearningPhase.TOPIC_SPECIALIZATION
        self.is_learning = True

        topic_neuron = self.visualizer.add_neuron(topic, layer=2)
        strategies = [
            f"{topic} fundamentals basics introduction",
            f"{topic} advanced concepts expert guide",
            f"{topic} research papers latest studies",
            f"{topic} common questions FAQ",
            f"{topic} practical applications examples",
            f"{topic} criticism debates controversies",
        ]

        expertise_score = 0.0
        iteration = 0
        max_iterations = 20

        while self.is_learning and expertise_score < 0.95 and iteration < max_iterations:
            query = strategies[iteration % len(strategies)]
            print(f"🧠 Deep learning [{topic}]: {query} | Expertise: {expertise_score:.2%}")
            nodes = await self.crawler.search_and_learn(query, context=topic)

            for node in nodes:
                await self._integrate_knowledge(node)
                new_neuron = self._grow_neural_network(node)
                self.visualizer.connect(topic_neuron.id, new_neuron.id)
                expertise_score = self._calculate_expertise(topic)

            iteration += 1
            if iteration % 5 == 0:
                self._render_network()
            await asyncio.sleep(0.1)

        self.phase = LearningPhase.EXPERT_MODE
        print(f"🎯 Expertise achieved in {topic}. Network size: {len(self.visualizer.neurons)} neurons.")

    async def _integrate_knowledge(self, node: MemoryNode) -> None:
        self.memory.append(node)
        sentences = re.split(r"[.!?]+", node.content)
        for sentence in sentences:
            pattern = self._normalize_pattern(sentence.strip())
            if pattern:
                self.language_patterns[pattern] = self.language_patterns.get(pattern, 0) + 1

    def _normalize_pattern(self, text: str) -> Optional[str]:
        if len(text) < 10:
            return None
        text = re.sub(r"\b\d+\b", "[NUM]", text)
        text = re.sub(r"\b[A-Z][a-z]+\b", "[TERM]", text)
        return text.lower().strip()

    def _grow_neural_network(self, node: MemoryNode) -> Neuron:
        candidates = node.related_concepts[:3]
        existing_concepts = {n.concept for n in self.visualizer.neurons.values()}
        created: List[Neuron] = []

        for concept in candidates:
            if concept not in existing_concepts:
                created.append(self.visualizer.add_neuron(concept, layer=2 + random.randint(0, 2)))

        for i, n1 in enumerate(created):
            for n2 in created[i + 1 :]:
                self.visualizer.connect(n1.id, n2.id)

        return created[0] if created else next(iter(self.visualizer.neurons.values()))

    def _calculate_expertise(self, topic: str) -> float:
        topic_memories = [m for m in self.memory if topic.lower() in m.content.lower()]
        if not topic_memories:
            return 0.0
        volume = min(1.0, len(topic_memories) / 100)
        diversity = min(1.0, len({m.source for m in topic_memories}) / 20)
        avg_confidence = sum(m.confidence for m in topic_memories) / len(topic_memories)
        return volume * 0.3 + diversity * 0.3 + avg_confidence * 0.4

    def _render_network(self) -> None:
        _ = self.visualizer.generate_svg()
        print(f"📊 Network visualization updated: {len(self.visualizer.neurons)} neurons")

    def stop_learning(self) -> None:
        self.is_learning = False
        print("⏸️ Learning paused. Entering testing mode.")

    def answer_question(self, question: str) -> str:
        for neuron in self.visualizer.neurons.values():
            if any(token in question.lower() for token in neuron.concept.lower().split()):
                neuron.activate()

        relevant: List[Tuple[MemoryNode, float]] = []
        for memory in self.memory:
            score = self._relevance_score(question, memory)
            if score > 0.35:
                relevant.append((memory, score))
        relevant.sort(key=lambda x: x[1], reverse=True)

        if not relevant:
            return f"I'm still learning about {self.topic or 'that topic'}. Could you provide more context?"

        context = "\n".join(m.content[:180] for m, _ in relevant[:3])
        return self._synthesize_response(question, context)

    def _relevance_score(self, question: str, memory: MemoryNode) -> float:
        q_words = set(question.lower().split())
        m_words = set(memory.content.lower().split())
        overlap = len(q_words & m_words)
        return overlap / max(len(q_words), 1) * memory.confidence

    def _synthesize_response(self, question: str, context: str) -> str:
        q = question.lower()
        if "what is" in q:
            return f"Based on learned sources: {context[:300]}..."
        if "how" in q:
            return f"Here's the process based on learned material: {context[:300]}..."
        return f"From what I've learned so far: {context[:300]}..."


class LearningInterface:
    def __init__(self):
        self.engine = LanguageLearningEngine()
        self.command_history: List[str] = []

    async def run(self) -> None:
        print(
            """
🧠 NEURAL LINGUISTIC WEB-LEARNING SYSTEM
=======================================
Commands:
  learn general    - Start general language acquisition
  learn <topic>    - Specialize in specific topic
  stop             - Pause learning
  ask <question>   - Test knowledge
  visualize        - Print SVG preview
  status           - Check learning progress
  exit             - Shutdown system
"""
        )

        while True:
            cmd = input("\n> ").strip()
            self.command_history.append(cmd)
            lower = cmd.lower()

            if lower == "exit":
                print("👋 Shutting down NLWS.")
                break
            if lower == "learn general":
                print("🚀 Starting general language learning...")
                await self.engine.start_general_language_learning()
                continue
            if lower.startswith("learn "):
                topic = cmd[6:].strip()
                if topic:
                    print(f"🎯 Specializing in: {topic}")
                    await self.engine.specialize_topic(topic)
                continue
            if lower == "stop":
                self.engine.stop_learning()
                continue
            if lower.startswith("ask "):
                question = cmd[4:].strip()
                print(f"\n🤖 {self.engine.answer_question(question)}")
                continue
            if lower == "visualize":
                svg = self.engine.visualizer.generate_svg()
                print(svg[:700] + "\n...")
                continue
            if lower == "status":
                print(
                    f"""
Phase: {self.engine.phase.value}
Memories: {len(self.engine.memory)}
Neurons: {len(self.engine.visualizer.neurons)}
Connections: {len(self.engine.visualizer.connections)}
Topic: {self.engine.topic or 'None'}
"""
                )
                continue

            print("Unknown command. Try: learn general | learn <topic> | ask <question> | status | exit")


if __name__ == "__main__":
    asyncio.run(LearningInterface().run())
