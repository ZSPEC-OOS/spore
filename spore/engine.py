"""Language learning orchestration engine for SPORE.

LanguageLearningEngine coordinates the full learning pipeline:
  - Phase management (INITIALIZATION → GENERAL_LANGUAGE → … → EXPERT_MODE)
  - Neural network growth (visualizer) mirroring knowledge acquisition
  - Question answering via retrieval from the in-memory knowledge base

Extension points (explicitly left as stubs):
  - LanguageLearningEngine._synthesize_response() — route to a model/runtime
"""

from __future__ import annotations

import asyncio
import random
import re
from typing import Dict, List, Optional, Tuple

from .ai_client import ExternalAIClient
from .config import AIModelConfig
from .crawler import DuckDuckGoCrawler
from .models import LearningPhase, MemoryNode, Neuron
from .visualizer import NeuralNetworkVisualizer


class LanguageLearningEngine:

    def __init__(self) -> None:
        self.phase      = LearningPhase.INITIALIZATION
        self.visualizer = NeuralNetworkVisualizer()
        self.ai_config  = AIModelConfig.from_env()
        self.ai_client  = ExternalAIClient(self.ai_config)
        self.crawler    = DuckDuckGoCrawler(ai_client=self.ai_client)
        self.memory:             List[MemoryNode]      = []
        self.language_patterns:  Dict[str, int]        = {}
        self.topic:              Optional[str]         = None
        self.is_learning                               = False
        self._initialize_core_neurons()

    def configure_ai_model(
        self, name: str, model_id: str, base_url: str, api_key: str
    ) -> None:
        self.ai_config = AIModelConfig(
            name     = name.strip()     or "Default Search Model",
            model_id = model_id.strip(),
            base_url = base_url.strip(),
            api_key  = api_key.strip(),
        )
        self.ai_client         = ExternalAIClient(self.ai_config)
        self.crawler.ai_client = self.ai_client

    async def test_integrations(self) -> Dict[str, Tuple[bool, str]]:
        ai_ok,       ai_msg       = await self.ai_client.test_connection()
        provider_ok, provider_msg = await self.crawler.provider.health_check()
        return {
            "ai_model":        (ai_ok,       ai_msg),
            "search_provider": (provider_ok, provider_msg),
        }

    async def start_general_language_learning(self) -> None:
        self.phase       = LearningPhase.GENERAL_LANGUAGE
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
        print("✅ General language learning complete. Ready for testing or topic specialisation.")

    async def specialize_topic(self, topic: str) -> None:
        self.topic       = topic
        self.phase       = LearningPhase.TOPIC_SPECIALIZATION
        self.is_learning = True

        topic_neuron = self.visualizer.add_neuron(topic, layer=2)
        strategies   = [
            f"{topic} fundamentals basics introduction",
            f"{topic} advanced concepts expert guide",
            f"{topic} research papers latest studies",
            f"{topic} common questions FAQ",
            f"{topic} practical applications examples",
            f"{topic} criticism debates controversies",
        ]

        expertise_score = 0.0
        iteration       = 0
        max_iterations  = 20

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
        print(
            f"🎯 Expertise achieved in {topic}. "
            f"Network size: {len(self.visualizer.neurons)} neurons."
        )

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

    # ── Private helpers ───────────────────────────────────────────────────────

    def _initialize_core_neurons(self) -> None:
        root = self.visualizer.add_neuron("Language", layer=0)
        for concept in ["Syntax", "Semantics", "Context", "Intent", "Tone"]:
            neuron = self.visualizer.add_neuron(concept, layer=1)
            self.visualizer.connect(root.id, neuron.id)

    async def _integrate_knowledge(self, node: MemoryNode) -> None:
        self.memory.append(node)
        sentences = re.split(r"[.!?]+", node.content)
        for sentence in sentences:
            pattern = self._normalize_pattern(sentence.strip())
            if pattern:
                self.language_patterns[pattern] = (
                    self.language_patterns.get(pattern, 0) + 1
                )

    def _normalize_pattern(self, text: str) -> Optional[str]:
        if len(text) < 10:
            return None
        text = re.sub(r"\b\d+\b",          "[NUM]",  text)
        text = re.sub(r"\b[A-Z][a-z]+\b",  "[TERM]", text)
        return text.lower().strip()

    def _grow_neural_network(self, node: MemoryNode) -> Neuron:
        candidates        = node.related_concepts[:3]
        existing_concepts = {n.concept for n in self.visualizer.neurons.values()}
        created: List[Neuron] = []

        for concept in candidates:
            if concept not in existing_concepts:
                created.append(
                    self.visualizer.add_neuron(concept, layer=2 + random.randint(0, 2))
                )

        for i, n1 in enumerate(created):
            for n2 in created[i + 1:]:
                self.visualizer.connect(n1.id, n2.id)

        return created[0] if created else next(iter(self.visualizer.neurons.values()))

    def _calculate_expertise(self, topic: str) -> float:
        topic_memories = [m for m in self.memory if topic.lower() in m.content.lower()]
        if not topic_memories:
            return 0.0
        volume         = min(1.0, len(topic_memories) / 100)
        diversity      = min(1.0, len({m.source for m in topic_memories}) / 20)
        avg_confidence = sum(m.confidence for m in topic_memories) / len(topic_memories)
        return volume * 0.3 + diversity * 0.3 + avg_confidence * 0.4

    def _render_network(self) -> None:
        _ = self.visualizer.generate_svg()
        print(f"📊 Network visualisation updated: {len(self.visualizer.neurons)} neurons")

    def _relevance_score(self, question: str, memory: MemoryNode) -> float:
        q_words = set(question.lower().split())
        m_words = set(memory.content.lower().split())
        overlap = len(q_words & m_words)
        return overlap / max(len(q_words), 1) * memory.confidence

    def _synthesize_response(self, question: str, context: str) -> str:
        """Extension point: route to a model/runtime for richer answers."""
        q = question.lower()
        if "what is" in q:
            return f"Based on learned sources: {context[:300]}..."
        if "how" in q:
            return f"Here's the process based on learned material: {context[:300]}..."
        return f"From what I've learned so far: {context[:300]}..."
