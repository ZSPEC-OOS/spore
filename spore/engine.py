"""Language learning orchestration engine for SPORE.

LanguageLearningEngine coordinates crawling, memory integration, and retrieval.
Visualization is now exclusively handled by the Streamlit geometric dashboard
(`streamlit_app.py`) and no longer uses in-memory node-link graph structures.
"""

from __future__ import annotations

import asyncio
import re
from typing import Dict, List, Optional, Tuple

from .ai_client import ExternalAIClient
from .config import AIModelConfig
from .crawler import DuckDuckGoCrawler
from .models import LearningPhase, MemoryNode
from .visualizer import GeometricActivationVisualizer


class LanguageLearningEngine:

    def __init__(self) -> None:
        self.phase = LearningPhase.INITIALIZATION
        self.visualizer = GeometricActivationVisualizer()
        self.ai_config = AIModelConfig.from_env()
        self.ai_client = ExternalAIClient(self.ai_config)
        self.crawler = DuckDuckGoCrawler(ai_client=self.ai_client)
        self.memory: List[MemoryNode] = []
        self.language_patterns: Dict[str, int] = {}
        self.concept_frequency: Dict[str, int] = {}
        self.topic: Optional[str] = None
        self.is_learning = False

    def configure_ai_model(
        self, name: str, model_id: str, base_url: str, api_key: str
    ) -> None:
        self.ai_config = AIModelConfig(
            name=name.strip() or "Default Search Model",
            model_id=model_id.strip(),
            base_url=base_url.strip(),
            api_key=api_key.strip(),
        )
        self.ai_client = ExternalAIClient(self.ai_config)
        self.crawler.ai_client = self.ai_client

    async def test_integrations(self) -> Dict[str, Tuple[bool, str]]:
        ai_ok, ai_msg = await self.ai_client.test_connection()
        provider_ok, provider_msg = await self.crawler.provider.health_check()
        return {
            "ai_model": (ai_ok, ai_msg),
            "search_provider": (provider_ok, provider_msg),
        }

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
            await asyncio.sleep(0.1)

        self.phase = LearningPhase.TESTING
        print("✅ General language learning complete. Ready for testing or topic specialisation.")

    async def specialize_topic(self, topic: str) -> None:
        self.topic = topic
        self.phase = LearningPhase.TOPIC_SPECIALIZATION
        self.is_learning = True

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
                expertise_score = self._calculate_expertise(topic)

            iteration += 1
            await asyncio.sleep(0.1)

        self.phase = LearningPhase.EXPERT_MODE
        print(f"🎯 Expertise achieved in {topic}. Concepts tracked: {len(self.concept_frequency)}.")

    def stop_learning(self) -> None:
        self.is_learning = False
        print("⏸️ Learning paused. Entering testing mode.")

    def answer_question(self, question: str) -> str:
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

    async def _integrate_knowledge(self, node: MemoryNode) -> None:
        self.memory.append(node)
        for concept in node.related_concepts:
            key = concept.lower().strip()
            if key:
                self.concept_frequency[key] = self.concept_frequency.get(key, 0) + 1

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

    def _calculate_expertise(self, topic: str) -> float:
        topic_memories = [m for m in self.memory if topic.lower() in m.content.lower()]
        if not topic_memories:
            return 0.0
        volume = min(1.0, len(topic_memories) / 100)
        diversity = min(1.0, len({m.source for m in topic_memories}) / 20)
        avg_confidence = sum(m.confidence for m in topic_memories) / len(topic_memories)
        return volume * 0.3 + diversity * 0.3 + avg_confidence * 0.4

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
