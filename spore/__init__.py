"""SPORE — System for Progressive Online Research & Evolution.

Public API surface — import anything you need from here.
"""

from .ai_client import ExternalAIClient
from .config import AIModelConfig
from .crawler import DuckDuckGoCrawler
from .engine import LanguageLearningEngine
from .models import LearningPhase, MemoryNode, Neuron
from .search import DuckDuckGoSearchProvider, SearchProvider
from .visualizer import NeuralNetworkVisualizer

__all__ = [
    "LearningPhase",
    "Neuron",
    "MemoryNode",
    "AIModelConfig",
    "ExternalAIClient",
    "SearchProvider",
    "DuckDuckGoSearchProvider",
    "DuckDuckGoCrawler",
    "NeuralNetworkVisualizer",
    "LanguageLearningEngine",
]
