"""
activation_pipeline — modular activation hook and caching system.

Collects residual-stream activations from transformer models at selected
layers and checkpoints, then saves them to a structured on-disk cache for
downstream analysis (UMAP projection, SAE training, trajectory geometry, …).

Quick start
-----------
>>> from spore.activation_pipeline import ActivationCollector, CorpusLoader, ActivationCache
>>> texts = CorpusLoader.diverse_sentences(n=1000)
>>> collector = ActivationCollector("gpt2", layers=[0, 6, 11])
>>> run = collector.collect(texts, batch_size=32)
>>> ActivationCache.save(run, "activations/run_01")
"""

from .collector import ActivationCollector, CollectionRun, HookPoint
from .corpus import CorpusLoader
from .io import ActivationCache

__all__ = [
    "ActivationCollector",
    "CollectionRun",
    "HookPoint",
    "CorpusLoader",
    "ActivationCache",
]
