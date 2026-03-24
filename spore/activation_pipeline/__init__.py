"""
activation_pipeline — modular activation hook and caching system.

Collects residual-stream activations from transformer models at selected
layers and checkpoints, then saves them to a structured on-disk cache for
downstream analysis (UMAP projection, SAE training, trajectory geometry, …).

Quick start — collection
------------------------
>>> from spore.activation_pipeline import ActivationCollector, CorpusLoader, ActivationCache
>>> texts = CorpusLoader.diverse_sentences(n=1000)
>>> collector = ActivationCollector("gpt2", layers=[0, 6, 11])
>>> run = collector.collect(texts, batch_size=32)
>>> ActivationCache.save(run, "activations/run_01")

Quick start — reduction
-----------------------
>>> from spore.activation_pipeline import ActivationCache
>>> from spore.activation_pipeline.reduction import compute_pca, compute_umap, to_dataframe, save_projection
>>> run  = ActivationCache.load("activations/run_01")
>>> acts = run.activations[6].float().numpy()
>>> pca  = compute_pca(acts, n_components=50)
>>> umr  = compute_umap(pca.coords, n_components=2)
>>> df   = to_dataframe(umr, {"label": run.labels, "layer": 6})
>>> save_projection(df, "projections/layer06_umap.parquet")
"""

from .collector import ActivationCollector, CollectionRun, HookPoint
from .corpus import CorpusLoader
from .io import ActivationCache
from .reduction import (
    ProjectionResult,
    ProjectionSuite,
    compute_pca,
    compute_umap,
    load_projection,
    save_projection,
    to_dataframe,
)
from .sae_dataset import SAEDataset, SAEDatasetConfig, SnippetResult

__all__ = [
    # collection
    "ActivationCollector",
    "CollectionRun",
    "HookPoint",
    "CorpusLoader",
    "ActivationCache",
    # reduction
    "ProjectionResult",
    "ProjectionSuite",
    "compute_pca",
    "compute_umap",
    "to_dataframe",
    "save_projection",
    "load_projection",
    # SAE dataset
    "SAEDataset",
    "SAEDatasetConfig",
    "SnippetResult",
]
