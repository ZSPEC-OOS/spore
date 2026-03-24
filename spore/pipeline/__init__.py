"""
spore.pipeline — clean public alias for spore.activation_pipeline.

Both import paths are valid and will remain so indefinitely:

    # New canonical path
    from spore.pipeline import ActivationCollector, ActivationCache

    # Legacy path (still works)
    from spore.activation_pipeline import ActivationCollector, ActivationCache

Sub-modules are also accessible via the aliased path:

    from spore.pipeline.reduction   import compute_pca, compute_umap
    from spore.pipeline.sae         import SparseAutoencoder, SAETrainer
    from spore.pipeline.sae_dataset import SAEDataset, SAEDatasetConfig
    from spore.pipeline.collector   import ActivationCollector, HookPoint
    from spore.pipeline.corpus      import CorpusLoader
    from spore.pipeline.io          import ActivationCache
"""
# Re-export all public symbols from activation_pipeline
from spore.activation_pipeline import (  # noqa: F401
    ActivationCache,
    ActivationCollector,
    CollectionRun,
    CorpusLoader,
    HookPoint,
    ProjectionResult,
    ProjectionSuite,
    SAEConfig,
    SAEDataset,
    SAEDatasetConfig,
    SAEOutput,
    SAETrainer,
    SnippetResult,
    SparseAutoencoder,
    compute_pca,
    compute_umap,
    load_projection,
    save_projection,
    to_dataframe,
)

# Expose sub-modules so ``from spore.pipeline.reduction import …`` also works
import spore.activation_pipeline.collector   as collector    # noqa: F401
import spore.activation_pipeline.corpus      as corpus       # noqa: F401
import spore.activation_pipeline.io          as io           # noqa: F401
import spore.activation_pipeline.reduction   as reduction    # noqa: F401
import spore.activation_pipeline.sae         as sae          # noqa: F401
import spore.activation_pipeline.sae_dataset as sae_dataset  # noqa: F401

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
    # SAE model + training
    "SAEConfig",
    "SAEOutput",
    "SparseAutoencoder",
    "SAETrainer",
    # sub-modules
    "collector",
    "corpus",
    "io",
    "reduction",
    "sae",
    "sae_dataset",
]
