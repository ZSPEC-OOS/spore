"""Geometric interpretability visualizer metadata for SPORE.

This module intentionally replaces the deprecated node-link / ball-and-web SVG
visualizer with a single mechanistic-interpretability-first dashboard contract.
The runnable UI entrypoint is ``streamlit_app.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


VERBATIM_MIGRATION_BRIEF = """Replace the existing ball-and-web (node-link graph with growing node sizes) visualizer entirely with the new High-Dimensional Geometric Activation Visualizer for the transformer-based AI.
Core concept: The model’s ‘mind’ is visualized as a continuously sculpted high-dimensional manifold in activation space. Knowledge is represented through distributed linear directions (features) via superposition rather than discrete nodes or explicit links.
Key visualization components to implement as the single main visualizer:
\t•\tLayer-wise UMAP/PCA projections of residual stream activations showing evolving semantic clusters and manifold refinement across training checkpoints and layers.
\t•\tSparse Autoencoder (SAE) feature dashboard: top-activating text examples, logit effects, and UMAP of monosemantic-ish feature directions.
\t•\tPrompt trajectory plots: animated paths of token representations through layers in projected space, with curvature and distance metrics.
\t•\tSupporting views: attention rollout heatmaps and logit lens (evolving predictions per layer).
The new visualizer must be built as a unified Streamlit multi-tab dashboard following the exact 10-request modular implementation (activation hooks, UMAP/PCA pipeline, SAE training, feature explorer, trajectory viewer, etc.). Discontinue and remove all code related to the old ball/web node-link visualization. All future development and displays will use only this geometric, mechanistic interpretability-style framework, which accurately reflects transformer internals: polysemantic neurons, superposition, and progressive geometric folding of the latent space."""


@dataclass(frozen=True)
class VisualizerModule:
    """Named dashboard module in the 10-request geometric visualizer plan."""

    name: str
    purpose: str


class GeometricActivationVisualizer:
    """Represents the unified geometric interpretability dashboard contract.

    This class does not render UI directly; it provides a stable API that the CLI,
    engine, and tests can use to verify that the runtime is configured for the
    Streamlit dashboard workflow.
    """

    streamlit_entrypoint: str = "streamlit_app.py"

    def migration_brief(self) -> str:
        """Return the required verbatim migration description for AI coders."""
        return VERBATIM_MIGRATION_BRIEF

    def module_plan(self) -> List[VisualizerModule]:
        """Return the canonical 10-module implementation plan."""
        return [
            VisualizerModule("activation_hooks", "Capture residual stream and attention activations per layer/checkpoint."),
            VisualizerModule("activation_storage", "Persist activations and metadata for reproducible downstream analysis."),
            VisualizerModule("reduction_pipeline", "Fit and cache PCA/UMAP reducers across checkpoints and layers."),
            VisualizerModule("latent_projection_view", "Render layer-wise projected manifolds with semantic clustering controls."),
            VisualizerModule("sae_dataset_builder", "Build token-level datasets for sparse autoencoder training."),
            VisualizerModule("sae_training", "Train sparse autoencoders with dead-feature handling and checkpoints."),
            VisualizerModule("sae_feature_explorer", "Inspect feature activations, top examples, and logit effects."),
            VisualizerModule("feature_umap_view", "Project SAE feature directions for monosemantic-ish neighborhood analysis."),
            VisualizerModule("prompt_trajectory_viewer", "Animate token trajectories through layers with curvature/distance metrics."),
            VisualizerModule("attention_logit_lens_view", "Show attention rollout heatmaps and per-layer logit-lens evolution."),
        ]

    def readiness(self, artifacts_root: str | Path = "artifacts") -> Dict[str, str | bool]:
        """Return whether artifact directories exist for immediate visualization.

        "Mind is empty" corresponds to a valid dashboard setup with no generated
        artifacts yet.
        """
        root = Path(artifacts_root)
        empty_but_ready = not root.exists() or not any(root.iterdir())
        return {
            "dashboard_only": True,
            "entrypoint": self.streamlit_entrypoint,
            "artifacts_root": str(root),
            "empty_but_ready": empty_but_ready,
        }
