"""
spore.dashboard.pages — one module per dashboard tab.

Each module re-exports its public render function from spore.app.*
so both import paths continue to work:

    # New path
    from spore.dashboard.pages.latent_space import render_tab

    # Legacy path (still works)
    from spore.app.latent_space import render_tab
"""

from spore.app.latent_space          import render_tab as render_latent_space      # noqa: F401
from spore.app.sae_dashboard         import render_tab as render_feature_dict      # noqa: F401
from spore.app.prompt_trajectory     import render_prompt_trajectory_viewer        # noqa: F401
from spore.app.attention_logit_lens  import render_tab as render_attention_maps    # noqa: F401
from spore.app.metrics_comparison    import render_tab as render_metrics           # noqa: F401

__all__ = [
    "render_latent_space",
    "render_feature_dict",
    "render_prompt_trajectory_viewer",
    "render_attention_maps",
    "render_metrics",
]
