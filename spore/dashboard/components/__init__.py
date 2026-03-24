"""
spore.dashboard.components — reusable Streamlit widgets and plot builders.

    from spore.dashboard.components.scatter      import ScatterConfig, build_scatter
    from spore.dashboard.components.status_panel import render_status_panel, scan_artifacts
"""

from spore.app.scatter      import ScatterConfig, build_scatter, color_column_options  # noqa: F401
from spore.app.status_panel import (  # noqa: F401
    ArtifactStatus,
    StepStatus,
    render_sidebar_status,
    render_status_panel,
    scan_artifacts,
)

__all__ = [
    "ScatterConfig",
    "build_scatter",
    "color_column_options",
    "ArtifactStatus",
    "StepStatus",
    "render_sidebar_status",
    "render_status_panel",
    "scan_artifacts",
]
