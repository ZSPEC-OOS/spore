"""
scatter.py — Plotly figure builder for latent-space scatter plots.

Public API
----------
build_scatter(df, cfg)           → go.Figure
ScatterConfig                    — typed config dataclass

Design notes
------------
• Uses go.Scattergl (WebGL) for smooth rendering of 10k+ points.
• Categorical color_col  → one Scattergl trace per category so the
  Plotly legend lets users toggle categories on/off.
• Numeric color_col      → single trace with a continuous colorscale.
• Hover template is built dynamically from whichever "rich" columns
  are present: text, label, activation_norm, layer, index.
• Supports 2-D (x, y) and 3-D (x, y, z) via go.Scatter3d / Scattergl.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# Palette catalogue
# ---------------------------------------------------------------------------

_QUALITATIVE_PALETTES: Dict[str, List[str]] = {
    "D3":      px.colors.qualitative.D3,
    "Plotly":  px.colors.qualitative.Plotly,
    "Safe":    px.colors.qualitative.Safe,
    "Vivid":   px.colors.qualitative.Vivid,
    "Pastel":  px.colors.qualitative.Pastel,
    "Bold":    px.colors.qualitative.Bold,
    "Antique": px.colors.qualitative.Antique,
}

_SEQUENTIAL_COLORSCALES = [
    "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
    "Turbo",   "RdBu",   "Spectral", "Jet",
]

# Columns that carry per-sample "rich" context for hover tooltips
_HOVER_PRIORITY = ["text", "label", "activation_norm", "layer", "index"]


# ---------------------------------------------------------------------------
# ScatterConfig
# ---------------------------------------------------------------------------

@dataclass
class ScatterConfig:
    """
    All display parameters for :func:`build_scatter`.

    Attributes
    ----------
    color_col : str
        Column name to drive point colour.
    x_col, y_col, z_col : str
        Axis columns in the DataFrame.
    point_size : int
    opacity : float  (0–1)
    palette : str
        Qualitative palette name (for categorical color_col).
    colorscale : str
        Continuous colorscale name (for numeric color_col).
    max_points : int | None
        Subsample to this many points for performance.  None → no limit.
    text_col : str | None
        Column used as the primary hover text (truncated to max_text_chars).
    max_text_chars : int
        Truncate text snippets in hover to this length.
    show_legend : bool
    drag_mode : str
        Plotly dragmode: "pan" | "lasso" | "select" | "zoom"
    title : str
    height : int
    three_d : bool
        If True, render a 3-D scatter (requires z_col present in df).
    """

    color_col:      str   = "label"
    x_col:          str   = "x"
    y_col:          str   = "y"
    z_col:          str   = "z"
    point_size:     int   = 5
    opacity:        float = 0.75
    palette:        str   = "D3"
    colorscale:     str   = "Viridis"
    max_points:     Optional[int] = None
    text_col:       Optional[str] = "text"
    max_text_chars: int   = 100
    show_legend:    bool  = True
    drag_mode:      str   = "pan"
    title:          str   = ""
    height:         int   = 600
    three_d:        bool  = False


# ---------------------------------------------------------------------------
# build_scatter
# ---------------------------------------------------------------------------

def build_scatter(df: pd.DataFrame, cfg: ScatterConfig) -> go.Figure:
    """
    Build a Plotly figure from *df* using *cfg*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ``cfg.x_col`` and ``cfg.y_col`` columns.
        Optional: ``cfg.z_col``, ``cfg.color_col``, text/label/norm columns.
    cfg : ScatterConfig

    Returns
    -------
    go.Figure
    """
    if df is None or df.empty:
        return _empty_figure("No projection data loaded.")

    # ── Subsample ────────────────────────────────────────────────────────────
    if cfg.max_points and len(df) > cfg.max_points:
        df = df.sample(cfg.max_points, random_state=0).reset_index(drop=True)

    # ── Validate axis columns ────────────────────────────────────────────────
    for col in (cfg.x_col, cfg.y_col):
        if col not in df.columns:
            return _empty_figure(f"Column '{col}' not found in projection data.")

    three_d = cfg.three_d and cfg.z_col in df.columns

    # ── Detect colour type ───────────────────────────────────────────────────
    is_numeric = (
        cfg.color_col in df.columns
        and pd.api.types.is_numeric_dtype(df[cfg.color_col])
        and df[cfg.color_col].nunique() > 10
    )

    # ── Build hover payload ──────────────────────────────────────────────────
    hover_cols, hover_tmpl = _build_hover(df, cfg)

    # ── Build figure ─────────────────────────────────────────────────────────
    fig = go.Figure()

    if is_numeric:
        _add_numeric_traces(fig, df, cfg, hover_cols, hover_tmpl, three_d)
    elif cfg.color_col in df.columns:
        _add_categorical_traces(fig, df, cfg, hover_cols, hover_tmpl, three_d)
    else:
        _add_single_trace(fig, df, cfg, hover_cols, hover_tmpl, three_d)

    # ── Layout ───────────────────────────────────────────────────────────────
    _apply_layout(fig, cfg, three_d)
    return fig


# ---------------------------------------------------------------------------
# Trace builders
# ---------------------------------------------------------------------------

def _add_categorical_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    cfg: ScatterConfig,
    hover_cols: List[str],
    hover_tmpl: str,
    three_d: bool,
) -> None:
    """One Scattergl trace per category — enables legend toggling."""
    palette   = _QUALITATIVE_PALETTES.get(cfg.palette, _QUALITATIVE_PALETTES["D3"])
    categories = _sorted_categories(df[cfg.color_col])
    color_map  = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

    TraceClass = go.Scatter3d if three_d else go.Scattergl

    for cat in categories:
        mask     = df[cfg.color_col] == cat
        sub      = df[mask]
        cdata    = sub[hover_cols].values if hover_cols else None

        kwargs: Dict[str, Any] = dict(
            x          = sub[cfg.x_col].values,
            y          = sub[cfg.y_col].values,
            mode       = "markers",
            name       = str(cat),
            marker     = dict(
                color   = color_map[cat],
                size    = cfg.point_size if three_d else cfg.point_size,
                opacity = cfg.opacity,
                line    = dict(width=0),
            ),
            hovertemplate = hover_tmpl,
            customdata    = cdata,
            showlegend    = cfg.show_legend,
        )
        if three_d:
            kwargs["z"] = sub[cfg.z_col].values

        fig.add_trace(TraceClass(**kwargs))


def _add_numeric_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    cfg: ScatterConfig,
    hover_cols: List[str],
    hover_tmpl: str,
    three_d: bool,
) -> None:
    """Single trace with continuous colorscale."""
    cdata     = df[hover_cols].values if hover_cols else None
    TraceClass = go.Scatter3d if three_d else go.Scattergl

    kwargs: Dict[str, Any] = dict(
        x    = df[cfg.x_col].values,
        y    = df[cfg.y_col].values,
        mode = "markers",
        name = cfg.color_col,
        marker = dict(
            color      = df[cfg.color_col].values,
            colorscale = cfg.colorscale,
            size       = cfg.point_size,
            opacity    = cfg.opacity,
            line       = dict(width=0),
            showscale  = True,
            colorbar   = dict(title=cfg.color_col, thickness=14, len=0.7),
        ),
        hovertemplate = hover_tmpl,
        customdata    = cdata,
        showlegend    = False,
    )
    if three_d:
        kwargs["z"] = df[cfg.z_col].values

    fig.add_trace(TraceClass(**kwargs))


def _add_single_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    cfg: ScatterConfig,
    hover_cols: List[str],
    hover_tmpl: str,
    three_d: bool,
) -> None:
    """Fallback: single trace, uniform colour."""
    cdata     = df[hover_cols].values if hover_cols else None
    TraceClass = go.Scatter3d if three_d else go.Scattergl

    kwargs: Dict[str, Any] = dict(
        x    = df[cfg.x_col].values,
        y    = df[cfg.y_col].values,
        mode = "markers",
        name = "points",
        marker = dict(
            color   = "#4C78A8",
            size    = cfg.point_size,
            opacity = cfg.opacity,
            line    = dict(width=0),
        ),
        hovertemplate = hover_tmpl,
        customdata    = cdata,
    )
    if three_d:
        kwargs["z"] = df[cfg.z_col].values

    fig.add_trace(TraceClass(**kwargs))


# ---------------------------------------------------------------------------
# Hover template builder
# ---------------------------------------------------------------------------

def _build_hover(
    df: pd.DataFrame,
    cfg: ScatterConfig,
) -> tuple[List[str], str]:
    """
    Build (hover_cols, hovertemplate) dynamically from available columns.

    Returns
    -------
    hover_cols : list[str]
        Columns to pack into customdata (in order).
    hovertemplate : str
        Plotly hovertemplate string referencing %{customdata[i]}.
    """
    # Determine which rich columns are actually present
    present = [c for c in _HOVER_PRIORITY if c in df.columns and c != cfg.color_col]
    # Also include color_col itself so users can see the value
    if cfg.color_col in df.columns:
        present = [cfg.color_col] + [c for c in present if c != cfg.color_col]

    hover_cols = present
    lines = []

    for i, col in enumerate(hover_cols):
        ref = f"%{{customdata[{i}]}}"
        if col == "text":
            lines.append(f"<b>{ref}</b>")
        elif col == "activation_norm":
            lines.append(f"norm: {ref:.3f}" if False else f"‖act‖₂: {ref}")
        else:
            lines.append(f"{col}: {ref}")

    lines.append("x: %{x:.3f} &nbsp; y: %{y:.3f}")
    lines.append("<extra></extra>")   # removes the trace-name box
    return hover_cols, "<br>".join(lines)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _apply_layout(fig: go.Figure, cfg: ScatterConfig, three_d: bool) -> None:
    axis_style = dict(
        showgrid       = True,
        gridcolor      = "rgba(200,200,200,0.25)",
        zeroline       = False,
        showline       = False,
        showticklabels = False,
        title          = "",
    )

    common = dict(
        title           = dict(text=cfg.title, font=dict(size=16), x=0.5),
        paper_bgcolor   = "rgba(0,0,0,0)",
        plot_bgcolor    = "rgba(14,17,23,1)",
        font            = dict(color="#c8d0e0"),
        legend          = dict(
            bgcolor      = "rgba(14,17,23,0.85)",
            bordercolor  = "rgba(100,100,100,0.3)",
            borderwidth  = 1,
            itemsizing   = "constant",
            font         = dict(size=11),
        ),
        margin          = dict(l=8, r=8, t=48, b=8),
        height          = cfg.height,
        uirevision      = "keep",    # preserve zoom/pan across reruns
    )

    if three_d:
        fig.update_layout(
            **common,
            scene=dict(
                xaxis=axis_style,
                yaxis=axis_style,
                zaxis=axis_style,
                bgcolor="rgba(14,17,23,1)",
            ),
        )
    else:
        fig.update_layout(
            **common,
            dragmode = cfg.drag_mode,
            xaxis    = axis_style,
            yaxis    = dict(**axis_style, scaleanchor="x", scaleratio=1),
            modebar  = dict(
                bgcolor    = "rgba(0,0,0,0)",
                color      = "#888",
                activecolor= "#ddd",
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_categories(series: pd.Series) -> List[Any]:
    """Return category values sorted numerically if possible, else alphabetically."""
    cats = series.dropna().unique()
    try:
        return sorted(cats, key=lambda x: (int(x) if str(x).lstrip("-").isdigit() else x))
    except TypeError:
        return sorted(cats, key=str)


def _empty_figure(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(14,17,23,1)",
        font          = dict(color="#c8d0e0"),
        annotations   = [dict(text=msg, x=0.5, y=0.5, showarrow=False,
                              font=dict(size=14), xref="paper", yref="paper")],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=400,
    )
    return fig


def color_column_options(df: pd.DataFrame) -> List[str]:
    """
    Return a ranked list of good color column candidates from *df*.

    Axis columns (x/y/z) and the index are excluded.
    High-cardinality numeric columns (like activation_norm) are placed
    after categorical ones.
    """
    exclude    = {"x", "y", "z", "c4", "c5"}
    cat_cols   = []
    num_cols   = []
    for col in df.columns:
        if col in exclude:
            continue
        if col == "text":
            continue   # too long to colour by
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return cat_cols + num_cols
