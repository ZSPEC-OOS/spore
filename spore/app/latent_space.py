"""
latent_space.py — Tab 1: Interactive Latent Space Projection Explorer.

Call ``render_tab()`` from the Streamlit entry point.

Layout
------
  [Sidebar]
    📂 Data Source    — projection directory path + method selector
    🔢 Layer          — single / overlay / animate view modes
    🎨 Display        — color-by, point size, opacity, palette
    🖱️ Interaction    — drag mode, 3-D toggle, subsample cap

  [Main area]
    Title + subtitle row
    Scatter plot (Plotly, WebGL, full width)
    ── PCA Explained Variance bar chart (PCA method only)
    ── Selected Points table (after lasso/box selection)
    ── Corpus preview table (bottom of page)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .loader import ProjectionStore
from .scatter import ScatterConfig, build_scatter, color_column_options
from .prompt_trajectory import render_prompt_trajectory_viewer

# ---------------------------------------------------------------------------
# Session-state keys
# ---------------------------------------------------------------------------
_K_STORE     = "_ls_store"       # ProjectionStore
_K_ROOT      = "_ls_root"        # last loaded directory
_K_PLAYING   = "_ls_playing"     # bool — animation running?
_K_LAYER_IDX = "_ls_layer_idx"   # int  — current layer index in animation

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* darken the default Streamlit background slightly */
[data-testid="stAppViewContainer"] { background: #0e1117; }
[data-testid="stSidebar"]          { background: #161b22; }

/* tighter sidebar sections */
.sidebar-section-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #8b949e;
    text-transform: uppercase;
    margin: 1.2rem 0 0.3rem 0;
}

/* stat badges */
.stat-badge {
    display: inline-block;
    padding: 0.18rem 0.55rem;
    border-radius: 4px;
    background: #21262d;
    color: #c9d1d9;
    font-size: 0.82rem;
    margin-right: 0.4rem;
    border: 1px solid #30363d;
}

/* selection info box */
.selection-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin-top: 0.6rem;
}
</style>
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render_tab() -> None:
    """Render the full Latent Space tab (sidebar + main area)."""
    st.markdown(_CSS, unsafe_allow_html=True)

    controls = _sidebar_controls()
    _main_panel(controls)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar_controls() -> Dict[str, Any]:
    """Render all sidebar widgets and return a controls dict."""
    st.sidebar.markdown("## 🧠 SPORE Visualiser")

    # ── 📂 Data Source ───────────────────────────────────────────────────────
    st.sidebar.markdown('<p class="sidebar-section-title">📂 Data Source</p>',
                        unsafe_allow_html=True)

    default_root = str(Path("projections/run").resolve())
    root = st.sidebar.text_input(
        "Projections directory",
        value=st.session_state.get(_K_ROOT, default_root),
        help="Directory produced by reduce_activations.py or ProjectionSuite.save()",
    )

    reload = st.sidebar.button("⟳ Reload", use_container_width=True)

    # Build/refresh the store
    if (
        _K_STORE not in st.session_state
        or st.session_state.get(_K_ROOT) != root
        or reload
    ):
        st.session_state[_K_STORE] = ProjectionStore(root)
        st.session_state[_K_ROOT]  = root

    store: ProjectionStore = st.session_state[_K_STORE]

    if store.is_empty():
        st.sidebar.warning(
            "No projection files found.\n\n"
            "Run `python collect_activations.py` then "
            "`python reduce_activations.py` first, or set a valid path above."
        )
        # Return a minimal controls dict so the main panel can show the guide
        return {"store": store, "ready": False}

    # ── Method ───────────────────────────────────────────────────────────────
    methods  = store.available_methods()
    method   = st.sidebar.radio(
        "Reduction method",
        options=methods,
        format_func=str.upper,
        horizontal=True,
    )

    # ── 🔢 Layer ─────────────────────────────────────────────────────────────
    st.sidebar.markdown('<p class="sidebar-section-title">🔢 Layer</p>',
                        unsafe_allow_html=True)

    layers = store.available_layers(method)

    view_mode = st.sidebar.radio(
        "View mode",
        ["Single", "Overlay", "Animate"],
        horizontal=True,
        help=(
            "Single — one layer at a time\n"
            "Overlay — stack multiple layers (colour by layer)\n"
            "Animate — auto-advance through layers"
        ),
    )

    # Single / Animate: one layer selected via slider
    if view_mode in ("Single", "Animate"):
        if len(layers) == 1:
            layer_sel = [layers[0]]
            st.sidebar.caption(f"Layer {layers[0]} (only one available)")
        else:
            idx = st.sidebar.select_slider(
                "Layer",
                options=layers,
                value=layers[st.session_state.get(_K_LAYER_IDX, 0)],
                key="layer_slider",
            )
            layer_sel = [idx]
    else:  # Overlay
        layer_sel = st.sidebar.multiselect(
            "Layers to overlay",
            options=layers,
            default=layers[:min(3, len(layers))],
            help="Select 2–5 layers for a readable overlay",
        )
        if not layer_sel:
            layer_sel = layers[:1]

    # Animate controls
    anim_interval = 0.8
    if view_mode == "Animate":
        anim_interval = st.sidebar.slider(
            "Advance interval (s)", min_value=0.3, max_value=3.0,
            value=0.8, step=0.1,
        )
        col1, col2 = st.sidebar.columns(2)
        if col1.button("▶ Play",  use_container_width=True):
            st.session_state[_K_PLAYING] = True
        if col2.button("⏹ Stop",  use_container_width=True):
            st.session_state[_K_PLAYING] = False

    # ── 🎨 Display ───────────────────────────────────────────────────────────
    st.sidebar.markdown('<p class="sidebar-section-title">🎨 Display</p>',
                        unsafe_allow_html=True)

    # Determine available colour columns from the first loaded layer
    sample_df = store.load(layer_sel[0], method)
    color_options = color_column_options(sample_df) if sample_df is not None else ["layer"]

    # Override color_col in overlay mode to colour by layer
    if view_mode == "Overlay":
        default_color = "layer" if "layer" in color_options else color_options[0]
    else:
        default_color = color_options[0] if color_options else "layer"

    color_col = st.sidebar.selectbox(
        "Color by",
        options=color_options,
        index=color_options.index(default_color) if default_color in color_options else 0,
    )

    # Palette / colorscale selector (shown contextually)
    if sample_df is not None and color_col in sample_df.columns:
        is_numeric_col = (
            pd.api.types.is_numeric_dtype(sample_df[color_col])
            and sample_df[color_col].nunique() > 10
        )
    else:
        is_numeric_col = False

    if is_numeric_col:
        colorscale = st.sidebar.selectbox(
            "Colorscale",
            ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "RdBu"],
            index=0,
        )
        palette = "D3"
    else:
        palette = st.sidebar.selectbox(
            "Palette",
            ["D3", "Plotly", "Safe", "Vivid", "Pastel", "Bold", "Antique"],
            index=0,
        )
        colorscale = "Viridis"

    point_size = st.sidebar.slider("Point size", min_value=2, max_value=14, value=5)
    opacity    = st.sidebar.slider("Opacity",    min_value=0.1, max_value=1.0,
                                   value=0.75, step=0.05)

    # ── 🖱️ Interaction ──────────────────────────────────────────────────────
    st.sidebar.markdown('<p class="sidebar-section-title">🖱️ Interaction</p>',
                        unsafe_allow_html=True)

    drag_mode = st.sidebar.radio(
        "Default drag mode",
        ["pan", "lasso", "select", "zoom"],
        horizontal=True,
        index=0,
        format_func={"pan": "🖐 Pan", "lasso": "🌀 Lasso",
                     "select": "⬜ Box", "zoom": "🔍 Zoom"}.get,
    )

    three_d = st.sidebar.checkbox(
        "3-D view",
        value=False,
        disabled=("z" not in (sample_df.columns if sample_df is not None else [])),
        help="Requires a 3-component projection.  Re-run reduce_activations.py with --umap-components 3",
    )

    with st.sidebar.expander("⚙️ Advanced"):
        max_points = st.number_input(
            "Max points rendered",
            min_value=500, max_value=50_000,
            value=10_000, step=500,
            help="Subsample large datasets for smoother interaction",
        )
        plot_height = st.slider(
            "Plot height (px)", min_value=300, max_value=1200, value=620, step=20,
        )

    return {
        "store":         store,
        "method":        method,
        "layer_sel":     layer_sel,
        "view_mode":     view_mode,
        "anim_interval": anim_interval,
        "color_col":     color_col,
        "palette":       palette,
        "colorscale":    colorscale,
        "point_size":    point_size,
        "opacity":       opacity,
        "drag_mode":     drag_mode,
        "three_d":       three_d,
        "max_points":    int(max_points),
        "plot_height":   int(plot_height),
        "ready":         True,
    }


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

def _main_panel(ctrl: Dict[str, Any]) -> None:
    store: ProjectionStore = ctrl["store"]

    if not ctrl.get("ready"):
        _render_onboarding()
        return

    method      = ctrl["method"]
    layer_sel   = ctrl["layer_sel"]
    view_mode   = ctrl["view_mode"]
    color_col   = ctrl["color_col"]

    # ── Animate: advance the layer index ─────────────────────────────────────
    if view_mode == "Animate" and st.session_state.get(_K_PLAYING):
        layers = store.available_layers(method)
        cur    = st.session_state.get(_K_LAYER_IDX, 0)
        nxt    = (cur + 1) % len(layers)
        st.session_state[_K_LAYER_IDX] = nxt
        layer_sel = [layers[nxt]]
        time.sleep(ctrl["anim_interval"])
        st.rerun()

    # ── Load data ─────────────────────────────────────────────────────────────
    if view_mode == "Overlay" and len(layer_sel) > 1:
        df = store.load_multi(layer_sel, method)
    else:
        df = store.load(layer_sel[0], method) if layer_sel else None

    if df is None or df.empty:
        st.error("Failed to load projection data.  Check the directory and method.")
        return

    # Truncate hover text for readability
    if "text" in df.columns:
        df = df.copy()
        df["text"] = df["text"].str[:100]

    # ── Title row ────────────────────────────────────────────────────────────
    model_name = _infer_model_name(store, method, layer_sel[0])
    layer_label = (
        f"Layer {layer_sel[0]}"
        if len(layer_sel) == 1
        else f"Layers {', '.join(str(l) for l in layer_sel)}"
    )

    col_title, col_stats = st.columns([3, 2])
    with col_title:
        st.markdown(
            f"### 🗺️ Latent Space — {method.upper()}",
            help="Points represent mean-pooled residual-stream activations.",
        )
        st.caption(
            f"{model_name}  ·  {layer_label}  ·  colored by **{color_col}**"
        )
    with col_stats:
        st.markdown(
            f'<div style="padding-top:1.2rem">'
            f'<span class="stat-badge">N = {len(df):,}</span>'
            f'<span class="stat-badge">{method.upper()}</span>'
            f'<span class="stat-badge">{layer_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Build figure ─────────────────────────────────────────────────────────
    cfg = ScatterConfig(
        color_col   = color_col,
        point_size  = ctrl["point_size"],
        opacity     = ctrl["opacity"],
        palette     = ctrl["palette"],
        colorscale  = ctrl["colorscale"],
        max_points  = ctrl["max_points"],
        drag_mode   = ctrl["drag_mode"],
        three_d     = ctrl["three_d"],
        height      = ctrl["plot_height"],
        title       = "",
    )
    fig = build_scatter(df, cfg)

    # ── Render scatter ───────────────────────────────────────────────────────
    try:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            key=f"scatter_{method}_{'-'.join(str(l) for l in layer_sel)}_{color_col}",
            config={
                "scrollZoom":              True,
                "displaylogo":             False,
                "modeBarButtonsToAdd":     ["lasso2d", "select2d"],
                "modeBarButtonsToRemove":  ["toImage"],
                "toImageButtonOptions":    {"format": "png", "scale": 2},
            },
        )
        selected_pts = event.selection.points if event and event.selection else []
    except TypeError:
        # Streamlit < 1.35 — on_select not supported
        st.plotly_chart(fig, use_container_width=True)
        selected_pts = []

    # ── PCA explained-variance panel ─────────────────────────────────────────
    if method == "pca" and store.summary is not None:
        with st.expander("📊 PCA Explained Variance", expanded=False):
            _render_pca_variance(store.summary, layer_sel)

    # ── Selection info ────────────────────────────────────────────────────────
    if selected_pts:
        _render_selection_panel(df, selected_pts)

    # ── Corpus preview ───────────────────────────────────────────────────────
    # ── Prompt trajectory viewer ─────────────────────────────────────────────
    render_prompt_trajectory_viewer(default_root=str(Path("activations/run").resolve()))

    with st.expander("📄 Corpus preview (first 50 rows)", expanded=False):
        preview_cols = [c for c in ["index", "text", "label", "activation_norm", "layer"]
                        if c in df.columns]
        st.dataframe(
            df[preview_cols].head(50),
            use_container_width=True,
            height=300,
        )


# ---------------------------------------------------------------------------
# Sub-panels
# ---------------------------------------------------------------------------

def _render_pca_variance(summary: pd.DataFrame, layer_sel: List[int]) -> None:
    import plotly.express as px

    sub = summary[summary["layer"].isin(layer_sel)] if "layer" in summary.columns else summary

    if "cumulative_expl_var" in sub.columns:
        sub = sub.copy()
        sub["pct"] = sub["cumulative_expl_var"] * 100

        fig = px.bar(
            sub, x="layer", y="pct",
            text=sub["pct"].map("{:.1f}%".format),
            labels={"pct": "Cumulative explained variance (%)", "layer": "Layer"},
            color="pct",
            color_continuous_scale="Viridis",
            range_y=[0, 105],
            height=250,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(14,17,23,0)",
            font          = dict(color="#c9d1d9"),
            coloraxis_showscale = False,
            margin        = dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(sub.round(4), use_container_width=True, hide_index=True)


def _render_selection_panel(df: pd.DataFrame, selected_pts: list) -> None:
    st.markdown('<div class="selection-box">', unsafe_allow_html=True)
    st.markdown(f"**✅ {len(selected_pts)} point(s) selected**")

    # Recover row indices from pointIndex / point_number
    indices = []
    for pt in selected_pts:
        idx = pt.get("point_index") if pt.get("point_index") is not None else pt.get("point_number")
        if idx is not None:
            indices.append(idx)

    if indices:
        show_cols = [c for c in ["index", "text", "label", "activation_norm", "layer", "x", "y"]
                     if c in df.columns]
        sel_df = df.iloc[indices][show_cols].reset_index(drop=True)
        st.dataframe(sel_df, use_container_width=True, height=min(300, 40 + 35 * len(sel_df)))

        # Download selected points
        csv_bytes = sel_df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download selection (.csv)",
            data      = csv_bytes,
            file_name = "selected_points.csv",
            mime      = "text/csv",
            use_container_width = False,
        )

    st.markdown('</div>', unsafe_allow_html=True)


def _render_onboarding() -> None:
    st.markdown("## 🧠 SPORE Latent Space Explorer")
    st.info(
        "**No projection data found.**\n\n"
        "Follow these steps to get started:\n\n"
        "```bash\n"
        "# Step 1 — collect activations\n"
        "python collect_activations.py \\\n"
        "    --model gpt2 --n 1000 --layers 0,3,6,9,11 \\\n"
        "    --out activations/run\n\n"
        "# Step 2 — compute UMAP + PCA projections\n"
        "python reduce_activations.py \\\n"
        "    --src activations/run \\\n"
        "    --method both --layers all \\\n"
        "    --out projections/run\n\n"
        "# Step 3 — launch this dashboard\n"
        "streamlit run streamlit_app.py\n"
        "```\n\n"
        "Then set the **Projections directory** in the sidebar to `projections/run`."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_model_name(store: ProjectionStore, method: str, layer: int) -> str:
    """Try to read model name from the projection DataFrame."""
    df = store.load(layer, method)
    if df is not None and "model_name" in df.columns:
        return str(df["model_name"].iloc[0])
    return "transformer"
