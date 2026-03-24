"""
feature_map.py — Tab 3: SAE Feature UMAP (global feature-space view).

Call ``render_tab()`` from the Streamlit entry point.

This tab renders a 2-D scatter of every SAE feature's decoder direction
projected into UMAP space (or PCA fallback when umap-learn is unavailable).

Layout
------
  [Sidebar — shared with 🔬 SAE Features]
    Reads ckpt/dataset/layer from session state (set in the SAE Features tab).

  [Sidebar — Feature Map controls]
    🌐 FEATURE MAP   — color-by selector, UMAP params, cluster count,
                       dataset scan cap, Compute button.

  [Main area]
    ── Info banner (computation status, dataset info)
    ── Scatter plot (Plotly WebGL):
         Each point = one SAE feature direction in UMAP space.
         Color by: cluster (categorical) | max activation | frequency | mean.
         Hover: feature index + top snippet + activation stats.
         Click: select feature → highlight + show detail panel below.
    ── Selected Feature detail panel (inline):
         Stats, top-3 snippets, note to switch to 🔬 SAE Features tab.

Click → jump
-----------
  Clicking a scatter point sets ``st.session_state["_sae_feat"]`` so that
  switching to the **🔬 SAE Features** tab immediately shows that feature's
  full histogram, examples, and logit effects.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from spore.app.feature_umap import FeatureMapData, compute_feature_map

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session-state keys
# ---------------------------------------------------------------------------
# Paths are shared with the SAE Features tab (_sae_* keys).
_K_CKPT       = "_sae_ckpt"
_K_DS         = "_sae_ds"
_K_LAYER      = "_sae_layer"
_K_FEAT       = "_sae_feat"           # shared → jump to detail
_K_FEAT_SL    = "_sae_feat_sl"        # shared slider

_K_COLOR_BY   = "_fm_color"
_K_N_CLUSTERS = "_fm_k"
_K_NEIGHBORS  = "_fm_nn"
_K_MIN_DIST   = "_fm_md"
_K_MAX_TOK    = "_fm_maxtok"
_K_PTSIZE     = "_fm_ptsize"
_K_OPACITY    = "_fm_opacity"
_K_SELECTED   = "_fm_selected"        # currently highlighted feature
_K_TRIGGER    = "_fm_trigger"         # int — incremented to force recompute

# ---------------------------------------------------------------------------
# Categorical colour palette (Plotly D3 + extended)
# ---------------------------------------------------------------------------
_CLUSTER_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#5254a3", "#8ca252", "#bd9e39", "#ad494a", "#a55194",
    "#6b6ecf", "#b5cf6b", "#e7ba52", "#d6616b", "#ce6dbd",
]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Computing UMAP over SAE features …", ttl=None)
def _load_feature_map(
    ckpt_path:    str,
    dataset_root: str,
    layer:        int,
    n_pca:        int,
    umap_neighbors: int,
    umap_min_dist:  float,
    n_clusters:   int,
    max_tokens:   int,
    _trigger:     int = 0,   # increment to force recompute
) -> Optional[dict]:
    """
    Load FeatureAnalyzer + run compute_feature_map.  Returns a plain dict
    (all numpy arrays → lists) so @st.cache_data can serialise it.
    """
    from spore.app.sae_feature import FeatureAnalyzer

    try:
        analyzer = FeatureAnalyzer.from_checkpoint(ckpt_path, dataset_root, layer)
    except Exception as exc:
        return {"error": str(exc)}

    try:
        data = compute_feature_map(
            analyzer,
            n_pca          = n_pca,
            umap_neighbors = umap_neighbors,
            umap_min_dist  = umap_min_dist,
            n_clusters     = n_clusters,
            max_tokens     = max_tokens,
        )
    except Exception as exc:
        return {"error": f"compute_feature_map failed: {exc}"}

    # Serialise to plain Python/lists for @st.cache_data
    return {
        "umap_xy":           data.umap_xy.tolist(),
        "max_act":           data.max_act.tolist(),
        "mean_active":       data.mean_active.tolist(),
        "pct_active":        data.pct_active.tolist(),
        "cluster_labels":    data.cluster_labels.tolist(),
        "hover_texts":       data.hover_texts,
        "n_features":        data.n_features,
        "d_model":           data.d_model,
        "layer":             data.layer,
        "n_clusters":        data.n_clusters,
        "n_tokens_scanned":  data.n_tokens_scanned,
        "pca_n_components":  data.pca_n_components,
        "pca_explained_var": data.pca_explained_var,
        "umap_neighbors":    data.umap_neighbors,
        "umap_min_dist":     data.umap_min_dist,
        "used_umap":         data.used_umap,
    }


@st.cache_data(show_spinner=False, ttl=600)
def _load_top_snippets(
    ckpt_path:    str,
    dataset_root: str,
    layer:        int,
    feat_idx:     int,
    k:            int = 3,
) -> List[dict]:
    """Top-k snippets for the detail panel (small k, fast)."""
    from spore.app.sae_dashboard import _compute_top_k
    return _compute_top_k(ckpt_path, dataset_root, layer, feat_idx, k, 0.0, 6)


def _dict_to_data(d: dict) -> FeatureMapData:
    """Reconstruct numpy arrays from the serialised cache dict."""
    return FeatureMapData(
        umap_xy           = np.array(d["umap_xy"],        dtype=np.float32),
        max_act           = np.array(d["max_act"],         dtype=np.float32),
        mean_active       = np.array(d["mean_active"],     dtype=np.float32),
        pct_active        = np.array(d["pct_active"],      dtype=np.float32),
        cluster_labels    = np.array(d["cluster_labels"],  dtype=np.int32),
        hover_texts       = d["hover_texts"],
        n_features        = d["n_features"],
        d_model           = d["d_model"],
        layer             = d["layer"],
        n_clusters        = d["n_clusters"],
        n_tokens_scanned  = d["n_tokens_scanned"],
        pca_n_components  = d["pca_n_components"],
        pca_explained_var = d.get("pca_explained_var"),
        umap_neighbors    = d["umap_neighbors"],
        umap_min_dist     = d["umap_min_dist"],
        used_umap         = d.get("used_umap", False),
    )


# ---------------------------------------------------------------------------
# Plotly figure builder
# ---------------------------------------------------------------------------

def _build_map_figure(
    data:         FeatureMapData,
    color_by:     str,
    point_size:   int,
    opacity:      float,
    selected_idx: Optional[int],
) -> go.Figure:
    """
    Build the feature-map scatter figure.

    Parameters
    ----------
    data : FeatureMapData
    color_by : str
        ``"cluster"`` | ``"max_act"`` | ``"pct_active"`` | ``"mean_active"``
    point_size : int
    opacity : float
    selected_idx : int | None
        If set, highlight that feature with a larger white-ringed marker.

    Returns
    -------
    go.Figure
    """
    xy    = data.umap_xy          # [F, 2]
    n     = data.n_features
    texts = data.hover_texts

    fig = go.Figure()

    if color_by == "cluster":
        # ── Categorical: one trace per cluster ───────────────────────
        k       = data.n_clusters
        labels  = data.cluster_labels
        palette = (_CLUSTER_PALETTE * ((k // len(_CLUSTER_PALETTE)) + 1))[:k]

        for c in range(k):
            mask   = labels == c
            if not mask.any():
                continue
            fidxs  = np.where(mask)[0]    # global feature indices
            x_c    = xy[mask, 0]
            y_c    = xy[mask, 1]
            txt_c  = [texts[i] for i in fidxs]
            # customdata = global feature index (scalar per point)
            cdata  = fidxs.tolist()

            fig.add_trace(go.Scattergl(
                x            = x_c,
                y            = y_c,
                mode         = "markers",
                name         = f"Cluster {c}",
                marker       = dict(
                    size    = point_size,
                    color   = palette[c],
                    opacity = opacity,
                    line    = dict(width=0),
                ),
                customdata   = cdata,
                hovertemplate = "%{customdata}<br>%{text}<extra></extra>",
                text         = txt_c,
                showlegend   = k <= 30,
            ))

    else:
        # ── Continuous: single trace with colorscale ─────────────────
        color_map = {
            "max_act":     (data.max_act,    "Max Activation",    "Viridis"),
            "pct_active":  (data.pct_active, "Activation Freq %", "Plasma"),
            "mean_active": (data.mean_active,"Mean Act (active)", "Magma"),
        }
        values, cb_title, cscale = color_map.get(
            color_by, (data.max_act, "Max Activation", "Viridis")
        )
        feat_indices = np.arange(n, dtype=np.int32)

        fig.add_trace(go.Scattergl(
            x            = xy[:, 0],
            y            = xy[:, 1],
            mode         = "markers",
            marker       = dict(
                size       = point_size,
                color      = values,
                colorscale = cscale,
                opacity    = opacity,
                line       = dict(width=0),
                colorbar   = dict(
                    title      = cb_title,
                    thickness  = 14,
                    len        = 0.8,
                    tickfont   = dict(color="#8b949e", size=10),
                    titlefont  = dict(color="#8b949e", size=11),
                ),
            ),
            customdata   = feat_indices.tolist(),
            hovertemplate = "%{customdata}<br>%{text}<extra></extra>",
            text         = texts,
            showlegend   = False,
            name         = "",
        ))

    # ── Highlight selected feature ────────────────────────────────────
    if selected_idx is not None and 0 <= selected_idx < n:
        sx = float(xy[selected_idx, 0])
        sy = float(xy[selected_idx, 1])
        fig.add_trace(go.Scatter(
            x            = [sx],
            y            = [sy],
            mode         = "markers",
            marker       = dict(
                size   = point_size + 8,
                color  = "rgba(0,0,0,0)",
                line   = dict(color="#f0f6fc", width=2.5),
            ),
            customdata   = [selected_idx],
            hovertemplate = f"<b>Selected: Feature {selected_idx}</b><extra></extra>",
            name         = f"Feature {selected_idx}",
            showlegend   = True,
        ))
        # Crosshair annotations for selected point
        fig.add_annotation(
            x=sx, y=sy,
            text=f" {selected_idx}",
            showarrow=False,
            font=dict(color="#f0f6fc", size=10),
            xanchor="left",
        )

    # ── Layout ────────────────────────────────────────────────────────
    method_str = "UMAP" if data.used_umap else "PCA 2D"
    ev_str = ""
    if data.pca_explained_var is not None:
        ev_str = f"  ·  PCA {data.pca_n_components}d ({data.pca_explained_var:.0%} var)"

    fig.update_layout(
        title=dict(
            text=(
                f"{method_str} of {n:,} SAE features  "
                f"(layer {data.layer}, d={data.d_model}){ev_str}"
            ),
            font=dict(size=12, color="#8b949e"),
            x=0,
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=600,
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="#21262d",
            borderwidth=1,
            font=dict(size=10, color="#cdd9e5"),
        ),
        dragmode="pan",
        hoverlabel=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            font=dict(color="#cdd9e5", size=11, family="SF Mono, monospace"),
        ),
        uirevision="feature_map",   # keep zoom/pan across rerenders
    )
    return fig


# ---------------------------------------------------------------------------
# Detail panel for selected feature
# ---------------------------------------------------------------------------

def _render_selected_panel(
    feat_idx:     int,
    data:         FeatureMapData,
    ckpt_path:    str,
    dataset_root: str,
    layer:        int,
) -> None:
    """
    Show a compact detail card for the clicked feature.
    """
    max_a  = float(data.max_act[feat_idx])
    pct_a  = float(data.pct_active[feat_idx])
    mean_a = float(data.mean_active[feat_idx])
    cl     = int(data.cluster_labels[feat_idx])
    xy     = data.umap_xy[feat_idx]

    st.markdown(
        f"### Feature `{feat_idx}`"
        f"<span style='color:#6e7681;font-size:0.85rem;font-weight:400'>"
        f"  cluster {cl}  ·  umap ({xy[0]:.2f}, {xy[1]:.2f})"
        f"</span>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Activation", f"{max_a:.4f}" if max_a > 0 else "dead")
    c2.metric("% Active",       f"{pct_a:.2f}%")
    c3.metric("Mean (active)",  f"{mean_a:.4f}" if mean_a > 0 else "—")
    c4.metric("Cluster",        cl)

    if max_a <= 0:
        st.markdown(
            '<div style="background:#2d1f0e;border:1px solid #5a3a1a;'
            'border-radius:6px;padding:10px 14px;color:#e3b341;font-size:0.85rem">'
            "⚠️ Dead feature — never activated on the scanned tokens."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Top-3 snippets
    st.markdown("**Top activating examples:**")
    with st.spinner("Loading snippets …"):
        snippets = _load_top_snippets(ckpt_path, dataset_root, layer, feat_idx, k=3)

    if snippets:
        rows = []
        for s in snippets:
            before = html.escape(s.get("context_before") or "")
            tok    = html.escape(s.get("token_str")      or "?")
            after  = html.escape(s.get("context_after")  or "")
            score  = s.get("score", 0.0)
            rows.append(
                f'<tr style="border-bottom:1px solid #21262d">'
                f'<td style="padding:4px 8px;color:#8b949e;font-size:0.75rem">'
                f'  {score:.3f}'
                f'</td>'
                f'<td style="padding:4px 8px;font-family:monospace;font-size:0.8rem">'
                f'  <span style="color:#8b949e">{before}</span>'
                f'  <span style="background:#fff3cd;color:#1a1a1a;padding:1px 3px;'
                f'    border-radius:3px;font-weight:700">{tok}</span>'
                f'  <span style="color:#8b949e">{after}</span>'
                f'</td>'
                f'</tr>'
            )
        table = (
            '<table style="border-collapse:collapse;font-size:0.82rem;'
            'margin:6px 0 10px 0;width:100%">'
            + "".join(rows)
            + "</table>"
        )
        st.markdown(table, unsafe_allow_html=True)
    else:
        st.caption("No examples above threshold.")

    # Jump note + sync
    st.info(
        f"**Feature {feat_idx}** is pre-selected — switch to the "
        "**🔬 SAE Features** tab for the full histogram, all top examples, "
        "and logit lens.",
        icon="💡",
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
.fm-section {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #8b949e;
    text-transform: uppercase;
    margin: 12px 0 4px 0;
    padding-top: 8px;
    border-top: 1px solid #21262d;
}
.fm-info-row {
    display: flex;
    gap: 18px;
    font-size: 0.78rem;
    color: #8b949e;
    margin: 4px 0 10px 0;
}
.fm-info-row span { color: #cdd9e5; }
</style>
"""


# ---------------------------------------------------------------------------
# Render function
# ---------------------------------------------------------------------------

def render_tab() -> None:
    """
    Render the Feature Map tab.

    Reads SAE paths/layer from session state shared with the
    🔬 SAE Features tab.  All UMAP-specific controls live in this
    tab's sidebar section.
    """
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Read shared paths ─────────────────────────────────────────────────────
    ckpt_path    = st.session_state.get(_K_CKPT, "")
    dataset_root = st.session_state.get(_K_DS, "")
    layer        = int(st.session_state.get(_K_LAYER, 0))

    # ── Sidebar — Feature Map controls ────────────────────────────────────────
    with st.sidebar:
        st.markdown('<p class="fm-section">🌐 FEATURE MAP</p>', unsafe_allow_html=True)

        if not ckpt_path or not dataset_root:
            st.caption("Configure SAE paths in **🔬 SAE Features** first.")
        else:
            st.caption(f"`{Path(ckpt_path).name}` · layer {layer}")

        color_by = st.selectbox(
            "Color by",
            options=["cluster", "max_act", "pct_active", "mean_active"],
            format_func=lambda x: {
                "cluster":     "Cluster (k-means)",
                "max_act":     "Max Activation",
                "pct_active":  "Activation Freq %",
                "mean_active": "Mean Act (active)",
            }[x],
            key=_K_COLOR_BY,
        )

        with st.expander("UMAP / clustering", expanded=False):
            n_clusters = st.slider(
                "K-means clusters k",
                min_value=5, max_value=60, value=20, step=5,
                key=_K_N_CLUSTERS,
            )
            n_neighbors = st.slider(
                "UMAP n_neighbors",
                min_value=5, max_value=50, value=15, step=5,
                key=_K_NEIGHBORS,
            )
            min_dist = st.slider(
                "UMAP min_dist",
                min_value=0.01, max_value=0.5, value=0.1, step=0.01,
                format="%.2f",
                key=_K_MIN_DIST,
            )
            max_tokens_k = st.slider(
                "Scan tokens (k)",
                min_value=0, max_value=500, value=200, step=25,
                key=_K_MAX_TOK,
                help="Thousands of tokens to scan for activation stats. "
                     "0 = skip scan (UMAP only, no activation colouring).",
            )

        with st.expander("Display", expanded=False):
            point_size = st.slider(
                "Point size", min_value=2, max_value=12, value=4, key=_K_PTSIZE
            )
            opacity = st.slider(
                "Opacity", min_value=0.1, max_value=1.0,
                value=0.75, step=0.05, key=_K_OPACITY
            )

        recompute = st.button(
            "🔄 Recompute",
            use_container_width=True,
            help="Force recompute with current UMAP/clustering parameters.",
        )
        if recompute:
            st.session_state[_K_TRIGGER] = st.session_state.get(_K_TRIGGER, 0) + 1

    # ── Main area ─────────────────────────────────────────────────────────────

    if not ckpt_path or not dataset_root:
        st.markdown(
            "## 🌐 Feature Map\n\n"
            "Configure the **SAE Checkpoint** and **Dataset Root** paths in "
            "the **🔬 SAE Features** tab sidebar first, then return here.\n\n"
            "---\n"
            "This tab shows a UMAP projection of all SAE decoder directions so "
            "you can explore semantic clusters, spot dead features, and click "
            "any point to jump to its full inspection view.\n"
        )
        return

    # ── Compute (cached) ──────────────────────────────────────────────────────
    trigger   = st.session_state.get(_K_TRIGGER, 0)
    max_tokens_val = int(max_tokens_k) * 1000

    raw = _load_feature_map(
        ckpt_path    = ckpt_path,
        dataset_root = dataset_root,
        layer        = layer,
        n_pca        = 50,
        umap_neighbors = n_neighbors,
        umap_min_dist  = float(min_dist),
        n_clusters   = n_clusters,
        max_tokens   = max_tokens_val,
        _trigger     = trigger,
    )

    if raw is None:
        st.error("No data returned from compute_feature_map.")
        return

    if "error" in raw:
        st.error(f"**Computation error:** {raw['error']}")
        return

    data = _dict_to_data(raw)

    # ── Info row ──────────────────────────────────────────────────────────────
    method_str = "UMAP" if data.used_umap else "PCA 2D (umap-learn not installed)"
    ev_str = (
        f" · PCA {data.pca_n_components}d ({data.pca_explained_var:.0%} var)"
        if data.pca_explained_var is not None else ""
    )
    n_dead  = int((data.max_act == 0).sum())
    n_alive = data.n_features - n_dead

    st.markdown(
        f'<div class="fm-info-row">'
        f"<div>Method: <span>{method_str}{ev_str}</span></div>"
        f"<div>Features: <span>{data.n_features:,}</span></div>"
        f"<div>Alive: <span>{n_alive:,}</span></div>"
        f"<div>Dead: <span>{n_dead:,}</span></div>"
        f"<div>Tokens scanned: <span>{data.n_tokens_scanned:,}</span></div>"
        f"<div>Clusters: <span>{data.n_clusters}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not data.used_umap:
        st.warning(
            "**umap-learn** not installed — showing PCA 2D projection. "
            "Install with `pip install umap-learn` for the full UMAP view.",
            icon="⚠️",
        )

    # ── Scatter plot ──────────────────────────────────────────────────────────
    selected_idx = st.session_state.get(_K_SELECTED)

    fig = _build_map_figure(
        data         = data,
        color_by     = color_by,
        point_size   = point_size,
        opacity      = opacity,
        selected_idx = selected_idx,
    )

    event = st.plotly_chart(
        fig,
        use_container_width = True,
        on_select           = "rerun",
        key                 = "feature_map_scatter",
        config              = {
            "scrollZoom":     True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "toImageButtonOptions": {"filename": "sae_feature_map"},
        },
    )

    # ── Handle click ──────────────────────────────────────────────────────────
    if event and hasattr(event, "selection") and event.selection.points:
        pt = event.selection.points[0]
        # customdata is the global feature index (stored per trace)
        cdata = pt.get("customdata")
        if cdata is not None:
            clicked_feat = int(cdata) if not isinstance(cdata, (list, tuple)) else int(cdata[0])
            if 0 <= clicked_feat < data.n_features:
                st.session_state[_K_SELECTED] = clicked_feat
                # Sync with SAE Features tab
                st.session_state[_K_FEAT]    = clicked_feat
                st.session_state[_K_FEAT_SL] = clicked_feat

    # ── Selected feature detail panel ─────────────────────────────────────────
    if selected_idx is not None and 0 <= selected_idx < data.n_features:
        st.markdown("---")
        _render_selected_panel(
            selected_idx, data, ckpt_path, dataset_root, layer
        )
    else:
        st.caption(
            "Click any point to inspect that feature.  "
            "Scroll to zoom · drag to pan."
        )
