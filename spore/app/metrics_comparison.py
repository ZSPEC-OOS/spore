"""
metrics_comparison.py — Tab 5: Metrics & Comparison.

Provides quantitative views across layers and checkpoints:
  • Layer-wise activation norm statistics (mean / std / max)
  • PCA explained variance per layer (from stored projection summaries)
  • Inter-layer distance profile derived from stored projections
  • Side-by-side checkpoint comparison for any of the above
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=600)
def _load_pca_summary(proj_root: str) -> Optional[pd.DataFrame]:
    """
    Read the PCA summary Parquet/CSV produced by reduce_activations.py.

    The file is expected at ``<proj_root>/pca_summary.parquet`` or
    ``<proj_root>/pca_summary.csv``.  Falls back to scanning per-layer
    parquet files and collecting ``pca_explained_var`` rows if present.
    """
    root = Path(proj_root)
    for name in ("pca_summary.parquet", "pca_summary.csv"):
        p = root / name
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    # Fallback: scan layer files for cumulative_expl_var
    rows = []
    for f in sorted(root.glob("layer_*_pca.parquet")):
        try:
            df = pd.read_parquet(f)
            if "cumulative_expl_var" in df.columns:
                layer_val = df["layer"].iloc[0] if "layer" in df.columns else -1
                rows.append({
                    "layer": layer_val,
                    "cumulative_expl_var": float(df["cumulative_expl_var"].iloc[0]),
                })
        except Exception:
            continue
    for f in sorted(root.glob("layer_*_pca.csv")):
        try:
            df = pd.read_csv(f)
            if "cumulative_expl_var" in df.columns:
                layer_val = df["layer"].iloc[0] if "layer" in df.columns else -1
                rows.append({
                    "layer": layer_val,
                    "cumulative_expl_var": float(df["cumulative_expl_var"].iloc[0]),
                })
        except Exception:
            continue

    return pd.DataFrame(rows) if rows else None


@st.cache_data(show_spinner=False, ttl=600)
def _load_activation_norms(proj_root: str, method: str = "pca") -> Optional[pd.DataFrame]:
    """
    Aggregate per-layer activation norm statistics from projection DataFrames.

    Reads ``layer_XX_{method}.parquet`` files and computes mean/std/max of
    the ``activation_norm`` column if present.
    """
    root = Path(proj_root)
    rows = []

    for f in sorted(root.glob(f"layer_*_{method}.parquet")):
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if "activation_norm" not in df.columns:
            continue
        layer_val = df["layer"].iloc[0] if "layer" in df.columns else -1
        norms = df["activation_norm"].dropna().to_numpy(dtype=np.float32)
        if len(norms) == 0:
            continue
        rows.append({
            "layer": int(layer_val),
            "norm_mean": float(norms.mean()),
            "norm_std":  float(norms.std()),
            "norm_max":  float(norms.max()),
            "norm_p25":  float(np.percentile(norms, 25)),
            "norm_p75":  float(np.percentile(norms, 75)),
            "n_points":  len(norms),
        })

    for f in sorted(root.glob(f"layer_*_{method}.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "activation_norm" not in df.columns:
            continue
        layer_val = df["layer"].iloc[0] if "layer" in df.columns else -1
        norms = df["activation_norm"].dropna().to_numpy(dtype=np.float32)
        if len(norms) == 0:
            continue
        rows.append({
            "layer": int(layer_val),
            "norm_mean": float(norms.mean()),
            "norm_std":  float(norms.std()),
            "norm_max":  float(norms.max()),
            "norm_p25":  float(np.percentile(norms, 25)),
            "norm_p75":  float(np.percentile(norms, 75)),
            "n_points":  len(norms),
        })

    if not rows:
        return None
    df_out = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    return df_out


@st.cache_data(show_spinner=False, ttl=600)
def _load_projection_coords(proj_root: str, method: str = "pca") -> Optional[pd.DataFrame]:
    """
    Load 2-D projection coordinates from all layer files.
    Used to compute inter-layer centroid distances.
    """
    root = Path(proj_root)
    frames = []
    for f in sorted(root.glob(f"layer_*_{method}.parquet")):
        try:
            df = pd.read_parquet(f)
            if "x" in df.columns and "y" in df.columns and "layer" in df.columns:
                frames.append(df[["layer", "x", "y"]])
        except Exception:
            continue
    for f in sorted(root.glob(f"layer_*_{method}.csv")):
        try:
            df = pd.read_csv(f)
            if "x" in df.columns and "y" in df.columns and "layer" in df.columns:
                frames.append(df[["layer", "x", "y"]])
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else None


def _centroid_distances(coords_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Euclidean distance between consecutive layer centroids."""
    layers = sorted(coords_df["layer"].unique())
    centroids = {
        int(l): coords_df[coords_df["layer"] == l][["x", "y"]].mean().to_numpy()
        for l in layers
    }
    rows = []
    for i in range(1, len(layers)):
        prev, curr = layers[i - 1], layers[i]
        dist = float(np.linalg.norm(centroids[curr] - centroids[prev]))
        rows.append({"layer": int(curr), "centroid_distance": dist})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------


def _plot_norm_stats(df: pd.DataFrame, label: str = "", color: str = "#58a6ff") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["layer"], y=df["norm_mean"],
        mode="lines+markers",
        name=f"Mean {label}",
        line=dict(color=color, width=2),
        marker=dict(size=6),
        error_y=dict(type="data", array=df["norm_std"].tolist(), visible=True,
                     color=color, thickness=1.5, width=4),
        hovertemplate="Layer %{x}<br>Mean: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["layer"], y=df["norm_max"],
        mode="lines",
        name=f"Max {label}",
        line=dict(color=color, width=1, dash="dot"),
        hovertemplate="Layer %{x}<br>Max: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=360,
        xaxis_title="Layer",
        yaxis_title="Activation norm",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=10, b=30),
    )
    return fig


def _plot_pca_variance(df: pd.DataFrame, label: str = "", color: str = "#2ea043") -> go.Figure:
    fig = go.Figure()
    if "cumulative_expl_var" in df.columns:
        fig.add_trace(go.Bar(
            x=df["layer"],
            y=(df["cumulative_expl_var"] * 100).round(1),
            name=f"Cumul. explained var {label}",
            marker_color=color,
            hovertemplate="Layer %{x}<br>%{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_dark",
        height=320,
        xaxis_title="Layer",
        yaxis_title="Cumulative explained variance (%)",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def _plot_centroid_dist(df: pd.DataFrame, label: str = "", color: str = "#e3b341") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["layer"], y=df["centroid_distance"],
        mode="lines+markers",
        name=f"Centroid dist {label}",
        line=dict(color=color, width=2),
        marker=dict(size=7),
        hovertemplate="Layer %{x}<br>Dist: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=300,
        xaxis_title="Layer",
        yaxis_title="Centroid distance (2-D projection)",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def render_tab() -> None:
    """Render the Metrics & Comparison tab."""
    st.markdown("### 📊 Metrics & Comparison")
    st.caption(
        "Quantitative layer-wise statistics and side-by-side checkpoint "
        "comparison for activation norms, PCA explained variance, and "
        "projection geometry."
    )

    # ── Data source controls ─────────────────────────────────────────────────
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        proj_root_a = st.text_input(
            "Projections root (A)",
            value=st.session_state.get("_ls_root", str(Path("projections/run").resolve())),
            key="_mc_root_a",
            help="Primary projection directory (reduce_activations.py output).",
        )
        method_a = st.selectbox("Method (A)", ["pca", "umap"], index=0, key="_mc_method_a")
    with dcol2:
        compare_on = st.checkbox(
            "Compare with a second checkpoint",
            value=False,
            key="_mc_compare",
        )
        if compare_on:
            proj_root_b = st.text_input(
                "Projections root (B)",
                value="",
                key="_mc_root_b",
                placeholder="projections/run_b",
                help="Second projection directory for comparison.",
            )
            method_b = st.selectbox("Method (B)", ["pca", "umap"], index=0, key="_mc_method_b")
        else:
            proj_root_b = None
            method_b   = method_a

    if not proj_root_a:
        st.info(
            "Set the **Projections root (A)** above (same directory used in "
            "the **Latent Space Explorer** tab)."
        )
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading metrics …"):
        norms_a  = _load_activation_norms(proj_root_a, method_a)
        pca_a    = _load_pca_summary(proj_root_a)
        coords_a = _load_projection_coords(proj_root_a, method_a)
        cdist_a  = _centroid_distances(coords_a) if coords_a is not None else None

        norms_b = pca_b = cdist_b = None
        if compare_on and proj_root_b:
            norms_b  = _load_activation_norms(proj_root_b, method_b)
            pca_b    = _load_pca_summary(proj_root_b)
            coords_b = _load_projection_coords(proj_root_b, method_b)
            cdist_b  = _centroid_distances(coords_b) if coords_b is not None else None

    # ── Check that at least something is available ─────────────────────────
    has_data = any(x is not None for x in [norms_a, pca_a, cdist_a])
    if not has_data:
        st.warning(
            "No metrics data found in the selected directory.  Make sure you "
            "have run `reduce_activations.py` and that the output directory "
            "contains `layer_XX_pca.parquet` or `layer_XX_pca.csv` files with "
            "an `activation_norm` column."
        )
        return

    compare_label_a = "Checkpoint A" if compare_on else ""
    compare_label_b = "Checkpoint B"

    # ══════════════════════════════════════════════════════════════════════════
    # Section 1 — Activation Norm Statistics
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("#### Activation Norm — Layer Profile")
    st.caption(
        "Mean ± std (error bars) and max (dotted) residual-stream activation "
        "norm at each layer.  Rising norms indicate growing representation "
        "magnitude; sudden spikes can hint at layer-level anomalies."
    )

    if norms_a is not None:
        if compare_on and norms_b is not None:
            ncol1, ncol2 = st.columns(2)
            with ncol1:
                st.markdown(f"**{compare_label_a}**")
                st.plotly_chart(_plot_norm_stats(norms_a, compare_label_a),
                                use_container_width=True)
            with ncol2:
                st.markdown(f"**{compare_label_b}**")
                st.plotly_chart(_plot_norm_stats(norms_b, compare_label_b, color="#f78166"),
                                use_container_width=True)
        else:
            st.plotly_chart(_plot_norm_stats(norms_a), use_container_width=True)

        with st.expander("Raw norm statistics table"):
            if compare_on and norms_b is not None:
                mc1, mc2 = st.columns(2)
                mc1.dataframe(norms_a.round(4), use_container_width=True, hide_index=True)
                mc2.dataframe(norms_b.round(4), use_container_width=True, hide_index=True)
            else:
                st.dataframe(norms_a.round(4), use_container_width=True, hide_index=True)
    else:
        st.info(
            "Activation norm data not available.  Re-run `reduce_activations.py` "
            "to include activation norm columns in the projection output."
        )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # Section 2 — PCA Explained Variance
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("#### PCA Explained Variance per Layer")
    st.caption(
        "Percentage of total variance captured by the first N principal "
        "components at each layer.  Higher values mean the residual stream "
        "is more \"low-dimensional\" at that depth."
    )

    if pca_a is not None:
        if compare_on and pca_b is not None:
            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown(f"**{compare_label_a}**")
                st.plotly_chart(_plot_pca_variance(pca_a, compare_label_a),
                                use_container_width=True)
            with pc2:
                st.markdown(f"**{compare_label_b}**")
                st.plotly_chart(_plot_pca_variance(pca_b, compare_label_b, color="#58a6ff"),
                                use_container_width=True)
        else:
            st.plotly_chart(_plot_pca_variance(pca_a), use_container_width=True)
    else:
        st.info(
            "PCA summary not found.  Run `reduce_activations.py --method pca` "
            "to generate this data."
        )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # Section 3 — Inter-Layer Centroid Distance
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("#### Inter-Layer Centroid Distance (2-D Projection)")
    st.caption(
        "Euclidean distance between consecutive layer centroids in the "
        "2-D projection space.  Large jumps indicate layers where the "
        "representation undergoes significant geometric transformation."
    )

    if cdist_a is not None and not cdist_a.empty:
        if compare_on and cdist_b is not None and not cdist_b.empty:
            dd1, dd2 = st.columns(2)
            with dd1:
                st.markdown(f"**{compare_label_a}**")
                st.plotly_chart(_plot_centroid_dist(cdist_a, compare_label_a),
                                use_container_width=True)
            with dd2:
                st.markdown(f"**{compare_label_b}**")
                st.plotly_chart(_plot_centroid_dist(cdist_b, compare_label_b, color="#da3633"),
                                use_container_width=True)
        else:
            st.plotly_chart(_plot_centroid_dist(cdist_a), use_container_width=True)
    else:
        st.info(
            "Projection coordinates with `x`, `y`, `layer` columns not found. "
            "Re-run `reduce_activations.py` to generate them."
        )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # Section 4 — Summary table
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("#### Summary")

    _summary_parts = []
    if norms_a is not None:
        _summary_parts.append(
            f"- Layers with norm data: **{len(norms_a)}**  "
            f"(mean norm range: {norms_a['norm_mean'].min():.3f} – "
            f"{norms_a['norm_mean'].max():.3f})"
        )
    if pca_a is not None and "cumulative_expl_var" in pca_a.columns:
        _avg_var = pca_a["cumulative_expl_var"].mean() * 100
        _summary_parts.append(
            f"- Average PCA explained variance: **{_avg_var:.1f}%**"
        )
    if cdist_a is not None and not cdist_a.empty:
        _max_jump_layer = int(cdist_a.loc[cdist_a["centroid_distance"].idxmax(), "layer"])
        _summary_parts.append(
            f"- Largest centroid jump at layer **{_max_jump_layer}** "
            f"({cdist_a['centroid_distance'].max():.4f})"
        )

    if _summary_parts:
        st.markdown("\n".join(_summary_parts))
    else:
        st.caption("No summary data available.")
