from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from spore.activation_pipeline.io import ActivationCache


@dataclass
class TrajectoryResult:
    df: pd.DataFrame
    metrics: pd.DataFrame
    meta: Dict[str, Any]


def render_prompt_trajectory_viewer(default_root: str | None = None) -> None:
    """Render inference-time trajectory controls + figures."""
    st.markdown("---")
    st.markdown("### 🧭 Prompt Trajectory Viewer")
    st.caption(
        "Run a prompt through the model layer-by-layer, then project token trajectories "
        "through a reducer fit on training activations."
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        train_root = st.text_input(
            "Training activations root",
            value=default_root or str(Path("activations/run").resolve()),
            help="ActivationCache directory used to fit PCA/UMAP for trajectory projection.",
            key="traj_train_root",
        )
        prompt = st.text_area(
            "Prompt",
            value="The quick brown fox jumps over the lazy dog.",
            height=90,
            key="traj_prompt",
        )
    with c2:
        method = st.selectbox("Projection", ["pca", "umap"], index=0, key="traj_method")
        color_by = st.selectbox("Color path by", ["layer", "token_type"], index=0, key="traj_color")
        max_train = st.number_input("Max training samples", min_value=100, max_value=100_000, value=5_000, step=100)

    if not st.button("Generate trajectory", use_container_width=True, type="primary"):
        return

    if not prompt.strip():
        st.warning("Enter a non-empty prompt.")
        return

    try:
        reducer, reducer_meta = _fit_reducer(train_root, method=method, max_samples=int(max_train))
        result = _compute_trajectory(prompt=prompt, reducer=reducer, reducer_method=method)
    except Exception as exc:  # surfaced to UI
        st.error(f"Trajectory generation failed: {exc}")
        return

    st.success(
        f"Projected {result.meta['n_points']} points "
        f"({result.meta['seq_len']} tokens × {result.meta['n_layers']} layers)"
    )

    fig = _build_animated_figure(result.df, color_by=color_by)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Metrics")
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.metric("Mean layer distance", f"{result.metrics['layer_distance'].mean():.3f}")
    with mcol2:
        st.metric("Mean curvature", f"{result.metrics['curvature'].mean():.3f}")
    with mcol3:
        st.metric("Mean subspace rank", f"{result.metrics['subspace_rank'].mean():.2f}")

    mfig = go.Figure()
    for col in ["layer_distance", "curvature", "subspace_rank"]:
        mfig.add_trace(go.Scatter(
            x=result.metrics["layer"],
            y=result.metrics[col],
            mode="lines+markers",
            name=col.replace("_", " ").title(),
        ))
    mfig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Layer",
        yaxis_title="Metric value",
    )
    st.plotly_chart(mfig, use_container_width=True)

    with st.expander("Trajectory table"):
        st.dataframe(result.df, use_container_width=True, height=280)


@st.cache_resource(show_spinner=False)
def _fit_reducer(
    run_root: str,
    method: str = "pca",
    max_samples: int = 5_000,
):
    run = ActivationCache.load(run_root, device="cpu")
    arrays = []
    for layer in run.layers:
        acts = run.activations[layer]
        if acts.ndim != 2:
            raise ValueError(
                "Training run must contain pooled 2D activations per layer "
                f"(got shape={tuple(acts.shape)} at layer {layer})."
            )
        arrays.append(acts.float().numpy())

    X = np.concatenate(arrays, axis=0)
    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=42)
        reducer.fit(X)
    else:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        reducer.fit(X)

    return reducer, {"n_train": len(X), "d_model": X.shape[1], "layers": run.layers}


def _compute_trajectory(prompt: str, reducer: Any, reducer_method: str) -> TrajectoryResult:
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise ImportError("transformer_lens is required for trajectory inference.") from exc

    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()

    toks = model.to_tokens(prompt)
    tok_ids = toks[0].tolist()
    token_text = [model.to_string([tid]) for tid in tok_ids]

    _, cache = model.run_with_cache(
        toks,
        names_filter=lambda n: n.endswith("hook_resid_post"),
        return_type=None,
    )

    rows: List[Dict[str, Any]] = []
    vectors: List[np.ndarray] = []
    n_layers = model.cfg.n_layers
    seq_len = toks.shape[1]

    for layer in range(n_layers):
        acts = cache[f"blocks.{layer}.hook_resid_post"][0].detach().cpu().numpy()
        for t in range(seq_len):
            token = token_text[t]
            rows.append(
                {
                    "layer": layer,
                    "token_idx": t,
                    "token": token,
                    "token_type": _token_type(token),
                }
            )
            vectors.append(acts[t])

    X = np.stack(vectors, axis=0)
    proj = reducer.transform(X)

    df = pd.DataFrame(rows)
    df["proj_x"] = proj[:, 0]
    df["proj_y"] = proj[:, 1]

    metrics = _compute_metrics(df, X)

    return TrajectoryResult(
        df=df,
        metrics=metrics,
        meta={"seq_len": seq_len, "n_layers": n_layers, "n_points": len(df), "method": reducer_method},
    )


def _compute_metrics(df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    """Compute per-layer metrics over trajectory tensors."""
    seq_len = int(df["token_idx"].max()) + 1
    n_layers = int(df["layer"].max()) + 1
    d_model = X.shape[1]
    tensor = X.reshape(n_layers, seq_len, d_model)

    rows = []
    for layer in range(n_layers):
        prev = tensor[layer - 1] if layer > 0 else tensor[layer]
        dist = np.linalg.norm(tensor[layer] - prev, axis=-1).mean()

        if layer >= 2:
            second = tensor[layer] - 2 * tensor[layer - 1] + tensor[layer - 2]
            curvature = np.linalg.norm(second, axis=-1).mean()
        else:
            curvature = 0.0

        rank = np.linalg.matrix_rank(tensor[layer])
        rows.append(
            {
                "layer": layer,
                "layer_distance": float(dist),
                "curvature": float(curvature),
                "subspace_rank": float(rank),
            }
        )

    return pd.DataFrame(rows)


def _build_animated_figure(df: pd.DataFrame, color_by: str = "layer") -> go.Figure:
    max_layer = int(df["layer"].max())

    # Build cumulative frames so each step reveals more of each token path.
    frames = []
    for frame_layer in range(max_layer + 1):
        sub = df[df["layer"] <= frame_layer].copy()
        sub["frame_layer"] = frame_layer
        frames.append(sub)
    anim_df = pd.concat(frames, ignore_index=True)

    fig = px.line(
        anim_df,
        x="proj_x",
        y="proj_y",
        animation_frame="frame_layer",
        line_group="token_idx",
        color=color_by,
        hover_data=["token", "token_idx", "layer", "token_type"],
        markers=True,
        template="plotly_dark",
    )
    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Prompt trajectory through layer-wise residual stream",
        legend_title=color_by.replace("_", " ").title(),
    )
    fig.update_traces(line=dict(width=2), marker=dict(size=7))
    return fig


def _token_type(tok: str) -> str:
    t = tok.strip()
    if not t:
        return "whitespace"
    if t.isalpha():
        return "alpha"
    if t.isnumeric():
        return "numeric"
    if any(ch.isalpha() for ch in t) and any(ch.isnumeric() for ch in t):
        return "alnum"
    return "punct/mixed"
