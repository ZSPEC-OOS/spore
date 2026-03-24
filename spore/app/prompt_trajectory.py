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
from spore.activation_pipeline.reduction import load_projection_model


@dataclass
class TrajectoryResult:
    df: pd.DataFrame
    metrics: pd.DataFrame
    meta: Dict[str, Any]


_EXAMPLE_PROMPTS: dict[str, str] = {
    "Factual recall":   "The capital of France is Paris, and the capital of Germany is",
    "Reasoning":        "If all mammals are warm-blooded, and whales are mammals, then whales are",
    "Negation":         "The dog did not chase the cat because the cat was too fast for the",
    "Arithmetic":       "The result of multiplying 7 by 8 and then subtracting 6 is",
    "Sentence completion": "The quick brown fox jumps over the lazy dog.",
}


def render_prompt_trajectory_viewer(default_root: str | None = None) -> None:
    """Render inference-time trajectory controls + figures."""
    st.markdown("### 🧭 Trajectory Analyzer")
    st.caption(
        "Run one or two prompts through the model and project token trajectories "
        "through a reducer fit on training activations."
    )

    # ── Example prompts ───────────────────────────────────────────────────────
    st.markdown("**Example prompts:**")
    ex_cols = st.columns(len(_EXAMPLE_PROMPTS))
    for col, (label, text) in zip(ex_cols, _EXAMPLE_PROMPTS.items()):
        if col.button(label, use_container_width=True, key=f"_ex_{label[:8]}"):
            st.session_state["traj_prompt"] = text

    st.markdown("---")

    # ── Comparison mode toggle ────────────────────────────────────────────────
    compare_mode = st.toggle(
        "Comparison mode — overlay two prompts",
        value=False,
        key="_traj_compare_mode",
        help="Run two prompts and render their trajectories in the same projection space.",
    )

    # ── Input fields ──────────────────────────────────────────────────────────
    cfg_col, prompt_col = st.columns([1, 2])

    with cfg_col:
        train_root = st.text_input(
            "Training activations root",
            value=default_root or str(Path("activations/run").resolve()),
            help="ActivationCache directory for fitting PCA/UMAP.",
            key="traj_train_root",
        )
        method = st.selectbox("Projection", ["pca", "umap"], index=0, key="traj_method")
        color_by = st.selectbox(
            "Color path by", ["layer", "token_type"], index=0, key="traj_color"
        )
        max_train = st.number_input(
            "Max training samples",
            min_value=100, max_value=100_000, value=5_000, step=100,
            key="_traj_max_train",
        )
        projection_root = st.text_input(
            "Projection artifacts root",
            value=st.session_state.get("_ls_root", str(Path("projections/run").resolve())),
            help="Directory with models/layer_XX_{pca|umap}.pkl",
            key="traj_proj_root",
        )
        projection_layer = st.number_input(
            "Projection layer",
            min_value=0, max_value=95,
            value=int(st.session_state.get("_global_layer_choice", 0)),
            step=1,
            key="traj_proj_layer",
        )

    with prompt_col:
        prompt = st.text_area(
            "Prompt A",
            value=st.session_state.get("traj_prompt", "The quick brown fox jumps over the lazy dog."),
            height=100,
            key="traj_prompt",
        )
        if compare_mode:
            prompt_b = st.text_area(
                "Prompt B",
                value=st.session_state.get("_traj_prompt_b", "The slow gray wolf sleeps under the old oak."),
                height=100,
                key="_traj_prompt_b",
                help="Second prompt for trajectory comparison.",
            )
        else:
            prompt_b = None

    run_btn = st.button(
        "▶ Run Trajectory" if not compare_mode else "▶ Run Comparison",
        use_container_width=True,
        type="primary",
    )

    if not run_btn:
        return

    if not prompt.strip():
        st.warning("Enter a non-empty prompt.")
        return
    if compare_mode and (not prompt_b or not prompt_b.strip()):
        st.warning("Enter a non-empty Prompt B for comparison.")
        return

    # ── Load reducer (shared for both prompts) ─────────────────────────────
    try:
        reducer, reducer_meta = _load_reducer(
            projection_root=projection_root,
            projection_layer=int(projection_layer),
            method=method,
            run_root=train_root,
            max_samples=int(max_train),
        )
    except Exception as exc:
        st.error(f"Failed to load/fit reducer: {exc}")
        return

    source_msg = reducer_meta.get("source", "in-memory fit")

    # ── Compute trajectories ──────────────────────────────────────────────────
    try:
        with st.spinner("Running forward pass on Prompt A …"):
            result_a = _compute_trajectory(prompt=prompt, reducer=reducer, reducer_method=method)
        if compare_mode:
            with st.spinner("Running forward pass on Prompt B …"):
                result_b = _compute_trajectory(prompt=prompt_b, reducer=reducer, reducer_method=method)
    except Exception as exc:
        st.error(f"Trajectory generation failed: {exc}")
        return

    st.success(
        f"Projected {result_a.meta['n_points']} points "
        f"({result_a.meta['seq_len']} tokens × {result_a.meta['n_layers']} layers) "
        f"· reducer: {source_msg}"
    )

    # ── Animated figure ───────────────────────────────────────────────────────
    if compare_mode:
        fig = _build_comparison_figure(result_a.df, result_b.df, color_by=color_by,
                                       label_a="Prompt A", label_b="Prompt B")
    else:
        fig = _build_animated_figure(result_a.df, color_by=color_by)

    st.plotly_chart(fig, use_container_width=True)

    # ── Layer step slider (static view) ──────────────────────────────────────
    max_layer = int(result_a.df["layer"].max())
    layer_step = st.slider("Step through layer", 0, max_layer, value=max_layer,
                           key="_traj_layer_step")
    static_sub = result_a.df[result_a.df["layer"] == layer_step]
    if not static_sub.empty:
        sfig = px.scatter(
            static_sub, x="proj_x", y="proj_y",
            color=color_by, hover_data=["token", "token_idx", "layer"],
            template="plotly_dark",
        )
        sfig.update_traces(marker=dict(size=10))
        sfig.update_layout(
            height=380, margin=dict(l=10, r=10, t=30, b=10),
            title=f"Static view — layer {layer_step}",
        )
        st.plotly_chart(sfig, use_container_width=True)

    # ── Metrics cards ─────────────────────────────────────────────────────────
    st.markdown("#### Metrics — Prompt A")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Mean layer distance", f"{result_a.metrics['layer_distance'].mean():.3f}")
    mc2.metric("Mean curvature",       f"{result_a.metrics['curvature'].mean():.3f}")
    mc3.metric("Mean subspace rank",   f"{result_a.metrics['subspace_rank'].mean():.2f}")

    mfig = go.Figure()
    for col_name in ["layer_distance", "curvature", "subspace_rank"]:
        mfig.add_trace(go.Scatter(
            x=result_a.metrics["layer"], y=result_a.metrics[col_name],
            mode="lines+markers", name=col_name.replace("_", " ").title(),
        ))
    if compare_mode:
        for col_name in ["layer_distance", "curvature", "subspace_rank"]:
            mfig.add_trace(go.Scatter(
                x=result_b.metrics["layer"], y=result_b.metrics[col_name],
                mode="lines+markers", line=dict(dash="dash"),
                name=f"{col_name.replace('_', ' ').title()} (B)",
            ))
    mfig.update_layout(
        template="plotly_dark", height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Layer", yaxis_title="Metric value",
    )
    st.plotly_chart(mfig, use_container_width=True)

    with st.expander("Trajectory data table"):
        st.dataframe(result_a.df, use_container_width=True, height=260)


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


@st.cache_resource(show_spinner=False)
def _load_reducer(
    projection_root: str,
    projection_layer: int,
    method: str,
    run_root: str,
    max_samples: int,
):
    model_path = Path(projection_root) / "models" / f"layer_{projection_layer:02d}_{method}.pkl"
    if model_path.exists():
        reducer = load_projection_model(model_path)
        return reducer, {"source": str(model_path), "fit": "precomputed"}
    return _fit_reducer(run_root, method=method, max_samples=max_samples)


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


def _build_comparison_figure(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    color_by: str = "layer",
    label_a: str = "Prompt A",
    label_b: str = "Prompt B",
) -> go.Figure:
    """Overlay two prompt trajectories in the same projection space."""
    fig = go.Figure()
    for df, label, dash in [(df_a, label_a, "solid"), (df_b, label_b, "dash")]:
        for tok_idx in df["token_idx"].unique():
            sub = df[df["token_idx"] == tok_idx].sort_values("layer")
            tok = sub["token"].iloc[0]
            fig.add_trace(go.Scatter(
                x=sub["proj_x"], y=sub["proj_y"],
                mode="lines+markers",
                name=f"{label}: '{tok}'",
                line=dict(dash=dash, width=2),
                marker=dict(size=6),
                hovertemplate=(
                    f"<b>{label}</b><br>token: {tok}<br>"
                    "layer: %{customdata}<extra></extra>"
                ),
                customdata=sub["layer"].tolist(),
                legendgroup=label,
                showlegend=(tok_idx == 0),
            ))
    fig.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=10, r=10, t=50, b=10),
        title="Trajectory comparison — solid: Prompt A, dashed: Prompt B",
        legend_title="Token / Prompt",
    )
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
