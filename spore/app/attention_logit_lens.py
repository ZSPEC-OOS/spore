from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@st.cache_resource(show_spinner="Loading TransformerLens model …")
def load_model(model_name: str):
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(model_name)
    model.eval()
    return model


@dataclass
class LogitLensRow:
    layer: int
    token: str
    logit: float
    rank: int


def render_tab(
    model_name: str = "gpt2",
    layer_default: int | None = None,
    top_k_default: int = 8,
) -> None:
    st.markdown("### 👁️ Supporting Maps")
    st.caption(
        "Prompt-level attention rollout heatmaps and intermediate residual "
        "projections into vocabulary space (logit lens)."
    )

    # ── Shared prompt / model controls ───────────────────────────────────────
    ctrl_col, prompt_col = st.columns([1, 2])
    with ctrl_col:
        top_k = st.slider("Top-k tokens", min_value=3, max_value=20, value=top_k_default,
                          key="_sm_topk")
        token_mode = st.radio("Token position", ["Last token", "Custom index"], index=0,
                              key="_sm_tok_mode")
    with prompt_col:
        prompt = st.text_area(
            "Prompt",
            value=st.session_state.get("_attn_prompt",
                                       "The quick brown fox jumps over the lazy dog."),
            height=100,
            key="_attn_prompt",
        )

    if not st.button("Run Analysis", type="primary", use_container_width=True,
                     key="_sm_run_btn"):
        return

    if not prompt.strip():
        st.warning("Please enter a non-empty prompt.")
        return

    try:
        model = load_model(model_name)
    except Exception as exc:
        st.error(f"Could not load model `{model_name}`: {exc}")
        return

    toks = model.to_tokens(prompt)
    token_text = [model.to_string([tid]) for tid in toks[0].tolist()]
    seq_len = len(token_text)

    if token_mode == "Custom index":
        tok_idx = st.number_input(
            "Token index",
            min_value=0, max_value=max(seq_len - 1, 0),
            value=max(seq_len - 1, 0), step=1,
            key="_attn_tok_idx",
        )
        token_index = int(tok_idx)
    else:
        token_index = seq_len - 1

    try:
        _, cache = model.run_with_cache(
            toks,
            names_filter=lambda n: (
                n.endswith("hook_attn_scores") or n.endswith("hook_resid_post")
            ),
            return_type=None,
        )
    except Exception as exc:
        st.error(f"Model forward pass failed: {exc}")
        return

    attn_probs = _extract_attention_probs(cache, model.cfg.n_layers)
    if attn_probs is None:
        st.warning("No attention probabilities found in cache.")
        return

    layer_count, head_count, _, _ = attn_probs.shape

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    sub_attn, sub_lens = st.tabs(["🔥 Attention Rollout Heatmap", "🔭 Logit Lens"])

    # ── Sub-tab 1: Attention Rollout Heatmap ──────────────────────────────────
    with sub_attn:
        st.caption(
            "Head-level attention matrix and cumulative rollout up to a chosen layer."
        )

        acol1, acol2 = st.columns(2)
        with acol1:
            layer_sel = st.slider(
                "Layer",
                0, layer_count - 1,
                value=min(
                    layer_default if layer_default is not None else layer_count - 1,
                    layer_count - 1,
                ),
                key="_attn_layer_slider",
            )
        with acol2:
            head_sel = st.slider("Head", 0, head_count - 1, value=0,
                                 key="_attn_head_slider")

        # Axis labels: truncate long tokens
        _labels = [t[:12] for t in token_text]

        hcol1, hcol2 = st.columns(2)
        with hcol1:
            head_mat = attn_probs[layer_sel, head_sel]
            fig_head = px.imshow(
                head_mat,
                color_continuous_scale="Viridis",
                labels={"x": "Source", "y": "Destination", "color": "Attention"},
                x=_labels, y=_labels,
                title=f"Layer {layer_sel} · Head {head_sel}",
                aspect="auto",
            )
            fig_head.update_layout(template="plotly_dark", height=440)
            st.plotly_chart(fig_head, use_container_width=True)

        with hcol2:
            rollout = _attention_rollout(attn_probs)
            rollout_mat = rollout[layer_sel]
            fig_roll = px.imshow(
                rollout_mat,
                color_continuous_scale="Plasma",
                labels={"x": "Source", "y": "Destination", "color": "Rollout"},
                x=_labels, y=_labels,
                title=f"Attention rollout — layer {layer_sel}",
                aspect="auto",
            )
            fig_roll.update_layout(template="plotly_dark", height=440)
            st.plotly_chart(fig_roll, use_container_width=True)

        with st.expander("Token legend"):
            st.dataframe(
                pd.DataFrame({"index": list(range(seq_len)), "token": token_text}),
                use_container_width=True, hide_index=True,
            )

    # ── Sub-tab 2: Logit Lens ─────────────────────────────────────────────────
    with sub_lens:
        st.caption(
            "Per-layer top-k vocabulary predictions obtained by applying the "
            "unembedding matrix to each residual-stream state."
        )

        try:
            lens_df = _compute_logit_lens_table(
                model, cache, token_index=token_index, top_k=top_k
            )
        except Exception as exc:
            st.error(f"Logit lens computation failed: {exc}")
            return

        if lens_df.empty:
            st.info("No logit-lens rows available.")
            return

        layer_view = st.selectbox(
            "View mode",
            ["Table — all layers", "Bar chart — single layer"],
            index=0,
            key="_sm_lens_view",
        )

        if layer_view.startswith("Table"):
            st.dataframe(lens_df, use_container_width=True, height=460, hide_index=True)
        else:
            layer_pick = st.slider(
                "Layer",
                0, int(lens_df["layer"].max()),
                value=int(lens_df["layer"].max()),
                key="_sm_lens_layer",
            )
            sub = lens_df[lens_df["layer"] == layer_pick].sort_values(
                "logit", ascending=False
            )
            bar_fig = go.Figure(
                go.Bar(
                    x=sub["token"], y=sub["logit"],
                    marker_color="#58a6ff",
                    hovertemplate="%{x}: %{y:.3f}<extra></extra>",
                )
            )
            bar_fig.update_layout(
                template="plotly_dark", height=380,
                xaxis_title="Token", yaxis_title="Projected logit",
                title=f"Top-{top_k} predictions · layer {layer_pick} · "
                      f"token index {token_index}",
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        st.caption(
            "Logit lens: `unembed(ln_final(resid_post[layer, token_idx]))` "
            "via TransformerLens weights."
        )


def _extract_attention_probs(cache: Any, n_layers: int) -> np.ndarray | None:
    mats: List[np.ndarray] = []
    for layer in range(n_layers):
        probs_name = f"blocks.{layer}.attn.hook_pattern"
        score_name = f"blocks.{layer}.attn.hook_attn_scores"
        if probs_name in cache:
            pat = cache[probs_name][0].detach().cpu().float().numpy()  # [head, dest, src]
            mats.append(pat)
        elif score_name in cache:
            scores = cache[score_name][0].detach().cpu().float().numpy()
            scores = scores - scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            mats.append(probs)
        else:
            return None
    return np.stack(mats, axis=0)


def _attention_rollout(attn_probs: np.ndarray) -> np.ndarray:
    """
    Compute cumulative rollout per layer.
    attn_probs: [layer, head, dest, src]
    returns: [layer, dest, src]
    """
    n_layers, _, seq_len, _ = attn_probs.shape
    eye = np.eye(seq_len, dtype=np.float32)

    rollout = np.zeros((n_layers, seq_len, seq_len), dtype=np.float32)
    running = eye.copy()
    for layer in range(n_layers):
        a = attn_probs[layer].mean(axis=0)
        a_tilde = 0.5 * a + 0.5 * eye
        a_tilde /= np.clip(a_tilde.sum(axis=-1, keepdims=True), 1e-8, None)
        running = a_tilde @ running
        rollout[layer] = running
    return rollout


def _compute_logit_lens_table(model: Any, cache: Any, token_index: int, top_k: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for layer in range(model.cfg.n_layers):
        resid_name = f"blocks.{layer}.hook_resid_post"
        if resid_name not in cache:
            continue
        resid = cache[resid_name][0, token_index]
        normed = model.ln_final(resid)
        logits = model.unembed(normed)
        vals, idxs = logits.topk(top_k)
        vals = vals.detach().cpu().float().tolist()
        idxs = idxs.detach().cpu().tolist()
        toks = [model.to_string([int(i)]) for i in idxs]
        for rank, (tok, val) in enumerate(zip(toks, vals), start=1):
            rows.append({"layer": layer, "rank": rank, "token": repr(tok), "logit": float(val)})
    return pd.DataFrame(rows)
