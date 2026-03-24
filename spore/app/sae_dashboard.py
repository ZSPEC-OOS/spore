"""
sae_dashboard.py — Tab 2: SAE Feature Dashboard.

Call ``render_tab()`` from the Streamlit entry point.

Layout
------
  [Sidebar]
    📂 SAE Data         — checkpoint path, dataset root, layer selector
    🔎 Feature          — feature index (number + slider), top-N, threshold,
                          context window
    🔭 Logit Lens       — optional model load for W_dec @ W_U projection

  [Main area, given valid paths]
    Feature header row  — 4 metric columns (index, max act, % active, mean)
    ── Row 1: Activation Histogram | Logit Effects (if enabled)
    ── Row 2: Top-N Activating Examples (HTML table with highlighted tokens)
"""

from __future__ import annotations

import html
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from spore.activation_pipeline.sae_dataset import SnippetResult
from spore.app.sae_feature import FeatureAnalyzer, HistogramData, LogitEffects

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session-state keys
# ---------------------------------------------------------------------------
_K_CKPT      = "_sae_ckpt"
_K_DS        = "_sae_ds"
_K_LAYER     = "_sae_layer"
_K_FEAT      = "_sae_feat"
_K_FEAT_SL   = "_sae_feat_sl"
_K_TOPK      = "_sae_topk"
_K_THRESH    = "_sae_thresh"
_K_CTX       = "_sae_ctx"
_K_LOGIT     = "_sae_logit"
_K_MODEL     = "_sae_model"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* ── Context highlighting ─────────────────────────────────────── */
.ctx-before, .ctx-after {
    color: #cdd9e5;
    font-family: "SF Mono", "Fira Code", monospace;
    font-size: 0.82rem;
    white-space: pre-wrap;
}
.ctx-token {
    background: #fff3cd;
    color: #1a1a1a;
    padding: 1px 4px;
    border-radius: 4px;
    font-family: "SF Mono", "Fira Code", monospace;
    font-size: 0.82rem;
    font-weight: 700;
}

/* ── Top-examples table ───────────────────────────────────────── */
.feat-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin-top: 0.5rem;
}
.feat-table th {
    background: #21262d;
    color: #8b949e;
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 6px 10px;
    border-bottom: 1px solid #30363d;
    text-align: left;
}
.feat-table td {
    padding: 5px 10px;
    border-bottom: 1px solid #21262d;
    vertical-align: middle;
    color: #cdd9e5;
}
.feat-table tr:hover td { background: #1c2128; }
.score-badge {
    background: #1f4a3c;
    color: #3fb950;
    font-family: "SF Mono", monospace;
    font-size: 0.78rem;
    padding: 1px 6px;
    border-radius: 10px;
}
.label-badge {
    background: #1d2d3e;
    color: #58a6ff;
    font-size: 0.72rem;
    padding: 1px 6px;
    border-radius: 10px;
}
.rank-num {
    color: #6e7681;
    font-family: monospace;
    font-size: 0.78rem;
}

/* ── Dead-feature warning ─────────────────────────────────────── */
.dead-feature-box {
    background: #2d1f0e;
    border: 1px solid #5a3a1a;
    border-radius: 6px;
    padding: 12px 16px;
    color: #e3b341;
    font-size: 0.9rem;
}

/* ── Sidebar section title ────────────────────────────────────── */
.sae-section {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #8b949e;
    text-transform: uppercase;
    margin: 12px 0 4px 0;
    padding-top: 8px;
    border-top: 1px solid #21262d;
}
</style>
"""

# ---------------------------------------------------------------------------
# Cached resource + data loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading SAE checkpoint …")
def _load_analyzer(ckpt_path: str, dataset_root: str, layer: int) -> FeatureAnalyzer:
    """Load and cache the FeatureAnalyzer (expensive: loads SAE + dataset)."""
    return FeatureAnalyzer.from_checkpoint(
        ckpt_path=ckpt_path,
        dataset_root=dataset_root,
        layer=layer,
        device="cpu",
    )


@st.cache_data(show_spinner=False)
def _read_dataset_meta(dataset_root: str) -> Optional[dict]:
    """Read just metadata.json from the dataset root (fast, no tensor I/O)."""
    p = Path(dataset_root) / "metadata.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _read_ckpt_meta(ckpt_path: str) -> Optional[dict]:
    """Read just meta.json from the checkpoint (fast)."""
    p = Path(ckpt_path) / "meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=600)
def _compute_histogram(
    ckpt_path:    str,
    dataset_root: str,
    layer:        int,
    feat_idx:     int,
    max_tokens:   int = 200_000,
) -> Optional[dict]:
    """
    Stream shards through SAE encoder, compute activation histogram for
    feat_idx.  Returns a plain dict (serialisable) or None.
    """
    analyzer = _load_analyzer(ckpt_path, dataset_root, layer)
    hist = analyzer.feature_histogram_data(feat_idx, max_tokens=max_tokens)
    return {
        "values":       hist.values.tolist(),   # list[float] — only positive
        "n_total":      hist.n_total,
        "n_active":     hist.n_active,
        "pct_active":   hist.pct_active,
        "max_act":      hist.max_act,
        "mean_active":  hist.mean_active,
        "mean_all":     hist.mean_all,
        "is_dead":      hist.is_dead,
    }


@st.cache_data(show_spinner=False, ttl=600)
def _compute_top_k(
    ckpt_path:      str,
    dataset_root:   str,
    layer:          int,
    feat_idx:       int,
    k:              int,
    threshold:      float,
    context_tokens: int,
) -> List[dict]:
    """
    Stream shards through SAE encoder, return top-k snippets as plain dicts.
    """
    analyzer  = _load_analyzer(ckpt_path, dataset_root, layer)
    snippets  = analyzer.top_k_by_activation(
        feat_idx=feat_idx, k=k, threshold=threshold,
        context_tokens=context_tokens,
    )
    return [asdict(s) for s in snippets]


@st.cache_data(show_spinner=False, ttl=3600)
def _compute_logit_effects(
    ckpt_path:  str,
    dataset_root: str,
    layer:      int,
    feat_idx:   int,
    model_name: str,
    top_k:      int = 10,
) -> Optional[dict]:
    """
    Compute W_dec[feat_idx] @ W_U for logit-lens view.
    Returns a plain dict or None if model cannot be loaded.
    """
    analyzer = _load_analyzer(ckpt_path, dataset_root, layer)
    effects  = analyzer.logit_effects(feat_idx, model_name=model_name, top_k=top_k)
    if effects is None:
        return None
    return {
        "promoted":   effects.promoted,
        "suppressed": effects.suppressed,
        "feat_idx":   effects.feat_idx,
        "model_name": effects.model_name,
    }


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

def _build_histogram_fig(hist: dict, feat_idx: int) -> go.Figure:
    """
    Build an activation-distribution histogram for one feature.
    Positive activations only (SAE ReLU → most values are 0, excluded).
    """
    values = np.array(hist["values"], dtype=np.float32)

    if len(values) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Feature never activated",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#8b949e"),
        )
    else:
        fig = go.Figure(
            go.Histogram(
                x          = values,
                nbinsx     = 60,
                marker     = dict(
                    color = "#238636",
                    line  = dict(color="#2ea043", width=0.5),
                ),
                hovertemplate = "act ∈ [%{x:.3f}]<br>count: %{y}<extra></extra>",
            )
        )
        # Mark mean
        fig.add_vline(
            x          = float(hist["mean_active"]),
            line_dash  = "dash",
            line_color = "#e3b341",
            annotation_text  = f"μ={hist['mean_active']:.3f}",
            annotation_font  = dict(color="#e3b341", size=11),
            annotation_position = "top right",
        )
        # Mark max
        fig.add_vline(
            x          = float(hist["max_act"]),
            line_dash  = "dot",
            line_color = "#f78166",
            annotation_text  = f"max={hist['max_act']:.3f}",
            annotation_font  = dict(color="#f78166", size=11),
            annotation_position = "top left",
        )

    fig.update_layout(
        title       = dict(
            text    = f"Feature {feat_idx} — positive activations "
                      f"({hist['n_active']:,} / {hist['n_total']:,} tokens, "
                      f"{hist['pct_active']:.2f}% active)",
            font    = dict(size=12, color="#cdd9e5"),
            x       = 0,
        ),
        xaxis_title = "Activation value",
        yaxis_title = "Count",
        paper_bgcolor = "#0d1117",
        plot_bgcolor  = "#161b22",
        font          = dict(color="#8b949e"),
        xaxis         = dict(gridcolor="#21262d", zerolinecolor="#21262d"),
        yaxis         = dict(gridcolor="#21262d"),
        margin        = dict(l=50, r=20, t=50, b=40),
        bargap        = 0.05,
        height        = 320,
    )
    return fig


def _build_logit_fig(effects: dict) -> go.Figure:
    """
    Two-panel horizontal bar chart: top promoted (green) + suppressed (red).
    """
    promoted   = effects["promoted"]   # [(tok, val), …]
    suppressed = effects["suppressed"] # [(tok, val), …]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Promoted tokens", "Suppressed tokens"],
        vertical_spacing=0.18,
    )

    # ── Promoted ───────────────────────────────────────────────────
    p_tokens = [repr(t) for t, _ in reversed(promoted)]
    p_vals   = [v for _, v in reversed(promoted)]
    fig.add_trace(
        go.Bar(
            x            = p_vals,
            y            = p_tokens,
            orientation  = "h",
            marker_color = "#2ea043",
            hovertemplate = "%{y}: %{x:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # ── Suppressed ─────────────────────────────────────────────────
    s_tokens = [repr(t) for t, _ in suppressed]
    s_vals   = [v for _, v in suppressed]     # already positive (we negated)
    fig.add_trace(
        go.Bar(
            x            = s_vals,
            y            = s_tokens,
            orientation  = "h",
            marker_color = "#da3633",
            hovertemplate = "%{y}: -%{x:.4f}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        paper_bgcolor = "#0d1117",
        plot_bgcolor  = "#161b22",
        font          = dict(color="#8b949e", size=11),
        showlegend    = False,
        height        = 420,
        margin        = dict(l=100, r=20, t=60, b=40),
    )
    for i in (1, 2):
        fig.update_xaxes(gridcolor="#21262d", row=i, col=1)
        fig.update_yaxes(gridcolor="#21262d", row=i, col=1)

    fig.update_annotations(font_color="#8b949e", font_size=11)
    return fig


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _snippet_context_html(snippet: dict) -> str:
    """
    Render context_before + highlighted token + context_after as HTML.
    All strings are HTML-escaped before insertion.
    """
    before = html.escape(snippet.get("context_before") or "")
    tok    = html.escape(snippet.get("token_str")      or "?")
    after  = html.escape(snippet.get("context_after")  or "")
    return (
        f'<span class="ctx-before">{before}</span>'
        f'<span class="ctx-token">{tok}</span>'
        f'<span class="ctx-after">{after}</span>'
    )


def _render_examples_html(snippets: List[dict]) -> str:
    """
    Build an HTML table of top-activating examples with highlighted tokens.
    """
    rows = []
    for s in snippets:
        ctx   = _snippet_context_html(s)
        score = s.get("score", 0.0)
        rank  = s.get("rank", "?")
        label = s.get("label") or ""
        sent  = s.get("sentence_idx", "?")

        label_cell = (
            f'<span class="label-badge">{html.escape(str(label))}</span>'
            if label else '<span style="color:#6e7681">—</span>'
        )

        rows.append(
            f"<tr>"
            f'<td><span class="rank-num">#{rank}</span></td>'
            f'<td><span class="score-badge">{score:.3f}</span></td>'
            f"<td>{ctx}</td>"
            f"<td>{label_cell}</td>"
            f'<td><span style="color:#6e7681;font-size:0.75rem">{sent}</span></td>'
            f"</tr>"
        )

    header = (
        "<tr>"
        "<th>#</th>"
        "<th>Score</th>"
        "<th>Context</th>"
        "<th>Label</th>"
        "<th>Sent</th>"
        "</tr>"
    )

    return (
        '<table class="feat-table">'
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


# ---------------------------------------------------------------------------
# Sidebar helpers
# ---------------------------------------------------------------------------

def _sidebar_section(title: str) -> None:
    st.sidebar.markdown(
        f'<p class="sae-section">{title}</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_tab() -> None:
    """
    Render the SAE Feature Dashboard tab.
    Called from the Streamlit entry point inside a ``with tab_sae:`` block.
    """
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        _sidebar_section("📂 SAE DATA")

        ckpt_path = st.text_input(
            "SAE Checkpoint",
            value=st.session_state.get(_K_CKPT, ""),
            key=_K_CKPT,
            help="Path to a step_XXXXXXX/ directory (or the `latest/` symlink) "
                 "produced by train_sae.py",
            placeholder="sae_checkpoints/run/latest",
        )

        dataset_root = st.text_input(
            "Dataset Root",
            value=st.session_state.get(_K_DS, ""),
            key=_K_DS,
            help="Root directory of a SAEDataset (built with build_sae_dataset.py)",
            placeholder="sae_data/gpt2_l6",
        )

        # ── Read metadata without loading tensors ─────────────────────
        ds_meta   = _read_dataset_meta(dataset_root)  if dataset_root  else None
        ckpt_meta = _read_ckpt_meta(ckpt_path)        if ckpt_path     else None

        available_layers = ds_meta["layers"] if ds_meta else [0]
        layer = st.selectbox(
            "Layer",
            options=available_layers,
            index=0,
            key=_K_LAYER,
        )

        # Show quick info if metadata loaded
        if ds_meta:
            n_tokens = ds_meta.get("total_tokens", "?")
            n_tok_fmt = f"{n_tokens:,}" if isinstance(n_tokens, int) else str(n_tokens)
            st.caption(
                f"model: `{ds_meta.get('model_name','?')}` · "
                f"d_model: `{ds_meta.get('d_model','?')}` · "
                f"tokens: `{n_tok_fmt}`"
            )
        if ckpt_meta:
            cfg_d = ckpt_meta.get("cfg", {})
            st.caption(
                f"SAE: `{cfg_d.get('n_features','?')}` features · "
                f"`{cfg_d.get('activation','?')}` · "
                f"step `{ckpt_meta.get('step','?')}`"
            )

        # ── Feature controls (need n_features) ────────────────────────
        _sidebar_section("🔎 FEATURE")

        n_features = (
            ckpt_meta["cfg"]["n_features"]
            if ckpt_meta and "cfg" in ckpt_meta
            else 8192
        )

        # Number input + slider, kept in sync via session state
        feat_input = st.number_input(
            "Feature Index",
            min_value=0,
            max_value=n_features - 1,
            value=int(st.session_state.get(_K_FEAT, 0)),
            step=1,
            key=_K_FEAT,
            help=f"0 – {n_features - 1}",
        )

        feat_slider = st.slider(
            "Browse",
            min_value=0,
            max_value=n_features - 1,
            value=int(feat_input),
            key=_K_FEAT_SL,
            label_visibility="collapsed",
        )

        # Slider takes precedence if moved; number input if typed
        feat_idx = (
            feat_slider
            if feat_slider != int(st.session_state.get(_K_FEAT, 0))
            else int(feat_input)
        )

        top_k = st.slider(
            "Top N Examples",
            min_value=5,
            max_value=50,
            value=int(st.session_state.get(_K_TOPK, 20)),
            step=5,
            key=_K_TOPK,
        )

        threshold = st.slider(
            "Activation Threshold",
            min_value=0.0,
            max_value=5.0,
            value=float(st.session_state.get(_K_THRESH, 0.0)),
            step=0.05,
            key=_K_THRESH,
            help="Only show tokens with h[feat] > threshold",
        )

        context_tokens = st.slider(
            "Context Window (tokens)",
            min_value=2,
            max_value=20,
            value=int(st.session_state.get(_K_CTX, 8)),
            step=2,
            key=_K_CTX,
        )

        # ── Logit lens ────────────────────────────────────────────────
        _sidebar_section("🔭 LOGIT LENS")

        use_logit = st.checkbox(
            "Enable logit lens",
            value=bool(st.session_state.get(_K_LOGIT, False)),
            key=_K_LOGIT,
            help="Load the transformer model to compute W_dec @ W_U, showing "
                 "which output tokens this feature promotes or suppresses.",
        )
        model_name = st.text_input(
            "Model Name",
            value=st.session_state.get(_K_MODEL, "gpt2"),
            key=_K_MODEL,
            disabled=not use_logit,
            help="TransformerLens model name (e.g. gpt2, pythia-160m)",
        )

    # ── Main area ─────────────────────────────────────────────────────────────

    if not ckpt_path or not dataset_root:
        st.markdown(
            "## 🔬 SAE Feature Dashboard\n\n"
            "Set the **SAE Checkpoint** and **Dataset Root** paths in the sidebar "
            "to begin exploring features.\n\n"
            "---\n"
            "**Workflow:**\n"
            "1. Build a dataset: `python build_sae_dataset.py …`\n"
            "2. Train an SAE:    `python train_sae.py …`\n"
            "3. Enter the paths above and pick a feature to inspect.\n"
        )
        return

    # ── Load analyzer ─────────────────────────────────────────────────────────
    try:
        _load_analyzer(ckpt_path, dataset_root, layer)   # warms the cache
    except Exception as exc:
        st.error(
            f"**Failed to load SAE.**\n\n"
            f"Check that `{ckpt_path}` is a valid step_XXXXXXX/ checkpoint "
            f"and `{dataset_root}` is a valid SAEDataset root.\n\n"
            f"Error: `{exc}`"
        )
        return

    # ── Compute feature data ───────────────────────────────────────────────────
    with st.spinner(f"Computing feature {feat_idx} / {n_features - 1} …"):
        hist_d   = _compute_histogram(ckpt_path, dataset_root, layer, feat_idx)
        snippets = _compute_top_k(
            ckpt_path, dataset_root, layer, feat_idx,
            top_k, threshold, context_tokens,
        )

    # ── Feature header metrics ─────────────────────────────────────────────────
    st.markdown(
        f"### Feature `{feat_idx}` "
        f"<span style='color:#6e7681;font-size:0.85rem;font-weight:400'>"
        f"/ {n_features} features · layer {layer}</span>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Feature", feat_idx)

    if hist_d:
        c2.metric("Max Activation",  f"{hist_d['max_act']:.4f}")
        c3.metric("% Active",        f"{hist_d['pct_active']:.2f}%")
        c4.metric("Mean (active)",   f"{hist_d['mean_active']:.4f}")
        c5.metric("Mean (all)",      f"{hist_d['mean_all']:.5f}")
        if hist_d["is_dead"]:
            st.markdown(
                '<div class="dead-feature-box">'
                "⚠️ <b>Dead feature</b> — this feature never activated on the "
                "scanned tokens.  Consider resampling or increasing λ."
                "</div>",
                unsafe_allow_html=True,
            )
    else:
        c2.metric("Max Activation",  "—")
        c3.metric("% Active",        "—")
        c4.metric("Mean (active)",   "—")
        c5.metric("Mean (all)",      "—")

    st.markdown("---")

    # ── Row 1: Histogram + Logit effects ──────────────────────────────────────
    if use_logit:
        col_hist, col_logit = st.columns([3, 2])
    else:
        col_hist = st.container()

    with col_hist:
        st.subheader("Activation Distribution")
        if hist_d and not hist_d["is_dead"]:
            fig_hist = _build_histogram_fig(hist_d, feat_idx)
            st.plotly_chart(fig_hist, use_container_width=True)
        elif hist_d and hist_d["is_dead"]:
            st.info("This feature never activated — no distribution to show.")
        else:
            st.warning("Histogram data unavailable.")

    if use_logit:
        with col_logit:
            st.subheader("Logit Effects")
            with st.spinner("Computing logit effects …"):
                effects = _compute_logit_effects(
                    ckpt_path, dataset_root, layer, feat_idx,
                    model_name, top_k=10,
                )

            if effects:
                fig_logit = _build_logit_fig(effects)
                st.plotly_chart(fig_logit, use_container_width=True)
                st.caption(
                    f"W_dec[{feat_idx}] @ W_U  ·  model: `{model_name}`  ·  "
                    "Green = promoted, Red = suppressed"
                )
            else:
                st.warning(
                    f"Could not compute logit effects.  "
                    f"Make sure `{model_name}` is a valid TransformerLens model "
                    "and the model's d_model matches the SAE."
                )

    if not use_logit:
        st.caption(
            "💡 Enable **Logit Lens** in the sidebar to see which output tokens "
            "this feature promotes or suppresses via W_dec @ W_U."
        )

    st.markdown("---")

    # ── Row 2: Top-N Examples ─────────────────────────────────────────────────
    n_found = len(snippets)
    label_str = (
        f" above threshold {threshold:.2f}" if threshold > 0 else ""
    )
    st.subheader(f"Top {top_k} Activating Examples{label_str}")

    if n_found == 0:
        st.info(
            f"No tokens found with activation > {threshold:.2f}.  "
            "Try lowering the **Activation Threshold** in the sidebar."
        )
    else:
        if n_found < top_k:
            st.caption(f"Showing {n_found} examples (fewer than {top_k} exceeded threshold).")

        table_html = _render_examples_html(snippets)
        st.markdown(table_html, unsafe_allow_html=True)

        # Expandable raw data
        with st.expander("Raw snippet data"):
            import pandas as pd
            df = pd.DataFrame([
                {
                    "rank":       s.get("rank"),
                    "score":      s.get("score"),
                    "token":      s.get("token_str"),
                    "sent_idx":   s.get("sentence_idx"),
                    "tok_pos":    s.get("token_pos"),
                    "label":      s.get("label"),
                    "sentence":   (s.get("sentence_text") or "")[:120],
                }
                for s in snippets
            ])
            st.dataframe(df, use_container_width=True, height=280)
