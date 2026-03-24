"""
streamlit_app.py — Geometric Activation Visualizer unified dashboard.

Launch:
    streamlit run streamlit_app.py

The dashboard is designed to be always-visible: each tab renders a useful
placeholder even before any training data exists, and auto-refreshes to
pick up new artifacts as training progresses.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Geometric Activation Visualizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Tab bar */
    [data-testid="stTabs"] button       { padding: 0.4rem 1rem; font-size: 0.88rem; }
    [data-testid="stTabs"] button:hover { background: rgba(255,255,255,0.06); }
    .block-container { padding-top: 0.55rem !important; padding-bottom: 5rem !important; }

    /* Dashboard header */
    .gav-title {
        font-size: 1.4rem; font-weight: 700; color: #f0f6fc;
        letter-spacing: -0.01em; margin: 0; line-height: 1.25;
    }
    .gav-subtitle {
        font-size: 0.82rem; color: #8b949e;
        margin: 0.12rem 0 0 0; font-style: italic;
    }

    /* Sticky footer */
    .gav-footer {
        position: fixed; bottom: 0; left: 0; right: 0;
        background: #161b22; border-top: 1px solid #30363d;
        padding: 0.3rem 1.2rem;
        display: flex; align-items: center; gap: 1.5rem;
        font-size: 0.71rem; color: #8b949e; z-index: 1000;
    }
    .status-dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: #2ea043; display: inline-block;
        margin-right: 4px; vertical-align: middle;
    }
    .status-dot.live { background: #388bfd; animation: blink 1.4s infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.35} }

    /* Sidebar section titles */
    .sb-section {
        font-size: 0.69rem; font-weight: 700; letter-spacing: 0.09em;
        color: #8b949e; text-transform: uppercase;
        margin: 0.9rem 0 0.15rem 0; padding-top: 0.5rem;
        border-top: 1px solid #21262d;
    }

    /* Tab context banners */
    .tab-context {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 6px; padding: 0.55rem 0.9rem;
        font-size: 0.82rem; color: #8b949e;
        margin-bottom: 0.6rem;
        display: flex; align-items: center; gap: 8px;
    }
    .tab-context strong { color: #e6edf3; }

    /* Empty-state placeholder card */
    .empty-card {
        background: #0d1117; border: 1px dashed #30363d;
        border-radius: 8px; padding: 3rem 1.5rem;
        text-align: center; color: #484f58;
        font-size: 0.82rem;
    }
    .empty-card .icon { font-size: 2.5rem; display: block; margin-bottom: 0.5rem; }
    .empty-card strong { color: #8b949e; display: block; margin-bottom: 0.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── CLI arg parsing ────────────────────────────────────────────────────────────

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--proj",     default=None)
    p.add_argument("--sae-ckpt", default=None)
    p.add_argument("--sae-ds",   default=None)
    p.add_argument("--act-root", default=None)
    return p.parse_known_args(sys.argv[1:])[0]


def _discover_checkpoints(artifacts_root: str) -> list[str]:
    root = Path(artifacts_root)
    if not root.exists() or not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _apply_checkpoint_selection(artifacts_root: str, checkpoint: str) -> None:
    base = Path(artifacts_root) / checkpoint
    for path, key in [
        (base / "projections",                "_ls_root"),
        (base / "activations",                "_traj_train_root"),
        (base / "sae_checkpoints" / "latest", "_sae_ckpt"),
        (base / "sae_data",                   "_sae_ds"),
    ]:
        if path.exists():
            st.session_state[key] = str(path)


# ── Session bootstrap from CLI ─────────────────────────────────────────────────
cli = _parse_cli()
for cli_val, key in [
    (cli.proj,     "_ls_root"),
    (cli.sae_ckpt, "_sae_ckpt"),
    (cli.sae_ds,   "_sae_ds"),
    (cli.act_root, "_traj_train_root"),
]:
    if cli_val and key not in st.session_state:
        st.session_state[key] = cli_val


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

hcol_title, hcol_about = st.columns([8, 2])
with hcol_title:
    st.markdown(
        '<p class="gav-title">🧬 Geometric Activation Visualizer'
        " \u2013 Transformer Mechanistic Insights</p>"
        '<p class="gav-subtitle">High-dimensional manifold sculpting via '
        "superposition and sparse features \u2014 "
        "charts update live as training progresses</p>",
        unsafe_allow_html=True,
    )
with hcol_about:
    with st.expander("\u2139\ufe0f About", expanded=False):
        st.markdown(
            "Transformer residual-stream activations form structured "
            "high-dimensional manifolds encoding semantic and syntactic "
            "information through superposition. This dashboard projects those "
            "manifolds into 2-D/3-D via UMAP and PCA, decomposes them into "
            "sparse features with a Sparse Autoencoder (SAE), and traces how "
            "token representations evolve layer-by-layer \u2014 enabling "
            "mechanistic inspection of attention patterns, feature geometry, "
            "and internal predictions at each depth."
            "\n\n"
            "**New to this dashboard?** Run the pipeline commands shown in "
            "the *Pipeline Status* panel below, then come back here. "
            "Charts populate automatically."
        )

st.markdown(
    '<hr style="margin:0.3rem 0 0.4rem 0;border-color:#21262d">',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def _sb_section(title: str) -> None:
    st.sidebar.markdown(f'<p class="sb-section">{title}</p>', unsafe_allow_html=True)


st.sidebar.markdown("## 🧬 GAV Controls")
st.sidebar.caption(
    "Geometric Activation Visualizer \u2014 SPORE interpretability workspace.\n"
    "The dashboard stays open during training and refreshes automatically."
)

# ── Live refresh ───────────────────────────────────────────────────────────────
_sb_section("\U0001f504 Live Refresh")

auto_refresh = st.sidebar.checkbox(
    "Auto-refresh during training",
    value=False,
    key="_auto_refresh",
    help="Poll the filesystem every N seconds and rerun when new artifacts appear.",
)
_REFRESH_OPTIONS = {"10 s": 10, "30 s": 30, "1 min": 60, "2 min": 120, "5 min": 300}
refresh_label = st.sidebar.selectbox(
    "Interval",
    options=list(_REFRESH_OPTIONS.keys()),
    index=1,
    disabled=not auto_refresh,
    key="_refresh_interval",
    label_visibility="collapsed",
)
refresh_secs = _REFRESH_OPTIONS[refresh_label]

# ── Model ─────────────────────────────────────────────────────────────────────
_sb_section("\U0001f916 Model")

_MODEL_PRESETS: dict[str, str] = {
    "GPT-2 Small (gpt2)":   "gpt2",
    "GPT-2 Medium":          "gpt2-medium",
    "Pythia 160M":           "pythia-160m",
    "Pythia 410M":           "pythia-410m",
    "Pythia 1.4B":           "pythia-1.4b",
    "Custom path\u2026":     "__custom__",
}
_preset_rev = {v: k for k, v in _MODEL_PRESETS.items()}
_cur_model   = st.session_state.get("_global_model_name", "gpt2")
_def_preset  = _preset_rev.get(_cur_model, "Custom path\u2026")

model_preset = st.sidebar.selectbox(
    "Model",
    options=list(_MODEL_PRESETS.keys()),
    index=list(_MODEL_PRESETS.keys()).index(_def_preset),
    key="_model_preset",
    help="Which transformer model to inspect. Must match the model used to collect activations.",
)
if model_preset == "Custom path\u2026":
    model_name: str = st.sidebar.text_input(
        "Custom model name",
        value=_cur_model if _cur_model not in _MODEL_PRESETS.values() else "gpt2",
        key="_global_model_name",
        placeholder="e.g. EleutherAI/gpt-j-6B",
    )
else:
    model_name = _MODEL_PRESETS[model_preset]
    st.session_state["_global_model_name"] = model_name

# ── Layer ─────────────────────────────────────────────────────────────────────
_sb_section("\U0001f522 Layer")

_layer_mode = st.sidebar.radio(
    "Selection mode",
    ["Slider", "Number input"],
    horizontal=True,
    label_visibility="collapsed",
    key="_layer_mode",
)
_layer_val = int(st.session_state.get("_global_layer_choice", 0))
if _layer_mode == "Slider":
    layer_choice: int = st.sidebar.slider(
        "Layer", min_value=0, max_value=95, value=_layer_val,
        key="_global_layer_choice",
        help="Layer index 0 = embedding output, higher = deeper in the model.",
    )
else:
    layer_choice = st.sidebar.number_input(
        "Layer", min_value=0, max_value=95, value=_layer_val, step=1,
        key="_global_layer_choice",
    )

# ── Dataset & Checkpoint ──────────────────────────────────────────────────────
_sb_section("\U0001f4e6 Dataset & Checkpoint")

dataset_subset: int = st.sidebar.number_input(
    "Dataset subset size",
    min_value=100, max_value=100_000,
    value=int(st.session_state.get("_global_dataset_subset", 5000)),
    step=100, key="_global_dataset_subset",
    help="Number of tokens/sentences sampled from the corpus for collection.",
)

artifacts_root: str = st.sidebar.text_input(
    "Artifacts root",
    value=st.session_state.get("_artifacts_root", str(Path("artifacts").resolve())),
    key="_artifacts_root",
    help="Directory containing per-epoch/checkpoint folders.",
)
checkpoint_options = _discover_checkpoints(artifacts_root)
checkpoint: str = st.sidebar.selectbox(
    "Checkpoint",
    options=checkpoint_options if checkpoint_options else ["(none found)"],
    index=0, key="_checkpoint_sel",
    help="Selecting a checkpoint automatically updates projection/activation/SAE paths.",
)
if checkpoint_options and st.sidebar.button("Apply checkpoint", use_container_width=True):
    _apply_checkpoint_selection(artifacts_root, checkpoint)
    st.sidebar.success(f"\u2713 Applied: {checkpoint}")

# ── Display Options ────────────────────────────────────────────────────────────
_sb_section("\U0001f3a8 Display Options")

color_scheme: str = st.sidebar.selectbox(
    "Color scheme",
    options=["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo"],
    index=0, key="_global_color_scheme",
    help="Default colorscale for continuous scatter plots.",
)
scale_by_magnitude: bool = st.sidebar.checkbox(
    "Scale point size by activation magnitude",
    value=False, key="_global_scale_by_mag",
)

# ── Accessibility ─────────────────────────────────────────────────────────────
_sb_section("\u267f Accessibility")

high_contrast: bool = st.sidebar.checkbox(
    "High-contrast mode", value=False, key="_global_accessibility",
)
if high_contrast:
    st.markdown(
        "<style>"
        "body,[data-testid='stAppViewContainer'],[data-testid='stMarkdownContainer'] p"
        "{color:#ffffff!important}"
        "[data-testid='stSidebar']{background:#000000!important}"
        ".stMetric label{color:#ffffff!important}"
        "</style>",
        unsafe_allow_html=True,
    )

# ── Export ─────────────────────────────────────────────────────────────────────
_sb_section("\U0001f4be Export")
st.sidebar.caption("Export session artifacts:")
ecol1, ecol2 = st.sidebar.columns(2)
with ecol1:
    if st.button("\U0001f4f7 PNG", use_container_width=True, key="_export_png_tip"):
        st.sidebar.info("Click the \U0001f4f7 icon in any chart toolbar to download PNG.")
with ecol2:
    _proj_root = st.session_state.get("_ls_root", "")
    if st.button("\U0001f4c2 CSV", use_container_width=True, key="_export_csv_tip"):
        if _proj_root:
            st.sidebar.success(f"Files are in:\n`{_proj_root}`")
        else:
            st.sidebar.warning("No projection directory set.")

_session_export = {
    k: v for k, v in st.session_state.items()
    if isinstance(v, (str, int, float, bool, list))
}
st.sidebar.download_button(
    "\U0001f4be Session JSON",
    data=json.dumps(_session_export, indent=2, default=str),
    file_name="gav_session.json", mime="application/json",
    use_container_width=True,
)

# ── Refresh ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
if st.sidebar.button("\U0001f504 Refresh all caches", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state["_last_refresh"] = datetime.datetime.now().strftime("%H:%M:%S")
    st.rerun()

# Compact pipeline status in sidebar
from spore.app.status_panel import render_sidebar_status, scan_artifacts as _scan
_sidebar_status = _scan(
    artifacts_root=artifacts_root,
    proj_root=st.session_state.get("_ls_root"),
    act_root=st.session_state.get("_traj_train_root"),
    sae_ckpt=st.session_state.get("_sae_ckpt"),
    sae_ds=st.session_state.get("_sae_ds"),
)
render_sidebar_status(_sidebar_status)

st.sidebar.caption("Tip: each tab also has local controls.")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STATUS PANEL  (always visible, auto-collapses when all steps done)
# ══════════════════════════════════════════════════════════════════════════════

from spore.app.status_panel import render_status_panel

_pipeline_status = render_status_panel(
    artifacts_root=artifacts_root,
    proj_root=st.session_state.get("_ls_root"),
    act_root=st.session_state.get("_traj_train_root"),
    sae_ckpt=st.session_state.get("_sae_ckpt"),
    sae_ds=st.session_state.get("_sae_ds"),
    expanded=(_sidebar_status.steps_complete < 4),
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB CONTEXT BANNERS  (tell the user what each tab does before they interact)
# ══════════════════════════════════════════════════════════════════════════════

_TAB_CONTEXTS = {
    "latent_space":    (
        "🗺️", "Latent Space Explorer",
        "Projects residual-stream activations into 2-D/3-D via UMAP or PCA. "
        "Points = sentences; clusters = semantic similarity; "
        "color = layer, norm, or label."
    ),
    "feature_dict":    (
        "🔬", "Feature Dictionary",
        "Inspect individual SAE features: activation histogram, "
        "top-activating token examples, logit effects (which output "
        "tokens this feature promotes or suppresses), and a global "
        "feature-space map on the right."
    ),
    "trajectory":      (
        "🧭", "Trajectory Analyzer",
        "Traces how a specific prompt's token representations move "
        "through the model's residual stream layer-by-layer. "
        "Reveals which layers cause the most geometric transformation."
    ),
    "supporting_maps": (
        "👁️", "Supporting Maps",
        "Attention rollout heatmaps (which tokens attend to which) "
        "and logit lens (which vocabulary tokens each layer predicts "
        "before the final unembedding)."
    ),
    "metrics":         (
        "📊", "Metrics & Comparison",
        "Quantitative layer-wise statistics: activation norm profile, "
        "PCA explained variance per layer, inter-layer centroid distance. "
        "Supports side-by-side comparison of two checkpoints."
    ),
}


def _tab_banner(key: str) -> None:
    icon, name, desc = _TAB_CONTEXTS[key]
    st.markdown(
        f'<div class="tab-context">'
        f'<span style="font-size:1.2rem">{icon}</span>'
        f'<div><strong>{name}</strong><br>{desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Latent Space Explorer",
    "🔬 Feature Dictionary",
    "🧭 Trajectory Analyzer",
    "👁️ Supporting Maps",
    "📊 Metrics & Comparison",
])

with tab1:
    _tab_banner("latent_space")
    from spore.app.latent_space import render_tab as render_latent
    render_latent(include_prompt_trajectory=False)

with tab2:
    _tab_banner("feature_dict")
    from spore.app.sae_dashboard import render_tab as render_sae
    render_sae()

with tab3:
    _tab_banner("trajectory")
    from spore.app.prompt_trajectory import render_prompt_trajectory_viewer
    render_prompt_trajectory_viewer(
        default_root=st.session_state.get("_traj_train_root")
    )

with tab4:
    _tab_banner("supporting_maps")
    from spore.app.attention_logit_lens import render_tab as render_attention_lens
    render_attention_lens(
        model_name=model_name,
        layer_default=int(layer_choice),
        top_k_default=min(20, max(3, int(dataset_subset // 1000))),
    )

with tab5:
    _tab_banner("metrics")
    from spore.app.metrics_comparison import render_tab as render_metrics
    render_metrics()


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

_last_refresh = st.session_state.get("_last_refresh", "\u2014")
_live_cls = "live" if auto_refresh else ""

st.markdown(
    f'<div class="gav-footer">'
    f'<span><span class="status-dot {_live_cls}"></span>'
    f'{"Auto-refresh ON" if auto_refresh else "Cache: active"}</span>'
    f'<span>Last refresh: {_last_refresh}</span>'
    f'<span style="color:#30363d">\u2502</span>'
    f'<span style="color:#6e7681">'
    f'Geometric Activation Visualizer &middot; SPORE v0.1</span>'
    f"</div>",
    unsafe_allow_html=True,
)

# Legacy option (out-of-the-way expander)
with st.expander("\U0001f519 Legacy options", expanded=False):
    _legacy_clicked = st.session_state.get("_legacy_clicked", False)
    if not _legacy_clicked:
        if st.button("Switch to legacy ball-web (deprecated)", type="secondary",
                     key="_legacy_btn"):
            st.session_state["_legacy_clicked"] = True
            st.rerun()
    else:
        st.warning(
            "\u26d4 **The legacy ball-and-web visualizer has been discontinued** "
            "and is no longer maintained. This button is now disabled.\n\n"
            "All visualization workflows are available in the tabs above.",
        )
        st.button("Legacy ball-web (disabled)", disabled=True, key="_legacy_btn_dis")


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH  (runs at the very end so the page fully renders first)
# ══════════════════════════════════════════════════════════════════════════════

if auto_refresh:
    time.sleep(refresh_secs)
    st.session_state["_last_refresh"] = datetime.datetime.now().strftime("%H:%M:%S")
    st.rerun()
