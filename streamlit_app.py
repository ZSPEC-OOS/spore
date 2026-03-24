"""
streamlit_app.py — Geometric Activation Visualizer unified dashboard.

Launch:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
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
    [data-testid="stTabs"] button { padding: 0.4rem 1.1rem; font-size: 0.9rem; }
    .block-container { padding-top: 0.6rem !important; padding-bottom: 5rem !important; }

    .gav-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f0f6fc;
        letter-spacing: -0.01em;
        margin: 0;
        line-height: 1.25;
    }
    .gav-subtitle {
        font-size: 0.82rem;
        color: #8b949e;
        margin: 0.15rem 0 0 0;
        font-style: italic;
    }
    .gav-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #161b22;
        border-top: 1px solid #30363d;
        padding: 0.32rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 1.6rem;
        font-size: 0.72rem;
        color: #8b949e;
        z-index: 1000;
    }
    .status-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #2ea043;
        display: inline-block;
        margin-right: 4px;
        vertical-align: middle;
    }
    .sb-section {
        font-size: 0.69rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        color: #8b949e;
        text-transform: uppercase;
        margin: 1rem 0 0.2rem 0;
        padding-top: 0.55rem;
        border-top: 1px solid #21262d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── CLI arg parsing ────────────────────────────────────────────────────────────

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--proj",     default=None, help="Default projections directory")
    p.add_argument("--sae-ckpt", default=None, help="Default SAE checkpoint path")
    p.add_argument("--sae-ds",   default=None, help="Default SAE dataset root")
    p.add_argument("--act-root", default=None, help="Default activations root for trajectory")
    return p.parse_known_args(sys.argv[1:])[0]


def _discover_checkpoints(artifacts_root: str) -> list[str]:
    root = Path(artifacts_root)
    if not root.exists() or not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _apply_checkpoint_selection(artifacts_root: str, checkpoint: str) -> None:
    base = Path(artifacts_root) / checkpoint
    for path, key in [
        (base / "projections",               "_ls_root"),
        (base / "activations",               "_traj_train_root"),
        (base / "sae_checkpoints" / "latest", "_sae_ckpt"),
        (base / "sae_data",                  "_sae_ds"),
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
        "superposition and sparse features</p>",
        unsafe_allow_html=True,
    )
with hcol_about:
    with st.expander("ℹ️ About", expanded=False):
        st.markdown(
            "Transformer residual-stream activations form structured "
            "high-dimensional manifolds encoding semantic and syntactic "
            "information through superposition. This dashboard projects those "
            "manifolds into 2-D/3-D via UMAP and PCA, decomposes them into "
            "sparse features with a Sparse Autoencoder (SAE), and traces how "
            "individual token representations evolve layer-by-layer — enabling "
            "mechanistic inspection of attention patterns, feature geometry, "
            "and internal predictions at each depth."
        )

st.markdown(
    '<hr style="margin:0.35rem 0 0.55rem 0;border-color:#21262d">',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def _sb_section(title: str) -> None:
    st.sidebar.markdown(
        f'<p class="sb-section">{title}</p>', unsafe_allow_html=True
    )


st.sidebar.markdown("## 🧬 GAV Controls")
st.sidebar.caption("Geometric Activation Visualizer — SPORE interpretability workspace.")

# ── Model ─────────────────────────────────────────────────────────────────────
_sb_section("🤖 Model")

_MODEL_PRESETS: dict[str, str] = {
    "GPT-2 Small (gpt2)":        "gpt2",
    "GPT-2 Medium":               "gpt2-medium",
    "Pythia 160M":                "pythia-160m",
    "Pythia 410M":                "pythia-410m",
    "Pythia 1.4B":                "pythia-1.4b",
    "Custom path\u2026":          "__custom__",
}
_preset_rev = {v: k for k, v in _MODEL_PRESETS.items()}
_cur_model   = st.session_state.get("_global_model_name", "gpt2")
_def_preset  = _preset_rev.get(_cur_model, "Custom path\u2026")

model_preset = st.sidebar.selectbox(
    "Model",
    options=list(_MODEL_PRESETS.keys()),
    index=list(_MODEL_PRESETS.keys()).index(_def_preset),
    key="_model_preset",
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
    # Persist so individual tabs can read _global_model_name from session state
    st.session_state["_global_model_name"] = model_name

# ── Layer ─────────────────────────────────────────────────────────────────────
_sb_section("🔢 Layer")

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
        "Layer",
        min_value=0, max_value=95,
        value=_layer_val,
        key="_global_layer_choice",
    )
else:
    layer_choice = st.sidebar.number_input(
        "Layer",
        min_value=0, max_value=95,
        value=_layer_val,
        step=1,
        key="_global_layer_choice",
    )

# ── Dataset & Checkpoint ──────────────────────────────────────────────────────
_sb_section("📦 Dataset & Checkpoint")

dataset_subset: int = st.sidebar.number_input(
    "Dataset subset size",
    min_value=100,
    max_value=100_000,
    value=int(st.session_state.get("_global_dataset_subset", 5000)),
    step=100,
    key="_global_dataset_subset",
    help="Number of tokens/sentences sampled from the corpus.",
)

artifacts_root: str = st.sidebar.text_input(
    "Artifacts root",
    value=st.session_state.get("_artifacts_root", str(Path("artifacts").resolve())),
    key="_artifacts_root",
    help="Directory containing per-epoch folders (epoch_0001/, epoch_0002/, \u2026).",
)

checkpoint_options = _discover_checkpoints(artifacts_root)
checkpoint: str = st.sidebar.selectbox(
    "Checkpoint",
    options=checkpoint_options if checkpoint_options else ["(none found)"],
    index=0,
    help="Selecting a checkpoint updates the default projection/activation/SAE paths.",
    key="_checkpoint_sel",
)

if checkpoint_options and st.sidebar.button("Apply checkpoint", use_container_width=True):
    _apply_checkpoint_selection(artifacts_root, checkpoint)
    st.sidebar.success(f"\u2713 Applied: {checkpoint}")

# ── Display Options ────────────────────────────────────────────────────────────
_sb_section("🎨 Display Options")

color_scheme: str = st.sidebar.selectbox(
    "Color scheme",
    options=["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo"],
    index=0,
    key="_global_color_scheme",
    help="Default colorscale for continuous scatter plots.",
)

scale_by_magnitude: bool = st.sidebar.checkbox(
    "Scale point size by activation magnitude",
    value=False,
    key="_global_scale_by_mag",
    help="Larger points = higher activation norm in scatter views.",
)

# ── Accessibility ─────────────────────────────────────────────────────────────
_sb_section("\u267f Accessibility")

high_contrast: bool = st.sidebar.checkbox(
    "High-contrast mode",
    value=False,
    key="_global_accessibility",
    help="Boosts text/background contrast for visual accessibility.",
)

if high_contrast:
    st.markdown(
        "<style>"
        "body, [data-testid='stAppViewContainer'], "
        "[data-testid='stMarkdownContainer'] p { color: #ffffff !important; }"
        "[data-testid='stSidebar'] { background: #000000 !important; }"
        ".stMetric label { color: #ffffff !important; }"
        "</style>",
        unsafe_allow_html=True,
    )

# ── Export ─────────────────────────────────────────────────────────────────────
_sb_section("💾 Export")

st.sidebar.caption("Export session artifacts:")

ecol1, ecol2 = st.sidebar.columns(2)
with ecol1:
    if st.button(
        "📷 PNG",
        use_container_width=True,
        key="_export_png_tip",
        help="Use the camera icon in any Plotly chart toolbar.",
    ):
        st.sidebar.info(
            "Click the \U0001f4f7 icon in any chart's toolbar to download as PNG."
        )
with ecol2:
    _proj_root = st.session_state.get("_ls_root", "")
    if st.button(
        "📂 CSV",
        use_container_width=True,
        key="_export_csv_tip",
        help="Shows path to projection Parquet/CSV files.",
    ):
        if _proj_root:
            st.sidebar.success(f"Projection files:\n`{_proj_root}`")
        else:
            st.sidebar.warning("No projection directory set.")

_session_export = {
    k: v
    for k, v in st.session_state.items()
    if isinstance(v, (str, int, float, bool, list))
}
st.sidebar.download_button(
    "💾 Session state JSON",
    data=json.dumps(_session_export, indent=2, default=str),
    file_name="gav_session.json",
    mime="application/json",
    use_container_width=True,
    help="Download all current session state keys as JSON.",
)

# ── Refresh ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh all caches", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state["_last_refresh"] = datetime.datetime.now().strftime("%H:%M:%S")
    st.rerun()

st.sidebar.caption("Tip: each tab also has local controls for deeper analysis.")


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
    from spore.app.latent_space import render_tab as render_latent
    render_latent(include_prompt_trajectory=False)

with tab2:
    from spore.app.sae_dashboard import render_tab as render_sae
    render_sae()

with tab3:
    from spore.app.prompt_trajectory import render_prompt_trajectory_viewer
    render_prompt_trajectory_viewer(
        default_root=st.session_state.get("_traj_train_root")
    )

with tab4:
    from spore.app.attention_logit_lens import render_tab as render_attention_lens
    render_attention_lens(
        model_name=model_name,
        layer_default=int(layer_choice),
        top_k_default=min(20, max(3, int(dataset_subset // 1000))),
    )

with tab5:
    from spore.app.metrics_comparison import render_tab as render_metrics
    render_metrics()


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

_last_refresh = st.session_state.get("_last_refresh", "\u2014")

st.markdown(
    f'<div class="gav-footer">'
    f'<span><span class="status-dot"></span>Cache: active</span>'
    f'<span>Last refresh: {_last_refresh}</span>'
    f'<span style="color:#30363d">\u2502</span>'
    f'<span style="color:#6e7681">Geometric Activation Visualizer &middot; SPORE v0.1</span>'
    f"</div>",
    unsafe_allow_html=True,
)

# Legacy notice (as expander so it is accessible but out of the way)
with st.expander("🔙 Legacy options", expanded=False):
    _legacy_clicked = st.session_state.get("_legacy_clicked", False)
    if not _legacy_clicked:
        if st.button(
            "Switch to legacy ball-web (deprecated)",
            type="secondary",
            key="_legacy_btn",
        ):
            st.session_state["_legacy_clicked"] = True
            st.rerun()
    else:
        st.warning(
            "\u26d4 **The legacy ball-and-web visualizer has been discontinued** "
            "and is no longer maintained. This button is now disabled.\n\n"
            "All visualization workflows are available in the tabs above.",
            icon="\u26a0\ufe0f",
        )
        st.button(
            "Legacy ball-web (disabled)",
            disabled=True,
            key="_legacy_btn_disabled",
        )
