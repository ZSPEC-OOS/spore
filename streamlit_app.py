"""
streamlit_app.py — SPORE Activation Visualiser entry point.

Launch with:
    streamlit run streamlit_app.py

Or with non-default directories pre-filled:
    streamlit run streamlit_app.py -- --proj projections/my_run
    streamlit run streamlit_app.py -- --sae-ckpt sae_checkpoints/run/latest \\
                                       --sae-ds sae_data/gpt2_l6

Tabs
----
  Tab 1: 🗺️ Latent Space   — UMAP/PCA scatter, layer slider, color-by, lasso
  Tab 2: 🔬 SAE Features   — per-feature histogram, top examples, logit lens
  Tab 3: 🌐 Feature Map    — UMAP over all SAE decoder directions, click → jump
"""

from __future__ import annotations

import argparse
import sys

import streamlit as st

# ── Page config (must be the FIRST Streamlit call) ───────────────────────────
st.set_page_config(
    page_title = "SPORE — Activation Visualiser",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded",
    menu_items = {
        "Get help":    None,
        "Report a bug": None,
        "About": (
            "**SPORE Activation Visualiser**\n\n"
            "Geometric analysis of transformer residual-stream activations.\n\n"
            "Collect → Reduce → Visualise → Interpret."
        ),
    },
)

# ── Global CSS overrides ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Tab label padding */
    [data-testid="stTabs"] button { padding: 0.4rem 1.2rem; font-size: 0.9rem; }
    /* Tighten top margin */
    .block-container { padding-top: 1rem !important; }
    /* Subtle header rule */
    hr { border-color: #30363d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Parse any CLI args forwarded via `streamlit run app.py -- ...` ────────────
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--proj",    default=None,
                   help="Default projections directory (Latent Space tab)")
    p.add_argument("--sae-ckpt", default=None,
                   help="Default SAE checkpoint path (SAE Features tab)")
    p.add_argument("--sae-ds",   default=None,
                   help="Default SAE dataset root (SAE Features tab)")
    args, _ = p.parse_known_args(sys.argv[1:])
    return args


cli = _parse_cli()
if cli.proj and "_ls_root" not in st.session_state:
    st.session_state["_ls_root"] = cli.proj
if cli.sae_ckpt and "_sae_ckpt" not in st.session_state:
    st.session_state["_sae_ckpt"] = cli.sae_ckpt
if cli.sae_ds and "_sae_ds" not in st.session_state:
    st.session_state["_sae_ds"] = cli.sae_ds

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_latent, tab_sae, tab_map = st.tabs([
    "🗺️ Latent Space",
    "🔬 SAE Features",
    "🌐 Feature Map",
])

with tab_latent:
    from spore.app.latent_space import render_tab
    render_tab()

with tab_sae:
    from spore.app.sae_dashboard import render_tab as render_sae_tab
    render_sae_tab()

with tab_map:
    from spore.app.feature_map import render_tab as render_map_tab
    render_map_tab()
