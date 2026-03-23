"""
streamlit_app.py — SPORE Activation Visualiser entry point.

Launch with:
    streamlit run streamlit_app.py

Or with a non-default projections directory:
    streamlit run streamlit_app.py -- --proj projections/my_run

Tabs
----
  Tab 1: 🗺️ Latent Space   — UMAP/PCA scatter, layer slider, color-by, lasso
  (future tabs added here)
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
            "Collect → Reduce → Visualise."
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

# ── Parse any CLI args forwarded via `streamlit run app.py -- --proj ...` ─────
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--proj", default=None,
                   help="Default projections directory shown in the sidebar")
    args, _ = p.parse_known_args(sys.argv[1:])
    return args


cli = _parse_cli()
if cli.proj and "_ls_root" not in st.session_state:
    st.session_state["_ls_root"] = cli.proj

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_latent, = st.tabs(["🗺️ Latent Space"])

with tab_latent:
    from spore.app.latent_space import render_tab
    render_tab()
