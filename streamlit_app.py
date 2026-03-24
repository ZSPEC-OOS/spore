"""
streamlit_app.py — SPORE Activation Visualiser unified dashboard.

Launch:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="SPORE — Unified Interpretability Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stTabs"] button { padding: 0.4rem 1.1rem; font-size: 0.9rem; }
    .block-container { padding-top: 1rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--proj", default=None, help="Default projections directory")
    p.add_argument("--sae-ckpt", default=None, help="Default SAE checkpoint path")
    p.add_argument("--sae-ds", default=None, help="Default SAE dataset root")
    p.add_argument("--act-root", default=None, help="Default activations root for trajectory")
    return p.parse_known_args(sys.argv[1:])[0]


def _discover_checkpoints(artifacts_root: str) -> list[str]:
    root = Path(artifacts_root)
    if not root.exists() or not root.is_dir():
        return []
    ckpts = [p.name for p in root.iterdir() if p.is_dir()]
    return sorted(ckpts)


def _apply_checkpoint_selection(artifacts_root: str, checkpoint: str) -> None:
    base = Path(artifacts_root) / checkpoint
    proj = base / "projections"
    acts = base / "activations"
    sae_ckpt = base / "sae_checkpoints" / "latest"
    sae_ds = base / "sae_data"

    if proj.exists():
        st.session_state["_ls_root"] = str(proj)
    if acts.exists():
        st.session_state["_traj_train_root"] = str(acts)
    if sae_ckpt.exists():
        st.session_state["_sae_ckpt"] = str(sae_ckpt)
    if sae_ds.exists():
        st.session_state["_sae_ds"] = str(sae_ds)


cli = _parse_cli()
if cli.proj and "_ls_root" not in st.session_state:
    st.session_state["_ls_root"] = cli.proj
if cli.sae_ckpt and "_sae_ckpt" not in st.session_state:
    st.session_state["_sae_ckpt"] = cli.sae_ckpt
if cli.sae_ds and "_sae_ds" not in st.session_state:
    st.session_state["_sae_ds"] = cli.sae_ds
if cli.act_root and "_traj_train_root" not in st.session_state:
    st.session_state["_traj_train_root"] = cli.act_root

st.sidebar.markdown("## 🧠 SPORE Dashboard")
st.sidebar.caption("Unified interpretability workspace with checkpoint comparison.")

model_name = st.sidebar.text_input(
    "Model selection",
    value=st.session_state.get("_global_model_name", "gpt2"),
    key="_global_model_name",
)
layer_choice = st.sidebar.number_input(
    "Layer choice",
    min_value=0,
    max_value=95,
    value=int(st.session_state.get("_global_layer_choice", 0)),
    step=1,
    key="_global_layer_choice",
)
dataset_subset = st.sidebar.number_input(
    "Dataset subset size",
    min_value=100,
    max_value=100_000,
    value=int(st.session_state.get("_global_dataset_subset", 5000)),
    step=100,
    key="_global_dataset_subset",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔁 Checkpoint comparison")
artifacts_root = st.sidebar.text_input(
    "Artifacts root",
    value=st.session_state.get("_artifacts_root", str(Path("artifacts").resolve())),
    key="_artifacts_root",
    help="Directory containing per-epoch folders (e.g., epoch_0001/, epoch_0002/).",
)
checkpoint_options = _discover_checkpoints(artifacts_root)
checkpoint = st.sidebar.selectbox(
    "Epoch/checkpoint",
    options=checkpoint_options if checkpoint_options else ["(none found)"],
    index=0,
    help="Switching this updates default projection/activation/SAE paths.",
)

if checkpoint_options and st.sidebar.button("Apply checkpoint", use_container_width=True):
    _apply_checkpoint_selection(artifacts_root, checkpoint)
    st.sidebar.success(f"Applied checkpoint: {checkpoint}")

if st.sidebar.button("🔄 Refresh data", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Tip: each page also exposes local controls for deeper analysis.")

page1, page2, page3, page4, page5 = st.tabs([
    "🗺️ Page 1 — Latent Space Projections",
    "🔬 Page 2 — SAE Feature Explorer",
    "🌐 Page 3 — SAE Feature UMAP",
    "🧭 Page 4 — Prompt Trajectory",
    "👀 Page 5 — Attention & Logit Lens",
])

with page1:
    from spore.app.latent_space import render_tab as render_latent

    render_latent(include_prompt_trajectory=False)

with page2:
    from spore.app.sae_dashboard import render_tab as render_sae

    render_sae()

with page3:
    from spore.app.feature_map import render_tab as render_feature_map

    render_feature_map()

with page4:
    from spore.app.prompt_trajectory import render_prompt_trajectory_viewer

    render_prompt_trajectory_viewer(default_root=st.session_state.get("_traj_train_root"))

with page5:
    from spore.app.attention_logit_lens import render_tab as render_attention_lens

    render_attention_lens(
        model_name=model_name,
        layer_default=int(layer_choice),
        top_k_default=min(20, max(3, int(dataset_subset // 1000))),
    )
