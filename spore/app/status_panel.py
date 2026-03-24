"""
status_panel.py — Live pipeline artifact scanner and status widget.

Scans the filesystem for data produced by each pipeline step and renders
a visual "training progress" panel with per-step health indicators.  This
makes the dashboard always useful even before any training has started.

Public API
----------
scan_artifacts(root)    → ArtifactStatus
render_status_panel()   → None   (Streamlit widget)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Artifact scanning
# ---------------------------------------------------------------------------

_PROJ_FILE_RE  = re.compile(r"^layer_\d+_(umap|pca)\.(parquet|csv)$", re.I)
_ACT_FILE_RE   = re.compile(r"^(activations|acts?)\.(pt|pkl|npz|parquet)$", re.I)
_LAYER_DIR_RE  = re.compile(r"^layer_\d+$", re.I)


@dataclass
class StepStatus:
    name: str
    icon: str
    done: bool
    detail: str = ""
    file_count: int = 0
    latest_mtime: Optional[float] = None


@dataclass
class ArtifactStatus:
    """Complete status of all four pipeline steps."""
    activations: StepStatus
    projections: StepStatus
    sae_dataset: StepStatus
    sae_checkpoint: StepStatus

    @property
    def steps_complete(self) -> int:
        return sum(
            1 for s in [self.activations, self.projections, self.sae_dataset, self.sae_checkpoint]
            if s.done
        )

    @property
    def ready_for_visualizer(self) -> bool:
        """True if at least projections are available (minimum for Tab 1)."""
        return self.projections.done

    @property
    def last_updated(self) -> Optional[float]:
        """Most recent mtime across all discovered artifacts."""
        mtimes = [
            s.latest_mtime
            for s in [self.activations, self.projections, self.sae_dataset, self.sae_checkpoint]
            if s.latest_mtime is not None
        ]
        return max(mtimes) if mtimes else None


def scan_artifacts(
    artifacts_root: str | Path,
    proj_root: Optional[str | Path] = None,
    act_root: Optional[str | Path] = None,
    sae_ckpt: Optional[str | Path] = None,
    sae_ds: Optional[str | Path] = None,
) -> ArtifactStatus:
    """
    Scan filesystem for pipeline artifacts under *artifacts_root* and the
    optional per-component overrides.

    Parameters
    ----------
    artifacts_root : str | Path
        Root directory that may contain epoch_XXXX/ subdirectories.
    proj_root, act_root, sae_ckpt, sae_ds
        Direct overrides for specific components (taken from session state
        in the dashboard).

    Returns
    -------
    ArtifactStatus
    """
    root = Path(artifacts_root) if artifacts_root else Path("artifacts")

    # ── Collect candidate directories ─────────────────────────────────────
    # Direct override paths
    _proj_dirs  = [Path(proj_root)] if proj_root else []
    _act_dirs   = [Path(act_root)]  if act_root  else []
    _sae_ckpts  = [Path(sae_ckpt)]  if sae_ckpt  else []
    _sae_dss    = [Path(sae_ds)]    if sae_ds    else []

    # Also scan artifacts_root/epoch_*/ subdirs
    if root.exists():
        for ep in sorted(root.iterdir()):
            if not ep.is_dir():
                continue
            for subdir, target in [
                ("projections", _proj_dirs),
                ("activations", _act_dirs),
                (Path("sae_checkpoints") / "latest", _sae_ckpts),
                ("sae_data", _sae_dss),
            ]:
                candidate = ep / subdir
                if candidate.exists():
                    target.append(candidate)

    # Also check default relative paths used by the pipeline scripts
    for default, target in [
        ("projections/run", _proj_dirs),
        ("activations/run", _act_dirs),
    ]:
        p = Path(default)
        if p.exists() and p not in target:
            target.append(p)

    # ── Step 1: Activations ───────────────────────────────────────────────
    act_files: List[Path] = []
    for d in _act_dirs:
        if d.exists():
            # Look for .pt tensors or parquet shards
            act_files.extend(d.rglob("*.pt"))
            act_files.extend(d.rglob("*.parquet"))
            act_files.extend(d.rglob("metadata.json"))

    act_done  = bool(act_files)
    act_mtime = max((f.stat().st_mtime for f in act_files if f.exists()), default=None)
    act_detail = (
        f"{len(act_files)} file(s) found across {len(_act_dirs)} location(s)"
        if act_done
        else "Run `collect_activations.py` to generate."
    )

    # ── Step 2: Projections ───────────────────────────────────────────────
    proj_files: List[Path] = []
    for d in _proj_dirs:
        if d.exists():
            for f in d.iterdir():
                if _PROJ_FILE_RE.match(f.name):
                    proj_files.append(f)

    proj_done  = bool(proj_files)
    proj_mtime = max((f.stat().st_mtime for f in proj_files if f.exists()), default=None)
    # Count unique layers
    unique_layers = set()
    for f in proj_files:
        m = re.match(r"layer_(\d+)", f.name, re.I)
        if m:
            unique_layers.add(int(m.group(1)))
    proj_detail = (
        f"{len(proj_files)} projection file(s) across {len(unique_layers)} layer(s)"
        if proj_done
        else "Run `reduce_activations.py --method both` to generate."
    )

    # ── Step 3: SAE Dataset ───────────────────────────────────────────────
    sae_ds_meta: List[Path] = []
    for d in _sae_dss:
        if d.exists():
            meta = d / "metadata.json"
            if meta.exists():
                sae_ds_meta.append(meta)

    sae_ds_done  = bool(sae_ds_meta)
    sae_ds_mtime = max((f.stat().st_mtime for f in sae_ds_meta if f.exists()), default=None)
    sae_ds_detail = (
        f"metadata.json found in {len(sae_ds_meta)} location(s)"
        if sae_ds_done
        else "Run `build_sae_dataset.py` to generate."
    )

    # ── Step 4: SAE Checkpoint ────────────────────────────────────────────
    sae_meta: List[Path] = []
    for d in _sae_ckpts:
        if d.exists():
            meta = d / "meta.json"
            if meta.exists():
                sae_meta.append(meta)

    sae_ckpt_done  = bool(sae_meta)
    sae_ckpt_mtime = max((f.stat().st_mtime for f in sae_meta if f.exists()), default=None)

    # Try to read step number from meta.json for richer detail
    sae_step_info = ""
    if sae_meta:
        try:
            import json
            meta_data = json.loads(sae_meta[0].read_text())
            step = meta_data.get("step", "?")
            sae_step_info = f" · step {step}"
        except Exception:
            pass
    sae_ckpt_detail = (
        f"checkpoint found{sae_step_info}"
        if sae_ckpt_done
        else "Run `train_sae.py` to generate."
    )

    return ArtifactStatus(
        activations    = StepStatus(
            name        = "Collect Activations",
            icon        = "⚡",
            done        = act_done,
            detail      = act_detail,
            file_count  = len(act_files),
            latest_mtime= act_mtime,
        ),
        projections    = StepStatus(
            name        = "Reduce Projections",
            icon        = "🗺️",
            done        = proj_done,
            detail      = proj_detail,
            file_count  = len(proj_files),
            latest_mtime= proj_mtime,
        ),
        sae_dataset    = StepStatus(
            name        = "Build SAE Dataset",
            icon        = "📦",
            done        = sae_ds_done,
            detail      = sae_ds_detail,
            file_count  = len(sae_ds_meta),
            latest_mtime= sae_ds_mtime,
        ),
        sae_checkpoint = StepStatus(
            name        = "Train SAE",
            icon        = "🧠",
            done        = sae_ckpt_done,
            detail      = sae_ckpt_detail,
            file_count  = len(sae_meta),
            latest_mtime= sae_ckpt_mtime,
        ),
    )


# ---------------------------------------------------------------------------
# Streamlit widget
# ---------------------------------------------------------------------------

_STATUS_CSS = """
<style>
.pip-step {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 10px;
    border-radius: 6px;
    margin-bottom: 6px;
    font-size: 0.82rem;
    border: 1px solid transparent;
}
.pip-step.done  { background:#0d2318; border-color:#238636; }
.pip-step.waiting { background:#161b22; border-color:#30363d; }
.pip-step.active  { background:#1a2433; border-color:#388bfd; animation: pip-pulse 2s infinite; }

@keyframes pip-pulse {
    0%,100% { border-color:#388bfd; }
    50%      { border-color:#58a6ff; }
}

.pip-icon { font-size:1.1rem; line-height:1.4; flex-shrink:0; }
.pip-name { font-weight:600; color:#e6edf3; }
.pip-detail { color:#8b949e; font-size:0.76rem; margin-top:1px; }
.pip-badge-done { color:#3fb950; font-size:0.72rem; }
.pip-badge-wait { color:#6e7681; font-size:0.72rem; }
.pip-connector {
    width:2px; height:12px;
    background:#30363d;
    margin-left:19px; margin-bottom:0;
}
</style>
"""


def render_status_panel(
    artifacts_root: str,
    proj_root:  Optional[str] = None,
    act_root:   Optional[str] = None,
    sae_ckpt:   Optional[str] = None,
    sae_ds:     Optional[str] = None,
    expanded: bool = True,
) -> ArtifactStatus:
    """
    Render the pipeline status panel inside a Streamlit expander.

    Returns the :class:`ArtifactStatus` so callers can gate functionality
    on what's available.
    """
    st.markdown(_STATUS_CSS, unsafe_allow_html=True)

    status = scan_artifacts(
        artifacts_root=artifacts_root,
        proj_root=proj_root,
        act_root=act_root,
        sae_ckpt=sae_ckpt,
        sae_ds=sae_ds,
    )

    n_done  = status.steps_complete
    n_total = 4
    bar_pct = int(n_done / n_total * 100)

    # Summary line
    last_upd = status.last_updated
    if last_upd:
        import datetime as _dt
        ts = _dt.datetime.fromtimestamp(last_upd).strftime("%H:%M:%S")
        summary_extra = f"  ·  last artifact: {ts}"
    else:
        summary_extra = "  ·  no artifacts found yet"

    title = (
        f"📡 Pipeline Status — {n_done}/{n_total} steps ready{summary_extra}"
    )

    with st.expander(title, expanded=expanded and n_done < n_total):
        # Progress bar
        st.markdown(
            f'<div style="background:#21262d;border-radius:4px;height:6px;margin-bottom:10px">'
            f'<div style="background:{"#2ea043" if n_done==n_total else "#388bfd"};'
            f'width:{bar_pct}%;height:100%;border-radius:4px;'
            f'transition:width 0.4s"></div></div>',
            unsafe_allow_html=True,
        )

        steps = [
            status.activations,
            status.projections,
            status.sae_dataset,
            status.sae_checkpoint,
        ]
        for i, step in enumerate(steps):
            cls  = "done" if step.done else "waiting"
            tick = "✅" if step.done else "⏳"
            badge_cls = "pip-badge-done" if step.done else "pip-badge-wait"
            badge_txt = "ready" if step.done else "pending"

            st.markdown(
                f'<div class="pip-step {cls}">'
                f'  <span class="pip-icon">{step.icon}</span>'
                f'  <div>'
                f'    <div class="pip-name">'
                f'      Step {i+1}: {step.name}'
                f'      <span class="{badge_cls}"> · {badge_txt}</span>'
                f'    </div>'
                f'    <div class="pip-detail">{step.detail}</div>'
                f'  </div>'
                f'  <span style="margin-left:auto;font-size:0.9rem">{tick}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if i < len(steps) - 1:
                st.markdown('<div class="pip-connector"></div>', unsafe_allow_html=True)

        if n_done == 0:
            st.markdown(
                "---\n"
                "**Quick start:**\n"
                "```bash\n"
                "python scripts/collect_activations.py --model gpt2 --n 500\n"
                "python scripts/reduce_activations.py  --method both\n"
                "```\n"
                "Then refresh this panel — charts will populate automatically.",
            )
        elif n_done == n_total:
            st.success("All pipeline steps complete — full dashboard unlocked! 🎉", icon="✅")
        else:
            # Show next command
            next_step = [s for s in steps if not s.done][0]
            cmd_map = {
                "Collect Activations": "python scripts/collect_activations.py --model gpt2 --n 500",
                "Reduce Projections":  "python scripts/reduce_activations.py  --method both",
                "Build SAE Dataset":   "python scripts/build_sae_dataset.py",
                "Train SAE":           "python scripts/train_sae.py",
            }
            cmd = cmd_map.get(next_step.name, "")
            if cmd:
                st.markdown(f"**Next step:** `{cmd}`")

    return status


def render_sidebar_status(status: ArtifactStatus) -> None:
    """Compact one-line status for the sidebar."""
    n_done = status.steps_complete
    color  = "#2ea043" if n_done == 4 else ("#388bfd" if n_done > 0 else "#6e7681")
    label  = "All ready" if n_done == 4 else f"{n_done}/4 steps ready"
    st.sidebar.markdown(
        f'<div style="font-size:0.72rem;color:{color};margin-bottom:0.3rem">'
        f'● {label}</div>',
        unsafe_allow_html=True,
    )
