"""
io.py — save and load CollectionRun objects.

Disk layout
-----------
  <root>/
    metadata.json             model name, layers, hook point, pool, timestamp
    corpus.json               original texts + optional labels
    tokens.pt                 int64 [N, seq_len]
    attention_mask.pt         bool  [N, seq_len]
    layer_{L:02d}.pt          float16 [N, d_model]  (or [N, seq, d_model])
    ...

All tensor files are saved with torch.save and loaded with torch.load.
Using float16 halves disk size vs float32.

Multi-checkpoint sweeps
-----------------------
Call ActivationCache.save(run, root / f"ckpt_{step:06d}") once per
checkpoint; then compare projections across the resulting subdirectories.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .collector import CollectionRun, HookPoint, PoolStrategy

logger = logging.getLogger(__name__)


class ActivationCache:
    """Save and load :class:`CollectionRun` objects to / from disk."""

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    @staticmethod
    def save(run: CollectionRun, root: str | Path, overwrite: bool = False) -> Path:
        """
        Persist *run* to *root* directory.

        Parameters
        ----------
        run : CollectionRun
        root : str | Path
            Target directory.  Created if it does not exist.
        overwrite : bool
            If False and *root* already contains a ``metadata.json``,
            raise FileExistsError.

        Returns
        -------
        Path
            The resolved *root* directory.
        """
        root = Path(root).resolve()
        if root.exists() and (root / "metadata.json").exists() and not overwrite:
            raise FileExistsError(
                f"{root} already contains a saved run.  "
                "Pass overwrite=True to replace it."
            )
        root.mkdir(parents=True, exist_ok=True)

        # ── metadata ────────────────────────────────────────────────────
        meta = {
            "model_name":     run.model_name,
            "layers":         run.layers,
            "hook_point":     run.hook_point.value,
            "pool":           run.pool.value,
            "d_model":        run.d_model,
            "n_layers_total": run.n_layers_total,
            "n_sentences":    len(run.texts),
            "seq_len":        run.tokens.shape[1],
            "elapsed_sec":    round(run.elapsed_sec, 2),
            "saved_at":       datetime.now(tz=timezone.utc).isoformat(),
        }
        _write_json(root / "metadata.json", meta)
        logger.debug("Saved metadata → %s", root / "metadata.json")

        # ── corpus ──────────────────────────────────────────────────────
        corpus = {
            "texts":  run.texts,
            "labels": run.labels,
        }
        _write_json(root / "corpus.json", corpus)
        logger.debug("Saved corpus (%d texts) → %s", len(run.texts), root / "corpus.json")

        # ── tokens & mask ───────────────────────────────────────────────
        torch.save(run.tokens,         root / "tokens.pt")
        torch.save(run.attention_mask, root / "attention_mask.pt")
        logger.debug("Saved tokens + mask")

        # ── per-layer activations ────────────────────────────────────────
        total_bytes = 0
        for layer, tensor in sorted(run.activations.items()):
            path = root / f"layer_{layer:02d}.pt"
            torch.save(tensor, path)
            total_bytes += tensor.element_size() * tensor.nelement()
            logger.debug("Saved layer %02d %s → %s", layer, tuple(tensor.shape), path)

        logger.info(
            "Saved %d-layer run (%d sentences) to %s  [%.1f MB]",
            len(run.layers), len(run.texts), root,
            total_bytes / 1_048_576,
        )
        return root

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @staticmethod
    def load(
        root: str | Path,
        layers: Optional[List[int]] = None,
        device: str = "cpu",
        dtype:  Optional[torch.dtype] = None,
    ) -> CollectionRun:
        """
        Reconstruct a :class:`CollectionRun` from disk.

        Parameters
        ----------
        root : str | Path
        layers : list[int] | None
            Subset of layers to load.  None → load all saved layers.
        device : str
            Target device for tensors.
        dtype : torch.dtype | None
            Cast activations to this dtype on load.  None → keep as saved.

        Returns
        -------
        CollectionRun
        """
        root = Path(root).resolve()
        if not root.exists():
            raise FileNotFoundError(root)

        meta   = _read_json(root / "metadata.json")
        corpus = _read_json(root / "corpus.json")

        tokens         = torch.load(root / "tokens.pt",         map_location=device, weights_only=True)
        attention_mask = torch.load(root / "attention_mask.pt", map_location=device, weights_only=True)

        saved_layers = meta["layers"]
        if layers is not None:
            missing = [l for l in layers if l not in saved_layers]
            if missing:
                raise ValueError(f"Layers {missing} not found in saved run at {root}.")
            load_layers = layers
        else:
            load_layers = saved_layers

        activations: Dict[int, torch.Tensor] = {}
        for layer in load_layers:
            path = root / f"layer_{layer:02d}.pt"
            t = torch.load(path, map_location=device, weights_only=True)
            if dtype is not None:
                t = t.to(dtype=dtype)
            activations[layer] = t

        logger.info(
            "Loaded %d-layer run (%d sentences) from %s",
            len(load_layers), len(corpus["texts"]), root,
        )
        return CollectionRun(
            model_name      = meta["model_name"],
            layers          = load_layers,
            hook_point      = HookPoint(meta["hook_point"]),
            pool            = PoolStrategy(meta["pool"]),
            texts           = corpus["texts"],
            tokens          = tokens,
            attention_mask  = attention_mask,
            labels          = corpus.get("labels"),
            activations     = activations,
            elapsed_sec     = meta.get("elapsed_sec", 0.0),
            d_model         = meta.get("d_model", 0),
            n_layers_total  = meta.get("n_layers_total", 0),
        )

    # ------------------------------------------------------------------
    # Inspect
    # ------------------------------------------------------------------

    @staticmethod
    def describe(root: str | Path) -> dict:
        """
        Read and return the metadata for a saved run without loading tensors.

        Useful for quickly auditing what was collected.
        """
        return _read_json(Path(root) / "metadata.json")

    @staticmethod
    def list_checkpoints(sweep_dir: str | Path) -> List[Path]:
        """
        Return subdirectories of *sweep_dir* that contain saved runs,
        sorted lexicographically (which preserves numeric step order when
        directories are named ckpt_000000, ckpt_000100, …).
        """
        sweep_dir = Path(sweep_dir)
        return sorted(
            p for p in sweep_dir.iterdir()
            if p.is_dir() and (p / "metadata.json").exists()
        )


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
