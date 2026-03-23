#!/usr/bin/env python3
"""
reduce_activations.py — CLI runner for UMAP / PCA dimensionality reduction.

Loads a saved activation run (from collect_activations.py) and produces
projection DataFrames saved as .parquet or .csv files.

Usage examples
--------------
# UMAP 2D on layers 0, 6, 11 — default settings
python reduce_activations.py --src activations/run

# PCA 50-D + UMAP 2D for every layer, parquet output
python reduce_activations.py \\
    --src activations/run \\
    --method both \\
    --layers all \\
    --pca-components 50 \\
    --umap-components 2 \\
    --umap-neighbors 15 \\
    --umap-min-dist 0.1 \\
    --umap-metric cosine \\
    --pca-pre 50 \\
    --out projections/run \\
    --fmt parquet

# Single-layer quick UMAP, no pre-PCA, 3-D embedding, CSV
python reduce_activations.py \\
    --src activations/run \\
    --method umap \\
    --layers 6 \\
    --umap-components 3 \\
    --no-pca-pre \\
    --out projections/layer6 \\
    --fmt csv

# Load + describe a saved projection
python reduce_activations.py --inspect projections/run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reduce_activations")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.inspect:
        _inspect(args.inspect)
        return

    from spore.activation_pipeline import ActivationCache
    from spore.activation_pipeline.reduction import (
        ProjectionSuite,
        compute_pca,
        compute_umap,
        save_projection,
        to_dataframe,
    )

    # ── Load activations ──────────────────────────────────────────────────
    logger.info("Loading activations from %s …", args.src)
    run = ActivationCache.load(args.src)
    logger.info(
        "Loaded: model=%s  sentences=%d  layers=%s  d_model=%d",
        run.model_name, len(run.texts), run.layers, run.d_model,
    )

    # ── Layer selection ───────────────────────────────────────────────────
    if args.layers == "all":
        layers = run.layers
    else:
        layers = [int(x.strip()) for x in args.layers.split(",")]
        missing = [l for l in layers if l not in run.layers]
        if missing:
            sys.exit(
                f"Layers {missing} not found in the run.  "
                f"Available: {run.layers}"
            )

    logger.info("Reducing layers: %s  method=%s", layers, args.method)

    # ── PCA pre-reduction setting ─────────────────────────────────────────
    pca_pre: int | None = None if args.no_pca_pre else args.pca_pre

    # ── Build ProjectionSuite ─────────────────────────────────────────────
    # Filter the run to the requested layers (avoid re-loading heavy tensors)
    import dataclasses
    sub_run = dataclasses.replace(
        run,
        layers      = layers,
        activations = {l: run.activations[l] for l in layers},
    )

    suite = ProjectionSuite.from_run(
        sub_run,
        method           = args.method,
        pca_components   = args.pca_components,
        umap_components  = args.umap_components,
        umap_neighbors   = args.umap_neighbors,
        umap_min_dist    = args.umap_min_dist,
        umap_metric      = args.umap_metric,
        pca_pre          = pca_pre,
        show_progress    = True,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = suite.save(args.out, fmt=args.fmt)
    logger.info("All projections saved to %s", out_path)

    # ── Summary ───────────────────────────────────────────────────────────
    _print_summary(suite, args.method, layers, out_path)


# ---------------------------------------------------------------------------
# Inspect helper
# ---------------------------------------------------------------------------

def _inspect(root: str) -> None:
    from pathlib import Path
    import pandas as pd

    root_path = Path(root)
    if not root_path.exists():
        sys.exit(f"Directory not found: {root}")

    files = sorted(root_path.glob("layer_*.*"))
    if not files:
        sys.exit(f"No projection files found in {root}")

    print(f"\n── Projections in {root_path} ──────────────────────────────")
    for f in files:
        size_kb = f.stat().st_size / 1024
        if f.suffix == ".parquet":
            try:
                df = pd.read_parquet(f)
            except Exception:
                df = None
        else:
            df = pd.read_csv(f, nrows=0)

        shape = f"{len(df)} rows × {len(df.columns)} cols" if df is not None else "?"
        print(f"  {f.name:40s}  {size_kb:7.1f} KB   {shape}")

    # PCA summary if present
    summary = root_path / "suite_summary.csv"
    if summary.exists():
        df = pd.read_csv(summary)
        print(f"\n── PCA explained variance ({summary.name}) ────────────────")
        print(df.to_string(index=False, float_format="%.4f"))
    print()


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(suite, method: str, layers: list[int], out_path: Path) -> None:
    import pandas as pd

    print("\n" + "─" * 64)
    print(f"  Method(s)   : {method}")
    print(f"  Layers      : {layers}")
    print(f"  Output dir  : {out_path}")

    methods_done = suite.methods
    for m in methods_done:
        df = suite.get(layers[0], m)
        if df is not None:
            cols = [c for c in df.columns]
            print(f"\n  [{m.upper()}] sample DataFrame  (layer {layers[0]}, first 3 rows)")
            print(df.head(3).to_string(index=False))
            print(f"\n  Columns: {cols}")

    # PCA explained variance plot hint
    if "pca" in methods_done:
        result = suite.get_result(layers[0], "pca")
        if result is not None and result.explained_ratio is not None:
            top5 = result.explained_ratio[:5] * 100
            print(f"\n  PCA (layer {layers[0]}) top-5 component variance (%):")
            print("    " + "  ".join(f"PC{i+1}:{v:.1f}%" for i, v in enumerate(top5)))
            print(f"    Cumulative total: {result.explained_ratio.sum()*100:.1f}%")

    print("─" * 64 + "\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply UMAP / PCA to cached transformer activations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input / output
    p.add_argument("--src", default="activations/run",
                   help="Activation run directory (default: activations/run)")
    p.add_argument("--out", default="projections/run",
                   help="Output directory (default: projections/run)")
    p.add_argument("--fmt", default="parquet", choices=["parquet", "csv"],
                   help="Output file format (default: parquet)")

    # Layer selection
    p.add_argument("--layers", default="all",
                   help="Comma-separated layer indices, or 'all' (default: all)")

    # Method
    p.add_argument("--method", default="both", choices=["pca", "umap", "both"],
                   help="Reduction method (default: both)")

    # PCA params
    p.add_argument("--pca-components", type=int, default=50,
                   help="PCA output dimensions (default: 50)")

    # UMAP params
    p.add_argument("--umap-components", type=int, default=2,
                   help="UMAP output dimensions — 2 or 3 (default: 2)")
    p.add_argument("--umap-neighbors",  type=int,   default=15,
                   help="UMAP n_neighbors (default: 15)")
    p.add_argument("--umap-min-dist",   type=float, default=0.1,
                   help="UMAP min_dist (default: 0.1)")
    p.add_argument("--umap-metric",     default="cosine",
                   help="UMAP distance metric (default: cosine)")

    # PCA pre-reduction
    p.add_argument("--pca-pre",    type=int, default=50,
                   help="PCA pre-reduction dims before UMAP (default: 50)")
    p.add_argument("--no-pca-pre", action="store_true",
                   help="Skip PCA pre-reduction before UMAP")

    # Inspect mode
    p.add_argument("--inspect", metavar="DIR", default=None,
                   help="Describe a saved projection directory and exit")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
