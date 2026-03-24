#!/usr/bin/env python
"""
train_sae.py — CLI for training a Sparse Autoencoder on token activations.

The SAE is trained from scratch (no external SAE library required).  For an
equivalent sae-lens config, see the docstring at the top of
``spore/activation_pipeline/sae.py``.

Modes
-----
  train (default)     train a new SAE on an existing SAEDataset
  --inspect PATH      print an existing checkpoint's metadata and metrics
  --search-l1         sweep a small grid of l1_coeff values, print L0 table
  --eval PATH         evaluate a checkpoint against the dataset and print stats

Quick start
-----------
  # 1. Build token-level activation dataset (Request 4)
  python build_sae_dataset.py --model gpt2 --n 20000 --layers 6 \\
      --out sae_data/gpt2_l6

  # 2. Train a standard ReLU SAE (expansion 8×, 50 k steps)
  python train_sae.py \\
      --dataset sae_data/gpt2_l6 \\
      --layer 6 \\
      --expansion 8 \\
      --l1 1e-3 \\
      --steps 50000 \\
      --out sae_checkpoints/gpt2_l6_relu8x

  # 3. Train a Gated SAE with a custom feature count
  python train_sae.py \\
      --dataset sae_data/gpt2_l6 \\
      --layer 6 \\
      --n-features 4096 \\
      --activation gated \\
      --l1 5e-4 \\
      --out sae_checkpoints/gpt2_l6_gated

  # 4. Inspect a checkpoint
  python train_sae.py --inspect sae_checkpoints/gpt2_l6_relu8x/latest

  # 5. Evaluate a checkpoint
  python train_sae.py --eval sae_checkpoints/gpt2_l6_relu8x/latest \\
      --dataset sae_data/gpt2_l6 --layer 6

  # 6. L1 coefficient grid search
  python train_sae.py --search-l1 \\
      --dataset sae_data/gpt2_l6 --layer 6 --steps 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("train_sae")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description     = "Train a Sparse Autoencoder on token-level activations.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog          = __doc__,
    )

    # ── Mode ─────────────────────────────────────────────────────────────────
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--inspect",   metavar="CKPT_PATH",
                      help="Print checkpoint metadata (no training).")
    mode.add_argument("--eval",      metavar="CKPT_PATH",
                      help="Evaluate checkpoint against dataset.")
    mode.add_argument("--search-l1", action="store_true",
                      help="Run a short L1 grid search and print L0 table.")

    # ── Dataset / layer ───────────────────────────────────────────────────────
    io = p.add_argument_group("Dataset")
    io.add_argument("--dataset", default=None, metavar="PATH",
                    help="Root of an SAEDataset built with build_sae_dataset.py.")
    io.add_argument("--layer", type=int, default=None,
                    help="Which layer's activations to train on.")

    # ── Architecture ─────────────────────────────────────────────────────────
    arch = p.add_argument_group("Architecture")
    arch.add_argument("--expansion", type=float, default=8.0,
                      help="n_features = expansion × d_model (default: 8).")
    arch.add_argument("--n-features", type=int, default=None,
                      help="Exact feature count (overrides --expansion).")
    arch.add_argument("--activation", choices=["relu", "gated"], default="relu",
                      help="Activation type (default: relu).")

    # ── Training ─────────────────────────────────────────────────────────────
    train = p.add_argument_group("Training")
    train.add_argument("--l1",           type=float, default=1e-3,
                       help="L1 sparsity coefficient (default: 1e-3).")
    train.add_argument("--l1-gate",      type=float, default=1e-4,
                       help="Gate pathway L1 weight for gated SAE (default: 1e-4).")
    train.add_argument("--lr",           type=float, default=2e-4,
                       help="Peak learning rate (default: 2e-4).")
    train.add_argument("--lr-end",       type=float, default=2e-5,
                       help="Final LR after cosine decay (default: 2e-5).")
    train.add_argument("--batch",        type=int,   default=4096,
                       help="Batch size in tokens (default: 4096).")
    train.add_argument("--steps",        type=int,   default=50_000,
                       help="Total gradient steps (default: 50000).")
    train.add_argument("--warmup",       type=int,   default=1_000,
                       help="LR warmup steps (default: 1000).")
    train.add_argument("--grad-clip",    type=float, default=1.0,
                       help="Gradient norm clip (default: 1.0).")
    train.add_argument("--resample-every",   type=int, default=2_500,
                       help="Resample dead neurons every N steps (default: 2500).")
    train.add_argument("--dead-after",       type=int, default=2_500,
                       help="Steps inactive before a neuron is 'dead' (default: 2500).")
    train.add_argument("--no-resample", action="store_true",
                       help="Disable neuron resampling.")
    train.add_argument("--dtype", choices=["float32", "float16", "bfloat16"],
                       default="float32",
                       help="Compute dtype (default: float32).")
    train.add_argument("--device", default="auto",
                       help="'auto' | 'cpu' | 'cuda' | 'mps' (default: auto).")
    train.add_argument("--seed", type=int, default=42)

    # ── Checkpointing ─────────────────────────────────────────────────────────
    ck = p.add_argument_group("Checkpointing")
    ck.add_argument("--out",      default="sae_checkpoints/run",
                    help="Output directory (default: sae_checkpoints/run).")
    ck.add_argument("--save-every", type=int, default=5_000,
                    help="Save checkpoint every N steps (default: 5000).")
    ck.add_argument("--log-every",  type=int, default=100,
                    help="Log metrics every N steps (default: 100).")
    ck.add_argument("--resume",     default=None, metavar="CKPT_PATH",
                    help="Resume training from a checkpoint directory.")

    # ── Grid search ───────────────────────────────────────────────────────────
    gs = p.add_argument_group("L1 grid search (--search-l1)")
    gs.add_argument("--l1-values", default="1e-4,5e-4,1e-3,5e-3,1e-2",
                    help="Comma-separated l1_coeff values to try.")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _mode_inspect(ckpt_path: str) -> None:
    ck_dir = Path(ckpt_path)
    meta_file = ck_dir / "meta.json"
    if not meta_file.exists():
        logger.error("No meta.json found in %s", ck_dir)
        sys.exit(1)

    meta = json.loads(meta_file.read_text())
    cfg_d = meta.get("cfg", {})
    m = meta.get("metrics", {})

    print(f"\n── SAE Checkpoint ─────────────────────────────────────────")
    print(f"  path          : {ck_dir}")
    print(f"  step          : {meta.get('step', '?')}")
    print()
    print(f"  model         : {cfg_d.get('model_name', '?')}")
    print(f"  layer         : {cfg_d.get('layer', '?')}  ({cfg_d.get('hook_point', '?')})")
    print(f"  d_model       : {cfg_d.get('d_model', '?')}")
    print(f"  n_features    : {cfg_d.get('n_features', '?')}")
    print(f"  activation    : {cfg_d.get('activation', '?')}")
    print(f"  l1_coeff      : {cfg_d.get('l1_coeff', '?')}")
    print(f"  batch_size    : {cfg_d.get('batch_size', '?')}")
    print(f"  n_steps       : {cfg_d.get('n_steps', '?')}")
    print(f"  dtype         : {cfg_d.get('dtype', '?')}")
    print()

    if m:
        print(f"── Last logged metrics ─────────────────────────────────────")
        for k, v in m.items():
            print(f"  {k:<18}  {v}")

    # Check for weights file
    wfile = ck_dir / "weights.pt"
    if wfile.exists():
        size_mb = wfile.stat().st_size / 1e6
        print(f"\n  weights.pt    : {size_mb:.1f} MB")

    print()


def _mode_eval(ckpt_path: str, dataset_root: str, layer: int) -> None:
    from spore.activation_pipeline import SAEDataset
    from spore.activation_pipeline.sae import SAEConfig, SparseAutoencoder

    ck_dir   = Path(ckpt_path)
    meta     = json.loads((ck_dir / "meta.json").read_text())
    cfg_d    = meta["cfg"]
    cfg      = SAEConfig(**cfg_d)

    device = _resolve_device("auto")

    logger.info("Loading model …")
    model = SparseAutoencoder(cfg).to(device)
    weights = torch.load(ck_dir / "weights.pt", weights_only=True, map_location=device)
    model.load_state_dict_sae(weights)
    model.eval()

    logger.info("Loading dataset …")
    ds   = SAEDataset.load(dataset_root)
    L    = layer if layer is not None else cfg.layer

    # Eval over all shards
    all_m: dict = {}
    n_shards = 0
    for _, shard in ds.iter_shards(L):
        x   = shard.float().to(device)
        out = model(x)
        m   = model.metrics(x, out)
        for k, v in m.items():
            if isinstance(v, (int, float)):
                all_m[k] = all_m.get(k, 0.0) + v
        n_shards += 1

    print(f"\n── SAE Evaluation  [{ck_dir.name}] ─────────────────────────")
    print(f"  dataset       : {dataset_root}")
    print(f"  layer         : {L}  ({cfg.hook_point})")
    print(f"  shards        : {n_shards}")
    print()
    for k, v in all_m.items():
        avg = v / max(n_shards, 1)
        print(f"  {k:<18}  {avg:.6f}")
    print()


def _mode_train(args: argparse.Namespace) -> None:
    from spore.activation_pipeline import SAEDataset
    from spore.activation_pipeline.sae import SAEConfig, SAETrainer

    # ── Load dataset ─────────────────────────────────────────────────────────
    if not args.dataset:
        logger.error("--dataset is required for training.")
        sys.exit(1)

    ds   = SAEDataset.load(args.dataset)
    meta = ds._meta

    # ── Resolve layer ─────────────────────────────────────────────────────────
    layer = args.layer
    if layer is None:
        if len(ds.layers) == 1:
            layer = ds.layers[0]
        else:
            logger.error(
                "Dataset has multiple layers %s — specify --layer", ds.layers
            )
            sys.exit(1)

    if layer not in ds.layers:
        logger.error("Layer %d not in dataset %s", layer, ds.layers)
        sys.exit(1)

    d_model    = meta["d_model"]
    n_features = args.n_features or max(1, int(args.expansion * d_model))

    logger.info(
        "Dataset: %s | layer=%d | d_model=%d | n_features=%d | n_tokens=%d",
        args.dataset, layer, d_model, n_features, ds.n_tokens,
    )

    # ── Build SAEConfig ───────────────────────────────────────────────────────
    cfg = SAEConfig(
        d_model          = d_model,
        n_features       = n_features,
        activation       = args.activation,
        l1_coeff         = args.l1,
        l1_gate_coeff    = args.l1_gate,
        lr               = args.lr,
        lr_end           = args.lr_end,
        betas            = (0.9, 0.999),
        eps              = 1e-8,
        weight_decay     = 0.0,
        batch_size       = args.batch,
        n_steps          = args.steps,
        warmup_steps     = args.warmup,
        grad_clip_norm   = args.grad_clip,
        resample_every   = 0 if args.no_resample else args.resample_every,
        dead_after_steps = 0 if args.no_resample else args.dead_after,
        checkpoint_every = args.save_every,
        log_every        = args.log_every,
        out_dir          = args.out,
        dtype            = args.dtype,
        seed             = args.seed,
        dataset_root     = args.dataset,
        layer            = layer,
        hook_point       = meta.get("hook_point", "resid_post"),
        model_name       = meta.get("model_name", ""),
    )

    _print_train_plan(cfg, ds)

    # ── Train ────────────────────────────────────────────────────────────────
    device = _resolve_device(args.device)
    trainer = SAETrainer(cfg, device=device, resume_from=args.resume)
    history = trainer.train(ds, show_progress=True)

    # ── Print final metrics ──────────────────────────────────────────────────
    if history:
        last = history[-1]
        print(f"\n── Final metrics (step {last.get('step', '?')}) ────────────────────")
        _print_metrics(last)

    print(f"\nCheckpoints in: {args.out}\n")


def _mode_search_l1(args: argparse.Namespace) -> None:
    """
    Short training run at each l1_coeff value in --l1-values.
    Reports L0, explained variance, and alive features.
    """
    from spore.activation_pipeline import SAEDataset
    from spore.activation_pipeline.sae import SAEConfig, SAETrainer

    if not args.dataset:
        logger.error("--dataset is required for --search-l1.")
        sys.exit(1)

    ds      = SAEDataset.load(args.dataset)
    meta    = ds._meta
    d_model = meta["d_model"]

    layer = args.layer
    if layer is None:
        layer = ds.layers[0]

    n_features = args.n_features or max(1, int(args.expansion * d_model))
    short_steps = min(args.steps, 5_000)
    device = _resolve_device(args.device)

    try:
        l1_values = [float(v.strip()) for v in args.l1_values.split(",")]
    except ValueError:
        logger.error("--l1-values must be comma-separated floats.")
        sys.exit(1)

    print(f"\n── L1 coefficient grid search ──────────────────────────────")
    print(f"  dataset    : {args.dataset}  (layer {layer})")
    print(f"  n_features : {n_features}  ({args.expansion:.1f}×)")
    print(f"  steps      : {short_steps}")
    print(f"  l1 values  : {l1_values}")
    print()
    print(f"{'l1_coeff':>12} {'L0_mean':>10} {'L0_pct':>8} {'alive_pct':>10} "
          f"{'expl_var':>10} {'recon_loss':>12} {'total_loss':>12}")
    print("─" * 80)

    for l1 in l1_values:
        out_dir = f"/tmp/sae_grid_l1_{l1:.0e}"
        cfg = SAEConfig(
            d_model          = d_model,
            n_features       = n_features,
            activation       = args.activation,
            l1_coeff         = l1,
            lr               = args.lr,
            lr_end           = args.lr_end,
            batch_size       = args.batch,
            n_steps          = short_steps,
            warmup_steps     = min(args.warmup, short_steps // 5),
            resample_every   = 0,
            dead_after_steps = 0,
            checkpoint_every = short_steps + 1,  # no mid-run saves
            log_every        = short_steps,       # only final log
            out_dir          = out_dir,
            seed             = args.seed,
            dataset_root     = args.dataset,
            layer            = layer,
            hook_point       = meta.get("hook_point", "resid_post"),
            model_name       = meta.get("model_name", ""),
        )
        trainer = SAETrainer(cfg, device=device)
        history = trainer.train(ds, show_progress=False)
        m = history[-1] if history else {}
        print(
            f"{l1:>12.1e} "
            f"{m.get('l0_mean', 0):>10.2f} "
            f"{m.get('l0_pct', 0):>8.3f} "
            f"{m.get('alive_pct', 0):>10.2f} "
            f"{m.get('explained_var', 0):>10.4f} "
            f"{m.get('recon_loss', 0):>12.6f} "
            f"{m.get('total_loss', 0):>12.6f}"
        )

    print()
    print("Target: L0_pct ≈ 1–5%  |  alive_pct > 30%  |  explained_var > 0.70")
    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _print_train_plan(cfg, ds) -> None:
    print(f"\n── SAE training plan ───────────────────────────────────────")
    print(f"  model         : {cfg.model_name or '(from dataset)'}  layer={cfg.layer}")
    print(f"  hook_point    : {cfg.hook_point}")
    print(f"  d_model       : {cfg.d_model}")
    print(f"  n_features    : {cfg.n_features}  "
          f"({cfg.n_features / cfg.d_model:.1f}× expansion)")
    print(f"  activation    : {cfg.activation}")
    print(f"  l1_coeff      : {cfg.l1_coeff}")
    if cfg.activation == "gated":
        print(f"  l1_gate_coeff : {cfg.l1_gate_coeff}")
    print(f"  n_steps       : {cfg.n_steps:,}")
    print(f"  batch_size    : {cfg.batch_size:,}  tokens")
    print(f"  total tokens  : {cfg.n_steps * cfg.batch_size:,}")
    print(f"  lr            : {cfg.lr}  → {cfg.lr_end} (cosine)")
    print(f"  warmup        : {cfg.warmup_steps:,} steps")
    if cfg.resample_every > 0:
        print(f"  resampling    : every {cfg.resample_every} steps  "
              f"(dead_after={cfg.dead_after_steps})")
    else:
        print(f"  resampling    : disabled")
    print(f"  dataset       : {ds.n_tokens:,} tokens  ({ds.n_shards} shards)")
    print(f"  out_dir       : {cfg.out_dir}")
    print()


def _print_metrics(m: dict) -> None:
    order = [
        "total_loss", "recon_loss", "l1_loss",
        "l0_mean", "l0_pct", "alive_pct", "dead_features",
        "explained_var", "mean_act", "lr", "elapsed",
    ]
    for k in order:
        if k in m:
            print(f"  {k:<18}  {m[k]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)

    if args.inspect:
        _mode_inspect(args.inspect)

    elif args.eval:
        if not args.dataset:
            logger.error("--dataset is required for --eval.")
            sys.exit(1)
        _mode_eval(args.eval, args.dataset, args.layer)

    elif args.search_l1:
        _mode_search_l1(args)

    else:
        _mode_train(args)


if __name__ == "__main__":
    main()
