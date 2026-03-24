#!/usr/bin/env python
"""
build_sae_dataset.py — CLI for building a token-level SAE training dataset.

Modes
-----
  build     (default)   collect token activations and save to shards
  --inspect <path>      print summary of an existing dataset
  --top-k   <feature>   demo retrieval: top-K text snippets for a feature

Examples
--------
  # Build with 20k sentences from the built-in diverse corpus
  python build_sae_dataset.py \\
      --model gpt2 \\
      --n 20000 \\
      --layers 6 \\
      --out sae_data/gpt2_l6

  # Multiple layers, OpenWebText, bfloat16, 2M token cap
  python build_sae_dataset.py \\
      --model EleutherAI/pythia-160m \\
      --corpus openwebtext \\
      --n 100000 \\
      --layers 3,6,9 \\
      --hook resid_mid \\
      --max-tokens 2000000 \\
      --dtype bfloat16 \\
      --shard-size 250000 \\
      --out sae_data/pythia160m_layers369

  # Inspect an existing dataset
  python build_sae_dataset.py --inspect sae_data/gpt2_l6

  # Top-K retrieval demo (by neuron index)
  python build_sae_dataset.py --top-k 42 --layer 6 --k 20 sae_data/gpt2_l6

  # Top-K retrieval with a learned direction tensor
  python build_sae_dataset.py \\
      --top-k-tensor my_sae/feature_42.pt \\
      --layer 6 --k 20 sae_data/gpt2_l6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("build_sae_dataset")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description = "Build a token-level activation dataset for SAE training.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = __doc__,
    )

    # ── Mode ─────────────────────────────────────────────────────────────────
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--inspect", metavar="PATH",
        help="Inspect an existing SAEDataset (no build).",
    )
    mode.add_argument(
        "--top-k", dest="topk_feature", type=int, metavar="FEATURE_IDX",
        help="Run top-K retrieval for a scalar feature/neuron index.",
    )
    mode.add_argument(
        "--top-k-tensor", dest="topk_tensor", metavar="PATH",
        help="Run top-K retrieval using a saved [d_model] direction tensor.",
    )

    # ── Dataset path (required for top-k modes) ───────────────────────────
    p.add_argument(
        "dataset_path", nargs="?", metavar="DATASET_PATH",
        help="Existing dataset root (required for --top-k / --top-k-tensor).",
    )

    # ── Build parameters ─────────────────────────────────────────────────────
    build = p.add_argument_group("Build options")
    build.add_argument("--model",  default="gpt2",
                       help="TransformerLens model name (default: gpt2)")
    build.add_argument("--corpus", default="diverse",
                       metavar="SOURCE",
                       help=(
                           "'diverse' (default), 'openwebtext', "
                           "or a path to a plain-text file (one sentence per line)."
                       ))
    build.add_argument("-n", "--n", dest="n_sentences", type=int, default=10_000,
                       help="Number of sentences to process (default: 10000)")
    build.add_argument("--layers", default="6",
                       help="Comma-separated layer indices, e.g. '3,6,9' (default: 6)")
    build.add_argument("--hook", dest="hook_point", default="resid_post",
                       choices=["resid_pre", "resid_mid", "resid_post", "mlp_out"],
                       help="Hook point (default: resid_post)")
    build.add_argument("--max-seq-len", type=int, default=128,
                       help="Tokenizer max length (default: 128)")
    build.add_argument("--batch-size", type=int, default=32,
                       help="Sentences per forward pass (default: 32)")
    build.add_argument("--shard-size", type=int, default=500_000,
                       help="Tokens per shard file (default: 500000)")
    build.add_argument("--max-tokens", type=int, default=None,
                       help="Stop after collecting this many tokens (default: no cap)")
    build.add_argument("--dtype",
                       choices=["float32", "float16", "bfloat16"],
                       default="float32",
                       help="Storage dtype (default: float32)")
    build.add_argument("--device", default="auto",
                       help="'auto', 'cpu', 'cuda', 'mps' (default: auto)")
    build.add_argument("--include-bos", action="store_true",
                       help="Include BOS token activations in the dataset")
    build.add_argument("--min-tokens", type=int, default=3,
                       help="Skip sentences with fewer real tokens (default: 3)")
    build.add_argument("--out", default="sae_data/run",
                       help="Output directory (default: sae_data/run)")
    build.add_argument("--overwrite", action="store_true",
                       help="Overwrite an existing dataset at --out")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--quiet", action="store_true",
                       help="Suppress progress bars")

    # ── Retrieval parameters ─────────────────────────────────────────────────
    ret = p.add_argument_group("Retrieval options (--top-k / --top-k-tensor)")
    ret.add_argument("--layer", type=int, default=None,
                     help="Layer to retrieve from (required for retrieval modes)")
    ret.add_argument("-k", type=int, default=20,
                     help="Number of top snippets (default: 20)")
    ret.add_argument("--threshold", type=float, default=None,
                     help="Minimum activation score to include")
    ret.add_argument("--context", type=int, default=8,
                     help="Context tokens before/after each result (default: 8)")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------

def _mode_inspect(path: str) -> None:
    from spore.activation_pipeline.sae_dataset import SAEDataset

    ds = SAEDataset.load(path)
    info = ds.describe()

    print("\n── SAEDataset summary ──────────────────────────────────")
    for k, v in info.items():
        print(f"  {k:<18}  {v}")

    print(f"\n  Token map shape : {tuple(ds._token_map.shape)}")
    print(f"  Sentences       : {len(ds.texts):,}")

    # List shard files
    for layer in ds.layers:
        hp = ds.hook_point
        layer_dir = ds._root / f"layer_{layer:02d}_{hp}"
        shards = sorted(layer_dir.glob("shard_*.pt"))
        total_sz = sum(s.stat().st_size for s in shards) / 1e9
        print(
            f"  Layer {layer:>2} shards : {len(shards)} files  "
            f"({total_sz:.2f} GB)"
        )

    print()


def _mode_build(args: argparse.Namespace) -> None:
    import torch
    from spore.activation_pipeline import CorpusLoader
    from spore.activation_pipeline.sae_dataset import SAEDataset, SAEDatasetConfig

    # ── Parse layers ─────────────────────────────────────────────────────────
    try:
        layers = [int(l.strip()) for l in args.layers.split(",") if l.strip()]
    except ValueError:
        logger.error("--layers must be comma-separated integers, e.g. '3,6,9'")
        sys.exit(1)

    # ── Load corpus ───────────────────────────────────────────────────────────
    n = args.n_sentences
    src = args.corpus
    logger.info("Loading corpus: %s  (n=%d)", src, n)

    if src == "diverse":
        texts, labels = _load_diverse(n, args.seed)
    elif src == "openwebtext":
        texts, labels = _load_openwebtext(n, args.seed)
    elif Path(src).is_file():
        texts, labels = _load_file(src, n, args.seed)
    else:
        logger.error(
            "Unknown corpus source %r.  "
            "Use 'diverse', 'openwebtext', or a file path.", src
        )
        sys.exit(1)

    logger.info("Corpus ready: %d sentences", len(texts))

    # ── Build config ──────────────────────────────────────────────────────────
    cfg = SAEDatasetConfig(
        model_name   = args.model,
        layers       = layers,
        hook_point   = args.hook_point,
        n_sentences  = n,
        max_seq_len  = args.max_seq_len,
        batch_size   = args.batch_size,
        shard_size   = args.shard_size,
        max_tokens   = args.max_tokens,
        dtype        = args.dtype,
        device       = args.device,
        seed         = args.seed,
        include_bos  = args.include_bos,
        min_tokens   = args.min_tokens,
        out_root     = args.out,
        overwrite    = args.overwrite,
    )

    # ── Build ─────────────────────────────────────────────────────────────────
    ds = SAEDataset.build(cfg, texts, labels=labels,
                          show_progress=not args.quiet)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── Build complete ──────────────────────────────────────")
    for k, v in ds.describe().items():
        print(f"  {k:<18}  {v}")
    print()

    # Quick activation stats for the first layer
    first_layer = layers[0]
    print(f"Computing activation stats for layer {first_layer} …")
    s = ds.stats(first_layer)
    mean_norm = float(s["l2_norm_mean"])
    mean_act  = float(s["mean"].mean())
    std_act   = float(s["std"].mean())
    print(
        f"  layer {first_layer}  |  "
        f"mean_act={mean_act:.4f}  std_act={std_act:.4f}  "
        f"mean_L2_norm={mean_norm:.4f}"
    )
    print(f"\nDataset saved to: {args.out}\n")


def _mode_topk(
    dataset_path: str,
    layer: int | None,
    feature,   # int or str (path to tensor)
    k: int,
    threshold: float | None,
    context: int,
) -> None:
    import torch
    from spore.activation_pipeline.sae_dataset import SAEDataset

    ds = SAEDataset.load(dataset_path)

    if layer is None:
        if len(ds.layers) == 1:
            layer = ds.layers[0]
        else:
            logger.error(
                "Multiple layers available %s — specify --layer", ds.layers
            )
            sys.exit(1)

    # Resolve feature argument
    if isinstance(feature, str):
        feat = torch.load(feature, weights_only=True)
        desc = f"direction from {feature}"
    else:
        feat = int(feature)
        desc = f"neuron {feature}"

    print(
        f"\nTop-{k} snippets for {desc}  "
        f"[layer {layer}, hook={ds.hook_point}]"
        + (f"  threshold={threshold}" if threshold is not None else "")
    )
    print("─" * 72)

    hits = ds.top_k_snippets(
        layer          = layer,
        feature        = feat,
        k              = k,
        threshold      = threshold,
        context_tokens = context,
    )

    if not hits:
        print("  (no results above threshold)")
    else:
        for h in hits:
            print(h.pretty())
            if h.label:
                print(f"     label={h.label}  sent_idx={h.sentence_idx}")

    print()


# ---------------------------------------------------------------------------
# Corpus loaders
# ---------------------------------------------------------------------------

def _load_diverse(n: int, seed: int):
    from spore.activation_pipeline import CorpusLoader
    texts = CorpusLoader.diverse_sentences(n=n, seed=seed)
    labels = _label_by_domain(texts)
    return texts, labels


def _load_openwebtext(n: int, seed: int):
    try:
        from spore.activation_pipeline import CorpusLoader
        texts = CorpusLoader.from_openwebtext(n=n, seed=seed)
    except Exception as exc:
        logger.error(
            "Failed to load OpenWebText (%s). "
            "Install datasets: pip install datasets", exc
        )
        sys.exit(1)
    return texts, None


def _load_file(path: str, n: int | None, seed: int):
    from spore.activation_pipeline import CorpusLoader
    texts = CorpusLoader.from_file(path, n=n, seed=seed)
    return texts, None


def _label_by_domain(texts: list[str]) -> list[str]:
    """Keyword-based domain labelling (mirrors collect_activations.py)."""
    _DOMAINS = [
        ("science",     ["photosynthesis", "dna", "atom", "quantum", "evolution",
                          "cell", "molecule", "enzyme", "orbit", "spectrum"]),
        ("technology",  ["neural network", "algorithm", "software", "robot",
                          "computer", "internet", "processor", "transistor"]),
        ("language",    ["grammar", "syntax", "metaphor", "dialect", "phoneme",
                          "vocabulary", "morpheme", "linguistics", "semantics"]),
        ("math",        ["theorem", "prime", "integral", "matrix", "polynomial",
                          "derivative", "proof", "equation", "function", "topology"]),
        ("history",     ["century", "civilization", "empire", "revolution",
                          "ancient", "medieval", "dynasty", "treaty", "war"]),
        ("philosophy",  ["consciousness", "ethics", "epistemology", "metaphysics",
                          "logic", "morality", "ontology", "dialectic"]),
        ("health",      ["neuron", "muscle", "hormone", "immune", "metabolism",
                          "protein", "tissue", "diagnosis", "symptom"]),
        ("environment", ["ecosystem", "climate", "carbon", "biodiversity",
                          "species", "ocean", "habitat", "fossil", "renewable"]),
    ]

    labels = []
    for text in texts:
        lower = text.lower()
        matched = "general"
        for domain, kws in _DOMAINS:
            if any(kw in lower for kw in kws):
                matched = domain
                break
        labels.append(matched)
    return labels


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.inspect:
        _mode_inspect(args.inspect)

    elif args.topk_feature is not None:
        if not args.dataset_path:
            logger.error("Provide a dataset path as a positional argument.")
            sys.exit(1)
        _mode_topk(
            dataset_path = args.dataset_path,
            layer        = args.layer,
            feature      = args.topk_feature,
            k            = args.k,
            threshold    = args.threshold,
            context      = args.context,
        )

    elif args.topk_tensor is not None:
        if not args.dataset_path:
            logger.error("Provide a dataset path as a positional argument.")
            sys.exit(1)
        _mode_topk(
            dataset_path = args.dataset_path,
            layer        = args.layer,
            feature      = args.topk_tensor,   # file path string
            k            = args.k,
            threshold    = args.threshold,
            context      = args.context,
        )

    else:
        _mode_build(args)


if __name__ == "__main__":
    main()
