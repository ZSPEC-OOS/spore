#!/usr/bin/env python3
"""
collect_activations.py — Example / CLI runner for the activation pipeline.

Usage examples
--------------
# Quick run — gpt2, built-in corpus, 3 layers, 500 sentences
python collect_activations.py

# Pythia-160m, all layers, 1000 sentences, batch 64
python collect_activations.py \\
    --model EleutherAI/pythia-160m \\
    --n 1000 \\
    --layers all \\
    --batch 64 \\
    --out activations/pythia160m_run1

# Use openwebtext subset instead of built-in corpus
python collect_activations.py \\
    --model gpt2 \\
    --corpus openwebtext \\
    --n 2000 \\
    --layers 0,3,6,9,11 \\
    --hook resid_post \\
    --pool mean \\
    --dtype float16 \\
    --out activations/gpt2_owt

# Load and inspect a saved run
python collect_activations.py --inspect activations/gpt2_owt
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
logger = logging.getLogger("collect_activations")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # ── Inspect-only mode ────────────────────────────────────────────────────
    if args.inspect:
        _inspect(args.inspect)
        return

    from spore.activation_pipeline import ActivationCollector, CorpusLoader, ActivationCache
    from spore.activation_pipeline.collector import HookPoint, PoolStrategy
    import torch

    # ── Corpus ───────────────────────────────────────────────────────────────
    if args.corpus == "builtin":
        logger.info("Using built-in diverse corpus (n=%d, seed=%d)", args.n, args.seed)
        texts  = CorpusLoader.diverse_sentences(n=args.n, seed=args.seed)
        labels = _assign_domain_labels(texts)
    elif args.corpus == "openwebtext":
        logger.info("Streaming openwebtext corpus (n=%d)", args.n)
        texts  = CorpusLoader.from_openwebtext(n=args.n, seed=args.seed)
        labels = None
    elif Path(args.corpus).exists():
        logger.info("Loading corpus from file: %s (n=%s)", args.corpus, args.n or "all")
        texts  = CorpusLoader.from_file(args.corpus, n=args.n, seed=args.seed)
        labels = None
    else:
        sys.exit(f"Unknown --corpus value: {args.corpus!r}")

    logger.info("Corpus ready: %d sentences", len(texts))

    # ── Layers ───────────────────────────────────────────────────────────────
    layers: list[int] | None
    if args.layers == "all":
        layers = None          # ActivationCollector will collect every layer
    else:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    # ── Dtype ────────────────────────────────────────────────────────────────
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    if args.dtype not in dtype_map:
        sys.exit(f"--dtype must be one of: {list(dtype_map)}")
    dtype = dtype_map[args.dtype]

    # ── Collector ────────────────────────────────────────────────────────────
    collector = ActivationCollector(
        model_name  = args.model,
        layers      = layers,
        hook_point  = HookPoint(args.hook),
        pool        = PoolStrategy(args.pool),
        dtype       = dtype,
        max_seq_len = args.max_seq_len,
        device      = args.device or None,
    )

    # ── Collect ──────────────────────────────────────────────────────────────
    logger.info(
        "Collecting: model=%s  hook=%s  pool=%s  batch=%d  seq=%d",
        args.model, args.hook, args.pool, args.batch, args.max_seq_len,
    )
    run = collector.collect(
        texts       = texts,
        batch_size  = args.batch,
        labels      = labels,
        show_progress = True,
        checkpoint_tag = Path(args.out).name,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = ActivationCache.save(run, args.out, overwrite=args.overwrite)
    logger.info("Saved to %s", out_path)

    # ── Quick summary ─────────────────────────────────────────────────────────
    _print_summary(run, out_path)


# ---------------------------------------------------------------------------
# Inspect helper
# ---------------------------------------------------------------------------

def _inspect(root: str) -> None:
    from spore.activation_pipeline import ActivationCache
    import json

    try:
        meta = ActivationCache.describe(root)
    except FileNotFoundError:
        sys.exit(f"No saved run found at: {root}")

    print("\n── Saved run metadata ─────────────────────────────────────────")
    print(json.dumps(meta, indent=2))

    # Also show per-layer tensor paths
    from pathlib import Path
    layer_files = sorted(Path(root).glob("layer_*.pt"))
    if layer_files:
        print(f"\nLayer files ({len(layer_files)}):")
        for f in layer_files:
            size_mb = f.stat().st_size / 1_048_576
            print(f"  {f.name}  ({size_mb:.1f} MB)")
    print()


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(run, out_path: Path) -> None:
    import torch

    print("\n" + "─" * 60)
    print(f"  Model         : {run.model_name}")
    print(f"  Sentences     : {len(run.texts)}")
    print(f"  Layers saved  : {run.layers}")
    print(f"  Hook point    : {run.hook_point.value}")
    print(f"  Pool strategy : {run.pool.value}")
    print(f"  d_model       : {run.d_model}")
    print(f"  seq_len       : {run.tokens.shape[1]}")
    print(f"  Elapsed       : {run.elapsed_sec:.1f} s")

    total_mb = sum(
        t.element_size() * t.nelement()
        for t in run.activations.values()
    ) / 1_048_576
    print(f"  Activation MB : {total_mb:.1f} MB  ({len(run.activations)} layers)")

    first_layer = run.layers[0]
    act = run.activations[first_layer]
    print(f"\n  Layer {first_layer:02d} tensor  : shape={tuple(act.shape)}, dtype={act.dtype}")
    print(f"  Saved to      : {out_path}")

    if run.labels:
        from collections import Counter
        counts = Counter(run.labels)
        print(f"\n  Label distribution ({len(counts)} classes):")
        for label, cnt in sorted(counts.items()):
            print(f"    {label:30s}  {cnt:4d}")
    print("─" * 60 + "\n")


# ---------------------------------------------------------------------------
# Domain label assignment (built-in corpus only)
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: list[tuple[str, list[str]]] = [
    ("science",     ["photosynthesis", "dna", "cell", "neuron", "enzyme", "virus",
                     "atmosphere", "ozone", "evolution", "quantum", "particle",
                     "black hole", "star", "supernova", "coral", "bioluminescence"]),
    ("technology",  ["neural network", "gradient", "transformer", "tokenis", "attention",
                     "backprop", "convolution", "recurrent", "cache", "compiler",
                     "database", "api", "distributed", "gpu", "binary search"]),
    ("language",    ["syntax", "semantic", "pragmatic", "morphology", "phoneme",
                     "lexical", "metaphor", "discourse", "sign language", "sociolinguistic"]),
    ("mathematics", ["pythagorean", "prime", "euler", "fourier", "eigenvalue",
                     "calculus", "manifold", "graph theory", "probability", "proof"]),
    ("history",     ["industrial revolution", "printing press", "democracy", "cold war",
                     "renaissance", "colonialism", "world war", "silk road", "nationalism"]),
    ("philosophy",  ["epistemology", "mind-body", "free will", "consciousness",
                     "heuristic", "cognitive bias", "phenomenology", "falsifiab"]),
    ("arts",        ["narrative", "rhythm", "rhyme", "sonnet", "jazz", "motif",
                     "irony", "stream of consciousness", "colour symbolism", "editing"]),
    ("health",      ["immune", "sleep", "vaccine", "antibiotic", "cortisol",
                     "epigenetic", "placebo", "mental health", "nutrition", "clinical trial"]),
    ("economics",   ["supply", "demand", "inflation", "compound interest",
                     "market failure", "game theory", "monetary policy", "network effect"]),
    ("environment", ["deforestation", "ocean acidification", "renewable", "water cycle",
                     "microplastic", "biodiversity", "carbon capture", "permafrost"]),
    ("psychology",  ["classical conditioning", "attachment", "amygdala", "self-efficacy",
                     "social identity", "intrinsic motivation", "cognitive dissonance"]),
    ("astronomy",   ["general relativity", "dark matter", "big bang", "neutron star",
                     "quantum mechanics", "cosmic microwave", "exoplanet", "laser"]),
    ("everyday",    ["learning", "writing", "sleep", "reading", "cooking", "listening",
                     "travelling", "journal", "boredom", "habit", "public speaking"]),
]


def _assign_domain_labels(texts: list[str]) -> list[str]:
    labels = []
    for text in texts:
        lower = text.lower()
        assigned = "other"
        for domain, keywords in _DOMAIN_KEYWORDS:
            if any(kw in lower for kw in keywords):
                assigned = domain
                break
        labels.append(assigned)
    return labels


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect residual-stream activations from a transformer model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model
    p.add_argument("--model",   default="gpt2",
                   help="TransformerLens model name (default: gpt2)")
    p.add_argument("--device",  default=None,
                   help="cuda / mps / cpu  (default: auto-detect)")

    # Corpus
    p.add_argument("--corpus",  default="builtin",
                   help="builtin | openwebtext | /path/to/file.txt  (default: builtin)")
    p.add_argument("--n",       type=int, default=500,
                   help="Number of sentences to collect (default: 500)")
    p.add_argument("--seed",    type=int, default=42,
                   help="Random seed for corpus sampling (default: 42)")

    # Collection
    p.add_argument("--layers",      default="0,3,6,9,11",
                   help="Comma-separated layer indices, or 'all' (default: 0,3,6,9,11)")
    p.add_argument("--hook",        default="resid_post",
                   choices=["resid_pre", "resid_mid", "resid_post", "mlp_out"],
                   help="Hook point in each transformer block (default: resid_post)")
    p.add_argument("--pool",        default="mean",
                   choices=["mean", "last", "all"],
                   help="Sequence pooling strategy (default: mean)")
    p.add_argument("--batch",       type=int, default=32,
                   help="Batch size for forward passes (default: 32)")
    p.add_argument("--max-seq-len", type=int, default=128, dest="max_seq_len",
                   help="Truncate / pad to this token length (default: 128)")
    p.add_argument("--dtype",       default="float16",
                   choices=["float16", "float32", "bfloat16"],
                   help="Storage dtype for activations (default: float16)")

    # Output
    p.add_argument("--out",       default="activations/run",
                   help="Output directory (default: activations/run)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite an existing run at --out")

    # Inspect mode
    p.add_argument("--inspect", metavar="DIR", default=None,
                   help="Print metadata for a saved run and exit")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
