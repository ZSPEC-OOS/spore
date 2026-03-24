#!/usr/bin/env python
"""
Backward-compatibility shim.
The canonical implementation lives at scripts/train_sae.py.

Usage (unchanged):
    python train_sae.py --dataset sae_data/gpt2_l6 --layer 6 --steps 50000
"""
import pathlib, runpy
runpy.run_path(
    str(pathlib.Path(__file__).parent / "scripts" / "train_sae.py"),
    run_name="__main__",
)
