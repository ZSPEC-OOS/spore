#!/usr/bin/env python
"""
Backward-compatibility shim.
The canonical implementation lives at scripts/build_sae_dataset.py.

Usage (unchanged):
    python build_sae_dataset.py --model gpt2 --n 20000 --layers 6 --out sae_data/gpt2_l6
"""
import pathlib, runpy
runpy.run_path(
    str(pathlib.Path(__file__).parent / "scripts" / "build_sae_dataset.py"),
    run_name="__main__",
)
