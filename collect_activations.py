#!/usr/bin/env python3
"""
Backward-compatibility shim.
The canonical implementation lives at scripts/collect_activations.py.

Usage (unchanged):
    python collect_activations.py --model gpt2 --n 500 --out activations/run
"""
import pathlib, runpy
runpy.run_path(
    str(pathlib.Path(__file__).parent / "scripts" / "collect_activations.py"),
    run_name="__main__",
)
