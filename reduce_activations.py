#!/usr/bin/env python3
"""
Backward-compatibility shim.
The canonical implementation lives at scripts/reduce_activations.py.

Usage (unchanged):
    python reduce_activations.py --src activations/run --method both
"""
import pathlib, runpy
runpy.run_path(
    str(pathlib.Path(__file__).parent / "scripts" / "reduce_activations.py"),
    run_name="__main__",
)
