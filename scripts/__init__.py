"""
scripts/ — pipeline CLI entry points.

Each script can be run directly::

    python scripts/collect_activations.py --model gpt2 --n 500
    python scripts/reduce_activations.py  --method both
    python scripts/build_sae_dataset.py   --model gpt2 --n 20000 --layers 6
    python scripts/train_sae.py           --dataset sae_data/gpt2_l6 --layer 6

Root-level shims (collect_activations.py, etc.) delegate here for backwards
compatibility.
"""
