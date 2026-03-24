import numpy as np
import pandas as pd

from spore.app.prompt_trajectory import _compute_metrics, _token_type


def test_token_type_buckets():
    assert _token_type(" ") == "whitespace"
    assert _token_type("hello") == "alpha"
    assert _token_type("123") == "numeric"
    assert _token_type("abc123") == "alnum"
    assert _token_type("!") == "punct/mixed"


def test_compute_metrics_shapes_and_values():
    # 3 layers, 2 tokens, d_model=2
    X = np.array([
        [0.0, 0.0], [1.0, 0.0],   # layer 0
        [1.0, 0.0], [2.0, 0.0],   # layer 1
        [3.0, 0.0], [4.0, 0.0],   # layer 2
    ], dtype=np.float32)
    df = pd.DataFrame({
        "layer": [0, 0, 1, 1, 2, 2],
        "token_idx": [0, 1, 0, 1, 0, 1],
    })

    metrics = _compute_metrics(df, X)

    assert list(metrics.columns) == ["layer", "layer_distance", "curvature", "subspace_rank"]
    assert len(metrics) == 3
    assert metrics.loc[0, "layer_distance"] == 0.0
    assert metrics.loc[1, "layer_distance"] > 0.0
    assert metrics.loc[2, "curvature"] > 0.0
