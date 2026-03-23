"""
reduction.py — UMAP and PCA dimensionality reduction for cached activations.

Public API
----------
compute_pca(activations, n_components=50)   → ProjectionResult
compute_umap(activations, n_components=2, …) → ProjectionResult

to_dataframe(result, metadata)              → pd.DataFrame (colored scatter)
save_projection(df, path, fmt)              → Path
load_projection(path)                       → pd.DataFrame

ProjectionResult
    .coords          np.ndarray [N, n_components]
    .model           fitted PCA | UMAP object  (callable for .transform())
    .method          "pca" | "umap"
    .explained_var   np.ndarray | None  (PCA: variance per component)
    .explained_ratio np.ndarray | None  (PCA: fraction of total variance)
    .params          dict  (hyperparameters used)

Typical workflow
----------------
>>> run   = ActivationCache.load("activations/run_01")
>>> acts  = run.activations[6].float().numpy()          # [N, d_model]

>>> pca   = compute_pca(acts, n_components=50)
>>> umap_ = compute_umap(pca.coords, n_components=2)    # 2-stage: PCA → UMAP

>>> meta  = {"label": run.labels, "layer": 6}
>>> df    = to_dataframe(umap_, meta)
>>> save_projection(df, "projections/layer06_umap.parquet")

Multi-layer sweep
-----------------
>>> suite = ProjectionSuite.from_run(run, method="umap", pca_pre=50)
>>> suite.save("projections/run_01/")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ProjectionResult
# ---------------------------------------------------------------------------

@dataclass
class ProjectionResult:
    """
    Output of :func:`compute_pca` or :func:`compute_umap`.

    Attributes
    ----------
    coords : np.ndarray
        Shape [N, n_components].  float32.
    model : object
        The fitted sklearn PCA or umap.UMAP object.
        Call ``model.transform(X)`` to project new data (PCA always works;
        UMAP parametric mode required for new-data transform).
    method : str
        "pca" or "umap".
    explained_var : np.ndarray | None
        PCA only — absolute explained variance per component [n_components].
    explained_ratio : np.ndarray | None
        PCA only — fraction of total variance per component [n_components].
    params : dict
        Hyperparameters used to produce this result.
    elapsed_sec : float
        Wall-clock fitting + transform time.
    n_samples : int
        Number of data points.
    n_input_dims : int
        Dimensionality of the input activations.
    """

    coords:          np.ndarray
    model:           Any
    method:          str
    params:          Dict[str, Any]
    n_samples:       int
    n_input_dims:    int
    elapsed_sec:     float           = 0.0
    explained_var:   Optional[np.ndarray] = None
    explained_ratio: Optional[np.ndarray] = None

    # ── convenience ─────────────────────────────────────────────────────────

    @property
    def n_components(self) -> int:
        return self.coords.shape[1]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data using the fitted model (PCA always; UMAP if parametric)."""
        return self.model.transform(X).astype(np.float32)

    def __repr__(self) -> str:
        ev = ""
        if self.explained_ratio is not None:
            cumulative = self.explained_ratio.sum() * 100
            ev = f", expl_var={cumulative:.1f}%"
        return (
            f"ProjectionResult(method={self.method!r}, "
            f"shape={self.coords.shape}{ev}, "
            f"elapsed={self.elapsed_sec:.1f}s)"
        )


# ---------------------------------------------------------------------------
# compute_pca
# ---------------------------------------------------------------------------

def compute_pca(
    activations:  np.ndarray,
    n_components: int = 50,
    whiten:       bool = False,
    random_state: int  = 42,
    center:       bool = True,
) -> ProjectionResult:
    """
    Fit and apply PCA to *activations*.

    Parameters
    ----------
    activations : np.ndarray
        Shape [N, d_model].  Accepts float16 / float32 / float64;
        automatically cast to float32 for sklearn.
    n_components : int
        Number of principal components to keep.
        Capped at min(N, d_model) automatically.
    whiten : bool
        Divide projected values by the square root of each component's
        variance, making all components unit-variance.
    random_state : int
    center : bool
        Subtract the mean before fitting (standard PCA pre-processing).

    Returns
    -------
    ProjectionResult
        ``.model`` is a fitted ``sklearn.decomposition.PCA`` object.
        ``.explained_ratio`` is an array of per-component variance fractions.
    """
    from sklearn.decomposition import PCA

    X = _to_float32(activations)
    n, d = X.shape
    n_components = min(n_components, n, d)

    logger.info("PCA  n=%d  d=%d  →  k=%d  (whiten=%s)", n, d, n_components, whiten)
    t0 = time.perf_counter()

    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        random_state=random_state,
    )
    coords = pca.fit_transform(X).astype(np.float32)  # [N, k]
    elapsed = time.perf_counter() - t0

    logger.info(
        "PCA done in %.2f s  —  cumulative explained variance: %.1f%%",
        elapsed, pca.explained_variance_ratio_.sum() * 100,
    )

    return ProjectionResult(
        coords          = coords,
        model           = pca,
        method          = "pca",
        params          = dict(n_components=n_components, whiten=whiten,
                               random_state=random_state, center=center),
        n_samples       = n,
        n_input_dims    = d,
        elapsed_sec     = elapsed,
        explained_var   = pca.explained_variance_.astype(np.float32),
        explained_ratio = pca.explained_variance_ratio_.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# compute_umap
# ---------------------------------------------------------------------------

def compute_umap(
    activations:  np.ndarray,
    n_components: int   = 2,
    n_neighbors:  int   = 15,
    min_dist:     float = 0.1,
    metric:       str   = "cosine",
    random_state: int   = 42,
    n_epochs:     Optional[int] = None,
    low_memory:   bool  = False,
    pca_pre:      Optional[int] = None,
    verbose:      bool  = True,
) -> ProjectionResult:
    """
    Fit and apply UMAP to *activations*.

    Parameters
    ----------
    activations : np.ndarray
        Shape [N, d].  For large d (e.g. 768), set ``pca_pre`` to first
        reduce to 50 components; this speeds up UMAP without meaningfully
        distorting the neighbourhood structure.
    n_components : int
        2 (default) or 3 for 3-D scatter plots.
    n_neighbors : int
        Controls local vs. global structure trade-off.
        Smaller → more local detail.  Larger → broader topology.
    min_dist : float
        Minimum distance between points in the low-d embedding.
        Smaller → tighter clusters.
    metric : str
        Distance metric.  "cosine" works well for residual-stream
        activations.  Also try "euclidean" or "manhattan".
    random_state : int
    n_epochs : int | None
        Training epochs.  None → UMAP default (200 for large datasets).
    low_memory : bool
        Activates UMAP's low-memory mode (slower but saves RAM).
    pca_pre : int | None
        If given, first reduce to this many PCA dimensions before UMAP.
        Recommended: 50 for d ≥ 256.
    verbose : bool

    Returns
    -------
    ProjectionResult
        ``.model`` is a fitted ``umap.UMAP`` object.

    Raises
    ------
    ImportError
        If ``umap-learn`` is not installed.
    """
    try:
        import umap as umap_lib
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required.  Install it with:\n  pip install umap-learn"
        ) from exc

    X = _to_float32(activations)
    n, d = X.shape
    input_d = d  # track original dim for metadata

    # Optional PCA pre-processing
    pca_result: Optional[ProjectionResult] = None
    if pca_pre is not None and d > pca_pre:
        logger.info("Pre-reducing with PCA: %d → %d dims …", d, pca_pre)
        pca_result = compute_pca(X, n_components=pca_pre)
        X = pca_result.coords
        d = X.shape[1]

    logger.info(
        "UMAP  n=%d  d=%d  →  %dD  "
        "(neighbors=%d, min_dist=%.2f, metric=%s)",
        n, d, n_components, n_neighbors, min_dist, metric,
    )
    t0 = time.perf_counter()

    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_epochs=n_epochs,
        low_memory=low_memory,
        verbose=verbose,
    )
    coords = reducer.fit_transform(X).astype(np.float32)  # [N, n_components]
    elapsed = time.perf_counter() - t0

    logger.info("UMAP done in %.2f s", elapsed)

    params = dict(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        pca_pre=pca_pre,
    )
    if n_epochs is not None:
        params["n_epochs"] = n_epochs

    return ProjectionResult(
        coords       = coords,
        model        = reducer,
        method       = "umap",
        params       = params,
        n_samples    = n,
        n_input_dims = input_d,
        elapsed_sec  = elapsed,
    )


# ---------------------------------------------------------------------------
# to_dataframe
# ---------------------------------------------------------------------------

def to_dataframe(
    result:   ProjectionResult,
    metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Build a tidy scatter DataFrame from a :class:`ProjectionResult`.

    Parameters
    ----------
    result : ProjectionResult
    metadata : dict | None
        Keys become column names.  Values may be:
          • list/array of length N  — one entry per sample
          • scalar                   — broadcast to all rows

        Example::

            {"label": run.labels,
             "layer": 6,
             "checkpoint": 1000,
             "text": run.texts}

    Returns
    -------
    pd.DataFrame
        Columns: x, [y], [z], [metadata columns …]
        The "x/y/z" labels are chosen according to n_components.
    """
    n = result.n_samples
    cols: Dict[str, Any] = {}

    # ── coordinate columns ───────────────────────────────────────────────────
    axis_names = ["x", "y", "z", "c4", "c5"]  # handles up to 5 components
    for i in range(result.n_components):
        cols[axis_names[i]] = result.coords[:, i]

    # ── metadata columns ─────────────────────────────────────────────────────
    if metadata:
        for key, val in metadata.items():
            if isinstance(val, (list, np.ndarray, pd.Series)):
                arr = list(val)
                if len(arr) != n:
                    raise ValueError(
                        f"metadata[{key!r}] has length {len(arr)}, "
                        f"expected {n} (n_samples)."
                    )
                cols[key] = arr
            else:
                # scalar → broadcast
                cols[key] = [val] * n

    df = pd.DataFrame(cols)
    logger.debug(
        "to_dataframe: %d rows × %d cols  (axes=%s)",
        len(df), len(df.columns), axis_names[:result.n_components],
    )
    return df


# ---------------------------------------------------------------------------
# save / load projections
# ---------------------------------------------------------------------------

Format = Literal["csv", "parquet", "auto"]


def save_projection(
    df:     pd.DataFrame,
    path:   Union[str, Path],
    fmt:    Format = "auto",
    index:  bool = False,
    **kwargs,
) -> Path:
    """
    Save *df* to disk as CSV or Parquet.

    Parameters
    ----------
    df : pd.DataFrame
    path : str | Path
        File path (including extension if fmt="auto").
    fmt : "csv" | "parquet" | "auto"
        "auto" infers from the file extension.
    index : bool
        Whether to write the DataFrame index.
    **kwargs
        Passed through to ``df.to_csv()`` or ``df.to_parquet()``.

    Returns
    -------
    Path
        Resolved output path.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "auto":
        fmt = "parquet" if path.suffix == ".parquet" else "csv"

    if fmt == "parquet":
        _require_parquet()
        df.to_parquet(path, index=index, **kwargs)
    else:
        df.to_csv(path, index=index, **kwargs)

    size_kb = path.stat().st_size / 1024
    logger.info("Saved projection → %s  (%.1f KB,  %d rows)", path, size_kb, len(df))
    return path


def load_projection(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a projection saved by :func:`save_projection`.

    Automatically detects CSV vs Parquet from the file extension.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".parquet":
        _require_parquet()
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    logger.info("Loaded projection ← %s  (%d rows × %d cols)", path, len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# ProjectionSuite — multi-layer sweep
# ---------------------------------------------------------------------------

class ProjectionSuite:
    """
    Run PCA and/or UMAP for every layer in a :class:`CollectionRun` and
    store the resulting DataFrames to a single directory.

    Usage
    -----
    >>> suite = ProjectionSuite.from_run(
    ...     run,
    ...     method="both",
    ...     pca_pre=50,
    ...     umap_components=2,
    ... )
    >>> suite.save("projections/run_01/")

    The output directory will contain::

      projections/run_01/
        layer_00_pca.parquet
        layer_00_umap.parquet
        layer_06_pca.parquet
        layer_06_umap.parquet
        …
        suite_summary.csv     ← per-layer PCA explained variance
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []   # {layer, method, df, result}
        self._pca_summaries: List[Dict[str, Any]] = []

    @classmethod
    def from_run(
        cls,
        run,                                  # CollectionRun
        method:          str   = "both",      # "pca" | "umap" | "both"
        pca_components:  int   = 50,
        umap_components: int   = 2,
        umap_neighbors:  int   = 15,
        umap_min_dist:   float = 0.1,
        umap_metric:     str   = "cosine",
        pca_pre:         Optional[int] = 50,
        extra_metadata:  Optional[Dict[str, Any]] = None,
        show_progress:   bool  = True,
    ) -> "ProjectionSuite":
        """
        Build a ProjectionSuite from a loaded :class:`CollectionRun`.

        Parameters
        ----------
        run : CollectionRun
        method : "pca" | "umap" | "both"
        pca_components : int
            Components to keep in PCA.
        umap_components : int
            2 or 3.
        umap_neighbors, umap_min_dist, umap_metric :
            Passed to :func:`compute_umap`.
        pca_pre : int | None
            Pre-PCA step before UMAP.  None → skip.
        extra_metadata : dict | None
            Additional columns to add to every DataFrame
            (e.g. {"checkpoint": 1000}).
        show_progress : bool
        """
        suite = cls()
        layers = run.layers

        bar = tqdm(layers, desc="ProjectionSuite", unit="layer") if show_progress else layers

        for layer in bar:
            if show_progress:
                bar.set_postfix(layer=layer)  # type: ignore[union-attr]

            acts = run.activations[layer].float().numpy()   # [N, d_model]

            # base metadata for this layer
            meta: Dict[str, Any] = {
                "layer": layer,
                "index": list(range(len(run.texts))),
                "text":  run.texts,
            }
            if run.labels is not None:
                meta["label"] = run.labels
            if extra_metadata:
                meta.update(extra_metadata)

            # PCA
            pca_result: Optional[ProjectionResult] = None
            if method in ("pca", "both"):
                pca_result = compute_pca(acts, n_components=pca_components)
                pca_df     = to_dataframe(pca_result, meta)
                suite._entries.append({
                    "layer": layer, "method": "pca",
                    "df": pca_df, "result": pca_result,
                })
                suite._pca_summaries.append({
                    "layer":              layer,
                    "n_components":       pca_result.n_components,
                    "cumulative_expl_var": float(pca_result.explained_ratio.sum()),
                    "top1_expl_var":      float(pca_result.explained_ratio[0]),
                    "top10_expl_var":     float(pca_result.explained_ratio[:10].sum()),
                })

            # UMAP
            if method in ("umap", "both"):
                # Use PCA coords as UMAP input when pca_pre is not None
                umap_input = (
                    pca_result.coords
                    if (pca_result is not None and pca_pre is None)
                    else acts
                )
                umap_result = compute_umap(
                    umap_input,
                    n_components=umap_components,
                    n_neighbors=umap_neighbors,
                    min_dist=umap_min_dist,
                    metric=umap_metric,
                    pca_pre=pca_pre,
                    verbose=False,
                )
                umap_df = to_dataframe(umap_result, meta)
                suite._entries.append({
                    "layer": layer, "method": "umap",
                    "df": umap_df, "result": umap_result,
                })

        logger.info("ProjectionSuite: %d projections across %d layers.", len(suite._entries), len(layers))
        return suite

    def save(self, root: Union[str, Path], fmt: Format = "parquet") -> Path:
        """
        Persist all projection DataFrames to *root*.

        Returns
        -------
        Path
            The resolved root directory.
        """
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)

        for entry in tqdm(self._entries, desc="Saving projections"):
            layer  = entry["layer"]
            method = entry["method"]
            ext    = "parquet" if fmt == "parquet" else "csv"
            path   = root / f"layer_{layer:02d}_{method}.{ext}"
            save_projection(entry["df"], path, fmt=fmt)

        # PCA explained-variance summary
        if self._pca_summaries:
            summary_df = pd.DataFrame(self._pca_summaries)
            summary_path = root / "suite_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info("PCA summary → %s", summary_path)

        logger.info("ProjectionSuite saved to %s  (%d files)", root, len(self._entries))
        return root

    def get(self, layer: int, method: str = "umap") -> Optional[pd.DataFrame]:
        """Retrieve a specific layer/method DataFrame."""
        for entry in self._entries:
            if entry["layer"] == layer and entry["method"] == method:
                return entry["df"]
        return None

    def get_result(self, layer: int, method: str = "umap") -> Optional[ProjectionResult]:
        """Retrieve the fitted ProjectionResult for a specific layer/method."""
        for entry in self._entries:
            if entry["layer"] == layer and entry["method"] == method:
                return entry["result"]
        return None

    @property
    def layers(self) -> List[int]:
        return sorted({e["layer"] for e in self._entries})

    @property
    def methods(self) -> List[str]:
        return sorted({e["method"] for e in self._entries})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_float32(X: np.ndarray) -> np.ndarray:
    """Cast any numeric array to float32 for sklearn/umap compatibility."""
    if isinstance(X, np.ndarray):
        return X.astype(np.float32, copy=False)
    # Handle torch tensors passed accidentally
    try:
        import torch
        if isinstance(X, torch.Tensor):
            return X.float().numpy()
    except ImportError:
        pass
    return np.asarray(X, dtype=np.float32)


def _require_parquet() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError:
            raise ImportError(
                "Parquet I/O requires pyarrow or fastparquet.\n"
                "Install with:  pip install pyarrow"
            )
