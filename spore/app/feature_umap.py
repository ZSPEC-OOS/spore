"""
feature_umap.py — Pre-compute UMAP over SAE decoder directions.

Provides :class:`FeatureMapData` (the result) and
:func:`compute_feature_map` (the computation pipeline):

  1. Extract ``W_dec`` rows [n_features, d_model]  — unit-norm decoder directions.
  2. PCA pre-reduction (default 50 dims) — speeds up UMAP, reduces noise.
  3. UMAP to 2D — uses cosine metric, appropriate for unit-sphere directions.
  4. K-means clustering — semantic groupings of feature directions.
  5. Dataset scan — stream all tokens through the SAE encoder to collect
     per-feature statistics (max_act, mean_active, pct_active, top-1 token).
  6. Hover text — decode the top-activating token context per feature.

All heavy work is done once and cached by the Streamlit dashboard via
``@st.cache_data``; subsequent rerenders are instant.

Fallback behaviour
------------------
  • If ``umap-learn`` is not installed, falls back to PCA 2D coordinates.
  • If ``scikit-learn`` is not installed, clustering is skipped (all label 0).
  • If the dataset scan is skipped (``max_tokens=0``), activation arrays are
    all-zeros (UMAP still renders; activation coloring is disabled).

Usage::

    from spore.app.sae_feature import FeatureAnalyzer
    from spore.app.feature_umap import compute_feature_map

    analyzer = FeatureAnalyzer.from_checkpoint(ckpt, ds, layer=6)
    data = compute_feature_map(analyzer, n_clusters=20, max_tokens=150_000)

    import plotly.express as px
    fig = px.scatter(
        x=data.umap_xy[:, 0], y=data.umap_xy[:, 1],
        color=data.cluster_labels.astype(str),
        hover_name=data.hover_texts,
    )
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from spore.app.sae_feature import FeatureAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FeatureMapData
# ---------------------------------------------------------------------------

@dataclass
class FeatureMapData:
    """
    All pre-computed data needed to render the Feature UMAP scatter plot.

    Arrays are indexed ``[feature_idx]`` throughout.

    Attributes
    ----------
    umap_xy : np.ndarray [n_features, 2]
        2-D UMAP (or PCA fallback) coordinates.
    max_act : np.ndarray [n_features]
        Highest activation value seen across the scanned tokens.
    mean_active : np.ndarray [n_features]
        Mean activation among tokens where h[i] > 0.
    pct_active : np.ndarray [n_features]
        Percentage of tokens where feature fired (h[i] > 0).
    cluster_labels : np.ndarray [n_features]
        K-means cluster assignment (0 … n_clusters-1).
    hover_texts : list[str]
        Pre-built Plotly hover string per feature (includes snippet).
    n_features, d_model, layer : int
    n_clusters : int
    n_tokens_scanned : int
    pca_n_components : int
    pca_explained_var : float | None
        Cumulative PCA explained variance (0–1) or None if no PCA.
    umap_neighbors : int
    umap_min_dist : float
    used_umap : bool
        True if UMAP succeeded; False if fell back to PCA 2D.
    """

    umap_xy:           np.ndarray
    max_act:           np.ndarray
    mean_active:       np.ndarray
    pct_active:        np.ndarray
    cluster_labels:    np.ndarray
    hover_texts:       List[str]

    n_features:        int
    d_model:           int
    layer:             int
    n_clusters:        int
    n_tokens_scanned:  int
    pca_n_components:  int
    pca_explained_var: Optional[float]
    umap_neighbors:    int
    umap_min_dist:     float
    used_umap:         bool


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_feature_map(
    analyzer:       FeatureAnalyzer,
    n_pca:          int   = 50,
    umap_neighbors: int   = 15,
    umap_min_dist:  float = 0.1,
    n_clusters:     int   = 20,
    max_tokens:     int   = 200_000,
    pca_seed:       int   = 42,
    umap_seed:      int   = 42,
    kmeans_seed:    int   = 42,
) -> FeatureMapData:
    """
    Build a :class:`FeatureMapData` for *analyzer*'s SAE.

    Parameters
    ----------
    analyzer : FeatureAnalyzer
    n_pca : int
        PCA target dims (pre-reduction before UMAP).  Set 0 to skip PCA.
    umap_neighbors : int
        UMAP ``n_neighbors`` — controls local vs global structure.
    umap_min_dist : float
        UMAP ``min_dist`` — how tightly to pack points.
    n_clusters : int
        K-means cluster count for the cluster-coloring option.
    max_tokens : int
        Token scan cap.  Set 0 to skip the dataset scan (faster, no
        activation stats).
    pca_seed, umap_seed, kmeans_seed : int
        Random seeds for reproducibility.

    Returns
    -------
    FeatureMapData
    """
    F, D = analyzer.n_features, analyzer.d_model
    logger.info(
        "compute_feature_map: F=%d d=%d n_pca=%d k=%d n_neighbors=%d",
        F, D, n_pca, n_clusters, umap_neighbors,
    )

    # ── 1. Decoder directions ─────────────────────────────────────────
    W_dec = analyzer.sae.W_dec.data.float().cpu().numpy()  # [F, D], unit-norm rows
    norms = np.linalg.norm(W_dec, axis=1, keepdims=True)
    W_dec = W_dec / np.clip(norms, 1e-8, None)            # ensure unit-norm

    # ── 2. PCA pre-reduction ──────────────────────────────────────────
    n_pca_actual, pca_explained_var, W_dec_pca = _run_pca(W_dec, n_pca, pca_seed)

    # ── 3. UMAP ───────────────────────────────────────────────────────
    nn_safe = max(2, min(umap_neighbors, F - 2))
    umap_xy, used_umap = _run_umap(W_dec_pca, nn_safe, umap_min_dist, umap_seed)

    # ── 4. K-means ────────────────────────────────────────────────────
    cluster_labels = _run_kmeans(W_dec_pca, n_clusters, F, kmeans_seed)

    # ── 5. Dataset scan ───────────────────────────────────────────────
    if max_tokens > 0:
        max_act, mean_active, pct_active, top1_flat_idx, n_scanned = \
            _scan_activations(analyzer, F, max_tokens)
    else:
        max_act      = np.zeros(F, dtype=np.float32)
        mean_active  = np.zeros(F, dtype=np.float32)
        pct_active   = np.zeros(F, dtype=np.float32)
        top1_flat_idx = np.zeros(F, dtype=np.int64)
        n_scanned    = 0

    # ── 6. Hover texts ────────────────────────────────────────────────
    hover_texts = _build_hover_texts(
        analyzer, top1_flat_idx, max_act, pct_active
    )

    return FeatureMapData(
        umap_xy           = umap_xy.astype(np.float32),
        max_act           = max_act,
        mean_active       = mean_active,
        pct_active        = pct_active,
        cluster_labels    = cluster_labels.astype(np.int32),
        hover_texts       = hover_texts,
        n_features        = F,
        d_model           = D,
        layer             = analyzer.layer,
        n_clusters        = int(cluster_labels.max()) + 1,
        n_tokens_scanned  = n_scanned,
        pca_n_components  = n_pca_actual,
        pca_explained_var = pca_explained_var,
        umap_neighbors    = nn_safe,
        umap_min_dist     = umap_min_dist,
        used_umap         = used_umap,
    )


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _run_pca(
    W: np.ndarray,
    n_pca: int,
    seed: int,
) -> tuple:
    """
    PCA-reduce W [F, D] to at most n_pca dims.

    Returns (n_components_actual, explained_var_ratio, W_reduced).
    """
    F, D = W.shape
    if n_pca <= 0 or n_pca >= D:
        return D, None, W.astype(np.float32)

    n_comp = min(n_pca, D, F - 1)
    try:
        from sklearn.decomposition import PCA
        logger.info("PCA %d → %d dims …", D, n_comp)
        pca = PCA(n_components=n_comp, random_state=seed)
        W_pca = pca.fit_transform(W).astype(np.float32)
        ev = float(pca.explained_variance_ratio_.sum())
        logger.info("PCA explained variance: %.1f%%", ev * 100)
        return n_comp, ev, W_pca
    except ImportError:
        logger.warning("scikit-learn not available — skipping PCA")
        return D, None, W.astype(np.float32)
    except Exception as exc:
        logger.warning("PCA failed: %s — using raw directions", exc)
        return D, None, W.astype(np.float32)


def _run_umap(
    W: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> tuple:
    """
    Run UMAP on W [F, dim] → [F, 2].

    Returns (coords [F, 2], used_umap bool).
    Falls back to PCA 2D if umap-learn is unavailable.
    """
    try:
        import umap as umap_lib   # umap-learn package
        logger.info("UMAP: n_neighbors=%d min_dist=%.2f …", n_neighbors, min_dist)
        reducer = umap_lib.UMAP(
            n_components = 2,
            n_neighbors  = n_neighbors,
            min_dist     = min_dist,
            metric       = "cosine",
            random_state = seed,
            verbose      = False,
            low_memory   = True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coords = reducer.fit_transform(W).astype(np.float32)
        return coords, True
    except ImportError:
        logger.info("umap-learn not installed — falling back to PCA 2D")
    except Exception as exc:
        logger.warning("UMAP failed (%s) — falling back to PCA 2D", exc)

    # PCA 2D fallback
    try:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=seed).fit_transform(W).astype(np.float32)
    except ImportError:
        # Last resort: random 2D coords
        rng = np.random.default_rng(seed)
        coords = rng.standard_normal((len(W), 2)).astype(np.float32)

    return coords, False


def _run_kmeans(
    W: np.ndarray,
    n_clusters: int,
    n_features: int,
    seed: int,
) -> np.ndarray:
    """
    K-means on W [F, dim] → integer cluster labels [F].
    Falls back to zeros if sklearn is unavailable or fails.
    """
    k = min(n_clusters, n_features)
    try:
        from sklearn.cluster import MiniBatchKMeans
        logger.info("K-means k=%d …", k)
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, max_iter=200)
        return km.fit_predict(W).astype(np.int32)
    except ImportError:
        logger.warning("scikit-learn not available — skipping clustering")
    except Exception as exc:
        logger.warning("K-means failed: %s", exc)

    return np.zeros(n_features, dtype=np.int32)


def _scan_activations(
    analyzer:   FeatureAnalyzer,
    n_features: int,
    max_tokens: int,
) -> tuple:
    """
    Stream dataset shards through the SAE encoder.

    Returns
    -------
    (max_act, mean_active, pct_active, top1_flat_idx, n_scanned)
    All arrays have shape [n_features].
    """
    max_act_t    = torch.zeros(n_features)
    sum_act_t    = torch.zeros(n_features)
    n_active_t   = torch.zeros(n_features, dtype=torch.long)
    top1_flat_t  = torch.zeros(n_features, dtype=torch.long)
    flat_offset  = 0
    n_scanned    = 0

    logger.info("Scanning activations (max_tokens=%d) …", max_tokens)

    for _, shard in analyzer.dataset.iter_shards(analyzer.layer):
        x = shard.float().to(analyzer.device)
        with torch.no_grad():
            h = analyzer.sae(x).h.cpu()           # [batch, F]

        # Per-feature max in this shard and its local position
        batch_max, batch_argmax = h.max(dim=0)    # [F]

        # Update global top-1
        update_mask  = batch_max > max_act_t
        max_act_t    = torch.where(update_mask, batch_max, max_act_t)
        new_flat     = flat_offset + batch_argmax
        top1_flat_t  = torch.where(update_mask, new_flat, top1_flat_t)

        # Activation stats (ReLU SAE: h ≥ 0 always)
        sum_act_t   += h.sum(dim=0)
        n_active_t  += (h > 0).sum(dim=0)

        flat_offset += len(shard)
        n_scanned   += len(shard)
        if n_scanned >= max_tokens:
            break

    n_act  = n_active_t.numpy().astype(np.float64)
    max_a  = max_act_t.numpy().astype(np.float32)
    pct_a  = (100.0 * n_act / max(n_scanned, 1)).astype(np.float32)
    mean_a = np.where(
        n_act > 0,
        sum_act_t.numpy() / np.clip(n_act, 1, None),
        0.0,
    ).astype(np.float32)

    return max_a, mean_a, pct_a, top1_flat_t.numpy(), n_scanned


def _build_hover_texts(
    analyzer:      FeatureAnalyzer,
    top1_flat_idx: np.ndarray,   # [n_features] int
    max_act:       np.ndarray,   # [n_features] float
    pct_active:    np.ndarray,   # [n_features] float
) -> List[str]:
    """
    Build one Plotly hover string per feature.

    Shows: feature index, top-activating token in context, activation stats.
    HTML is used (Plotly hover supports it).
    """
    dataset = analyzer.dataset
    n       = len(top1_flat_idx)
    texts: List[str] = []

    for i in range(n):
        max_a = float(max_act[i])
        pct_a = float(pct_active[i])

        if max_a <= 0.0:
            texts.append(
                f"<b>Feature {i}</b><br>"
                f"<i style='color:#6e7681'>never activated</i>"
            )
            continue

        flat_i = int(top1_flat_idx[i])
        try:
            sent_idx = int(dataset._token_map[flat_i, 0])
            tok_pos  = int(dataset._token_map[flat_i, 1])
            sentence = dataset.texts[sent_idx]
            tok_str, ctx_before, ctx_after = dataset._decode_context(
                sentence, tok_pos, window=4
            )
            # Compact context: trim to ≤ 25 chars each side
            cb = (ctx_before[-25:] if len(ctx_before) > 25 else ctx_before)
            ca = (ctx_after[:25]   if len(ctx_after)  > 25 else ctx_after)
            snippet = f"{cb}<b>[{tok_str}]</b>{ca}"
        except Exception:
            snippet = "(context unavailable)"

        texts.append(
            f"<b>Feature {i}</b><br>"
            f"{snippet}<br>"
            f"max: {max_a:.3f} · freq: {pct_a:.2f}%"
        )

    return texts
