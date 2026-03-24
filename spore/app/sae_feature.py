"""
sae_feature.py — Data-analysis backend for the SAE Feature Dashboard.

Provides :class:`FeatureAnalyzer`, which wraps a loaded
:class:`~spore.activation_pipeline.sae.SparseAutoencoder` and a
:class:`~spore.activation_pipeline.sae_dataset.SAEDataset` and exposes the
three core queries the dashboard needs:

1. :meth:`FeatureAnalyzer.feature_histogram_data`
   Streams all shards through the SAE encoder, collects positive-valued
   activations for one feature, returns histogram statistics.

2. :meth:`FeatureAnalyzer.top_k_by_activation`
   Same streaming scan but maintains a min-heap to return the *k* tokens
   with the highest h[feat_idx], decorated with decoded text context.

3. :meth:`FeatureAnalyzer.logit_effects`
   Computes W_dec[feat_idx] @ W_U (TransformerLens unembedding) and
   returns the top promoted / suppressed output tokens — the "logit lens"
   view of what this feature writes into the residual stream.

Usage::

    from spore.app.sae_feature import FeatureAnalyzer

    analyzer = FeatureAnalyzer.from_checkpoint(
        ckpt_path    = "sae_checkpoints/gpt2_l6/latest",
        dataset_root = "sae_data/gpt2_l6",
        layer        = 6,
        device       = "cpu",
    )

    hist   = analyzer.feature_histogram_data(feat_idx=42)
    top20  = analyzer.top_k_by_activation(feat_idx=42, k=20, threshold=0.1)
    logits = analyzer.logit_effects(feat_idx=42, model_name="gpt2", top_k=10)
"""

from __future__ import annotations

import heapq
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from spore.activation_pipeline.sae import SAEConfig, SparseAutoencoder
from spore.activation_pipeline.sae_dataset import SAEDataset, SnippetResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HistogramData — lightweight result returned by feature_histogram_data()
# ---------------------------------------------------------------------------

@dataclass
class HistogramData:
    """
    Histogram statistics for one SAE feature over a token dataset.

    Attributes
    ----------
    values : np.ndarray
        Positive activation values sampled from the dataset (shape [n_active]).
        Used to build the histogram.
    n_total : int
        Total tokens scanned (may be < dataset size if max_tokens was set).
    n_active : int
        Tokens on which this feature activated (h > 0).
    pct_active : float
        Percentage of tokens where feature was active.
    max_act : float
        Maximum activation value seen.
    mean_active : float
        Mean activation among tokens where feature fired.
    mean_all : float
        Mean activation across all tokens (sparse mean, includes zeros).
    """
    values:       np.ndarray
    n_total:      int
    n_active:     int
    pct_active:   float
    max_act:      float
    mean_active:  float
    mean_all:     float

    @property
    def is_dead(self) -> bool:
        return self.n_active == 0


# ---------------------------------------------------------------------------
# LogitEffects — result of the logit-lens computation
# ---------------------------------------------------------------------------

@dataclass
class LogitEffects:
    """
    Top promoted and suppressed output tokens for one SAE feature.

    Each entry is ``(decoded_token_str, logit_delta)`` where
    ``logit_delta`` is the dot product of W_dec[feat_idx] with the
    corresponding column of the unembedding matrix W_U.

    Attributes
    ----------
    promoted : list[(str, float)]
        Top-k tokens most promoted by this feature (positive logit delta).
    suppressed : list[(str, float)]
        Top-k tokens most suppressed (negative logit delta).
    feat_idx : int
    model_name : str
    """
    promoted:   List[Tuple[str, float]]
    suppressed: List[Tuple[str, float]]
    feat_idx:   int
    model_name: str


# ---------------------------------------------------------------------------
# FeatureAnalyzer
# ---------------------------------------------------------------------------

class FeatureAnalyzer:
    """
    Wraps a trained :class:`SparseAutoencoder` and an :class:`SAEDataset`
    to answer per-feature inspection queries efficiently.

    Load once (expensive), then call any method cheaply:

    >>> analyzer = FeatureAnalyzer.from_checkpoint(ckpt, ds_root, layer=6)
    >>> hist  = analyzer.feature_histogram_data(42)
    >>> top20 = analyzer.top_k_by_activation(42, k=20)

    Parameters
    ----------
    sae : SparseAutoencoder
    dataset : SAEDataset
    layer : int
    device : str
    """

    def __init__(
        self,
        sae:     SparseAutoencoder,
        dataset: SAEDataset,
        layer:   int,
        device:  str = "cpu",
    ) -> None:
        self.sae     = sae
        self.dataset = dataset
        self.layer   = layer
        self.device  = device
        self._tl_model = None   # loaded lazily for logit lens

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path:    str,
        dataset_root: str,
        layer:        int,
        device:       str = "cpu",
    ) -> "FeatureAnalyzer":
        """
        Load a trained SAE from a checkpoint directory and pair it with
        an :class:`SAEDataset`.

        Parameters
        ----------
        ckpt_path : str
            A step_XXXXXXX/ directory (or the ``latest/`` symlink) produced
            by :class:`~spore.activation_pipeline.sae.SAETrainer`.
        dataset_root : str
            Root directory of an :class:`SAEDataset`.
        layer : int
            Which layer's activations to analyse.
        device : str
            ``"cpu"`` | ``"cuda"`` | ``"mps"``
        """
        ck  = Path(ckpt_path).resolve()
        meta_path = ck / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No meta.json in {ck}.  "
                "Make sure ckpt_path points to a step_XXXXXXX/ directory."
            )

        meta = json.loads(meta_path.read_text())
        cfg  = SAEConfig(**meta["cfg"])

        sae = SparseAutoencoder(cfg)
        weights = torch.load(
            ck / "weights.pt",
            weights_only=True,
            map_location=device,
        )
        sae.load_state_dict_sae(weights)
        sae.eval()
        sae.to(device)

        logger.info(
            "Loaded SAE: %s  layer=%d  n_features=%d  device=%s",
            ck.name, layer, cfg.n_features, device,
        )

        dataset = SAEDataset.load(dataset_root)
        return cls(sae, dataset, layer, device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        return self.sae.cfg.n_features

    @property
    def d_model(self) -> int:
        return self.sae.cfg.d_model

    @property
    def cfg(self) -> SAEConfig:
        return self.sae.cfg

    # ------------------------------------------------------------------
    # 1. Histogram data
    # ------------------------------------------------------------------

    def feature_histogram_data(
        self,
        feat_idx:   int,
        max_tokens: int = 200_000,
    ) -> HistogramData:
        """
        Stream shards through the SAE encoder and collect activation
        statistics for *feat_idx*.

        Parameters
        ----------
        feat_idx : int
            Feature index (0 ≤ feat_idx < n_features).
        max_tokens : int
            Cap on how many tokens to scan (for speed on large datasets).

        Returns
        -------
        HistogramData
        """
        self._validate_feat(feat_idx)

        pos_chunks: List[torch.Tensor] = []
        n_total = 0

        for _, shard in self.dataset.iter_shards(self.layer):
            x = shard.float().to(self.device)
            with torch.no_grad():
                out = self.sae(x)
            h = out.h[:, feat_idx].cpu()

            active = h > 0
            pos_chunks.append(h[active])
            n_total += len(shard)
            if n_total >= max_tokens:
                break

        if not pos_chunks:
            vals = np.empty(0, dtype=np.float32)
        else:
            vals = torch.cat(pos_chunks).numpy().astype(np.float32)

        n_active = len(vals)
        pct      = 100.0 * n_active / max(n_total, 1)

        return HistogramData(
            values      = vals,
            n_total     = n_total,
            n_active    = n_active,
            pct_active  = round(pct, 4),
            max_act     = float(vals.max()) if n_active > 0 else 0.0,
            mean_active = float(vals.mean()) if n_active > 0 else 0.0,
            mean_all    = float(vals.sum() / max(n_total, 1)),
        )

    # ------------------------------------------------------------------
    # 2. Top-k activating snippets
    # ------------------------------------------------------------------

    def top_k_by_activation(
        self,
        feat_idx:       int,
        k:              int = 20,
        threshold:      float = 0.0,
        context_tokens: int = 8,
        max_tokens:     int = 0,      # 0 = scan entire dataset
    ) -> List[SnippetResult]:
        """
        Return the *k* tokens whose SAE feature activation h[feat_idx] is
        highest (streaming, O(n_tokens) time, O(k) memory).

        Parameters
        ----------
        feat_idx : int
        k : int
            Number of results.
        threshold : float
            Minimum activation value; tokens below this are skipped.
        context_tokens : int
            Window size (tokens before/after) for text context.
        max_tokens : int
            If > 0, stop scanning after this many tokens.

        Returns
        -------
        List[SnippetResult]  — sorted by score descending.
        """
        self._validate_feat(feat_idx)

        heap:        list = []   # min-heap of (score, flat_token_idx)
        flat_offset: int  = 0
        n_scanned:   int  = 0

        for _, shard in self.dataset.iter_shards(self.layer):
            x = shard.float().to(self.device)
            with torch.no_grad():
                out = self.sae(x)
            scores = out.h[:, feat_idx].cpu().tolist()

            for local_i, score in enumerate(scores):
                if score <= threshold:
                    continue
                flat_i = flat_offset + local_i
                if len(heap) < k:
                    heapq.heappush(heap, (score, flat_i))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, flat_i))

            flat_offset += len(shard)
            n_scanned   += len(shard)
            if max_tokens > 0 and n_scanned >= max_tokens:
                break

        # Sort descending by score
        top = sorted(heap, key=lambda x: x[0], reverse=True)

        results: List[SnippetResult] = []
        for rank, (score, flat_idx) in enumerate(top, start=1):
            sent_idx = int(self.dataset._token_map[flat_idx, 0])
            tok_pos  = int(self.dataset._token_map[flat_idx, 1])
            sentence = self.dataset.texts[sent_idx]
            labels   = self.dataset.labels
            label    = labels[sent_idx] if labels else None

            tok_str, ctx_before, ctx_after = self.dataset._decode_context(
                sentence, tok_pos, context_tokens
            )

            results.append(SnippetResult(
                rank           = rank,
                score          = round(score, 5),
                flat_token_idx = flat_idx,
                sentence_idx   = sent_idx,
                token_pos      = tok_pos,
                sentence_text  = sentence,
                token_str      = tok_str,
                context_before = ctx_before,
                context_after  = ctx_after,
                label          = label,
            ))

        return results

    # ------------------------------------------------------------------
    # 3. Logit effects (logit lens)
    # ------------------------------------------------------------------

    def logit_effects(
        self,
        feat_idx:   int,
        model_name: str,
        top_k:      int = 10,
    ) -> Optional[LogitEffects]:
        """
        Compute how much feature *feat_idx* promotes or suppresses each
        vocabulary token via the decoder direction W_dec[feat_idx] @ W_U.

        Requires loading the transformer model with TransformerLens (CPU-only,
        called lazily, cached on this object).

        Parameters
        ----------
        feat_idx : int
        model_name : str
            TransformerLens model name (e.g. ``"gpt2"``).
        top_k : int
            Number of promoted / suppressed tokens to return.

        Returns
        -------
        LogitEffects | None — None if the model cannot be loaded.
        """
        self._validate_feat(feat_idx)

        tl_model = self._get_tl_model(model_name)
        if tl_model is None:
            return None

        try:
            W_U = tl_model.W_U.float().cpu()       # [d_model, vocab_size]
            tokenizer = tl_model.tokenizer

            # Decoder direction for this feature: [d_model]
            dec_dir = self.sae.W_dec.data[feat_idx].float().cpu()

            # Logit delta: positive = promoted, negative = suppressed
            logit_eff = dec_dir @ W_U               # [vocab_size]

            top_pos_vals, top_pos_idx = logit_eff.topk(top_k)
            top_neg_vals, top_neg_idx = (-logit_eff).topk(top_k)

            def _decode(ids, vals):
                out = []
                for idx, v in zip(ids.tolist(), vals.tolist()):
                    try:
                        tok = tokenizer.decode([idx])
                    except Exception:
                        tok = f"[{idx}]"
                    out.append((tok, round(v, 4)))
                return out

            return LogitEffects(
                promoted   = _decode(top_pos_idx, top_pos_vals),
                suppressed = _decode(top_neg_idx, top_neg_vals),
                feat_idx   = feat_idx,
                model_name = model_name,
            )

        except Exception as exc:
            logger.warning("logit_effects for feat %d failed: %s", feat_idx, exc)
            return None

    # ------------------------------------------------------------------
    # Summary info for a feature (cheap — no forward pass)
    # ------------------------------------------------------------------

    def decoder_direction(self, feat_idx: int) -> torch.Tensor:
        """Return W_dec[feat_idx] — the feature's write direction. [d_model]"""
        self._validate_feat(feat_idx)
        return self.sae.W_dec.data[feat_idx].cpu()

    def encoder_direction(self, feat_idx: int) -> torch.Tensor:
        """Return W_enc[:, feat_idx] — the feature's read direction. [d_model]"""
        self._validate_feat(feat_idx)
        return self.sae.W_enc.data[:, feat_idx].cpu()

    def encoder_bias(self, feat_idx: int) -> float:
        """Return b_enc[feat_idx]."""
        self._validate_feat(feat_idx)
        return float(self.sae.b_enc.data[feat_idx].item())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_feat(self, feat_idx: int) -> None:
        if not (0 <= feat_idx < self.n_features):
            raise IndexError(
                f"feat_idx={feat_idx} out of range for n_features={self.n_features}"
            )

    def _get_tl_model(self, model_name: str):
        """
        Lazily load a TransformerLens model for logit-lens computation.
        Cached on the FeatureAnalyzer instance (model_name aware).
        """
        cache_key = f"_tl_{model_name}"
        if hasattr(self, cache_key):
            return getattr(self, cache_key)

        try:
            from transformer_lens import HookedTransformer
            logger.info("Loading %s on CPU for logit lens …", model_name)
            model = HookedTransformer.from_pretrained(
                model_name,
                fold_ln       = False,
                center_unembed = True,   # needed for logit-lens interpretation
                device        = "cpu",
            )
            model.eval()
            setattr(self, cache_key, model)
            return model
        except Exception as exc:
            logger.warning(
                "Could not load TransformerLens model %r: %s", model_name, exc
            )
            setattr(self, cache_key, None)
            return None

    def __repr__(self) -> str:
        return (
            f"FeatureAnalyzer("
            f"n_features={self.n_features}, "
            f"d_model={self.d_model}, "
            f"layer={self.layer}, "
            f"n_tokens={self.dataset.n_tokens:,})"
        )
