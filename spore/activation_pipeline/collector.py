"""
collector.py — ActivationCollector and CollectionRun.

Wraps TransformerLens (primary) or nnsight (optional fallback) to:
  • register forward hooks on residual-stream positions,
  • batch-process a text corpus with a progress bar,
  • return a CollectionRun holding tensors + metadata.

Hook points (TransformerLens naming)
-------------------------------------
  resid_pre   blocks.{L}.hook_resid_pre     before attention + MLP
  resid_mid   blocks.{L}.hook_resid_mid     after attention, before MLP
  resid_post  blocks.{L}.hook_resid_post    after full transformer block  ← default
  mlp_out     blocks.{L}.hook_mlp_out       MLP output only

Pooling strategies
------------------
  mean   — average over non-padding token positions  ← default
  last   — last non-padding token
  all    — return the full [seq_len, d_model] tensor (large)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums / constants
# ---------------------------------------------------------------------------

class HookPoint(str, Enum):
    """Named residual-stream positions understood by TransformerLens."""
    RESID_PRE  = "resid_pre"
    RESID_MID  = "resid_mid"
    RESID_POST = "resid_post"
    MLP_OUT    = "mlp_out"


class PoolStrategy(str, Enum):
    MEAN = "mean"
    LAST = "last"
    ALL  = "all"


# ---------------------------------------------------------------------------
# CollectionRun — result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CollectionRun:
    """
    All data collected in a single call to ActivationCollector.collect().

    Attributes
    ----------
    model_name : str
        e.g. "gpt2", "EleutherAI/pythia-160m"
    layers : list[int]
        Which layer indices were collected.
    hook_point : HookPoint
        Where in each block the hook was attached.
    pool : PoolStrategy
        How sequence positions were collapsed.
    texts : list[str]
        The original input sentences.
    tokens : torch.Tensor
        Shape [N, max_seq] — padded token IDs (int64).
    attention_mask : torch.Tensor
        Shape [N, max_seq] — 1 for real tokens, 0 for padding.
    labels : list[str] | None
        Optional per-sentence category labels.
    activations : dict[int, torch.Tensor]
        {layer_index: tensor}.
        • pool="mean" / "last" → [N, d_model]  (float16 by default)
        • pool="all"           → [N, max_seq, d_model]
    elapsed_sec : float
        Wall-clock collection time.
    """

    model_name:     str
    layers:         List[int]
    hook_point:     HookPoint
    pool:           PoolStrategy
    texts:          List[str]
    tokens:         torch.Tensor
    attention_mask: torch.Tensor
    labels:         Optional[List[str]]
    activations:    Dict[int, torch.Tensor]
    elapsed_sec:    float = 0.0
    d_model:        int   = 0
    n_layers_total: int   = 0


# ---------------------------------------------------------------------------
# ActivationCollector
# ---------------------------------------------------------------------------

class ActivationCollector:
    """
    Collect residual-stream activations from a HookedTransformer.

    Parameters
    ----------
    model_name : str
        Any model supported by TransformerLens (gpt2, pythia-*, etc.).
    layers : list[int] | None
        Indices of layers to collect.  None → collect every layer.
    hook_point : HookPoint | str
        Where in each block to tap the residual stream.
    pool : PoolStrategy | str
        How to reduce the sequence dimension.
    device : str | None
        "cuda", "mps", or "cpu".  None → auto-detect.
    dtype : torch.dtype
        Storage dtype for activations. float16 halves disk/RAM usage.
    max_seq_len : int
        Truncate / pad inputs to this length.
    center_unembed : bool
        Passed to TransformerLens; set True only for logit-lens work.
    """

    def __init__(
        self,
        model_name:    str = "gpt2",
        layers:        Optional[Sequence[int]] = None,
        hook_point:    HookPoint | str = HookPoint.RESID_POST,
        pool:          PoolStrategy | str = PoolStrategy.MEAN,
        device:        Optional[str] = None,
        dtype:         torch.dtype = torch.float16,
        max_seq_len:   int = 128,
        center_unembed: bool = False,
    ) -> None:
        self.model_name     = model_name
        self.hook_point     = HookPoint(hook_point)
        self.pool           = PoolStrategy(pool)
        self.dtype          = dtype
        self.max_seq_len    = max_seq_len
        self.center_unembed = center_unembed

        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps"  if torch.backends.mps.is_available()
            else "cpu"
        )

        logger.info("Loading model %s on %s …", model_name, self.device)
        self._model = self._load_model()

        total_layers = self._model.cfg.n_layers
        if layers is None:
            self.layers = list(range(total_layers))
        else:
            bad = [l for l in layers if l < 0 or l >= total_layers]
            if bad:
                raise ValueError(
                    f"Layer indices {bad} out of range for {model_name} "
                    f"({total_layers} layers, 0-indexed)."
                )
            self.layers = sorted(set(layers))

        logger.info(
            "Collecting %s @ %s for layers %s  (pool=%s, device=%s)",
            model_name, self.hook_point.value, self.layers,
            self.pool.value, self.device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        texts:      List[str],
        batch_size: int = 32,
        labels:     Optional[List[str]] = None,
        show_progress: bool = True,
        checkpoint_tag: str = "",
    ) -> CollectionRun:
        """
        Run the model on ``texts`` and return a :class:`CollectionRun`.

        Parameters
        ----------
        texts : list[str]
        batch_size : int
        labels : list[str] | None
            Optional category / topic label per sentence.
        show_progress : bool
            Show a tqdm progress bar.
        checkpoint_tag : str
            Informational tag stored in the run (useful for multi-checkpoint sweeps).
        """
        if labels is not None and len(labels) != len(texts):
            raise ValueError("len(labels) must equal len(texts).")

        t0 = time.perf_counter()

        # ── tokenise entire corpus once ──────────────────────────────
        logger.info("Tokenising %d texts …", len(texts))
        all_tokens, all_mask = self._tokenise_corpus(texts)

        # ── accumulators ─────────────────────────────────────────────
        layer_accum: Dict[int, List[torch.Tensor]] = {l: [] for l in self.layers}
        n = len(texts)

        # ── batch loop ───────────────────────────────────────────────
        batches = range(0, n, batch_size)
        if show_progress:
            batches = tqdm(
                batches,
                total=(n + batch_size - 1) // batch_size,
                desc=f"Collecting [{checkpoint_tag or self.model_name}]",
                unit="batch",
                dynamic_ncols=True,
            )

        for start in batches:
            end        = min(start + batch_size, n)
            batch_tok  = all_tokens[start:end].to(self.device)
            batch_mask = all_mask[start:end].to(self.device)

            with torch.no_grad():
                _, cache = self._model.run_with_cache(
                    batch_tok,
                    names_filter=self._names_filter(),
                    return_type=None,
                )

            for layer in self.layers:
                key   = self._hook_key(layer)
                acts  = cache[key]                          # [B, seq, d_model]
                acts  = self._pool(acts, batch_mask)        # [B, d_model] or [B, seq, d_model]
                layer_accum[layer].append(acts.to("cpu", dtype=self.dtype))

            del cache
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # ── concatenate and package ───────────────────────────────────
        activations = {
            layer: torch.cat(chunks, dim=0)
            for layer, chunks in layer_accum.items()
        }

        elapsed = time.perf_counter() - t0
        logger.info(
            "Collection complete — %d sentences, %d layers, %.1f s",
            n, len(self.layers), elapsed,
        )

        return CollectionRun(
            model_name      = self.model_name,
            layers          = self.layers,
            hook_point      = self.hook_point,
            pool            = self.pool,
            texts           = texts,
            tokens          = all_tokens,
            attention_mask  = all_mask,
            labels          = labels,
            activations     = activations,
            elapsed_sec     = elapsed,
            d_model         = self._model.cfg.d_model,
            n_layers_total  = self._model.cfg.n_layers,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        try:
            from transformer_lens import HookedTransformer
        except ImportError as exc:
            raise ImportError(
                "transformer_lens is required.  Install it with:\n"
                "  pip install transformer_lens"
            ) from exc

        model = HookedTransformer.from_pretrained(
            self.model_name,
            center_unembed=self.center_unembed,
            fold_ln=False,
            device=self.device,
        )
        model.eval()
        return model

    def _tokenise_corpus(
        self, texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenise all texts with right-padding to max_seq_len.

        Returns
        -------
        tokens : LongTensor [N, max_seq_len]
        mask   : BoolTensor [N, max_seq_len]  — True for real tokens
        """
        tokenizer = self._model.tokenizer
        if tokenizer.pad_token is None:
            # GPT-style tokenizers have no pad token by default
            tokenizer.pad_token = tokenizer.eos_token

        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"].bool()

    def _hook_key(self, layer: int) -> str:
        """TransformerLens cache key for a given layer and hook point."""
        hp = self.hook_point
        if hp == HookPoint.RESID_PRE:
            return f"blocks.{layer}.hook_resid_pre"
        if hp == HookPoint.RESID_MID:
            return f"blocks.{layer}.hook_resid_mid"
        if hp == HookPoint.RESID_POST:
            return f"blocks.{layer}.hook_resid_post"
        if hp == HookPoint.MLP_OUT:
            return f"blocks.{layer}.hook_mlp_out"
        raise ValueError(f"Unknown hook point: {hp}")

    def _names_filter(self):
        """Only cache the hook keys we actually need (saves memory)."""
        keys = {self._hook_key(l) for l in self.layers}
        return lambda name: name in keys

    def _pool(
        self,
        acts: torch.Tensor,     # [B, seq, d_model]
        mask: torch.Tensor,     # [B, seq]  bool
    ) -> torch.Tensor:
        if self.pool == PoolStrategy.ALL:
            return acts

        if self.pool == PoolStrategy.MEAN:
            mask_f  = mask.float().unsqueeze(-1)              # [B, seq, 1]
            summed  = (acts * mask_f).sum(dim=1)              # [B, d_model]
            counts  = mask_f.sum(dim=1).clamp(min=1)          # [B, 1]
            return summed / counts

        if self.pool == PoolStrategy.LAST:
            # index of last real token per sequence
            lengths = mask.long().sum(dim=1) - 1              # [B]
            lengths = lengths.clamp(min=0)
            idx = lengths.view(-1, 1, 1).expand(-1, 1, acts.size(-1))
            return acts.gather(dim=1, index=idx).squeeze(1)   # [B, d_model]

        raise ValueError(f"Unknown pool strategy: {self.pool}")
