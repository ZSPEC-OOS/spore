"""
sae_dataset.py — Token-level activation dataset for Sparse Autoencoder training.

Differences from the sentence-level pipeline (collector.py + io.py)
--------------------------------------------------------------------
• Activations are stored **per token**, not mean-pooled per sentence.
  Each row in a shard tensor is one real (non-padding) token's activation.
• The build loop streams directly to shard files — it never accumulates
  the full activation tensor in RAM.  Safe for 1M–5M token datasets.
• A compact token_map.pt  [total_tokens, 2] int32 maps every flat token
  index → (sentence_idx, token_pos) for text retrieval.
• Multiple layers are processed in a single forward pass and saved to
  separate per-layer shard directories.

Disk layout
-----------
  <root>/
    metadata.json
    corpus.json                 — texts + optional labels
    token_map.pt                — [total_tokens, 2] int32
    layer_06_resid_post/
      shard_0000.pt             — [shard_size, d_model] float32/16/bf16
      shard_0001.pt
      ...
    layer_11_resid_post/
      shard_0000.pt
      ...

Quick start
-----------
>>> from spore.activation_pipeline.sae_dataset import SAEDataset, SAEDatasetConfig
>>> from spore.activation_pipeline import CorpusLoader
>>>
>>> texts  = CorpusLoader.from_openwebtext(n=20_000)
>>> cfg    = SAEDatasetConfig(model_name="gpt2", layers=[6], n_sentences=20_000,
...                           out_root="sae_data/gpt2_l6")
>>> ds     = SAEDataset.build(cfg, texts)
>>>
>>> # Load later
>>> ds = SAEDataset.load("sae_data/gpt2_l6")
>>> ds.describe()
>>>
>>> # Top-k retrieval: which tokens maximally activate dimension 42?
>>> hits = ds.top_k_snippets(layer=6, feature=42, k=20)
>>> for h in hits:
...     print(h.pretty())
>>>
>>> # Or use a learned SAE direction:
>>> direction = torch.load("my_sae/features.pt")[42]   # [d_model]
>>> hits = ds.top_k_snippets(layer=6, feature=direction, k=20, threshold=0.0)
"""

from __future__ import annotations

import heapq
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported dtypes
# ---------------------------------------------------------------------------

_DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}

_VALID_HOOK_POINTS = {"resid_pre", "resid_mid", "resid_post", "mlp_out"}


# ---------------------------------------------------------------------------
# SAEDatasetConfig
# ---------------------------------------------------------------------------

@dataclass
class SAEDatasetConfig:
    """
    All parameters that control a single ``SAEDataset.build()`` run.

    Parameters
    ----------
    model_name : str
        Any model supported by TransformerLens (``"gpt2"``,
        ``"EleutherAI/pythia-160m"``, …).
    layers : list[int]
        Which transformer layers to collect.  Multiple layers are extracted
        in a **single** forward pass.
    hook_point : str
        Where in each block to tap activations:
        ``resid_pre | resid_mid | resid_post | mlp_out``
    n_sentences : int
        Number of sentences from the corpus to process.
    max_seq_len : int
        Tokenizer truncation / padding length.
    batch_size : int
        Sentences per forward pass.  Reduce if you hit OOM.
    shard_size : int
        Tokens per shard file.  500 k → ~1.5 GB/shard at float32, 768-dim.
    max_tokens : int | None
        Stop early once this many tokens have been collected.  None → no cap.
    dtype : str
        Storage dtype: ``"float32"`` | ``"float16"`` | ``"bfloat16"``.
    device : str
        ``"auto"`` | ``"cpu"`` | ``"cuda"`` | ``"mps"``
    seed : int
    include_bos : bool
        Include the BOS token activation in the dataset.
    min_tokens : int
        Skip sentences that contribute fewer than this many real tokens
        (after BOS exclusion).
    out_root : str
        Directory to write the dataset into.
    overwrite : bool
        If False (default), raise if ``out_root`` already contains a dataset.
    """

    model_name:   str            = "gpt2"
    layers:       List[int]      = field(default_factory=lambda: [6])
    hook_point:   str            = "resid_post"

    # corpus
    n_sentences:  int            = 10_000
    max_seq_len:  int            = 128

    # processing
    batch_size:   int            = 32
    shard_size:   int            = 500_000
    max_tokens:   Optional[int]  = None
    dtype:        str            = "float32"
    device:       str            = "auto"
    seed:         int            = 42

    # token filtering
    include_bos:  bool           = False
    min_tokens:   int            = 3

    # output
    out_root:     str            = "sae_data/run"
    overwrite:    bool           = False

    def __post_init__(self) -> None:
        if self.hook_point not in _VALID_HOOK_POINTS:
            raise ValueError(
                f"hook_point must be one of {_VALID_HOOK_POINTS}, "
                f"got {self.hook_point!r}"
            )
        if self.dtype not in _DTYPE_MAP:
            raise ValueError(f"dtype must be one of {list(_DTYPE_MAP)}, got {self.dtype!r}")
        if not self.layers:
            raise ValueError("layers must be a non-empty list.")


# ---------------------------------------------------------------------------
# SnippetResult
# ---------------------------------------------------------------------------

@dataclass
class SnippetResult:
    """
    A single result returned by :meth:`SAEDataset.top_k_snippets`.

    ``context_before`` and ``context_after`` are decoded token strings
    surrounding the activating token within the same sentence.
    """

    rank:           int
    score:          float
    flat_token_idx: int
    sentence_idx:   int
    token_pos:      int           # position in tokenized sequence
    sentence_text:  str
    token_str:      str           # decoded activating token
    context_before: str
    context_after:  str
    label:          Optional[str] = None

    def pretty(self, ctx_width: int = 60) -> str:
        """One-line human-readable summary."""
        ctx = f"{self.context_before}>>>{self.token_str}<<<{self.context_after}"
        if len(ctx) > ctx_width:
            ctx = "…" + ctx[-(ctx_width - 1):]
        return (
            f"#{self.rank:<3}  score={self.score:+.4f}  "
            f"sent={self.sentence_idx:<6}  pos={self.token_pos:<4}  {ctx}"
        )


# ---------------------------------------------------------------------------
# SAEDataset
# ---------------------------------------------------------------------------

class SAEDataset:
    """
    A flat, sharded token-level activation dataset ready for SAE training.

    Build a new dataset with :meth:`build`, or load an existing one with
    :meth:`load`.

    After building, iterate over shards for training:

    >>> for shard_idx, shard in ds.iter_shards(layer=6):
    ...     train_sae_one_step(shard)   # shard: [shard_size, d_model]

    Or retrieve text snippets that maximally activate a feature direction:

    >>> hits = ds.top_k_snippets(layer=6, feature=my_direction_tensor, k=20)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # Private — use build() or load() instead.
        self._root:      Path
        self._meta:      dict
        self._corpus:    dict          # {"texts": [...], "labels": [...] | None}
        self._token_map: torch.Tensor  # [total_tokens, 2] int32
        self._tokenizer = None         # lazy

    # ------------------------------------------------------------------
    # Public class methods
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        cfg: SAEDatasetConfig,
        texts: List[str],
        labels: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> "SAEDataset":
        """
        Build a new dataset from *texts*.

        Parameters
        ----------
        cfg : SAEDatasetConfig
        texts : list[str]
            Raw text corpus.  Will be truncated to ``cfg.n_sentences``.
        labels : list[str] | None
            Optional per-sentence label (domain, topic, …).
        show_progress : bool
            Display a tqdm bar over batches.

        Returns
        -------
        SAEDataset
            A fully built and saved dataset, ready for use.
        """
        # ── Validate / prepare output directory ──────────────────────
        root = Path(cfg.out_root).resolve()
        _prepare_out_dir(root, cfg.overwrite)

        # ── Trim corpus ───────────────────────────────────────────────
        texts = list(texts[: cfg.n_sentences])
        if labels is not None:
            labels = list(labels[: len(texts)])
        n_sentences = len(texts)
        logger.info("Building SAEDataset from %d sentences …", n_sentences)

        # ── Resolve device / dtype ────────────────────────────────────
        device = _resolve_device(cfg.device)
        torch_dtype = _DTYPE_MAP[cfg.dtype]

        # ── Load model ────────────────────────────────────────────────
        model = _load_model(cfg.model_name, device)
        d_model = model.cfg.d_model
        n_layers = model.cfg.n_layers

        # Validate layer indices
        bad = [l for l in cfg.layers if not (0 <= l < n_layers)]
        if bad:
            raise ValueError(
                f"Layer indices {bad} out of range for {cfg.model_name} "
                f"({n_layers} layers, 0-indexed)."
            )

        # ── Prepare tokenizer ─────────────────────────────────────────
        tokenizer = model.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── Create per-layer shard directories ────────────────────────
        layer_dirs = {
            L: root / f"layer_{L:02d}_{cfg.hook_point}"
            for L in cfg.layers
        }
        for d in layer_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # ── Names filter (cache only needed hooks) ────────────────────
        hook_keys = {
            f"blocks.{L}.hook_{cfg.hook_point}" for L in cfg.layers
        }
        names_filter = lambda name: name in hook_keys

        # ── Pre-allocate shard buffers ────────────────────────────────
        # One buffer per layer.  All layers share the same shard_pos cursor.
        shard_buf: Dict[int, torch.Tensor] = {
            L: torch.zeros(cfg.shard_size, d_model, dtype=torch_dtype)
            for L in cfg.layers
        }
        shard_pos   = 0       # tokens written into the current shard
        shard_count = 0       # shards flushed so far

        # token_map grows throughout the build; it's small (8 bytes/token)
        token_map_list: List[Tuple[int, int]] = []  # (sentence_idx, token_pos)
        total_tokens  = 0
        t0 = time.perf_counter()

        # ── Batch loop ────────────────────────────────────────────────
        n_batches = (n_sentences + cfg.batch_size - 1) // cfg.batch_size
        batch_iter = range(n_batches)
        if show_progress:
            batch_iter = tqdm(
                batch_iter,
                desc=f"Building SAEDataset [{cfg.model_name}]",
                unit="batch",
                dynamic_ncols=True,
            )

        done = False
        for batch_idx in batch_iter:
            if done:
                break

            b_start = batch_idx * cfg.batch_size
            b_end   = min(b_start + cfg.batch_size, n_sentences)
            batch_texts = texts[b_start:b_end]

            # ── Tokenise ─────────────────────────────────────────────
            enc = tokenizer(
                batch_texts,
                padding    = "max_length",
                truncation = True,
                max_length = cfg.max_seq_len,
                return_tensors = "pt",
            )
            batch_tokens = enc["input_ids"].to(device)           # [B, seq]
            batch_mask   = enc["attention_mask"].bool().to(device) # [B, seq]

            # ── Forward pass ─────────────────────────────────────────
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    batch_tokens,
                    names_filter = names_filter,
                    return_type  = None,
                )

            B = batch_tokens.shape[0]

            # ── Scatter tokens into shard buffers ─────────────────────
            for i in range(B):
                sentence_idx = b_start + i
                real_len = int(batch_mask[i].sum().item())
                tok_start = 0 if cfg.include_bos else 1
                tok_end   = real_len   # excludes padding

                n_tok = tok_end - tok_start
                if n_tok < cfg.min_tokens:
                    continue

                # Extend token_map for this sentence
                for pos in range(tok_start, tok_end):
                    token_map_list.append((sentence_idx, pos))

                # Write activations to shard buffer(s), splitting across
                # shard boundaries if necessary.
                written = 0
                while written < n_tok:
                    space  = cfg.shard_size - shard_pos
                    chunk  = min(n_tok - written, space)
                    src_s  = tok_start + written
                    src_e  = src_s + chunk

                    for L in cfg.layers:
                        acts_slice = (
                            cache[f"blocks.{L}.hook_{cfg.hook_point}"]
                            [i, src_s:src_e, :]
                            .to(dtype=torch_dtype, device="cpu")
                        )
                        shard_buf[L][shard_pos : shard_pos + chunk] = acts_slice

                    shard_pos    += chunk
                    written      += chunk
                    total_tokens += chunk

                    # Flush full shard
                    if shard_pos == cfg.shard_size:
                        _flush_shard(shard_buf, shard_pos, shard_count,
                                     layer_dirs, cfg.shard_size)
                        shard_count += 1
                        shard_pos    = 0

                    # Honour max_tokens cap
                    if cfg.max_tokens and total_tokens >= cfg.max_tokens:
                        done = True
                        break

                if done:
                    break

            # Free cache tensors immediately
            del cache

        # ── Flush partial final shard ─────────────────────────────────
        if shard_pos > 0:
            _flush_shard(shard_buf, shard_pos, shard_count,
                         layer_dirs, cfg.shard_size)
            shard_count += 1

        elapsed = time.perf_counter() - t0
        logger.info(
            "Build complete: %d tokens across %d shards in %.1f s",
            total_tokens, shard_count, elapsed,
        )

        # ── Save token_map ────────────────────────────────────────────
        if not token_map_list:
            raise RuntimeError("No tokens were collected.  Check corpus and config.")

        token_map_tensor = torch.tensor(token_map_list, dtype=torch.int32)
        torch.save(token_map_tensor, root / "token_map.pt")

        # ── Save corpus ───────────────────────────────────────────────
        corpus_payload = {"texts": texts, "labels": labels}
        (root / "corpus.json").write_text(
            json.dumps(corpus_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # ── Save metadata ─────────────────────────────────────────────
        meta = {
            "model_name":   cfg.model_name,
            "layers":       cfg.layers,
            "hook_point":   cfg.hook_point,
            "d_model":      d_model,
            "n_sentences":  n_sentences,
            "total_tokens": total_tokens,
            "n_shards":     shard_count,
            "shard_size":   cfg.shard_size,
            "max_seq_len":  cfg.max_seq_len,
            "dtype":        cfg.dtype,
            "include_bos":  cfg.include_bos,
            "elapsed_sec":  round(elapsed, 2),
            "built_with":   "SAEDataset.build",
        }
        (root / "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        return cls.load(root)

    @classmethod
    def load(cls, root: Union[str, Path]) -> "SAEDataset":
        """
        Load an existing dataset from *root*.

        Parameters
        ----------
        root : str | Path
            Directory previously created by :meth:`build`.

        Returns
        -------
        SAEDataset
        """
        root = Path(root).resolve()
        meta_path = root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No metadata.json found in {root}.  "
                "Did you run SAEDataset.build() first?"
            )

        meta   = json.loads(meta_path.read_text(encoding="utf-8"))
        corpus = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
        token_map = torch.load(root / "token_map.pt", weights_only=True)

        ds = cls.__new__(cls)
        ds._root      = root
        ds._meta      = meta
        ds._corpus    = corpus
        ds._token_map = token_map  # [total_tokens, 2] int32
        ds._tokenizer = None
        return ds

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_tokens(self) -> int:
        """Total number of tokens in the dataset."""
        return int(self._meta["total_tokens"])

    @property
    def n_shards(self) -> int:
        """Number of shard files per layer."""
        return int(self._meta["n_shards"])

    @property
    def shard_size(self) -> int:
        return int(self._meta["shard_size"])

    @property
    def d_model(self) -> int:
        return int(self._meta["d_model"])

    @property
    def layers(self) -> List[int]:
        return list(self._meta["layers"])

    @property
    def hook_point(self) -> str:
        return self._meta["hook_point"]

    @property
    def texts(self) -> List[str]:
        return self._corpus["texts"]

    @property
    def labels(self) -> Optional[List[str]]:
        return self._corpus.get("labels")

    # ------------------------------------------------------------------
    # Shard access (training interface)
    # ------------------------------------------------------------------

    def get_shard(self, layer: int, shard_idx: int) -> torch.Tensor:
        """
        Load one shard tensor for *layer*.

        Parameters
        ----------
        layer : int
        shard_idx : int

        Returns
        -------
        Tensor  [n_tokens_in_shard, d_model]
        """
        self._check_layer(layer)
        path = self._shard_path(layer, shard_idx)
        if not path.exists():
            raise FileNotFoundError(f"Shard not found: {path}")
        return torch.load(path, weights_only=True)

    def iter_shards(
        self,
        layer: int,
        shuffle_shards: bool = False,
        seed: int = 0,
    ) -> Iterator[Tuple[int, torch.Tensor]]:
        """
        Iterate over all shards for *layer*.

        Yields ``(shard_idx, tensor)`` where ``tensor`` has shape
        ``[n_tokens_in_shard, d_model]``.

        Parameters
        ----------
        layer : int
        shuffle_shards : bool
            Randomise the order of shard files.
        seed : int
            RNG seed used when *shuffle_shards* is True.
        """
        self._check_layer(layer)
        indices = list(range(self.n_shards))
        if shuffle_shards:
            import random
            rng = random.Random(seed)
            rng.shuffle(indices)
        for idx in indices:
            yield idx, self.get_shard(layer, idx)

    # ------------------------------------------------------------------
    # Top-k retrieval
    # ------------------------------------------------------------------

    def top_k_snippets(
        self,
        layer: int,
        feature: Union[int, torch.Tensor],
        k: int = 20,
        threshold: Optional[float] = None,
        context_tokens: int = 8,
    ) -> List[SnippetResult]:
        """
        Find the *k* tokens that most strongly activate a feature.

        Uses a streaming min-heap scan so only one shard is in memory at
        a time — safe for very large datasets.

        Parameters
        ----------
        layer : int
            Which layer's activations to search.
        feature : int or Tensor[d_model]
            **int** — score each token by its activation at dimension
            ``feature`` (a single neuron / basis direction).

            **Tensor[d_model]** — score by dot-product with this direction
            (e.g. a learned SAE decoder column).
        k : int
            Number of top results to return.
        threshold : float | None
            Minimum score.  Tokens below this are skipped.
        context_tokens : int
            How many tokens before and after the activating token to
            include in ``context_before`` / ``context_after``.

        Returns
        -------
        List[SnippetResult]  — sorted by score descending (rank 1 = highest).
        """
        self._check_layer(layer)

        if isinstance(feature, torch.Tensor):
            if feature.shape != (self.d_model,):
                raise ValueError(
                    f"feature tensor must have shape ({self.d_model},), "
                    f"got {tuple(feature.shape)}"
                )
            feature_vec = feature.float()
        else:
            feature_vec = None
            feature_idx = int(feature)
            if not (0 <= feature_idx < self.d_model):
                raise ValueError(
                    f"feature index {feature_idx} out of range "
                    f"for d_model={self.d_model}"
                )

        heap: list = []   # min-heap: (score, flat_token_idx)
        flat_offset = 0

        for shard_idx, shard in self.iter_shards(layer):
            shard_f = shard.float()  # [n, d_model]

            if feature_vec is not None:
                scores = shard_f @ feature_vec.to(shard_f.device)  # [n]
            else:
                scores = shard_f[:, feature_idx]                    # [n]

            for local_i, raw_score in enumerate(scores.tolist()):
                s = float(raw_score)
                if threshold is not None and s < threshold:
                    continue
                flat_i = flat_offset + local_i
                if len(heap) < k:
                    heapq.heappush(heap, (s, flat_i))
                elif s > heap[0][0]:
                    heapq.heapreplace(heap, (s, flat_i))

            flat_offset += len(shard)

        # Sort descending
        top_items = sorted(heap, key=lambda x: x[0], reverse=True)

        results: List[SnippetResult] = []
        for rank, (score, flat_idx) in enumerate(top_items, start=1):
            sentence_idx = int(self._token_map[flat_idx, 0])
            token_pos    = int(self._token_map[flat_idx, 1])

            sentence_text = self._corpus["texts"][sentence_idx]
            label_list    = self._corpus.get("labels")
            label = label_list[sentence_idx] if label_list else None

            tok_str, ctx_before, ctx_after = self._decode_context(
                sentence_text, token_pos, context_tokens
            )

            results.append(SnippetResult(
                rank           = rank,
                score          = score,
                flat_token_idx = flat_idx,
                sentence_idx   = sentence_idx,
                token_pos      = token_pos,
                sentence_text  = sentence_text,
                token_str      = tok_str,
                context_before = ctx_before,
                context_after  = ctx_after,
                label          = label,
            ))

        return results

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self, layer: int) -> Dict:
        """
        Compute summary statistics over all token activations for *layer*.

        Scans all shards; may take a moment for large datasets.

        Returns
        -------
        dict with keys: mean, std, min, max, l2_norm_mean, shape
        """
        self._check_layer(layer)

        n_total = 0
        sum_  = torch.zeros(self.d_model, dtype=torch.float64)
        sum2_ = torch.zeros(self.d_model, dtype=torch.float64)
        g_min = torch.full((self.d_model,), float("inf"),  dtype=torch.float64)
        g_max = torch.full((self.d_model,), float("-inf"), dtype=torch.float64)
        norm_sum = 0.0

        for _, shard in self.iter_shards(layer):
            s = shard.double()
            sum_  += s.sum(dim=0)
            sum2_ += (s ** 2).sum(dim=0)
            g_min  = torch.minimum(g_min, s.min(dim=0).values)
            g_max  = torch.maximum(g_max, s.max(dim=0).values)
            norm_sum += torch.linalg.norm(s, dim=1).sum().item()
            n_total  += len(shard)

        mean = (sum_ / n_total).float()
        var  = ((sum2_ / n_total) - (sum_ / n_total) ** 2).clamp(min=0)
        std  = var.sqrt().float()

        return {
            "n_tokens":      n_total,
            "d_model":       self.d_model,
            "mean":          mean,          # [d_model]
            "std":           std,           # [d_model]
            "min":           g_min.float(), # [d_model]
            "max":           g_max.float(), # [d_model]
            "l2_norm_mean":  norm_sum / n_total,
        }

    def describe(self) -> Dict:
        """Return a human-readable summary dict (no tensor loading)."""
        return {
            "root":         str(self._root),
            "model":        self._meta.get("model_name"),
            "hook_point":   self._meta.get("hook_point"),
            "layers":       self._meta.get("layers"),
            "d_model":      self.d_model,
            "n_sentences":  self._meta.get("n_sentences"),
            "total_tokens": self.n_tokens,
            "n_shards":     self.n_shards,
            "shard_size":   self.shard_size,
            "dtype":        self._meta.get("dtype"),
            "include_bos":  self._meta.get("include_bos"),
            "elapsed_sec":  self._meta.get("elapsed_sec"),
            "disk_gb":      self._estimate_disk_gb(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _shard_path(self, layer: int, shard_idx: int) -> Path:
        return (
            self._root
            / f"layer_{layer:02d}_{self.hook_point}"
            / f"shard_{shard_idx:04d}.pt"
        )

    def _check_layer(self, layer: int) -> None:
        if layer not in self.layers:
            raise ValueError(
                f"Layer {layer} not in this dataset.  Available: {self.layers}"
            )

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(self._meta["model_name"])
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                self._tokenizer = tok
            except Exception as exc:
                logger.warning("Could not load tokenizer: %s", exc)
                return None
        return self._tokenizer

    def _decode_context(
        self,
        sentence: str,
        token_pos: int,
        window: int,
    ) -> Tuple[str, str, str]:
        """
        Re-tokenise *sentence* and decode the token at *token_pos* plus
        surrounding context.

        Returns (token_str, context_before, context_after).
        """
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return ("[?]", "", "")

        enc = tokenizer(
            sentence,
            truncation  = True,
            max_length  = self._meta.get("max_seq_len", 128),
            return_tensors = "pt",
        )
        ids = enc["input_ids"][0]

        if token_pos >= len(ids):
            return ("[?]", "", "")

        tok_str    = tokenizer.decode([ids[token_pos].item()])
        before_ids = ids[max(0, token_pos - window) : token_pos]
        after_ids  = ids[token_pos + 1 : token_pos + 1 + window]

        ctx_before = tokenizer.decode(before_ids, skip_special_tokens=True)
        ctx_after  = tokenizer.decode(after_ids,  skip_special_tokens=True)

        return tok_str, ctx_before, ctx_after

    def _estimate_disk_gb(self) -> float:
        bytes_per_val = {"float32": 4, "float16": 2, "bfloat16": 2}.get(
            self._meta.get("dtype", "float32"), 4
        )
        n_layers = len(self._meta.get("layers", [1]))
        total_bytes = self.n_tokens * self.d_model * bytes_per_val * n_layers
        return round(total_bytes / 1e9, 3)

    def __repr__(self) -> str:
        return (
            f"SAEDataset("
            f"model={self._meta.get('model_name')!r}, "
            f"layers={self.layers}, "
            f"n_tokens={self.n_tokens:,}, "
            f"n_shards={self.n_shards}, "
            f"d_model={self.d_model})"
        )


# ---------------------------------------------------------------------------
# Internal helpers (module-level)
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model(model_name: str, device: str):
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise ImportError(
            "transformer_lens is required.  Install with:\n"
            "  pip install transformer_lens"
        ) from exc

    logger.info("Loading %s on %s …", model_name, device)
    model = HookedTransformer.from_pretrained(
        model_name,
        fold_ln       = False,
        center_unembed = False,
        device        = device,
    )
    model.eval()
    return model


def _flush_shard(
    shard_buf: Dict[int, torch.Tensor],
    n_tokens: int,
    shard_idx: int,
    layer_dirs: Dict[int, Path],
    shard_size: int,
) -> None:
    """Save the current shard for every layer and log progress."""
    fname = f"shard_{shard_idx:04d}.pt"
    for L, buf in shard_buf.items():
        data = buf[:n_tokens].clone()
        torch.save(data, layer_dirs[L] / fname)
    logger.info(
        "  flushed shard %04d  (%d / %d tokens)",
        shard_idx, n_tokens, shard_size,
    )


def _prepare_out_dir(root: Path, overwrite: bool) -> None:
    if root.exists() and (root / "metadata.json").exists():
        if not overwrite:
            raise FileExistsError(
                f"{root} already contains a dataset.  "
                "Pass overwrite=True to replace it."
            )
        import shutil
        shutil.rmtree(root)
        logger.info("Removed existing dataset at %s", root)
    root.mkdir(parents=True, exist_ok=True)
