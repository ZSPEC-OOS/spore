"""
sae.py — From-scratch Sparse Autoencoder for mechanistic interpretability.

Architecture
------------
  ReLU (activation="relu")  — standard Anthropic / EleutherAI SAE:
    x_c  = x − b_pre                           centre activations
    h    = ReLU(W_enc @ x_c + b_enc)           sparse feature acts   [F]
    x̂   = W_dec @ h + b_dec                    reconstruction        [D]
    L    = MSE(x, x̂)  +  λ · ‖h‖₁            total loss

  Gated (activation="gated")  — differentiable soft-gate variant:
    x_c     = x − b_pre
    π_gate  = W_gate @ x_c + b_gate            gate pre-act          [F]
    π_mag   = W_enc  @ x_c + b_enc             magnitude pre-act     [F]
    h       = sigmoid(π_gate) · ReLU(π_mag)    gated features        [F]
    x̂      = W_dec @ h + b_dec
    L       = MSE(x, x̂)  +  λ · ‖h‖₁  +  λ_gate · ‖ReLU(π_gate)‖₁

  Decoder unit-norm constraint
    W_dec rows (shape [F, D]) are normalised to ‖w‖₂ = 1 after every
    gradient step.  This prevents the model from absorbing scale into
    the decoder and making h arbitrarily small.

  Neuron resampling
    Dead features (‖h_i‖₀ = 0 for the last ``dead_after_steps`` steps)
    are periodically resampled:
      • W_enc[:, i] ← (x − x̂)[argmax_recon_err] / ‖…‖  (scaled)
      • W_dec[i, :] ← W_enc[:, i].T  (keeps encoder/decoder aligned)
      • b_enc[i]    ← 0
    This prevents permanently dead neurons and improves feature
    utilisation.

sae-lens equivalent
-------------------
  If you prefer the sae-lens library (``pip install sae-lens``), the
  equivalent config for a ReLU SAE on GPT-2 layer-6 residual stream is:

    from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

    cfg = LanguageModelSAERunnerConfig(
        model_name              = "gpt2",
        hook_name               = "blocks.6.hook_resid_post",
        hook_layer              = 6,
        d_in                    = 768,
        expansion_factor        = 8,          # n_features = 8 × 768 = 6144
        activation_fn_name      = "relu",
        l1_coefficient          = 1e-3,
        lr                      = 2e-4,
        train_batch_size_tokens = 4096,
        n_training_tokens       = 50_000_000,
        normalize_activations   = "expected_average_only_in",
        dtype                   = "float32",
        device                  = "cuda",
        checkpoint_path         = "sae_checkpoints",
    )
    SAETrainingRunner(cfg).run()

  The from-scratch implementation below targets the same architecture and
  produces compatible checkpoint dictionaries.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAEConfig
# ---------------------------------------------------------------------------

@dataclass
class SAEConfig:
    """
    All hyperparameters for a single SAE training run.

    Parameters
    ----------
    d_model : int
        Dimension of the transformer residual stream (e.g. 768 for GPT-2).
    n_features : int
        Number of SAE features (dictionary atoms).  Typically 4–32 × d_model.
    activation : str
        ``"relu"`` — standard ReLU SAE.
        ``"gated"`` — soft-gate SAE (sigmoid gate × ReLU magnitude).
    l1_coeff : float
        Sparsity regularisation weight λ.
    l1_gate_coeff : float
        Extra gate-pathway L1 weight (gated SAE only).
    lr : float
        Peak AdamW learning rate.
    lr_end : float
        Final learning rate at the end of cosine decay.
    betas : tuple
        AdamW (β₁, β₂).
    eps : float
        AdamW ε.
    weight_decay : float
        AdamW weight decay (0 is standard for SAEs).
    batch_size : int
        Tokens per gradient step.
    n_steps : int
        Total gradient steps to train for.
    warmup_steps : int
        Linear LR warmup period.
    grad_clip_norm : float
        Maximum gradient norm before clipping.
    resample_every : int
        Resample dead neurons every this many steps.
    dead_after_steps : int
        A neuron is considered dead if it hasn't fired in this many steps.
    resample_scale : float
        Scale applied to resampled encoder vectors.
    checkpoint_every : int
        Save a checkpoint every this many steps.
    log_every : int
        Print metrics every this many steps.
    out_dir : str
        Root directory for checkpoints and logs.
    dtype : str
        Storage dtype (also used for internal computation):
        ``"float32"`` | ``"float16"`` | ``"bfloat16"``.
    seed : int
    """

    # Architecture
    d_model:         int   = 768
    n_features:      int   = 6144          # default: ~8× expansion
    activation:      str   = "relu"        # "relu" | "gated"

    # Loss
    l1_coeff:        float = 1e-3
    l1_gate_coeff:   float = 1e-4          # gated SAE only

    # Optimiser
    lr:              float = 2e-4
    lr_end:          float = 2e-5
    betas:           Tuple[float, float] = (0.9, 0.999)
    eps:             float = 1e-8
    weight_decay:    float = 0.0
    batch_size:      int   = 4096
    n_steps:         int   = 50_000
    warmup_steps:    int   = 1_000
    grad_clip_norm:  float = 1.0

    # Neuron resampling
    resample_every:     int   = 2_500
    dead_after_steps:   int   = 2_500
    resample_scale:     float = 0.2

    # Checkpointing / logging
    checkpoint_every: int  = 5_000
    log_every:        int  = 100
    out_dir:          str  = "sae_checkpoints/run"

    # Misc
    dtype:   str = "float32"
    seed:    int = 42

    # ── Dataset reference (informational, stored in checkpoint) ──────────
    dataset_root: str = ""
    layer:        int = 0
    hook_point:   str = "resid_post"
    model_name:   str = ""

    def __post_init__(self) -> None:
        if self.activation not in ("relu", "gated"):
            raise ValueError(
                f"activation must be 'relu' or 'gated', got {self.activation!r}"
            )


# ---------------------------------------------------------------------------
# SAEOutput
# ---------------------------------------------------------------------------

class SAEOutput(NamedTuple):
    """
    Return value of :meth:`SparseAutoencoder.forward`.

    Attributes
    ----------
    h : Tensor [batch, n_features]
        Sparse feature activations.
    x_hat : Tensor [batch, d_model]
        Reconstructed input.
    gate_pre : Tensor [batch, n_features] | None
        Gate pre-activations π_gate (gated SAE only; None for ReLU).
    """
    h:        torch.Tensor
    x_hat:    torch.Tensor
    gate_pre: Optional[torch.Tensor]


# ---------------------------------------------------------------------------
# SparseAutoencoder
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder with optional soft-gate variant.

    Parameters
    ----------
    cfg : SAEConfig
    """

    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D, F = cfg.d_model, cfg.n_features

        # ── Shared ───────────────────────────────────────────────────
        # Pre-encoder bias: subtract from input to approximately centre it.
        # Initialised to 0; call initialize_b_pre() with empirical mean.
        self.b_pre = nn.Parameter(torch.zeros(D))

        # ── Encoder ──────────────────────────────────────────────────
        self.W_enc = nn.Parameter(torch.empty(D, F))
        self.b_enc = nn.Parameter(torch.zeros(F))

        if cfg.activation == "gated":
            self.W_gate = nn.Parameter(torch.empty(D, F))
            self.b_gate = nn.Parameter(torch.zeros(F))
        else:
            self.W_gate = None  # type: ignore[assignment]
            self.b_gate = None  # type: ignore[assignment]

        # ── Decoder ──────────────────────────────────────────────────
        # W_dec rows [F, D] are kept unit-norm.
        self.W_dec = nn.Parameter(torch.empty(F, D))
        self.b_dec = nn.Parameter(torch.zeros(D))

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        if self.W_gate is not None:
            nn.init.kaiming_uniform_(self.W_gate, a=math.sqrt(5))
        self.normalize_decoder()

    @torch.no_grad()
    def initialize_b_pre(self, mean_activation: torch.Tensor) -> None:
        """
        Set b_pre to the empirical mean of the training activations.

        Call once before training to improve convergence.

        Parameters
        ----------
        mean_activation : Tensor [d_model]
        """
        self.b_pre.data.copy_(mean_activation.to(self.b_pre.device))
        logger.info(
            "b_pre initialised from data mean  (‖μ‖=%.4f)",
            float(mean_activation.norm()),
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        x : Tensor [batch, d_model]

        Returns
        -------
        h : Tensor [batch, n_features]
        gate_pre : Tensor [batch, n_features] | None
        """
        x_c = x - self.b_pre          # centre
        pre_act = x_c @ self.W_enc + self.b_enc

        if self.cfg.activation == "relu":
            h = F.relu(pre_act)
            return h, None

        # Gated: soft gate (sigmoid) × magnitude (ReLU)
        gate_pre = x_c @ self.W_gate + self.b_gate  # type: ignore[operator]
        h = torch.sigmoid(gate_pre) * F.relu(pre_act)
        return h, gate_pre

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        h, gate_pre = self.encode(x)
        x_hat = self.decode(h)
        return SAEOutput(h=h, x_hat=x_hat, gate_pre=gate_pre)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        x: torch.Tensor,
        out: SAEOutput,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss, reconstruction loss, and L1 sparsity loss.

        Parameters
        ----------
        x : Tensor [batch, d_model]   — original activations
        out : SAEOutput

        Returns
        -------
        total_loss, recon_loss, l1_loss  — all scalar Tensors
        """
        recon_loss = F.mse_loss(out.x_hat, x)
        l1_loss    = out.h.abs().mean()
        total_loss = recon_loss + self.cfg.l1_coeff * l1_loss

        if self.cfg.activation == "gated" and out.gate_pre is not None:
            gate_l1    = F.relu(out.gate_pre).abs().mean()
            total_loss = total_loss + self.cfg.l1_gate_coeff * gate_l1

        return total_loss, recon_loss, l1_loss

    # ------------------------------------------------------------------
    # Decoder normalisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """
        Project W_dec rows onto the unit sphere.
        Called after every gradient step.
        """
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def metrics(self, x: torch.Tensor, out: SAEOutput) -> Dict:
        """
        Compute interpretability-relevant metrics for a batch.

        Returns
        -------
        dict with keys:
            recon_loss, l1_loss, total_loss,
            l0_mean          — mean number of active features per token,
            l0_pct           — l0_mean / n_features * 100,
            alive_pct        — % of features that fired at least once,
            explained_var    — fraction of variance explained by reconstruction,
            mean_act         — mean of h over active features,
            dead_features    — count of features that never fired in this batch.
        """
        total, recon, l1 = self.loss(x, out)
        h = out.h

        l0_per_token  = (h > 0).float().sum(dim=1)
        l0_mean       = l0_per_token.mean().item()
        l0_pct        = 100.0 * l0_mean / self.cfg.n_features

        alive_mask    = (h > 0).any(dim=0)
        alive_pct     = 100.0 * alive_mask.float().mean().item()
        dead_features = int((~alive_mask).sum().item())

        # Explained variance
        var_x   = (x - x.mean(dim=0, keepdim=True)).pow(2).mean()
        var_err = (x - out.x_hat).pow(2).mean()
        explained_var = 1.0 - (var_err / var_x.clamp(min=1e-8)).item()

        active_vals = h[h > 0]
        mean_act = active_vals.mean().item() if active_vals.numel() > 0 else 0.0

        return {
            "total_loss":    total.item(),
            "recon_loss":    recon.item(),
            "l1_loss":       l1.item(),
            "l0_mean":       round(l0_mean, 2),
            "l0_pct":        round(l0_pct, 4),
            "alive_pct":     round(alive_pct, 2),
            "dead_features": dead_features,
            "explained_var": round(explained_var, 4),
            "mean_act":      round(mean_act, 4),
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def state_dict_sae(self) -> Dict:
        """
        Compact state dict using named parameter aliases for readability.
        """
        d = {
            "W_enc": self.W_enc.data,
            "b_enc": self.b_enc.data,
            "W_dec": self.W_dec.data,
            "b_dec": self.b_dec.data,
            "b_pre": self.b_pre.data,
        }
        if self.W_gate is not None:
            d["W_gate"] = self.W_gate.data
            d["b_gate"] = self.b_gate.data
        return d

    def load_state_dict_sae(self, d: Dict) -> None:
        self.W_enc.data.copy_(d["W_enc"])
        self.b_enc.data.copy_(d["b_enc"])
        self.W_dec.data.copy_(d["W_dec"])
        self.b_dec.data.copy_(d["b_dec"])
        self.b_pre.data.copy_(d["b_pre"])
        if "W_gate" in d and self.W_gate is not None:
            self.W_gate.data.copy_(d["W_gate"])
            self.b_gate.data.copy_(d["b_gate"])  # type: ignore[union-attr]

    def __repr__(self) -> str:
        cfg = self.cfg
        params = sum(p.numel() for p in self.parameters())
        return (
            f"SparseAutoencoder("
            f"d_model={cfg.d_model}, n_features={cfg.n_features}, "
            f"activation={cfg.activation!r}, params={params:,})"
        )


# ---------------------------------------------------------------------------
# SAETrainer
# ---------------------------------------------------------------------------

class SAETrainer:
    """
    Wraps a :class:`SparseAutoencoder` with an AdamW optimiser, warmup +
    cosine LR schedule, neuron resampling, and checkpoint management.

    Usage::

        from spore.activation_pipeline import SAEDataset
        from spore.activation_pipeline.sae import SAEConfig, SAETrainer

        cfg = SAEConfig(d_model=768, n_features=6144, l1_coeff=1e-3,
                        n_steps=50_000, batch_size=4096,
                        dataset_root="sae_data/gpt2_l6", layer=6,
                        out_dir="sae_checkpoints/gpt2_l6")

        ds      = SAEDataset.load("sae_data/gpt2_l6")
        trainer = SAETrainer(cfg)
        trainer.train(ds)
    """

    def __init__(
        self,
        cfg: SAEConfig,
        device: Optional[str] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        cfg : SAEConfig
        device : str | None
            ``None`` → auto-detect (CUDA > MPS > CPU).
        resume_from : str | None
            Path to a checkpoint directory to resume from.
        """
        self.cfg = cfg
        self.device = device or _auto_device()
        torch.manual_seed(cfg.seed)

        # ── Model ────────────────────────────────────────────────────
        self.model = SparseAutoencoder(cfg).to(self.device)

        # ── Optimiser ────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = cfg.lr,
            betas        = cfg.betas,
            eps          = cfg.eps,
            weight_decay = cfg.weight_decay,
        )

        # ── LR schedule: warmup + cosine decay ───────────────────────
        self.scheduler = LambdaLR(self.optimizer, _make_lr_fn(cfg))

        # ── Activity tracker (for dead-neuron detection) ──────────────
        # steps_since_active[i] = how many steps feature i has been inactive
        self.steps_since_active = torch.zeros(
            cfg.n_features, dtype=torch.long, device=self.device
        )

        # ── Step counter ─────────────────────────────────────────────
        self.global_step = 0

        # ── Losses history (for logging / return) ────────────────────
        self.history: List[Dict] = []

        if resume_from is not None:
            self._load_checkpoint(resume_from)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, dataset, show_progress: bool = True) -> List[Dict]:
        """
        Run the full training loop.

        Parameters
        ----------
        dataset : SAEDataset
            Built with :class:`SAEDatasetConfig` for the desired layer.
        show_progress : bool

        Returns
        -------
        List[Dict]
            Logged metrics (one entry per ``log_every`` steps).
        """
        cfg  = self.cfg
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        layer = cfg.layer

        # ── Save config ──────────────────────────────────────────────
        (out_dir / "config.json").write_text(
            json.dumps(asdict(cfg), indent=2), encoding="utf-8"
        )

        # ── Initialise b_pre from data mean ──────────────────────────
        logger.info("Computing activation mean for b_pre initialisation …")
        data_mean = _compute_mean(dataset, layer, self.device, max_tokens=50_000)
        self.model.initialize_b_pre(data_mean)

        # ── Token iterator ────────────────────────────────────────────
        token_iter = _InfiniteTokenIter(
            dataset, layer, cfg.batch_size, cfg.seed
        )

        # ── Progress bar ──────────────────────────────────────────────
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(
                    total=cfg.n_steps,
                    initial=self.global_step,
                    desc=f"SAE [{cfg.activation}, F={cfg.n_features}]",
                    unit="step",
                    dynamic_ncols=True,
                )
            except ImportError:
                pbar = None
        else:
            pbar = None

        t_start = time.perf_counter()

        # ── Main loop ─────────────────────────────────────────────────
        while self.global_step < cfg.n_steps:
            self.global_step += 1
            batch = next(token_iter).to(self.device)

            # Forward
            out = self.model(batch)
            total_loss, recon_loss, l1_loss = self.model.loss(batch, out)

            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip_norm
            )
            self.optimizer.step()
            self.scheduler.step()

            # Decoder normalisation after every step
            self.model.normalize_decoder()

            # Track dead neurons
            with torch.no_grad():
                fired = (out.h.detach() > 0).any(dim=0)
                self.steps_since_active[fired]  = 0
                self.steps_since_active[~fired] += 1

            # Neuron resampling
            if (
                self.global_step % cfg.resample_every == 0
                and cfg.dead_after_steps > 0
            ):
                n_resampled = self._resample_dead_neurons(batch.detach())
                if n_resampled:
                    logger.info(
                        "step %d: resampled %d dead neurons",
                        self.global_step, n_resampled,
                    )

            # Logging
            if self.global_step % cfg.log_every == 0:
                with torch.no_grad():
                    m = self.model.metrics(batch, out)
                m["step"]    = self.global_step
                m["lr"]      = self.scheduler.get_last_lr()[0]
                m["elapsed"] = round(time.perf_counter() - t_start, 1)
                self.history.append(m)
                if pbar is not None:
                    pbar.set_postfix({
                        "loss":  f"{m['total_loss']:.4f}",
                        "L0":    f"{m['l0_mean']:.1f}",
                        "ev":    f"{m['explained_var']:.3f}",
                        "dead":  m["dead_features"],
                    })
                else:
                    logger.info(
                        "step %5d | loss=%.4f  recon=%.4f  l1=%.4f  "
                        "L0=%.1f (%.2f%%)  ev=%.3f  dead=%d  lr=%.2e",
                        self.global_step,
                        m["total_loss"], m["recon_loss"], m["l1_loss"],
                        m["l0_mean"], m["l0_pct"], m["explained_var"],
                        m["dead_features"], m["lr"],
                    )

            # Checkpoint
            if self.global_step % cfg.checkpoint_every == 0:
                self._save_checkpoint()

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        # Final checkpoint
        self._save_checkpoint()

        # Save training history
        (out_dir / "history.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )

        total_time = time.perf_counter() - t_start
        logger.info(
            "Training complete in %.1f s  (%d steps, %.0f steps/s)",
            total_time, cfg.n_steps,
            cfg.n_steps / max(total_time, 1e-6),
        )
        return self.history

    # ------------------------------------------------------------------
    # Dead-neuron resampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _resample_dead_neurons(self, batch: torch.Tensor) -> int:
        """
        For each dead neuron (inactive for ≥ dead_after_steps steps),
        reinitialise its encoder column and decoder row to a rescaled
        version of the highest-reconstruction-error token in *batch*.

        Returns the number of neurons resampled.
        """
        cfg  = self.cfg
        dead = (self.steps_since_active >= cfg.dead_after_steps).nonzero(as_tuple=True)[0]
        if len(dead) == 0:
            return 0

        # Reconstruction errors per token
        out    = self.model(batch)
        errors = (batch - out.x_hat).pow(2).sum(dim=1)   # [B]

        # Sample dead-neuron replacement directions proportional to error
        probs = (errors / errors.sum()).cpu()
        indices = torch.multinomial(probs, num_samples=len(dead), replacement=True)
        replacement_vecs = batch[indices.to(self.device)]  # [n_dead, D]

        # Normalise and scale
        norms = replacement_vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        replacement_vecs = replacement_vecs / norms * cfg.resample_scale

        # Reinitialise encoder column
        self.model.W_enc.data[:, dead] = replacement_vecs.T          # [D, n_dead]
        self.model.b_enc.data[dead]    = 0.0

        # Keep decoder row aligned with encoder
        self.model.W_dec.data[dead, :] = replacement_vecs            # [n_dead, D]
        self.model.normalize_decoder()

        # Also reinitialise gate weights for gated SAE
        if cfg.activation == "gated" and self.model.W_gate is not None:
            self.model.W_gate.data[:, dead] = replacement_vecs.T
            self.model.b_gate.data[dead]    = 0.0  # type: ignore[union-attr]

        # Reset activity counters
        self.steps_since_active[dead] = 0

        return int(len(dead))

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> Path:
        step    = self.global_step
        ck_dir  = Path(self.cfg.out_dir) / f"step_{step:07d}"
        ck_dir.mkdir(parents=True, exist_ok=True)

        # Model weights
        torch.save(self.model.state_dict_sae(), ck_dir / "weights.pt")

        # Optimiser + scheduler state (for resuming)
        torch.save(self.optimizer.state_dict(),  ck_dir / "optimizer.pt")
        torch.save(self.scheduler.state_dict(),  ck_dir / "scheduler.pt")

        # Activity tracker
        torch.save(self.steps_since_active.cpu(), ck_dir / "activity.pt")

        # Latest metrics
        last_m = self.history[-1] if self.history else {}
        meta = {
            "step":           step,
            "global_step":    step,
            "cfg":            asdict(self.cfg),
            "metrics":        last_m,
        }
        (ck_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        # Symlink "latest" → newest checkpoint
        latest = Path(self.cfg.out_dir) / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(ck_dir.name)

        logger.info("Checkpoint saved → %s", ck_dir)
        return ck_dir

    def _load_checkpoint(self, path: str) -> None:
        ck_dir = Path(path)
        if not ck_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ck_dir}")

        weights = torch.load(ck_dir / "weights.pt",    weights_only=True, map_location=self.device)
        opt_sd  = torch.load(ck_dir / "optimizer.pt",  weights_only=True, map_location="cpu")
        sch_sd  = torch.load(ck_dir / "scheduler.pt",  weights_only=True, map_location="cpu")
        act     = torch.load(ck_dir / "activity.pt",   weights_only=True)
        meta    = json.loads((ck_dir / "meta.json").read_text())

        self.model.load_state_dict_sae(weights)
        self.optimizer.load_state_dict(opt_sd)
        self.scheduler.load_state_dict(sch_sd)
        self.steps_since_active = act.to(self.device)
        self.global_step = meta["global_step"]

        logger.info("Resumed from checkpoint at step %d", self.global_step)


# ---------------------------------------------------------------------------
# _InfiniteTokenIter — streams shards → batches endlessly
# ---------------------------------------------------------------------------

class _InfiniteTokenIter:
    """
    Yields [batch_size, d_model] tensors endlessly from a SAEDataset.
    Shuffles within each shard; cycles through shards round-robin.
    """

    def __init__(self, dataset, layer: int, batch_size: int, seed: int) -> None:
        self.dataset    = dataset
        self.layer      = layer
        self.batch_size = batch_size
        self.rng        = torch.Generator()
        self.rng.manual_seed(seed)

        self._buf:    Optional[torch.Tensor] = None
        self._pos:    int = 0
        self._shard:  int = 0

    def __next__(self) -> torch.Tensor:
        while True:
            if self._buf is None or self._pos + self.batch_size > len(self._buf):
                self._load_next_shard()
            batch = self._buf[self._pos : self._pos + self.batch_size]
            self._pos += self.batch_size
            if len(batch) == self.batch_size:
                return batch
            # Partial tail: load next shard and concatenate
            self._load_next_shard()
            extra = self._buf[: self.batch_size - len(batch)]
            self._pos = self.batch_size - len(batch)
            return torch.cat([batch, extra], dim=0)

    def _load_next_shard(self) -> None:
        n = self.dataset.n_shards
        idx = self._shard % n
        self._shard += 1
        shard = self.dataset.get_shard(self.layer, idx).float()
        perm  = torch.randperm(len(shard), generator=self.rng)
        self._buf = shard[perm]
        self._pos = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_lr_fn(cfg: SAEConfig):
    """Return a LambdaLR multiplier: linear warmup then cosine decay."""
    warmup = cfg.warmup_steps
    total  = cfg.n_steps
    ratio  = cfg.lr_end / max(cfg.lr, 1e-12)

    def fn(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        t = (step - warmup) / max(total - warmup, 1)
        t = min(t, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return ratio + (1.0 - ratio) * cosine

    return fn


@torch.no_grad()
def _compute_mean(
    dataset,
    layer: int,
    device: str,
    max_tokens: int = 50_000,
) -> torch.Tensor:
    """Compute the empirical mean of activations (for b_pre init)."""
    acc   = None
    count = 0
    for _, shard in dataset.iter_shards(layer):
        n = min(len(shard), max_tokens - count)
        chunk = shard[:n].float()
        if acc is None:
            acc = chunk.sum(dim=0)
        else:
            acc += chunk.sum(dim=0)
        count += n
        if count >= max_tokens:
            break
    if acc is None or count == 0:
        raise RuntimeError("Dataset is empty — cannot compute mean.")
    return (acc / count).to(device)
