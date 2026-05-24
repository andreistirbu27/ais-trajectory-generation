"""
v13 (TrajDiff) — Whole-sequence diffusion Transformer for endpoint-conditioned
AIS trajectory generation.

Paper-track module. Do NOT use for school-report work; v10 remains the school
report's production model. See `paper/notes/v13_architecture.md` for the
design rationale.

Architecture (DiT-style):
  - Backbone: stack of `n_layer` blocks, each with
      (i) self-attention over T trajectory tokens,
     (ii) cross-attention to a conditioning memory built from
          {start, end, vessel_type, K retrieved routes},
    (iii) FFN.
    All sub-layers are pre-LayerNorm with adaLN-zero scale/shift modulated by
    the noise level sigma (Peebles & Xie 2023).
  - Reuses v10's `PerStepRouteEncoder` for the retrieval bank — verbatim,
    no copy-paste — so the conditioning is exactly comparable to v10's.
  - Network signature: F(x_noisy_in, c_noise, cond) -> same shape as x.
    Wired to work with `src.diffusion.edm_preconditioning` so the trained
    network is interpreted as an EDM denoiser via the standard
    (c_skip, c_out, c_in, c_noise) reparameterization.
  - CFG (classifier-free guidance): conditioning gets independently dropped
    during training with `p_uncond` (default 0.1). At inference time the
    sampler queries both conditional and unconditional forward passes; mixing
    is handled in `src.diffusion.edm_sample`.

Input/output shapes:
  - Trajectory (training target): (B, T=128, 2), normalized (lon, lat).
  - Retrieved routes: (B, K, T_orig=128, 2), normalized.
  - vessel_type: (B,) int64.
  - start, end: (B, 2), normalized.
  - Output of forward(): same shape as input trajectory (B, T, 2).

This module is intentionally short — no training loop, no eval — those live in
`scripts/paper/train_gen_v13.py` and `scripts/paper/eval_gen_v13.py`
(to be written).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_gen_v10 import PerStepRouteEncoder


# ─── Config ─────────────────────────────────────────────────────────────────

@dataclass
class TrajDiffConfig:
    seq_len: int = 128
    in_dim: int = 2          # (lon, lat) normalized
    n_vessel_types: int = 28
    vessel_emb_dim: int = 16
    d_model: int = 384
    n_heads: int = 8
    ffn_mult: int = 4
    n_layer: int = 8
    dropout: float = 0.0     # dropout in attn/FFN; usually 0 for diffusion
    p_uncond: float = 0.1    # CFG conditioning-drop probability at training
    max_retrieved: int = 5
    t_retr: int = 32
    # Sinusoidal time embedding dim before MLP. Reused for c_noise (=0.25*log sigma).
    time_emb_dim: int = 256


# ─── Time / sigma embedding ─────────────────────────────────────────────────

def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Standard sinusoidal positional embedding, fed by a scalar per sample.

    Args:
        t   : (B,) tensor of scalar inputs.
        dim : even integer.
    Returns:
        (B, dim) embedding.
    """
    assert dim % 2 == 0
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, device=device, dtype=t.dtype))
        * torch.arange(half, device=device, dtype=t.dtype) / half
    )
    angles = t.view(-1, 1) * freqs.view(1, -1)
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


# ─── Conditioning encoder ───────────────────────────────────────────────────

class ConditionEncoder(nn.Module):
    """Build a (B, M, d_model) memory tensor from (start, end, vessel_type, retrieved)."""

    def __init__(self, cfg: TrajDiffConfig):
        super().__init__()
        d = cfg.d_model
        self.start_proj = nn.Linear(2, d)
        self.end_proj = nn.Linear(2, d)
        self.vtype_emb = nn.Embedding(cfg.n_vessel_types, cfg.vessel_emb_dim)
        self.vtype_proj = nn.Linear(cfg.vessel_emb_dim, d)

        # Token-type embeddings for {start, end, vtype, retrieval_*}
        self.token_type_emb = nn.Embedding(4, d)

        # Retrieved routes — reuse v10's per-step encoder verbatim.
        self.retrieval_enc = PerStepRouteEncoder(
            d_model=d, max_retrieved=cfg.max_retrieved, t_retr=cfg.t_retr,
        )

        # Unconditional ("null") embedding for CFG. Used at training time when
        # conditioning is dropped, and at sampling time as the unconditional
        # branch. Same shape as a single base token so we can substitute the
        # entire conditioning memory.
        n_base = 3
        n_retr = cfg.max_retrieved * cfg.t_retr
        self.null_memory = nn.Parameter(torch.zeros(n_base + n_retr, d))
        nn.init.normal_(self.null_memory, std=0.02)

        nn.init.xavier_uniform_(self.start_proj.weight)
        nn.init.xavier_uniform_(self.end_proj.weight)
        nn.init.xavier_uniform_(self.vtype_proj.weight)
        nn.init.zeros_(self.start_proj.bias)
        nn.init.zeros_(self.end_proj.bias)
        nn.init.zeros_(self.vtype_proj.bias)

    def forward(
        self,
        start: torch.Tensor,        # (B, 2)
        end: torch.Tensor,          # (B, 2)
        vessel_type: torch.Tensor,  # (B,) int64
        retrieved: torch.Tensor,    # (B, K, T_orig, 2)
        drop_mask: Optional[torch.Tensor] = None,    # (B,) bool, True = drop conditioning
    ):
        B = start.shape[0]
        s_tok = self.start_proj(start) + self.token_type_emb.weight[0]
        e_tok = self.end_proj(end) + self.token_type_emb.weight[1]
        v_tok = self.vtype_proj(self.vtype_emb(vessel_type)) + self.token_type_emb.weight[2]
        base = torch.stack([s_tok, e_tok, v_tok], dim=1)        # (B, 3, d)

        r_tok = self.retrieval_enc(retrieved)                   # (B, K*T_retr, d)
        r_tok = r_tok + self.token_type_emb.weight[3].view(1, 1, -1)

        memory = torch.cat([base, r_tok], dim=1)                # (B, M, d)

        if drop_mask is not None and drop_mask.any():
            # Per-sample: replace entire memory with `null_memory` where dropped.
            null = self.null_memory.unsqueeze(0).expand(B, -1, -1)
            memory = torch.where(drop_mask.view(B, 1, 1), null, memory)

        return memory


# ─── DiT block with adaLN-zero ──────────────────────────────────────────────

class AdaLNZero(nn.Module):
    """Per-block conditioning: takes a sigma-embedding and emits
    (scale, shift, gate) triplets for two LayerNorms (self-attn + FFN).

    The "zero" trick: initialise the final linear to zero so each block starts
    as an identity at t=0 of training.
    """

    def __init__(self, d_model: int, n_groups: int = 6):
        super().__init__()
        # 6 = (shift, scale, gate) for each of (attn, ffn).
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, n_groups * d_model),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, c: torch.Tensor):
        return self.mlp(c).chunk(6, dim=-1)        # each (B, d_model)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, cfg: TrajDiffConfig):
        super().__init__()
        d, nh = cfg.d_model, cfg.n_heads
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, nh, dropout=cfg.dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(d, nh, dropout=cfg.dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.ffn_mult * d),
            nn.GELU(),
            nn.Linear(cfg.ffn_mult * d, d),
        )
        self.adaln = AdaLNZero(d)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor, memory: torch.Tensor):
        sa_shift, sa_scale, sa_gate, ff_shift, ff_scale, ff_gate = self.adaln(cond_emb)

        h = _modulate(self.norm1(x), sa_shift, sa_scale)
        sa_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + sa_gate.unsqueeze(1) * sa_out

        # Cross-attention to conditioning memory. NO adaLN modulation here:
        # the memory is already a function of the conditioning; gating once is
        # enough. This matches MotionDiffuser's design.
        h2 = self.cross_norm(x)
        ca_out, _ = self.cross_attn(h2, memory, memory, need_weights=False)
        x = x + ca_out

        h3 = _modulate(self.norm2(x), ff_shift, ff_scale)
        x = x + ff_gate.unsqueeze(1) * self.ffn(h3)
        return x


# ─── Backbone ───────────────────────────────────────────────────────────────

class TrajDiff(nn.Module):
    """v13 backbone. Trained as an EDM denoiser; see `src/diffusion.py`."""

    def __init__(self, cfg: TrajDiffConfig):
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.in_dim, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        # Sigma -> cond_emb path
        self.t_mlp = nn.Sequential(
            nn.Linear(cfg.time_emb_dim, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        self.cond_encoder = ConditionEncoder(cfg)
        self.blocks = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.n_layer)])

        self.norm_out = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.in_dim)
        # Final adaLN: shift + scale for the output norm.
        self.adaln_out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
        )
        nn.init.zeros_(self.adaln_out[-1].weight)
        nn.init.zeros_(self.adaln_out[-1].bias)
        # Zero-init the output proj so the network starts as identity.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _cond_embedding(self, c_noise: torch.Tensor) -> torch.Tensor:
        """Map per-sample noise scalar (c_noise from EDM precond) to (B, d_model)."""
        emb = sinusoidal_embedding(c_noise, self.cfg.time_emb_dim)
        return self.t_mlp(emb)

    def forward(
        self,
        x_in: torch.Tensor,         # (B, T, 2) — preconditioned input (c_in * x_noisy)
        c_noise: torch.Tensor,      # (B,) scalar per sample (= 0.25 * log sigma)
        cond: dict,                 # {"start", "end", "vessel_type", "retrieved", optional "drop_mask"}
    ) -> torch.Tensor:
        B, T, D = x_in.shape
        assert T == self.cfg.seq_len, f"got T={T}, expected {self.cfg.seq_len}"

        h = self.input_proj(x_in) + self.pos_emb[:, :T]
        cond_emb = self._cond_embedding(c_noise)

        memory = self.cond_encoder(
            start=cond["start"],
            end=cond["end"],
            vessel_type=cond["vessel_type"],
            retrieved=cond["retrieved"],
            drop_mask=cond.get("drop_mask"),
        )

        for block in self.blocks:
            h = block(h, cond_emb, memory)

        # Output adaLN + projection.
        out_shift, out_scale = self.adaln_out(cond_emb).chunk(2, dim=-1)
        h = _modulate(self.norm_out(h), out_shift, out_scale)
        return self.out_proj(h)

    # -------- Convenience helpers --------
    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Sampling-time CFG conditioning ─────────────────────────────────────────

def make_cfg_conds(
    cond: dict,
    p_uncond: float,
    device,
) -> dict:
    """Add a random drop_mask to `cond` for training-time CFG.

    The mask is True with probability `p_uncond`; ConditionEncoder reads it
    and substitutes the unconditional null memory at those samples.
    """
    B = cond["start"].shape[0]
    drop = (torch.rand(B, device=device) < p_uncond)
    return {**cond, "drop_mask": drop}


def make_uncond(cond: dict, device) -> dict:
    """Build an all-dropped conditioning dict for sampling-time unconditional pass."""
    B = cond["start"].shape[0]
    return {**cond, "drop_mask": torch.ones(B, dtype=torch.bool, device=device)}


__all__ = [
    "TrajDiffConfig",
    "ConditionEncoder",
    "DiTBlock",
    "TrajDiff",
    "make_cfg_conds",
    "make_uncond",
]
