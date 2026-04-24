"""
v12 trajectory generator — K-ring pointer over a partitioned water grid.

Motivation. v11 used per-step Halton candidates in a 3–25 km forward cone,
which was catastrophically mismatched with the dataset's 0.34 km median step
(ADE 30 km, path-length ratio 13×). v10's continuous regression produced
ADE 6.2 km but collapsed on long routes (46 km) and relied on post-hoc land
projection rather than structural water-validity. v12 replaces both with
**pointer selection over a precomputed water-cell graph**:

  - The trajectory space is quantized onto the same 0.005° grid as the SDF
    (~0.47 km × 0.55 km cells over bbox (-125, 10, -60, 55) → 9000 × 13000).
  - Each cell is labelled water iff `sdf_km ≤ -tau_km` (tau = 0.5 km).
  - At each decoder step, candidates are a K-ring of Chebyshev-distance-K
    cells around the current cell + an END token. K=10 covers 99% of
    observed consecutive-cell transitions in the training data (Exp 1).
  - A water-mask is applied to candidate logits, so the softmax only
    assigns probability to water cells. Water-validity is therefore
    structural, not a post-hoc projection.
  - A small sub-cell offset head regresses ±0.5 cell-edges within the
    selected cell to recover sub-km precision.

What is reused from v10 (via subclass):
  - PerStepRouteEncoder (K retrieval neighbors × T_retr points each).
  - `encode_condition_retrieval` (encoder stack + token-type embeddings).
  - The `nn.TransformerDecoder` stack, windowed/full causal masks, positional
    encoding.
  - `_bearing_features` for sin/cos to destination.

What is new:
  - `emb_x = nn.Embedding(N_x, d_model)`, `emb_y = nn.Embedding(N_y, d_model)`
    — axis-factored absolute-cell embeddings on the decoder input side.
  - `offset_proj = nn.Linear(2, d_model)` — projects sub-cell offset into the
    same space.
  - `bearing_proj = nn.Linear(2, d_model)` — sin/cos bearing projection.
  - `rel_emb_table` — learned embedding for the (2K+1)² Moore-K-ring relative
    offsets + 1 END slot. Pointer logits = h_t · rel_emb[offset] / √d.
  - `offset_head = nn.Linear(d_model, 2)` — sub-cell regression.

Decoder input at step t:
    emb_x(ix_t) + emb_y(iy_t) + offset_proj(offset_t) + bearing_proj(sin, cos)
Then + positional encoding, causal mask, TransformerDecoder (unchanged).

Output (pointer over K-ring + END):
    logits_t   = (h_t · rel_emb_table.T) / sqrt(d_model)     → (B, n_cand)
    offset_t+1 = 0.5 * tanh(offset_head(h_t))                → (B, 2)

Water mask is applied per-step: for each candidate cell absolute index
(ix_t + dx, iy_t + dy), we check water_mask[iy, ix]. Candidates failing
the mask (or out-of-bounds) get -inf logit. END token is always valid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_gen_v10 import TrajectoryGeneratorV10


# ─── Relative K-ring offsets ────────────────────────────────────────────────

def build_kring_offsets(K: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Enumerate the Moore-K-ring relative offsets in row-major order.

    Order: dy varies slowest, dx fastest.  Offset i at (2K+1)*(dy+K) + (dx+K).

    Returns:
        dx_table, dy_table: (n_ring,) int64 — each in [-K, +K].
        n_ring: (2K+1)**2
    """
    n_ring = (2 * K + 1) ** 2
    dy_grid, dx_grid = torch.meshgrid(
        torch.arange(-K, K + 1, dtype=torch.int64),
        torch.arange(-K, K + 1, dtype=torch.int64),
        indexing="ij",
    )
    dx_table = dx_grid.reshape(-1)       # (n_ring,)
    dy_table = dy_grid.reshape(-1)
    return dx_table, dy_table, n_ring


def offset_to_index(dx: torch.Tensor, dy: torch.Tensor, K: int) -> torch.Tensor:
    """Pointer-target index for (dx, dy) in K-ring. Returns -100 if out of range."""
    in_range = (dx.abs() <= K) & (dy.abs() <= K)
    idx = (dy + K) * (2 * K + 1) + (dx + K)
    return torch.where(in_range, idx, torch.full_like(idx, -100))


# ─── Model config ────────────────────────────────────────────────────────────

@dataclass
class V12Config:
    N_x: int           # grid width  (e.g. 13000 for 0.005° over 65° lon)
    N_y: int           # grid height (e.g.  9000 for 0.005° over 45° lat)
    K: int = 10        # pointer ring radius (Chebyshev)
    max_retrieved: int = 5
    t_retr: int = 32
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 2
    num_decoder_layers: int = 3
    dim_feedforward: int = 1024
    dropout: float = 0.1
    k_past: Optional[int] = None
    num_vessel_types: int = 28
    vessel_type_embed_dim: int = 8
    max_seq_len: int = 512

    @property
    def n_cand(self) -> int:
        return (2 * self.K + 1) ** 2 + 1   # K-ring + END


# ─── v12 model ───────────────────────────────────────────────────────────────

class TrajectoryGeneratorV12(TrajectoryGeneratorV10):
    """Pointer variant of v10.  Encoder + decoder core = v10.  Head = pointer."""

    END_TOKEN_OFFSET = 0   # the END slot is always at index n_ring (end of logits)

    def __init__(self, cfg: V12Config):
        super().__init__(
            max_retrieved=cfg.max_retrieved,
            t_retr=cfg.t_retr,
            n_points=128,                 # unused by v10 __init__ but required by signature
            k_past=cfg.k_past,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            num_vessel_types=cfg.num_vessel_types,
            vessel_type_embed_dim=cfg.vessel_type_embed_dim,
            max_seq_len=cfg.max_seq_len,
        )
        self.cfg = cfg
        d = cfg.d_model

        # ── New I/O machinery (replaces v10's waypoint_proj / delta_head) ──
        # v10's `waypoint_proj = Linear(5, d)` and `delta_head = Linear(d, 2)`
        # remain in the state dict but are unused; we deliberately don't reuse
        # them so the graph is explicit.
        self.emb_x = nn.Embedding(cfg.N_x, d)
        self.emb_y = nn.Embedding(cfg.N_y, d)
        self.offset_proj  = nn.Linear(2, d)
        self.bearing_proj = nn.Linear(2, d)

        self.rel_emb_table = nn.Embedding(cfg.n_cand, d)
        self.offset_head   = nn.Linear(d, 2)

        nn.init.normal_(self.emb_x.weight, std=0.02)
        nn.init.normal_(self.emb_y.weight, std=0.02)
        nn.init.xavier_uniform_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)
        nn.init.xavier_uniform_(self.bearing_proj.weight)
        nn.init.zeros_(self.bearing_proj.bias)
        nn.init.normal_(self.rel_emb_table.weight, std=0.02)
        nn.init.uniform_(self.offset_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.offset_head.bias)

        self._logit_scale = 1.0 / math.sqrt(d)

        # Precomputed relative offsets (n_ring,) — registered as buffers
        dx_table, dy_table, n_ring = build_kring_offsets(cfg.K)
        self.register_buffer("dx_table", dx_table, persistent=False)
        self.register_buffer("dy_table", dy_table, persistent=False)
        self.n_ring = n_ring   # n_cand == n_ring + 1

    # ── Water mask injection (one-shot at model init time) ──────────────────
    def attach_water_mask(self, water_mask: torch.Tensor):
        """Attach the (N_y, N_x) bool water-mask tensor as a non-persistent buffer.

        Called after the model is moved to device.  Decoupled from __init__
        so the bulky (~117 MB) mask doesn't have to live in the state dict.
        """
        assert water_mask.dtype == torch.bool
        assert water_mask.shape == (self.cfg.N_y, self.cfg.N_x), \
            f"water_mask shape {tuple(water_mask.shape)} != ({self.cfg.N_y}, {self.cfg.N_x})"
        # Use register_buffer so the mask follows .to(device) calls.
        # persistent=False keeps it out of state_dict — we always re-attach.
        self.register_buffer("water_mask", water_mask, persistent=False)

    # ── Bearing in cell-grid coordinates ────────────────────────────────────
    def _bearing_to_end_cell(
        self,
        cur_ix: torch.Tensor,     # (T, B) int64
        cur_iy: torch.Tensor,     # (T, B) int64
        end_ix: torch.Tensor,      # (B,)
        end_iy: torch.Tensor,      # (B,)
    ) -> torch.Tensor:
        """Return (sin, cos) of bearing from (cur_ix, cur_iy) to (end_ix, end_iy).

        Output shape (T, B, 2).  Row 0 = maxy, so +y means south; we flip sign
        so sin/cos are in conventional "north-is-up" terms.
        """
        dx = (end_ix.unsqueeze(0) - cur_ix).float()
        dy_raw = (end_iy.unsqueeze(0) - cur_iy).float()
        # Flip dy so +y = north (row index decreases going north)
        dy = -dy_raw
        angle = torch.atan2(dy, dx)
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)

    # ── Build decoder input tokens ──────────────────────────────────────────
    def _embed_step(
        self,
        ix: torch.Tensor,          # (T, B) int64
        iy: torch.Tensor,          # (T, B) int64
        offset: torch.Tensor,      # (T, B, 2) float
        bearing: torch.Tensor,     # (T, B, 2) float  (sin, cos)
    ) -> torch.Tensor:
        """Compose the per-step decoder input. Returns (T, B, d_model)."""
        ix_safe = ix.clamp(0, self.cfg.N_x - 1)
        iy_safe = iy.clamp(0, self.cfg.N_y - 1)
        return (self.emb_x(ix_safe)
                + self.emb_y(iy_safe)
                + self.offset_proj(offset)
                + self.bearing_proj(bearing))

    # ── Water-aware candidate mask ──────────────────────────────────────────
    def _candidate_mask(
        self,
        ix: torch.Tensor,          # (T, B) int64
        iy: torch.Tensor,          # (T, B) int64
    ) -> torch.Tensor:
        """Compute per-step candidate validity mask.

        Returns (T, B, n_cand) bool, True = valid.  END slot always True.
        """
        # Broadcast ring offsets: (T, B, n_ring)
        cand_ix = ix.unsqueeze(-1) + self.dx_table.view(1, 1, -1)
        cand_iy = iy.unsqueeze(-1) + self.dy_table.view(1, 1, -1)
        in_bounds = ((cand_ix >= 0) & (cand_ix < self.cfg.N_x)
                     & (cand_iy >= 0) & (cand_iy < self.cfg.N_y))
        clamped_ix = cand_ix.clamp(0, self.cfg.N_x - 1)
        clamped_iy = cand_iy.clamp(0, self.cfg.N_y - 1)
        is_water = self.water_mask[clamped_iy, clamped_ix]
        ring_valid = in_bounds & is_water
        # END token
        T, B = ix.shape
        end_slot = torch.ones(T, B, 1, dtype=torch.bool, device=ix.device)
        return torch.cat([ring_valid, end_slot], dim=-1)   # (T, B, n_cand)

    # ── Forward (teacher forcing) ───────────────────────────────────────────
    def forward(
        self,
        start_norm,                  # (B, 2) — normalized lon/lat (for encoder)
        end_norm,                     # (B, 2)
        vessel_type,                  # (B,)
        retrieved_norm,               # (B, K, T_orig, 2)
        retrieval_mask,               # (B, K)
        cell_ix,                      # (T, B) int64
        cell_iy,                      # (T, B) int64
        offset,                       # (T, B, 2) float
        end_ix,                       # (B,) int64
        end_iy,                       # (B,) int64
    ):
        """Teacher-forcing forward. Returns:
          logits (T-1, B, n_cand)
          offset_pred (T-1, B, 2)
        """
        T, B = cell_ix.shape
        device = cell_ix.device

        # Encoder memory (unchanged from v10)
        memory, pad_mask = self.encode_condition_retrieval(
            start_norm, end_norm, vessel_type, retrieved_norm, retrieval_mask)

        # Decoder input tokens
        bearing = self._bearing_to_end_cell(cell_ix, cell_iy, end_ix, end_iy)
        dec_input = self._embed_step(cell_ix, cell_iy, offset, bearing)
        dec_input = self.pos_enc(dec_input)

        causal_mask = self._windowed_causal_mask(T, device)
        out = self.decoder(
            dec_input, memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=pad_mask,
        )                                                    # (T, B, d)
        h = out[:-1]                                         # (T-1, B, d)

        # Pointer logits: h @ rel_emb.T / sqrt(d)
        logits = h @ self.rel_emb_table.weight.T * self._logit_scale  # (T-1, B, n_cand)

        # Water-adjacency mask on ring candidates (END always valid)
        valid = self._candidate_mask(cell_ix[:-1], cell_iy[:-1])      # (T-1, B, n_cand)
        logits = logits.masked_fill(~valid, float("-inf"))

        offset_pred = 0.5 * torch.tanh(self.offset_head(h))            # (T-1, B, 2)
        return logits, offset_pred

    # ── Inference (autoregressive, pointer argmax) ──────────────────────────
    @torch.no_grad()
    def generate(
        self,
        start_norm,                  # (B, 2)
        end_norm,                     # (B, 2)
        vessel_type,                  # (B,)
        retrieved_norm,               # (B, K, T_orig, 2)
        retrieval_mask,               # (B, K)
        start_ix,                     # (B,) int64
        start_iy,                     # (B,) int64
        start_offset,                 # (B, 2) float
        end_ix,                       # (B,) int64
        end_iy,                       # (B,) int64
        n_steps: int,                 # maximum decoder steps (incl. start)
        self_bias: float = 0.0,       # subtract this from the self-loop logit
    ) -> dict:
        """Argmax rollout. Stops per-sample when END is emitted or n_steps hit.

        Returns a dict with:
          cell_ix   (L_max, B) int64
          cell_iy   (L_max, B) int64
          offset    (L_max, B, 2) float
          alive     (L_max, B) bool  — True before END was emitted
          length    (B,) int — number of valid waypoints per sample
        """
        device = start_norm.device
        B = start_norm.shape[0]

        memory, pad_mask = self.encode_condition_retrieval(
            start_norm, end_norm, vessel_type, retrieved_norm, retrieval_mask)

        ix_hist  = [start_ix]
        iy_hist  = [start_iy]
        off_hist = [start_offset]
        alive = torch.ones(B, dtype=torch.bool, device=device)
        length = torch.ones(B, dtype=torch.int64, device=device)   # start counts

        for t in range(n_steps - 1):
            T_cur = len(ix_hist)
            cix = torch.stack(ix_hist, dim=0)                       # (T, B)
            ciy = torch.stack(iy_hist, dim=0)
            coff = torch.stack(off_hist, dim=0)                     # (T, B, 2)

            bearing = self._bearing_to_end_cell(cix, ciy, end_ix, end_iy)
            dec_in = self._embed_step(cix, ciy, coff, bearing)
            dec_in = self.pos_enc(dec_in)

            causal = self._windowed_causal_mask(T_cur, device)
            out = self.decoder(
                dec_in, memory,
                tgt_mask=causal,
                memory_key_padding_mask=pad_mask,
            )                                                       # (T, B, d)
            h_last = out[-1]                                        # (B, d)

            logits = h_last @ self.rel_emb_table.weight.T * self._logit_scale  # (B, n_cand)
            valid = self._candidate_mask(
                cix[-1:], ciy[-1:])[0]                              # (B, n_cand)
            logits = logits.masked_fill(~valid, float("-inf"))

            if self_bias != 0.0:
                # Self-loop lives at ring_idx = K*(2K+1) + K = center of the ring.
                self_slot = self.cfg.K * (2 * self.cfg.K + 1) + self.cfg.K
                logits[:, self_slot] = logits[:, self_slot] - self_bias

            next_idx = logits.argmax(dim=-1)                        # (B,)
            is_end = next_idx == self.n_ring                         # END slot
            # For END samples, stay at current cell from now on — doesn't affect metrics since alive=False
            ring_idx = next_idx.clamp(max=self.n_ring - 1)
            dx = self.dx_table[ring_idx]
            dy = self.dy_table[ring_idx]

            new_ix = cix[-1] + dx
            new_iy = ciy[-1] + dy
            new_offset = 0.5 * torch.tanh(self.offset_head(h_last)) # (B, 2)

            # Samples that had already ended keep their last cell
            # (prevents cell drift, though we also mark them dead).
            prev_alive = alive.clone()
            alive = alive & ~is_end
            stay = (~prev_alive).unsqueeze(-1)                      # (B, 1)
            new_ix = torch.where(prev_alive, new_ix, cix[-1])
            new_iy = torch.where(prev_alive, new_iy, ciy[-1])
            new_offset = torch.where(stay, coff[-1], new_offset)

            length = length + prev_alive.to(torch.int64)

            ix_hist.append(new_ix)
            iy_hist.append(new_iy)
            off_hist.append(new_offset)

            if not alive.any():
                break

        ix_out  = torch.stack(ix_hist, dim=0)                       # (L, B)
        iy_out  = torch.stack(iy_hist, dim=0)
        off_out = torch.stack(off_hist, dim=0)                      # (L, B, 2)
        # `alive` at the end of the loop is per-sample post-rollout; reconstruct per-step.
        # For simplicity return only the final `length` tensor.
        return {
            "cell_ix": ix_out,
            "cell_iy": iy_out,
            "offset":  off_out,
            "length":  length,
        }


# ─── Loss helpers (kept here so the model file is self-contained) ───────────

def v12_loss(
    logits: torch.Tensor,          # (T-1, B, n_cand)
    offset_pred: torch.Tensor,     # (T-1, B, 2)
    target_cell_ix: torch.Tensor,  # (T, B) int64
    target_cell_iy: torch.Tensor,  # (T, B) int64
    target_offset: torch.Tensor,   # (T, B, 2) float
    K: int,
    lambda_offset: float = 1.0,
    lambda_smooth: float = 0.5,
    ignore_self_loop: bool = True,
) -> dict:
    """Compute v12 pointer CE + offset MSE + smoothness penalty.

    Steps in which the GT transition exceeds the K-ring are masked out of the
    CE loss via ignore_index=-100.
    """
    T_minus_1 = logits.shape[0]
    assert target_cell_ix.shape[0] == T_minus_1 + 1

    dx = target_cell_ix[1:] - target_cell_ix[:-1]                   # (T-1, B)
    dy = target_cell_iy[1:] - target_cell_iy[:-1]
    gt_idx = offset_to_index(dx, dy, K)                             # (T-1, B)

    # Optionally skip self-loop steps from the CE: the 128-point resample
    # produces 26% consecutive-same-cell transitions which teach the model
    # to emit "stay" at inference, collapsing path length. With ignore_self_loop
    # the model only trains on genuine movements.
    if ignore_self_loop:
        self_loop = (dx == 0) & (dy == 0)
        gt_idx = torch.where(self_loop, torch.full_like(gt_idx, -100), gt_idx)

    # Some GT next-cells may be "water enough" for the quantizer (snap within
    # 10 km) but not water under the pointer's tau=0.5 km mask. Those steps'
    # GT logit is `-inf` after masking. Skip them in CE.
    n_cand = logits.shape[-1]
    gt_safe = gt_idx.clamp(min=0)
    gt_logit = logits.gather(-1, gt_safe.unsqueeze(-1)).squeeze(-1) # (T-1, B)
    unreachable = ~torch.isfinite(gt_logit) & (gt_idx != -100)
    gt_idx = torch.where(unreachable,
                          torch.full_like(gt_idx, -100), gt_idx)

    ce = F.cross_entropy(
        logits.reshape(-1, n_cand),
        gt_idx.reshape(-1),
        ignore_index=-100,
    )

    # Offset MSE: predicted offset is the next step's sub-cell offset.
    valid = gt_idx != -100                                           # (T-1, B)
    if valid.any():
        target_off = target_offset[1:]                               # (T-1, B, 2)
        offset_mse = ((offset_pred - target_off) ** 2).sum(-1)       # (T-1, B)
        offset_mse = (offset_mse * valid.float()).sum() / valid.float().sum()
    else:
        offset_mse = torch.zeros((), device=logits.device)

    # Smoothness penalty on predicted-transition deltas: (dx_t - dx_{t-1})^2
    # Uses the GT cell transitions since the pointer is argmax-supervised; this
    # penalizes offset-head zig-zag, not cell zig-zag.
    if offset_pred.shape[0] >= 2:
        accel = offset_pred[1:] - offset_pred[:-1]                   # (T-2, B, 2)
        smooth = (accel ** 2).mean()
    else:
        smooth = torch.zeros((), device=logits.device)

    total = ce + lambda_offset * offset_mse + lambda_smooth * smooth

    with torch.no_grad():
        # Top-1 accuracy over non-ignored steps
        pred_idx = logits.argmax(dim=-1)
        correct = (pred_idx == gt_idx) & valid
        top1 = correct.float().sum() / valid.float().sum().clamp_min(1)

    return {
        "total": total,
        "ce": ce,
        "offset": offset_mse,
        "smooth": smooth,
        "top1": top1,
    }
