"""
Retrieval-augmented trajectory generator (v9).

Extends TrajectoryGenerator with variable-count retrieved-route conditioning.
Architecture:
    Encoder memory = 3 base tokens + K retrieved-route tokens (padded to K)
    Decoder        = unchanged from v4/v5 (5-dim input with bearing features)

Route encoding (how to compress each (T, 2) retrieved route into one d_model vector):
  - mean_pool         : mean over T positions, then Linear(2 → d_model)
  - linear_flatten    : flatten (T, 2) → (2T,) then Linear(2T → d_model)
  - tiny_transformer  : small bidirectional encoder, mean-pool hidden states

Start with mean_pool (cheapest, proves concept). Swap if capacity is a limiter.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.model_gen import PositionalEncoding, TrajectoryGenerator


# ─── Route encoder variants ─────────────────────────────────────────────────

class MeanPoolRouteEncoder(nn.Module):
    """Mean-pool positions → Linear(2 → d_model)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(2, d_model)

    def forward(self, routes_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:   routes_norm: (B, K, T, 2) normalized [lon, lat]
        Returns: (B, K, d_model)
        """
        pooled = routes_norm.mean(dim=2)    # (B, K, 2)
        return self.proj(pooled)


class LinearFlattenRouteEncoder(nn.Module):
    """Flatten (T, 2) → (2T,) then Linear to d_model. Preserves full shape."""

    def __init__(self, d_model: int, n_points: int = 128):
        super().__init__()
        self.proj = nn.Linear(2 * n_points, d_model)

    def forward(self, routes_norm: torch.Tensor) -> torch.Tensor:
        B, K, T, _ = routes_norm.shape
        flat = routes_norm.reshape(B, K, 2 * T)
        return self.proj(flat)


# ─── Model ──────────────────────────────────────────────────────────────────

class RetrievalTrajectoryGenerator(TrajectoryGenerator):
    """v5 architecture + retrieved-route tokens as additional encoder memory.

    Design mirrors ObstacleTrajectoryGenerator: extends token_type_emb to
    4 slots (start, end, vtype, retrieved), builds a padding mask, runs
    through encoder, passes memory + mask to the unchanged decoder.
    """

    def __init__(
        self,
        max_retrieved: int = 5,
        route_encoder: str = "mean_pool",
        n_points: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_retrieved = max_retrieved

        # Extend token-type embedding to include retrieval slot
        # (base class defines Embedding(3, d_model) — rebuild as Embedding(4))
        d_model = kwargs.get("d_model", 128)
        self.token_type_emb = nn.Embedding(4, d_model)
        nn.init.normal_(self.token_type_emb.weight, std=0.02)

        if route_encoder == "mean_pool":
            self.route_encoder = MeanPoolRouteEncoder(d_model)
        elif route_encoder == "linear_flatten":
            self.route_encoder = LinearFlattenRouteEncoder(d_model, n_points)
        else:
            raise ValueError(f"Unknown route_encoder: {route_encoder}")

    def encode_condition_retrieval(
        self,
        start: torch.Tensor,            # (B, 2) normalized
        end: torch.Tensor,              # (B, 2) normalized
        vessel_type: torch.Tensor,      # (B,) int64
        retrieved_norm: torch.Tensor,   # (B, K, T, 2) normalized
        retrieval_mask: torch.Tensor,   # (B, K) bool — True = padding
    ):
        """Build encoder memory from base tokens + retrieved routes.

        Returns:
            memory       : (3 + K, B, d_model)
            pad_mask_bk  : (B, 3 + K) — True for padded slots (decoder needs this)
        """
        B = start.shape[0]
        K = retrieved_norm.shape[1]
        device = start.device

        # Base 3 tokens (using existing projections from parent class)
        tok_start = self.start_proj(start)
        tok_end   = self.end_proj(end)
        tok_vtype = self.vessel_type_proj(self.vessel_type_emb(vessel_type))

        # Retrieved-route tokens: (B, K, d_model)
        tok_retr = self.route_encoder(retrieved_norm)

        # Stack to (3 + K, B, d_model)
        base = torch.stack([tok_start, tok_end, tok_vtype], dim=0)  # (3, B, d)
        retr = tok_retr.permute(1, 0, 2)                            # (K, B, d)
        memory = torch.cat([base, retr], dim=0)                     # (3+K, B, d)

        # Token-type embeddings: 0=start, 1=end, 2=vtype, 3=retrieved
        type_ids = torch.cat([
            torch.arange(3, device=device),
            torch.full((K,), 3, device=device, dtype=torch.long),
        ])                                                           # (3+K,)
        type_embs = self.token_type_emb(type_ids)                    # (3+K, d)
        memory = memory + type_embs.unsqueeze(1)                     # broadcast over B

        # Padding mask: first 3 slots never padded; retrieval slots per-sample
        base_unpadded = torch.zeros(B, 3, device=device, dtype=torch.bool)
        pad_mask_bk = torch.cat([base_unpadded, retrieval_mask], dim=1)  # (B, 3+K)

        if self.num_encoder_layers > 0:
            memory = self.encoder(memory, src_key_padding_mask=pad_mask_bk)

        return memory, pad_mask_bk

    def forward(
        self,
        start,
        end,
        vessel_type,
        target_traj,
        retrieved_norm,                 # (B, K, T, 2)
        retrieval_mask,                 # (B, K)
        ss_prob: float = 0.0,
        delta_scaler=None,
        pos_scaler=None,
    ):
        """Teacher-forcing forward with retrieved-route conditioning.

        Overrides parent's forward: parent builds memory from 3 tokens via
        encode_condition(); we build it from 3+K tokens via
        encode_condition_retrieval() and pass the padding mask into the
        decoder's cross-attention.
        """
        T, B, _ = target_traj.shape
        device = target_traj.device

        memory, pad_mask = self.encode_condition_retrieval(
            start, end, vessel_type, retrieved_norm, retrieval_mask)

        if ss_prob <= 0 or delta_scaler is None or pos_scaler is None:
            fracs = torch.linspace(0, 1, T, device=device)
            fracs = fracs.unsqueeze(1).expand(T, B).unsqueeze(2)
            sin_b, cos_b = self._bearing_features(target_traj, end)
            dec_input = torch.cat([target_traj, fracs, sin_b, cos_b], dim=2)
            dec_input = self.waypoint_proj(dec_input)
            dec_input = self.pos_enc(dec_input)
            causal_mask = self._causal_mask(T, device)
            out = self.decoder(
                dec_input, memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=pad_mask,
            )
            pred_deltas = self.delta_head(out[:-1])
            return pred_deltas, target_traj

        # SS path: identical to parent's two-pass logic but with pad_mask
        # plumbed into both decoder calls. Copy/adapt from model_gen.py
        # forward() when enabling SS — not shown here to keep the sketch
        # focused on the retrieval additions.
        raise NotImplementedError(
            "SS path for retrieval model not yet implemented; "
            "use ss_prob=0 for v9 initial training (v5 showed SS no help)."
        )

    @torch.no_grad()
    def generate(
        self,
        start,
        end,
        vessel_type,
        retrieved_norm,
        retrieval_mask,
        n_steps: int = 128,
        pos_scaler=None,
        delta_scaler=None,
    ) -> torch.Tensor:
        """Autoregressive rollout with retrieval conditioning.

        Copy of parent's generate() with encode_condition swapped for
        encode_condition_retrieval() and pad_mask plumbed into the decoder.
        """
        B = start.shape[0]
        device = start.device

        memory, pad_mask = self.encode_condition_retrieval(
            start, end, vessel_type, retrieved_norm, retrieval_mask)

        generated = [start.unsqueeze(0)]

        for t in range(n_steps - 1):
            T_cur = len(generated)
            traj_so_far = torch.cat(generated, dim=0)

            fracs = torch.linspace(0, t / max(n_steps - 1, 1), T_cur, device=device)
            fracs = fracs.unsqueeze(1).expand(T_cur, B).unsqueeze(2)

            sin_b, cos_b = self._bearing_features(traj_so_far, end)
            dec_input = torch.cat([traj_so_far, fracs, sin_b, cos_b], dim=2)
            dec_input = self.waypoint_proj(dec_input)
            dec_input = self.pos_enc(dec_input)

            causal_mask = self._causal_mask(T_cur, device)
            out = self.decoder(
                dec_input, memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=pad_mask,
            )

            pred_delta_norm = self.delta_head(out[-1])

            if delta_scaler is not None and pos_scaler is not None:
                cur_pos_norm = traj_so_far[-1]
                cur_pos_raw = torch.tensor(
                    pos_scaler.inverse(cur_pos_norm.cpu().numpy()),
                    device=device, dtype=torch.float32)
                delta_raw = torch.tensor(
                    delta_scaler.inverse(pred_delta_norm.cpu().numpy()),
                    device=device, dtype=torch.float32)
                next_pos_raw = cur_pos_raw + delta_raw
                next_pos_norm = torch.tensor(
                    pos_scaler.transform(next_pos_raw.cpu().numpy()),
                    device=device, dtype=torch.float32)
            else:
                next_pos_norm = traj_so_far[-1] + pred_delta_norm

            generated.append(next_pos_norm.unsqueeze(0))

        trajectory = torch.cat(generated, dim=0)

        if pos_scaler is not None:
            traj_np = trajectory.cpu().numpy()
            end_np = end.cpu().numpy()
            T_len, B_len, _ = traj_np.shape
            traj_raw = pos_scaler.inverse(traj_np.reshape(-1, 2)).reshape(T_len, B_len, 2)
            end_raw = pos_scaler.inverse(end_np)
            error = end_raw - traj_raw[-1]
            fracs = np.linspace(0, 1, T_len).reshape(-1, 1, 1)
            traj_corrected = traj_raw + fracs * error[np.newaxis]
            traj_norm = pos_scaler.transform(traj_corrected.reshape(-1, 2)).reshape(T_len, B_len, 2)
            trajectory = torch.tensor(traj_norm, device=device, dtype=torch.float32)

        return trajectory
