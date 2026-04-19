"""
TrajectoryGenerator: conditional transformer encoder-decoder for trajectory generation.

Given (start, end, ship_type), generates a full trajectory as a sequence of
displacement deltas. The encoder produces 3 conditioning tokens (start, end,
vessel type). The decoder autoregressively generates waypoints.

Training: teacher forcing — ground truth positions fed as decoder input.
Inference: autoregressive — own predictions fed back as decoder input.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from src.model import PositionalEncoding


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TrajectoryGenerator(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 0,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_vessel_types: int = 28,
        vessel_type_embed_dim: int = 8,
        max_seq_len: int = 512,
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model

        # ── Encoder: conditioning tokens ─────────────────────────────────────
        self.start_proj = nn.Linear(2, d_model)
        self.end_proj   = nn.Linear(2, d_model)

        # Vessel type embedding
        self.vessel_type_emb  = nn.Embedding(num_vessel_types, vessel_type_embed_dim)
        self.vessel_type_proj = nn.Linear(vessel_type_embed_dim, d_model)

        # Learnable token-type embeddings for the 3 encoder tokens
        self.token_type_emb = nn.Embedding(3, d_model)

        # Optional transformer encoder for conditioning token interaction
        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=False,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # ── Decoder: waypoint generation ─────────────────────────────────────
        # Input per decoder step: [lon_norm, lat_norm, progress, sin_bearing, cos_bearing] → 5 dims
        self.waypoint_proj = nn.Linear(5, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Output head: displacement deltas
        self.delta_head = nn.Linear(d_model, 2)

        self._init_weights()

    def _init_weights(self):
        for name, proj in [("start_proj", self.start_proj),
                           ("end_proj", self.end_proj),
                           ("waypoint_proj", self.waypoint_proj)]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

        nn.init.normal_(self.vessel_type_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.vessel_type_proj.weight)
        nn.init.zeros_(self.vessel_type_proj.bias)

        nn.init.normal_(self.token_type_emb.weight, std=0.02)

        nn.init.uniform_(self.delta_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.delta_head.bias)

    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def encode_condition(
        self,
        start: torch.Tensor,        # (B, 2) normalized [lon, lat]
        end: torch.Tensor,           # (B, 2) normalized [lon, lat]
        vessel_type: torch.Tensor,   # (B,) int64 vocab index
    ) -> torch.Tensor:
        """Encode conditioning inputs into 3 memory tokens.

        Returns: (3, B, d_model)
        """
        B = start.shape[0]
        device = start.device

        tok_start = self.start_proj(start)                                  # (B, d_model)
        tok_end   = self.end_proj(end)                                      # (B, d_model)
        tok_vtype = self.vessel_type_proj(self.vessel_type_emb(vessel_type)) # (B, d_model)

        # Add token-type embeddings so the decoder can distinguish start/end/type
        type_ids = torch.arange(3, device=device)
        type_embs = self.token_type_emb(type_ids)                           # (3, d_model)

        memory = torch.stack([tok_start, tok_end, tok_vtype], dim=0)        # (3, B, d_model)
        memory = memory + type_embs.unsqueeze(1)                            # broadcast over B

        if self.num_encoder_layers > 0:
            memory = self.encoder(memory)                                   # (3, B, d_model)

        return memory

    def _bearing_features(self, traj, end):
        """Compute sin/cos bearing from each position to endpoint.

        Args:
            traj: (T, B, 2) positions [lon_norm, lat_norm]
            end: (B, 2) endpoint [lon_norm, lat_norm]

        Returns:
            sin_b, cos_b: each (T, B, 1)
        """
        diff = end.unsqueeze(0) - traj                              # (T, B, 2) [dlon, dlat]
        angle = torch.atan2(diff[..., 1], diff[..., 0])             # (T, B)
        return torch.sin(angle).unsqueeze(2), torch.cos(angle).unsqueeze(2)

    def forward(
        self,
        start: torch.Tensor,         # (B, 2)
        end: torch.Tensor,           # (B, 2)
        vessel_type: torch.Tensor,   # (B,)
        target_traj: torch.Tensor,   # (T, B, 2) normalized positions
        ss_prob: float = 0.0,        # scheduled sampling probability
        delta_scaler=None,           # needed for scheduled sampling
        pos_scaler=None,             # needed for scheduled sampling
    ):
        """Teacher-forcing forward pass with optional two-pass scheduled sampling.

        When ss_prob > 0, runs two parallel teacher-forcing passes:
          Pass 1 (no_grad): TF over GT to get one-step-ahead model predictions.
          Pass 2 (with grad): TF over a mixed trajectory where each step (except
            step 0) is replaced by the model's own prediction from pass 1 with
            probability ss_prob. The loss is computed on pass 2's deltas
            relative to the mixed input.

        This replaces the O(T) sequential decoder calls of step-by-step SS with
        2 parallel passes, bringing per-batch cost back near TF levels.

        Returns:
            pred_deltas: (T-1, B, 2) predicted normalized displacements
            input_traj:  (T, B, 2) what the decoder was actually fed (either
                         target_traj for TF, or mixed_traj for SS). The loss
                         needs this to recompute target deltas relative to
                         the mixed inputs.
        """
        T, B, _ = target_traj.shape
        device = target_traj.device

        # Encode conditioning
        memory = self.encode_condition(start, end, vessel_type)  # (3, B, d_model)

        if ss_prob <= 0 or delta_scaler is None or pos_scaler is None:
            # Pure teacher forcing (fast path)
            fracs = torch.linspace(0, 1, T, device=device)
            fracs = fracs.unsqueeze(1).expand(T, B).unsqueeze(2)     # (T, B, 1)

            sin_b, cos_b = self._bearing_features(target_traj, end)  # (T, B, 1) each
            dec_input = torch.cat([target_traj, fracs, sin_b, cos_b], dim=2)  # (T, B, 5)
            dec_input = self.waypoint_proj(dec_input)
            dec_input = self.pos_enc(dec_input)

            causal_mask = self._causal_mask(T, device)
            out = self.decoder(dec_input, memory, tgt_mask=causal_mask)
            pred_deltas = self.delta_head(out[:-1])                  # (T-1, B, 2)
            return pred_deltas, target_traj

        # Two-pass scheduled sampling
        delta_mean = torch.tensor(delta_scaler.mean, device=device, dtype=torch.float32)
        delta_std  = torch.tensor(delta_scaler.std,  device=device, dtype=torch.float32)
        pos_mean   = torch.tensor(pos_scaler.mean,   device=device, dtype=torch.float32)
        pos_std    = torch.tensor(pos_scaler.std,    device=device, dtype=torch.float32)

        fracs = torch.linspace(0, 1, T, device=device)
        fracs = fracs.unsqueeze(1).expand(T, B).unsqueeze(2)
        causal_mask = self._causal_mask(T, device)

        # Pass 1: teacher-forced forward, no grad — produces model's one-step predictions
        with torch.no_grad():
            sin_b_1, cos_b_1 = self._bearing_features(target_traj, end)
            dec_input_1 = torch.cat([target_traj, fracs, sin_b_1, cos_b_1], dim=2)
            dec_input_1 = self.waypoint_proj(dec_input_1)
            dec_input_1 = self.pos_enc(dec_input_1)
            out_1 = self.decoder(dec_input_1, memory, tgt_mask=causal_mask)
            pred_deltas_1 = self.delta_head(out_1[:-1])                    # (T-1, B, 2)

            # Reconstruct model's predicted next positions (in normalized pos space)
            target_raw = target_traj * pos_std + pos_mean                  # (T, B, 2)
            pred_deltas_1_raw = pred_deltas_1 * delta_std + delta_mean     # (T-1, B, 2)
            model_next_raw = target_raw[:-1] + pred_deltas_1_raw           # (T-1, B, 2)
            model_next_norm = (model_next_raw - pos_mean) / pos_std        # (T-1, B, 2)

        # Build mixed trajectory: step 0 always GT; later steps mix per-sample
        use_model = (torch.rand(T - 1, B, device=device) < ss_prob).unsqueeze(2)
        mixed_tail = torch.where(use_model, model_next_norm, target_traj[1:])
        mixed_traj = torch.cat([target_traj[:1], mixed_tail], dim=0)       # (T, B, 2)

        # Pass 2: teacher-forced forward with grad, over the mixed trajectory
        sin_b_2, cos_b_2 = self._bearing_features(mixed_traj, end)
        dec_input_2 = torch.cat([mixed_traj, fracs, sin_b_2, cos_b_2], dim=2)
        dec_input_2 = self.waypoint_proj(dec_input_2)
        dec_input_2 = self.pos_enc(dec_input_2)
        out_2 = self.decoder(dec_input_2, memory, tgt_mask=causal_mask)
        pred_deltas_2 = self.delta_head(out_2[:-1])                        # (T-1, B, 2)

        return pred_deltas_2, mixed_traj

    @torch.no_grad()
    def generate(
        self,
        start: torch.Tensor,        # (B, 2) normalized
        end: torch.Tensor,           # (B, 2) normalized
        vessel_type: torch.Tensor,   # (B,)
        n_steps: int = 128,
        pos_scaler=None,
        delta_scaler=None,
    ) -> torch.Tensor:
        """Autoregressive trajectory generation.

        Args:
            start, end: normalized start/end positions
            vessel_type: vessel type indices
            n_steps: number of trajectory points to generate
            pos_scaler: if provided, normalizes predicted positions for next step
            delta_scaler: if provided, inverse-transforms predicted deltas

        Returns:
            trajectory: (n_steps, B, 2) generated trajectory in normalized space
        """
        B = start.shape[0]
        device = start.device

        memory = self.encode_condition(start, end, vessel_type)  # (3, B, d_model)

        # Start with the start position
        generated = [start.unsqueeze(0)]  # list of (1, B, 2)

        for t in range(n_steps - 1):
            T_cur = len(generated)
            traj_so_far = torch.cat(generated, dim=0)          # (T_cur, B, 2)

            # Progress fractions
            fracs = torch.linspace(0, t / max(n_steps - 1, 1), T_cur, device=device)
            fracs = fracs.unsqueeze(1).expand(T_cur, B).unsqueeze(2)

            sin_b, cos_b = self._bearing_features(traj_so_far, end)
            dec_input = torch.cat([traj_so_far, fracs, sin_b, cos_b], dim=2)  # (T_cur, B, 5)
            dec_input = self.waypoint_proj(dec_input)
            dec_input = self.pos_enc(dec_input)

            causal_mask = self._causal_mask(T_cur, device)
            out = self.decoder(dec_input, memory, tgt_mask=causal_mask)

            # Take last output → predict delta
            pred_delta_norm = self.delta_head(out[-1])           # (B, 2)

            # Compute next position
            if delta_scaler is not None and pos_scaler is not None:
                # Work in raw space for position accumulation
                cur_pos_norm = traj_so_far[-1]                   # (B, 2)
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
                # Approximate: accumulate in normalized space
                next_pos_norm = traj_so_far[-1] + pred_delta_norm

            generated.append(next_pos_norm.unsqueeze(0))

        trajectory = torch.cat(generated, dim=0)  # (n_steps, B, 2)

        # Linear endpoint correction: distribute error evenly across points
        if pos_scaler is not None:
            traj_np = trajectory.cpu().numpy()                    # (T, B, 2)
            end_np = end.cpu().numpy()                            # (B, 2)
            # Convert to raw space
            T_len, B_len, _ = traj_np.shape
            traj_raw = pos_scaler.inverse(traj_np.reshape(-1, 2)).reshape(T_len, B_len, 2)
            end_raw = pos_scaler.inverse(end_np)
            # Compute error at endpoint
            error = end_raw - traj_raw[-1]                        # (B, 2)
            # Linear ramp: 0 at start, full correction at end
            fracs = np.linspace(0, 1, T_len).reshape(-1, 1, 1)   # (T, 1, 1)
            traj_corrected = traj_raw + fracs * error[np.newaxis]
            # Back to normalized space
            traj_norm = pos_scaler.transform(traj_corrected.reshape(-1, 2)).reshape(T_len, B_len, 2)
            trajectory = torch.tensor(traj_norm, device=device, dtype=torch.float32)

        return trajectory
