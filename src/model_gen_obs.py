"""
ObstacleTrajectoryGenerator: obstacle-conditioned trajectory generation.

Extends the TrajectoryGenerator architecture with obstacle encoder tokens.
Obstacles are circular exclusion zones encoded as (center_lon, center_lat, radius).
They are projected into d_model space and concatenated with the 3 base conditioning
tokens (start, end, vessel_type), forming the encoder memory.

The decoder cross-attends to both route conditioning and obstacle tokens,
learning to generate trajectories that avoid obstacles.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from src.model import PositionalEncoding


class ObstacleTrajectoryGenerator(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_vessel_types: int = 28,
        vessel_type_embed_dim: int = 8,
        max_seq_len: int = 512,
        max_obstacles: int = 3,
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.max_obstacles = max_obstacles

        # ── Encoder: conditioning tokens ─────────────────────────────────────
        self.start_proj = nn.Linear(2, d_model)
        self.end_proj   = nn.Linear(2, d_model)

        # Vessel type
        self.vessel_type_emb  = nn.Embedding(num_vessel_types, vessel_type_embed_dim)
        self.vessel_type_proj = nn.Linear(vessel_type_embed_dim, d_model)

        # Token-type embeddings: 0=start, 1=end, 2=vessel_type, 3=obstacle
        self.token_type_emb = nn.Embedding(4, d_model)

        # Obstacle projection: (center_lon, center_lat, radius) → d_model
        self.obstacle_proj = nn.Linear(3, d_model)

        # Transformer encoder for conditioning token interaction
        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=False,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # ── Decoder: waypoint generation ─────────────────────────────────────
        # Input per step: [lon_norm, lat_norm, progress, sin_bearing, cos_bearing]
        self.waypoint_proj = nn.Linear(5, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Output head
        self.delta_head = nn.Linear(d_model, 2)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.start_proj, self.end_proj, self.waypoint_proj,
                     self.obstacle_proj]:
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
        start: torch.Tensor,           # (B, 2)
        end: torch.Tensor,             # (B, 2)
        vessel_type: torch.Tensor,     # (B,)
        obstacles: torch.Tensor = None,       # (B, K, 3) or None
        obstacle_mask: torch.Tensor = None,   # (B, K) bool, True=padding; or None
    ):
        """Encode conditioning inputs into memory tokens.

        Returns:
            memory: (3+K, B, d_model) or (3, B, d_model) if no obstacles
            memory_key_padding_mask: (B, 3+K) or None
        """
        B = start.shape[0]
        device = start.device

        tok_start = self.start_proj(start)                                  # (B, d)
        tok_end   = self.end_proj(end)                                      # (B, d)
        tok_vtype = self.vessel_type_proj(self.vessel_type_emb(vessel_type)) # (B, d)

        # Token-type embeddings for the 3 base tokens
        type_ids_base = torch.arange(3, device=device)
        type_embs_base = self.token_type_emb(type_ids_base)                 # (3, d)

        base_tokens = torch.stack([tok_start, tok_end, tok_vtype], dim=0)   # (3, B, d)
        base_tokens = base_tokens + type_embs_base.unsqueeze(1)

        if obstacles is not None and obstacle_mask is not None:
            K = obstacles.shape[1]

            # Project obstacles
            obs_tokens = self.obstacle_proj(obstacles)                      # (B, K, d)
            obs_tokens = obs_tokens.permute(1, 0, 2)                       # (K, B, d)

            # Add obstacle token-type embedding (type index 3)
            obs_type_emb = self.token_type_emb(
                torch.tensor(3, device=device)).unsqueeze(0).unsqueeze(1)   # (1, 1, d)
            obs_tokens = obs_tokens + obs_type_emb

            # Concatenate: (3+K, B, d)
            memory = torch.cat([base_tokens, obs_tokens], dim=0)

            # Build padding mask: (B, 3+K), True = ignore
            base_mask = torch.zeros(B, 3, device=device, dtype=torch.bool)
            memory_key_padding_mask = torch.cat(
                [base_mask, obstacle_mask], dim=1)                          # (B, 3+K)
        else:
            memory = base_tokens
            memory_key_padding_mask = None

        if self.num_encoder_layers > 0:
            memory = self.encoder(
                memory,
                src_key_padding_mask=memory_key_padding_mask,
            )

        return memory, memory_key_padding_mask

    def _bearing_features(self, traj, end):
        """Compute sin/cos bearing from each position to endpoint.

        Args:
            traj: (T, B, 2) positions
            end: (B, 2) endpoint

        Returns:
            sin_b, cos_b: each (T, B, 1)
        """
        diff = end.unsqueeze(0) - traj
        angle = torch.atan2(diff[..., 1], diff[..., 0])
        return torch.sin(angle).unsqueeze(2), torch.cos(angle).unsqueeze(2)

    def forward(
        self,
        start: torch.Tensor,           # (B, 2)
        end: torch.Tensor,             # (B, 2)
        vessel_type: torch.Tensor,     # (B,)
        target_traj: torch.Tensor,     # (T, B, 2)
        obstacles: torch.Tensor = None,       # (B, K, 3)
        obstacle_mask: torch.Tensor = None,   # (B, K)
        ss_prob: float = 0.0,
        delta_scaler=None,
        pos_scaler=None,
    ):
        """Teacher-forcing forward pass with optional two-pass scheduled sampling.

        See `src/model_gen.py` for the two-pass SS rationale.

        Returns:
            pred_deltas: (T-1, B, 2)
            input_traj:  (T, B, 2) — target_traj for TF, mixed_traj for SS
        """
        T, B, _ = target_traj.shape
        device = target_traj.device

        memory, mem_mask = self.encode_condition(
            start, end, vessel_type, obstacles, obstacle_mask)

        if ss_prob <= 0 or delta_scaler is None or pos_scaler is None:
            # Pure teacher forcing
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
                memory_key_padding_mask=mem_mask,
            )
            return self.delta_head(out[:-1]), target_traj

        # Two-pass scheduled sampling
        delta_mean = torch.tensor(delta_scaler.mean, device=device, dtype=torch.float32)
        delta_std  = torch.tensor(delta_scaler.std,  device=device, dtype=torch.float32)
        pos_mean   = torch.tensor(pos_scaler.mean,   device=device, dtype=torch.float32)
        pos_std    = torch.tensor(pos_scaler.std,    device=device, dtype=torch.float32)

        fracs = torch.linspace(0, 1, T, device=device)
        fracs = fracs.unsqueeze(1).expand(T, B).unsqueeze(2)
        causal_mask = self._causal_mask(T, device)

        # Pass 1: TF under GT, no grad — collect model's one-step predictions
        with torch.no_grad():
            sin_b_1, cos_b_1 = self._bearing_features(target_traj, end)
            dec_input_1 = torch.cat([target_traj, fracs, sin_b_1, cos_b_1], dim=2)
            dec_input_1 = self.waypoint_proj(dec_input_1)
            dec_input_1 = self.pos_enc(dec_input_1)
            out_1 = self.decoder(
                dec_input_1, memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=mem_mask,
            )
            pred_deltas_1 = self.delta_head(out_1[:-1])                    # (T-1, B, 2)

            target_raw = target_traj * pos_std + pos_mean
            pred_deltas_1_raw = pred_deltas_1 * delta_std + delta_mean
            model_next_raw = target_raw[:-1] + pred_deltas_1_raw           # (T-1, B, 2)
            model_next_norm = (model_next_raw - pos_mean) / pos_std

        # Build mixed trajectory: step 0 always GT; per-sample mix otherwise
        use_model = (torch.rand(T - 1, B, device=device) < ss_prob).unsqueeze(2)
        mixed_tail = torch.where(use_model, model_next_norm, target_traj[1:])
        mixed_traj = torch.cat([target_traj[:1], mixed_tail], dim=0)       # (T, B, 2)

        # Pass 2: TF over mixed input, with grad
        sin_b_2, cos_b_2 = self._bearing_features(mixed_traj, end)
        dec_input_2 = torch.cat([mixed_traj, fracs, sin_b_2, cos_b_2], dim=2)
        dec_input_2 = self.waypoint_proj(dec_input_2)
        dec_input_2 = self.pos_enc(dec_input_2)
        out_2 = self.decoder(
            dec_input_2, memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=mem_mask,
        )
        pred_deltas_2 = self.delta_head(out_2[:-1])
        return pred_deltas_2, mixed_traj

    @torch.no_grad()
    def generate(
        self,
        start: torch.Tensor,           # (B, 2) normalized
        end: torch.Tensor,             # (B, 2) normalized
        vessel_type: torch.Tensor,     # (B,)
        n_steps: int = 128,
        pos_scaler=None,
        delta_scaler=None,
        obstacles: torch.Tensor = None,       # (B, K, 3)
        obstacle_mask: torch.Tensor = None,   # (B, K)
    ) -> torch.Tensor:
        """Autoregressive trajectory generation.

        Returns:
            trajectory: (n_steps, B, 2) in normalized space
        """
        B = start.shape[0]
        device = start.device

        memory, mem_mask = self.encode_condition(
            start, end, vessel_type, obstacles, obstacle_mask)

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
                memory_key_padding_mask=mem_mask,
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

        # Linear endpoint correction
        if pos_scaler is not None:
            traj_np = trajectory.cpu().numpy()
            end_np = end.cpu().numpy()
            T_len, B_len, _ = traj_np.shape
            traj_raw = pos_scaler.inverse(traj_np.reshape(-1, 2)).reshape(T_len, B_len, 2)
            end_raw = pos_scaler.inverse(end_np)
            error = end_raw - traj_raw[-1]
            fracs = np.linspace(0, 1, T_len).reshape(-1, 1, 1)
            traj_corrected = traj_raw + fracs * error[np.newaxis]
            traj_norm = pos_scaler.transform(
                traj_corrected.reshape(-1, 2)).reshape(T_len, B_len, 2)
            trajectory = torch.tensor(traj_norm, device=device, dtype=torch.float32)

        return trajectory
