"""
TrajectoryGenerator: conditional transformer encoder-decoder for trajectory generation.

Given (start, end, ship_type), generates a full trajectory as a sequence of
displacement deltas. The encoder produces 3 conditioning tokens (start, end,
vessel type). The decoder autoregressively generates waypoints.

Training: teacher forcing — ground truth positions fed as decoder input.
Inference: autoregressive — own predictions fed back as decoder input.
"""

import math

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

        # ── Decoder: waypoint generation ─────────────────────────────────────
        # Input per decoder step: [lon_norm, lat_norm, progress_fraction] → 3 dims
        self.waypoint_proj = nn.Linear(3, d_model)
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

        return memory

    def forward(
        self,
        start: torch.Tensor,         # (B, 2)
        end: torch.Tensor,           # (B, 2)
        vessel_type: torch.Tensor,   # (B,)
        target_traj: torch.Tensor,   # (T, B, 2) normalized positions
    ) -> torch.Tensor:
        """Teacher-forcing forward pass.

        The decoder receives ground truth positions at all T steps and
        predicts the delta from each position to the next.

        Args:
            target_traj: (T, B, 2) full ground truth trajectory (normalized)

        Returns:
            pred_deltas: (T-1, B, 2) predicted normalized displacements
                         (delta from step t to step t+1, for t=0..T-2)
        """
        T, B, _ = target_traj.shape
        device = target_traj.device

        # Encode conditioning
        memory = self.encode_condition(start, end, vessel_type)  # (3, B, d_model)

        # Build decoder input: positions 0..T-2 (predict delta to 1..T-1)
        # We feed all T positions but only use outputs 0..T-2 for delta prediction
        fracs = torch.linspace(0, 1, T, device=device)          # (T,)
        fracs = fracs.unsqueeze(1).expand(T, B).unsqueeze(2)     # (T, B, 1)

        dec_input = torch.cat([target_traj, fracs], dim=2)       # (T, B, 3)
        dec_input = self.waypoint_proj(dec_input)                # (T, B, d_model)
        dec_input = self.pos_enc(dec_input)                      # (T, B, d_model)

        # Causal mask: position t can only attend to 0..t
        causal_mask = self._causal_mask(T, device)

        # Decode
        out = self.decoder(dec_input, memory, tgt_mask=causal_mask)  # (T, B, d_model)

        # Predict deltas from positions 0..T-2 → positions 1..T-1
        pred_deltas = self.delta_head(out[:-1])                  # (T-1, B, 2)

        return pred_deltas

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

            dec_input = torch.cat([traj_so_far, fracs], dim=2)  # (T_cur, B, 3)
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

        return torch.cat(generated, dim=0)  # (n_steps, B, 2)
