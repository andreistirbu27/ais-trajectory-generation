"""
v10 trajectory generator — per-timestep retrieval + windowed past context.

Built to answer the v9 failure mode: v9's mean-pool route encoder compresses
each (T, 2) retrieved route into a single d_model vector, so per-timestep
shape is gone before the decoder sees it. The decoder ends up cross-attending
to K "roughly where this route went" summaries and can't read "this neighbor
is at (lat, lon) at step t." At epoch 17 v9 ADE (7980 m) was already 45%
behind zero-training retrieval-top-1 (5489 m).

v10 changes vs v9 (RetrievalTrajectoryGenerator):
  1. Route encoder produces K * T_retr tokens (per-timestep), NOT K tokens.
     Each retrieved route is subsampled to T_retr ≈ 32 points; each point
     becomes one encoder token with a route-index embedding + along-route
     step embedding. The decoder can now cross-attend to "neighbor k at
     step t" directly.
  2. Optional windowed past context (k_past). The plan calls for a
     "next-step head predicting (Δlon, Δlat) from a short past-context
     window of K_past ≈ 16–32 recent waypoints." Implemented as a windowed
     causal mask: at step t, the decoder attends only to positions
     max(0, t - k_past)..t. k_past=None keeps the fully causal v5/v9 mask.
  3. Land awareness lives outside the model class:
        - Training loss adds λ_land · mean(ReLU(sdf_km - threshold_km))²
          (src/land_mask.py::sample_km_torch is differentiable via grid_sample).
        - Inference hard-projects any waypoint with sdf_km > hard_threshold_km
          to the nearest water cell (LandMask.project_to_water).
     generate() accepts optional (land_mask, hard_threshold_km) kwargs to
     enable the hard projection in-line.

Retrieved route input shape (B, K, T_orig, 2). T_orig is the dataset's native
resolution (128). The encoder uniformly subsamples to T_retr internally.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.model_gen import TrajectoryGenerator


# ─── Per-timestep route encoder ─────────────────────────────────────────────

class PerStepRouteEncoder(nn.Module):
    """Encode each retrieved route as T_retr tokens (no pooling).

    Each token carries:
      - projected position (lon, lat normalized)
      - learned (route_idx, step) positional bias — stored as a single
        (max_retrieved, t_retr, d_model) parameter. Logically equivalent
        to the factored sum of a route-index embedding and a step
        embedding, but stored as one dense bias so the forward is a
        single broadcast add. The factored form triggered a torch.compile
        MPS bug (nested modulo/gather lowering to invalid Metal with
        undeclared `Where(...)` identifiers); dense bias avoids it.

    Output shape: (B, K * T_retr, d_model).
    """

    def __init__(self, d_model: int, max_retrieved: int = 5, t_retr: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_retrieved = max_retrieved
        self.t_retr = t_retr

        self.pos_proj = nn.Linear(2, d_model)
        self.route_pos_bias = nn.Parameter(
            torch.zeros(max_retrieved, t_retr, d_model))

        nn.init.xavier_uniform_(self.pos_proj.weight)
        nn.init.zeros_(self.pos_proj.bias)
        nn.init.normal_(self.route_pos_bias, std=0.02)

    def forward(self, routes_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:   routes_norm (B, K, T_orig, 2) normalized [lon, lat]
        Returns: (B, K * T_retr, d_model)
        """
        B, K, T_orig, _ = routes_norm.shape
        assert K <= self.max_retrieved, f"K={K} > max_retrieved={self.max_retrieved}"

        if T_orig != self.t_retr:
            idx = torch.linspace(0, T_orig - 1, self.t_retr,
                                 device=routes_norm.device).long()
            routes_norm = routes_norm[:, :, idx]            # (B, K, T_retr, 2)

        tok = self.pos_proj(routes_norm)                    # (B, K, T_retr, d)
        tok = tok + self.route_pos_bias[:K].unsqueeze(0)    # broadcast add

        return tok.reshape(B, K * self.t_retr, self.d_model)


# ─── v10 model ──────────────────────────────────────────────────────────────

class TrajectoryGeneratorV10(TrajectoryGenerator):
    """v5 architecture + per-timestep retrieval + optional windowed past context.

    Decoder is unchanged (5-dim waypoint input with bearing features). Only
    the encoder memory shape and the causal mask change.
    """

    def __init__(
        self,
        max_retrieved: int = 5,
        t_retr: int = 32,
        n_points: int = 128,
        k_past: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_retrieved = max_retrieved
        self.t_retr = t_retr
        self.k_past = k_past

        d_model = kwargs.get("d_model", 128)

        # Extend token-type embedding: 0=start, 1=end, 2=vtype, 3=retrieved
        self.token_type_emb = nn.Embedding(4, d_model)
        nn.init.normal_(self.token_type_emb.weight, std=0.02)

        self.route_encoder = PerStepRouteEncoder(d_model, max_retrieved, t_retr)

    # ── encoder memory ──────────────────────────────────────────────────────

    def encode_condition_retrieval(
        self,
        start: torch.Tensor,            # (B, 2) normalized
        end: torch.Tensor,              # (B, 2) normalized
        vessel_type: torch.Tensor,      # (B,) int64
        retrieved_norm: torch.Tensor,   # (B, K, T_orig, 2) normalized
        retrieval_mask: torch.Tensor,   # (B, K) bool — True = padding
    ):
        """Build encoder memory = 3 base tokens + K * T_retr retrieval tokens.

        Returns:
            memory:      (3 + K * T_retr, B, d_model)
            pad_mask_bk: (B, 3 + K * T_retr) — True for padded slots
        """
        B = start.shape[0]
        K = retrieved_norm.shape[1]
        device = start.device

        tok_start = self.start_proj(start)
        tok_end   = self.end_proj(end)
        tok_vtype = self.vessel_type_proj(self.vessel_type_emb(vessel_type))

        base = torch.stack([tok_start, tok_end, tok_vtype], dim=0)   # (3, B, d)

        retr = self.route_encoder(retrieved_norm)                    # (B, K*T_retr, d)
        retr = retr.permute(1, 0, 2)                                 # (K*T_retr, B, d)

        # Add token-type embeddings to base and retr SEPARATELY before concat.
        # Concatenating first and then gathering from cat([arange(3), full(3)])
        # fuses into an Inductor-MPS kernel that lowers to invalid Metal
        # (`Where(...)` with capital W, undeclared identifier).
        type_weight = self.token_type_emb.weight                     # (4, d)
        base = base + type_weight[:3].unsqueeze(1)                   # (3, 1, d)
        retr = retr + type_weight[3].view(1, 1, -1)                  # (1, 1, d)

        memory = torch.cat([base, retr], dim=0)                      # (3+K*T_retr, B, d)

        n_retr = K * self.t_retr

        # Expand per-route padding mask across T_retr steps
        base_unpadded = torch.zeros(B, 3, device=device, dtype=torch.bool)
        retr_pad = retrieval_mask.unsqueeze(2).expand(B, K, self.t_retr)
        retr_pad = retr_pad.reshape(B, n_retr)
        pad_mask_bk = torch.cat([base_unpadded, retr_pad], dim=1)

        if self.num_encoder_layers > 0:
            memory = self.encoder(memory, src_key_padding_mask=pad_mask_bk)

        return memory, pad_mask_bk

    # ── windowed causal mask ────────────────────────────────────────────────

    def _windowed_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Causal mask restricted to the last k_past positions.

        At step t, attend only to positions max(0, t - k_past)..t.
        If k_past is None, falls back to full causal.
        """
        if self.k_past is None:
            return self._causal_mask(sz, device)

        # Start with a fully-allowed (0.0) mask, then block future and too-old
        i = torch.arange(sz, device=device).unsqueeze(1)       # (sz, 1)
        j = torch.arange(sz, device=device).unsqueeze(0)       # (1, sz)
        # Future: j > i   → block
        # Too old: j < i - k_past → block
        block = (j > i) | (j < i - self.k_past)
        mask = torch.zeros(sz, sz, device=device)
        mask = mask.masked_fill(block, float("-inf"))
        return mask

    # ── forward (teacher forcing) ───────────────────────────────────────────

    def forward(
        self,
        start,
        end,
        vessel_type,
        target_traj,                    # (T, B, 2)
        retrieved_norm,                 # (B, K, T_orig, 2)
        retrieval_mask,                 # (B, K)
    ):
        T, B, _ = target_traj.shape
        device = target_traj.device

        memory, pad_mask = self.encode_condition_retrieval(
            start, end, vessel_type, retrieved_norm, retrieval_mask)

        fracs = torch.linspace(0, 1, T, device=device)
        fracs = fracs.unsqueeze(1).expand(T, B).unsqueeze(2)
        sin_b, cos_b = self._bearing_features(target_traj, end)

        dec_input = torch.cat([target_traj, fracs, sin_b, cos_b], dim=2)
        dec_input = self.waypoint_proj(dec_input)
        dec_input = self.pos_enc(dec_input)

        causal_mask = self._windowed_causal_mask(T, device)
        out = self.decoder(
            dec_input, memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=pad_mask,
        )
        pred_deltas = self.delta_head(out[:-1])              # (T-1, B, 2)
        return pred_deltas, target_traj

    # ── generate (autoregressive, optional hard land projection) ────────────

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
        land_mask=None,                 # optional LandMask
        hard_threshold_km: float = 10.0,
        endpoint_correction: bool = True,
    ) -> torch.Tensor:
        """Autoregressive rollout. If `land_mask` is given, any waypoint with
        sdf_km > hard_threshold_km is projected to the nearest water cell at
        the time it is produced (before being fed back into the decoder).
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

            causal_mask = self._windowed_causal_mask(T_cur, device)
            out = self.decoder(
                dec_input, memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=pad_mask,
            )
            pred_delta_norm = self.delta_head(out[-1])       # (B, 2)

            if delta_scaler is not None and pos_scaler is not None:
                cur_pos_norm = traj_so_far[-1]
                cur_pos_raw = pos_scaler.inverse(cur_pos_norm.cpu().numpy())
                delta_raw = delta_scaler.inverse(pred_delta_norm.cpu().numpy())
                next_pos_raw = cur_pos_raw + delta_raw       # (B, 2)

                if land_mask is not None:
                    lon_arr = next_pos_raw[:, 0]
                    lat_arr = next_pos_raw[:, 1]
                    sdf = land_mask.sample_km_np(lon_arr, lat_arr)
                    on_land = sdf > hard_threshold_km
                    if on_land.any():
                        lon_w, lat_w = land_mask.project_to_water(
                            lon_arr[on_land], lat_arr[on_land])
                        next_pos_raw[on_land, 0] = lon_w
                        next_pos_raw[on_land, 1] = lat_w

                next_pos_norm = torch.tensor(
                    pos_scaler.transform(next_pos_raw),
                    device=device, dtype=torch.float32)
            else:
                next_pos_norm = traj_so_far[-1] + pred_delta_norm

            generated.append(next_pos_norm.unsqueeze(0))

        trajectory = torch.cat(generated, dim=0)

        if endpoint_correction and pos_scaler is not None:
            traj_np = trajectory.cpu().numpy()
            end_np = end.cpu().numpy()
            T_len, B_len, _ = traj_np.shape
            traj_raw = pos_scaler.inverse(traj_np.reshape(-1, 2)).reshape(T_len, B_len, 2)
            end_raw = pos_scaler.inverse(end_np)
            error = end_raw - traj_raw[-1]
            fracs_np = np.linspace(0, 1, T_len).reshape(-1, 1, 1)
            traj_corrected = traj_raw + fracs_np * error[np.newaxis]

            if land_mask is not None:
                flat = traj_corrected.reshape(-1, 2)
                sdf_flat = land_mask.sample_km_np(flat[:, 0], flat[:, 1])
                on_land = sdf_flat > hard_threshold_km
                if on_land.any():
                    lon_w, lat_w = land_mask.project_to_water(
                        flat[on_land, 0], flat[on_land, 1])
                    flat[on_land, 0] = lon_w
                    flat[on_land, 1] = lat_w
                traj_corrected = flat.reshape(T_len, B_len, 2)

            traj_norm = pos_scaler.transform(traj_corrected.reshape(-1, 2))
            traj_norm = traj_norm.reshape(T_len, B_len, 2)
            trajectory = torch.tensor(traj_norm, device=device, dtype=torch.float32)

        return trajectory
