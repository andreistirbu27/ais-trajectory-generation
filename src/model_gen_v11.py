"""
v11 trajectory generator — pointer head over water-valid candidate pool.

v10's regression head can't structurally guarantee water-validity; v11
replaces it with a pointer (masked softmax) over a set of K candidate
waypoints generated and pre-filtered per step. Every waypoint and every
segment is in water by construction of the candidate pool.

What is reused from v10 verbatim (via subclass):
  - PerStepRouteEncoder (retrieval tokens)
  - encode_condition_retrieval (encoder memory + padding mask)
  - _windowed_causal_mask (for v11 use k_past=None = full causal)
  - waypoint_proj / pos_enc / decoder layers

What is new:
  - candidate_proj: Linear(4, d_model) projects each candidate's
    (Δlon_norm, Δlat_norm, sdf_km_norm, storm_dist_norm) → d_model.
  - Pointer head: logits = candidates · h_t / sqrt(d_model).
  - Offset head: auxiliary Linear(d_model, 2) refines the selected
    candidate in normalized delta space.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.model_gen_v10 import TrajectoryGeneratorV10


class TrajectoryGeneratorV11(TrajectoryGeneratorV10):
    """Pointer variant of v10. Encoder = v10. Decoder core = v10. Head = pointer."""

    CAND_FEAT_DIM = 4   # [Δlon_norm, Δlat_norm, sdf_km / 50, storm_km / 100]

    def __init__(
        self,
        sdf_scale_km: float = 50.0,
        storm_scale_km: float = 100.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sdf_scale_km = sdf_scale_km
        self.storm_scale_km = storm_scale_km

        d_model = kwargs.get("d_model", 128)
        self.candidate_proj = nn.Linear(self.CAND_FEAT_DIM, d_model)
        self.offset_head = nn.Linear(d_model, 2)
        self._logit_scale = 1.0 / math.sqrt(d_model)

        nn.init.xavier_uniform_(self.candidate_proj.weight)
        nn.init.zeros_(self.candidate_proj.bias)
        nn.init.uniform_(self.offset_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.offset_head.bias)

    # ── forward (teacher forcing over GT trajectory) ───────────────────────

    def forward(
        self,
        start,                     # (B, 2) normalized
        end,                        # (B, 2) normalized
        vessel_type,                # (B,)
        target_traj,                # (T, B, 2) normalized positions
        retrieved_norm,             # (B, K_retr, T_orig, 2)
        retrieval_mask,             # (B, K_retr)
        candidates_norm,            # (T-1, B, K, 2)  normalized absolute positions
        candidate_feats,            # (T-1, B, K, 2)  [sdf_km, storm_km] raw
        candidate_mask,             # (T-1, B, K)     bool, True = valid
    ):
        """Returns:
          logits:  (T-1, B, K)
          offset:  (T-1, B, 2) — auxiliary delta in normalized pos space
        """
        T, B, _ = target_traj.shape
        device = target_traj.device

        memory, pad_mask = self.encode_condition_retrieval(
            start, end, vessel_type, retrieved_norm, retrieval_mask)

        # Decoder input (5-dim, same as v5/v10)
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
        )                                                # (T, B, d_model)
        h = out[:-1]                                     # (T-1, B, d_model)

        # Candidate features: (T-1, B, K, 4)
        feats = self._build_cand_feats(candidates_norm, candidate_feats, target_traj)
        cand_emb = self.candidate_proj(feats)            # (T-1, B, K, d_model)

        # Pointer logits: dot(cand_emb, h) / sqrt(d_model)
        logits = (cand_emb * h.unsqueeze(2)).sum(dim=-1) * self._logit_scale

        # Mask invalid candidates
        logits = logits.masked_fill(~candidate_mask, float("-inf"))

        offset = self.offset_head(h)                     # (T-1, B, 2)
        return logits, offset

    # ── candidate feature construction ─────────────────────────────────────

    def _build_cand_feats(
        self,
        candidates_norm: torch.Tensor,    # (T-1, B, K, 2)
        candidate_feats: torch.Tensor,    # (T-1, B, K, 2) [sdf_km, storm_km]
        cur_traj: torch.Tensor,           # (T, B, 2) normalized positions
    ) -> torch.Tensor:
        """Assemble (Δlon_norm, Δlat_norm, sdf_norm, storm_norm)."""
        cur = cur_traj[:-1].unsqueeze(2)                 # (T-1, B, 1, 2)
        delta_norm = candidates_norm - cur               # (T-1, B, K, 2)

        sdf = candidate_feats[..., 0:1] / self.sdf_scale_km
        storm = candidate_feats[..., 1:2] / self.storm_scale_km
        return torch.cat([delta_norm, sdf, storm], dim=-1)

    # ── Inference ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_step(
        self,
        memory,
        pad_mask,
        traj_so_far: torch.Tensor,    # (T_cur, B, 2) normalized
        end: torch.Tensor,             # (B, 2) normalized
        n_steps: int,
        t: int,
    ) -> torch.Tensor:
        """Run one decoder step; return h_t of shape (B, d_model)."""
        T_cur = traj_so_far.shape[0]
        device = traj_so_far.device
        B = traj_so_far.shape[1]

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
        return out[-1]

    @torch.no_grad()
    def score_candidates(
        self,
        h: torch.Tensor,              # (B, d_model)
        cand_feats: torch.Tensor,     # (B, K, 4)
    ) -> torch.Tensor:
        """Pointer scores (B, K)."""
        cand_emb = self.candidate_proj(cand_feats)       # (B, K, d_model)
        return (cand_emb * h.unsqueeze(1)).sum(dim=-1) * self._logit_scale

    # ── Full autoregressive rollout with candidate generation ──────────────

    @torch.no_grad()
    def generate(
        self,
        start: torch.Tensor,          # (B, 2) normalized
        end: torch.Tensor,             # (B, 2) normalized
        vessel_type: torch.Tensor,     # (B,)
        retrieved_norm: torch.Tensor,  # (B, K_retr, T_orig, 2)
        retrieval_mask: torch.Tensor,  # (B, K_retr)
        land_mask,
        pos_scaler,
        n_steps: int = 128,
        K: int = 32,
        cone_deg: float = 60.0,
        d_min_km: float = 3.0,
        d_max_km: float = 25.0,
        tau_km: float = 2.0,
        n_segment_samples: int = 5,
        use_segment_check: bool = True,
        use_storm_check: bool = False,
        storms=None,
        use_offset: bool = False,
        deterministic: bool = True,
        force_end_snap: bool = True,
    ):
        """Water-by-construction autoregressive rollout.

        Returns:
          trajectory  (n_steps, B, 2) normalized positions
          info dict with per-step fallback flags and endpoint-snap distances.
        """
        from src.candidates import (generate_and_filter, segment_water_check,
                                     point_water_check)

        B = start.shape[0]
        device = start.device

        # Build encoder memory once (invariant across steps)
        memory, pad_mask = self.encode_condition_retrieval(
            start, end, vessel_type, retrieved_norm, retrieval_mask)

        end_raw = pos_scaler.inverse(end.cpu().numpy())              # (B, 2)
        generated = [start.unsqueeze(0)]

        fallback_flags = np.zeros((n_steps - 1, B), dtype=bool)
        endpoint_snap_km = np.zeros(B, dtype=np.float32)
        infeasible_mask = np.zeros(B, dtype=bool)

        for t in range(n_steps - 1):
            traj_so_far = torch.cat(generated, dim=0)                # (T_cur, B, 2)
            h = self.generate_step(memory, pad_mask, traj_so_far,
                                    end, n_steps, t)                  # (B, d_model)

            p_t_norm_np = traj_so_far[-1].cpu().numpy()
            p_t_raw = pos_scaler.inverse(p_t_norm_np)                # (B, 2)

            is_last = (t == n_steps - 2)
            if is_last and force_end_snap:
                # Force last step to land on `end` if the segment is water;
                # otherwise fall through to candidate selection and snap later.
                seg_ok = segment_water_check(
                    p_t_raw, end_raw, land_mask,
                    n_samples=n_segment_samples, tau_km=tau_km)       # (B,)
                next_raw = np.where(seg_ok[:, None], end_raw, p_t_raw).astype(np.float32)
                # For samples whose final segment IS water: snap to end.
                # For samples that aren't: produce a candidate step and snap later.
                need_cand = ~seg_ok
                if need_cand.any():
                    cands_sub, valid_sub, fb_sub = generate_and_filter(
                        p_t_raw[need_cand], end_raw[need_cand], land_mask,
                        K=K, cone_deg=cone_deg,
                        d_min_km=d_min_km, d_max_km=d_max_km,
                        tau_km=tau_km, n_segment_samples=n_segment_samples,
                        storms=storms,
                        use_segment_check=use_segment_check,
                        use_storm_check=use_storm_check,
                    )
                    cand_feats_sub = self._assemble_cand_feats(
                        p_t_norm_np[need_cand], cands_sub, land_mask,
                        pos_scaler, storms)
                    logits = self.score_candidates(
                        h[torch.from_numpy(need_cand).to(device)],
                        torch.from_numpy(cand_feats_sub).to(device=device,
                                                             dtype=torch.float32),
                    )
                    logits = logits.masked_fill(
                        ~torch.from_numpy(valid_sub).to(device), float("-inf"))
                    if deterministic:
                        idx = logits.argmax(dim=-1).cpu().numpy()
                    else:
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                        idx = np.array([np.random.choice(K, p=p) for p in probs])
                    # But ultimately still snap the endpoint_snap_km:
                    sub_indices = np.nonzero(need_cand)[0]
                    for j, b in enumerate(sub_indices):
                        picked = cands_sub[j, idx[j]]
                        next_raw[b] = picked
                        fallback_flags[t, b] = fb_sub[j]
                        # Distance from picked to true end
                        d = self._haversine_km(picked, end_raw[b])
                        endpoint_snap_km[b] = max(endpoint_snap_km[b], d)
                        if not valid_sub[j].any():
                            infeasible_mask[b] = True
                next_norm = pos_scaler.transform(next_raw)
                generated.append(torch.tensor(
                    next_norm, device=device, dtype=torch.float32
                ).unsqueeze(0))
                continue

            cands_raw, valid, fb = generate_and_filter(
                p_t_raw, end_raw, land_mask,
                K=K, cone_deg=cone_deg, d_min_km=d_min_km, d_max_km=d_max_km,
                tau_km=tau_km, n_segment_samples=n_segment_samples,
                storms=storms,
                use_segment_check=use_segment_check,
                use_storm_check=use_storm_check,
            )
            fallback_flags[t] = fb
            cand_feats_np = self._assemble_cand_feats(
                p_t_norm_np, cands_raw, land_mask, pos_scaler, storms)

            cand_feats_t = torch.from_numpy(cand_feats_np).to(
                device=device, dtype=torch.float32)
            logits = self.score_candidates(h, cand_feats_t)
            valid_t = torch.from_numpy(valid).to(device)
            logits = logits.masked_fill(~valid_t, float("-inf"))

            # Handle rows with no valid candidates: pick slot 0 anyway and flag.
            any_valid = valid.any(axis=1)
            if not any_valid.all():
                dead = ~any_valid
                infeasible_mask[dead] = True
                logits[torch.from_numpy(dead).to(device), 0] = 0.0  # break ties

            if deterministic:
                idx = logits.argmax(dim=-1).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                idx = np.array([np.random.choice(K, p=p) for p in probs])

            picked_raw = cands_raw[np.arange(B), idx]                # (B, 2)
            picked_norm = pos_scaler.transform(picked_raw).astype(np.float32)
            picked_t = torch.from_numpy(picked_norm).to(
                device=device, dtype=torch.float32)

            if use_offset:
                offset = self.offset_head(h)                          # (B, 2)
                candidate_t = picked_t + offset
                cand_raw_t = pos_scaler.inverse(candidate_t.cpu().numpy())
                # Verify offset result is still water + segment-valid
                pt_ok = point_water_check(cand_raw_t, land_mask, tau_km)
                seg_ok = segment_water_check(
                    p_t_raw, cand_raw_t, land_mask,
                    n_samples=n_segment_samples, tau_km=tau_km)
                keep = pt_ok & seg_ok
                # Replace final choice only where offset is water-valid
                final_raw = np.where(keep[:, None], cand_raw_t, picked_raw
                                      ).astype(np.float32)
                final_norm = pos_scaler.transform(final_raw).astype(np.float32)
                generated.append(torch.from_numpy(final_norm).to(
                    device=device, dtype=torch.float32).unsqueeze(0))
            else:
                generated.append(picked_t.unsqueeze(0))

        trajectory = torch.cat(generated, dim=0)                     # (T, B, 2)

        info = {
            "fallback_flags": fallback_flags,                        # (T-1, B)
            "endpoint_snap_km": endpoint_snap_km,                    # (B,)
            "infeasible_mask": infeasible_mask,                      # (B,)
        }
        return trajectory, info

    # ── Helpers for inference ──────────────────────────────────────────────

    def _assemble_cand_feats(
        self,
        p_t_norm_np: np.ndarray,        # (B, 2) normalized
        cands_raw: np.ndarray,           # (B, K, 2) raw degrees
        land_mask,
        pos_scaler,
        storms,
    ) -> np.ndarray:
        """Build (B, K, 4) float32 features in the same scheme as forward()."""
        B, K, _ = cands_raw.shape
        flat = cands_raw.reshape(-1, 2)
        cands_norm_flat = pos_scaler.transform(flat).astype(np.float32)
        cands_norm = cands_norm_flat.reshape(B, K, 2)

        # Δ in normalized space
        delta_norm = cands_norm - p_t_norm_np[:, None, :]           # (B, K, 2)

        sdf_flat = land_mask.sample_km_np(flat[:, 0], flat[:, 1])
        sdf = sdf_flat.reshape(B, K, 1) / self.sdf_scale_km

        storm = np.zeros((B, K, 1), dtype=np.float32)
        if storms:
            storm_dists = np.full(B * K, 1e6, dtype=np.float32)
            for c_lon, c_lat, r_km in storms:
                km_per_deg_lat = 111.0
                cos_lat = np.cos(np.deg2rad(c_lat))
                dlat_km = (flat[:, 1] - c_lat) * km_per_deg_lat
                dlon_km = (flat[:, 0] - c_lon) * km_per_deg_lat * cos_lat
                d_km = np.sqrt(dlat_km ** 2 + dlon_km ** 2) - r_km
                storm_dists = np.minimum(storm_dists, d_km)
            storm = (storm_dists / self.storm_scale_km).reshape(B, K, 1
                                                                   ).astype(np.float32)

        return np.concatenate([delta_norm.astype(np.float32),
                                sdf.astype(np.float32),
                                storm], axis=-1)

    @staticmethod
    def _haversine_km(p0: np.ndarray, p1: np.ndarray) -> float:
        """Distance between two (lon, lat) points in km."""
        R = 6371.0
        lat1 = math.radians(float(p0[1]))
        lat2 = math.radians(float(p1[1]))
        dlat = lat2 - lat1
        dlon = math.radians(float(p1[0] - p0[0]))
        a = (math.sin(dlat / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        return 2 * R * math.asin(math.sqrt(a))
