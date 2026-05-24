"""
diffusion.py — EDM noise schedule + DDIM sampler for v13 (TrajDiff).

Conventions follow Karras et al. 2022 "Elucidating the Design Space of
Diffusion-Based Generative Models" (EDM):

  - Continuous noise level sigma in [sigma_min, sigma_max].
  - Forward process: x_t = x_0 + sigma_t * eps,  eps ~ N(0, I).
  - Denoiser: D(x; sigma, c) = (x - sigma * eps_theta(x, sigma, c)).
  - Training loss: weighted MSE between D(x; sigma, c) and x_0.

This module is paper-track (do not modify for school-report work).
Used by `src/model_gen_v13.py` (the DiT backbone) and
`scripts/paper/train_gen_v13.py` (the training loop, to be written).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Noise schedule
# -----------------------------------------------------------------------------
@dataclass
class EDMSchedule:
    """EDM-style continuous noise schedule.

    Defaults match Karras et al. 2022 Table 1 row "EDM" for natural images;
    for trajectory data we expect sigma_max to be smaller (trajectories live in
    normalized [-1, 1]-ish space, not pixel space). Override at construction.
    """
    sigma_min: float = 0.002
    sigma_max: float = 5.0
    sigma_data: float = 0.5     # std of clean data after normalization
    rho: float = 7.0            # exponent for the time -> sigma schedule

    def t_to_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Map t in [0, 1] to sigma. t=0 -> sigma_max, t=1 -> sigma_min."""
        s_min, s_max, rho = self.sigma_min, self.sigma_max, self.rho
        return (s_max ** (1 / rho) + t * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho

    def sample_sigma_train(
        self, n: int, device, dtype=torch.float32, mean: float = -1.2, std: float = 1.2,
    ) -> torch.Tensor:
        """Sample a training sigma per example from EDM's log-normal distribution.

        Defaults (mean=-1.2, std=1.2) follow Karras et al. for images; consider
        retuning for trajectory data after first training pass.
        """
        log_sigma = torch.randn(n, device=device, dtype=dtype) * std + mean
        return log_sigma.exp().clamp(self.sigma_min, self.sigma_max)

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """EDM loss weight w(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2."""
        s_d = self.sigma_data
        return (sigma ** 2 + s_d ** 2) / (sigma * s_d) ** 2


# -----------------------------------------------------------------------------
# Preconditioning (EDM eq. 7)
# -----------------------------------------------------------------------------
def edm_preconditioning(sigma: torch.Tensor, sigma_data: float):
    """Returns (c_skip, c_out, c_in, c_noise) for EDM-style preconditioning.

    The trained network F_theta is mapped to a denoiser via
        D(x; sigma, c) = c_skip * x + c_out * F_theta(c_in * x, c_noise, c).
    """
    sigma_sq = sigma ** 2
    sd_sq = sigma_data ** 2
    c_skip = sd_sq / (sigma_sq + sd_sq)
    c_out  = sigma * sigma_data / (sigma_sq + sd_sq).sqrt()
    c_in   = 1.0 / (sigma_sq + sd_sq).sqrt()
    c_noise = 0.25 * sigma.log()
    return c_skip, c_out, c_in, c_noise


# -----------------------------------------------------------------------------
# Training loss
# -----------------------------------------------------------------------------
def edm_training_loss(
    network: Callable,           # F_theta(c_in*x, c_noise, c) -> tensor same shape as x
    x_clean: torch.Tensor,        # (B, T, D)
    cond: Optional[dict],         # passed through to the network
    schedule: EDMSchedule,
    anchor_mask: Optional[torch.Tensor] = None,    # (B, T) bool; True = fixed (don't noise)
) -> torch.Tensor:
    """One step of EDM training loss.

    For anchored points (e.g. start / end of a trajectory), we set the noisy
    input equal to the clean value and zero out the per-element loss
    contribution. The denoiser still sees the anchors in its input.
    """
    B = x_clean.shape[0]
    device = x_clean.device
    sigma = schedule.sample_sigma_train(B, device=device, dtype=x_clean.dtype)
    sigma_b = sigma.view(B, *([1] * (x_clean.dim() - 1)))   # broadcast

    eps = torch.randn_like(x_clean)
    x_noisy = x_clean + sigma_b * eps

    if anchor_mask is not None:
        # Replace noisy values with clean at anchored positions.
        x_noisy = torch.where(anchor_mask.unsqueeze(-1), x_clean, x_noisy)

    c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma, schedule.sigma_data)
    c_skip_b = c_skip.view_as(sigma_b)
    c_out_b  = c_out.view_as(sigma_b)
    c_in_b   = c_in.view_as(sigma_b)

    F = network(c_in_b * x_noisy, c_noise, cond)
    D = c_skip_b * x_noisy + c_out_b * F

    w = schedule.loss_weight(sigma).view_as(sigma_b)
    err = (D - x_clean) ** 2
    if anchor_mask is not None:
        # Don't penalize anchored positions.
        err = err * (~anchor_mask).unsqueeze(-1).to(err.dtype)
    loss = (w * err).mean()
    return loss


# -----------------------------------------------------------------------------
# Deterministic DDIM-like sampler (EDM Heun-1 / Euler)
# -----------------------------------------------------------------------------
@torch.no_grad()
def edm_sample(
    network: Callable,
    shape: tuple,                 # (B, T, D)
    cond: Optional[dict],
    schedule: EDMSchedule,
    n_steps: int = 50,
    device=None,
    dtype=torch.float32,
    anchor_mask: Optional[torch.Tensor] = None,
    anchor_values: Optional[torch.Tensor] = None,
    guidance_scale: float = 0.0,
    uncond_cond: Optional[dict] = None,
    score_guidance: Optional[Callable] = None,   # x_0_hat -> additive eps correction
):
    """Deterministic Euler sampler over the EDM noise schedule.

    Parameters
    ----------
    network : the (preconditioned) denoiser; see `edm_preconditioning`.
    shape   : output shape (B, T, D).
    cond    : conditioning dict, passed through.
    schedule: EDMSchedule.
    n_steps : number of denoising steps (50 is a sensible default).
    anchor_mask, anchor_values : optional fixed positions to inpaint each step.
    guidance_scale : CFG weight. 0 = no guidance. Requires uncond_cond.
    score_guidance : optional callable applied to x_0_hat; should return an
                     additive correction to eps_theta. Used for land guidance.
    """
    B = shape[0]
    if device is None:
        # Best-effort
        device = next(network.parameters()).device if hasattr(network, "parameters") else "cpu"

    # Time grid: t = 0 -> sigma_max, t = 1 -> sigma_min.
    t_steps = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=dtype)
    sigmas = schedule.t_to_sigma(t_steps)
    sigmas[-1] = 0.0       # final step lands at fully clean

    x = torch.randn(shape, device=device, dtype=dtype) * sigmas[0]
    if anchor_mask is not None and anchor_values is not None:
        x = torch.where(anchor_mask.unsqueeze(-1), anchor_values, x)

    for i in range(n_steps):
        s_cur = sigmas[i]
        s_next = sigmas[i + 1]
        sigma_b = s_cur.expand(B)
        c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma_b, schedule.sigma_data)
        c_skip_b = c_skip.view(B, *([1] * (x.dim() - 1)))
        c_out_b  = c_out.view_as(c_skip_b)
        c_in_b   = c_in.view_as(c_skip_b)

        F = network(c_in_b * x, c_noise, cond)
        D = c_skip_b * x + c_out_b * F   # denoiser output (x_0 estimate)

        if guidance_scale != 0.0 and uncond_cond is not None:
            F_u = network(c_in_b * x, c_noise, uncond_cond)
            D_u = c_skip_b * x + c_out_b * F_u
            D = D_u + (1.0 + guidance_scale) * (D - D_u)

        if score_guidance is not None:
            D = score_guidance(D, sigma=s_cur)

        # First-order Euler step from x at sigma_cur towards D at sigma_next.
        d = (x - D) / s_cur if s_cur > 0 else torch.zeros_like(x)
        x = x + (s_next - s_cur) * d

        # Re-clamp anchors after each step.
        if anchor_mask is not None and anchor_values is not None:
            x = torch.where(anchor_mask.unsqueeze(-1), anchor_values, x)

    return x


# -----------------------------------------------------------------------------
# Convenience: build an anchor mask + values from start/end positions
# -----------------------------------------------------------------------------
def make_endpoint_anchor(
    start: torch.Tensor,         # (B, 2)
    end: torch.Tensor,           # (B, 2)
    seq_len: int,
    device=None,
    dtype=torch.float32,
):
    """Build (anchor_mask, anchor_values) that fix positions 0 and T-1.

    anchor_mask  : (B, T) bool, True at t in {0, T-1}.
    anchor_values: (B, T, 2). Zeros except at anchored positions, where they
                   hold the start / end.
    """
    B = start.shape[0]
    if device is None:
        device = start.device
    mask = torch.zeros(B, seq_len, device=device, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, -1] = True
    values = torch.zeros(B, seq_len, start.shape[-1], device=device, dtype=dtype)
    values[:, 0, :] = start
    values[:, -1, :] = end
    return mask, values


__all__ = [
    "EDMSchedule",
    "edm_preconditioning",
    "edm_training_loss",
    "edm_sample",
    "make_endpoint_anchor",
]
