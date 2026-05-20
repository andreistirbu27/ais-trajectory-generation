# v13 (TrajDiff) — Implementation-Aware Architecture Sketch

This document refines `docs/v13_proposal.md` against actual codebase constraints. Source for the underlying spec: `docs/v13_proposal.md`.

## Goal

A diffusion Transformer that generates a 128-waypoint trajectory `p_{0:T} ∈ R^{128 × 2}` conditioned on `(start, end, vessel_type, retrieved_K)`, with classifier-free guidance and **score-based land guidance** during DDIM sampling.

## Why diffusion (vs v10 autoregressive)

- **Probability calibration.** Diffusion gives a posterior, not a point estimate. ADE drops in expectation; uncertainty quantification becomes natural.
- **Whole-sequence reasoning.** v10's AR rollout commits to bad decisions early in long routes. Diffusion sees the whole trajectory at every step.
- **Score-based constraints.** Adding land guidance is mathematically clean — just add a score correction term `∇_x log p_constraint(x)` during sampling. v10 needs a hard projection post-hoc.

## Architecture (concrete sizes)

### Input representation
- Trajectory: `(B, T=128, 2)` in normalized (lon, lat) space. **Continuous, not discretized.** Unlike TrAISformer, we do not quantize.
- Conditioning: encoder memory from `PerStepRouteEncoder` and base tokens (start, end, vessel_type) — reuse `src/model_gen_v10.py:PerStepRouteEncoder` directly.

### Backbone: Diffusion Transformer (DiT-style)
- `d_model = 384` (larger than v10's 256 to absorb diffusion-time conditioning + retrieval cross-attention).
- 8 transformer blocks, 8 heads each.
- Each block: self-attention (over T=128 trajectory tokens) + cross-attention to encoder memory + FFN.
- adaLN-zero conditioning on noise level σ_t (DiT trick from Peebles & Xie 2023).
- Output head: `Linear(d_model → 2)` predicting the noise `ε_θ(x_t, t, c)`.

### Noise schedule
- Linear σ schedule in log-SNR space, from σ_max ≈ 80 to σ_min ≈ 0.002 (EDM convention from Karras 2022).
- DDIM-style deterministic sampler with 50 steps at inference (compute budget permitting; can drop to 20 for ablation).

### Conditioning paths
1. **Encoder memory**: 3 base + K=5 × t_retr=32 = 163 tokens from `PerStepRouteEncoder` (reuse v10's encoder verbatim).
2. **Vessel type**: same 8-d embedding as v10.
3. **Noise level σ_t**: adaLN-zero, conditions every transformer block.
4. **Endpoint anchoring**: trajectory points 0 and T are *fixed* (not noisy) during both training and sampling — the model is conditioned by inpainting these anchors. This is the equivalent of v10's "predict deltas from a known start".

### Classifier-free guidance
- During training, drop the entire conditioning context with probability 0.1 (per CFG convention).
- At sampling, use `ε̂ = (1 + w) · ε_θ(x, t, c) - w · ε_θ(x, t, ∅)` with guidance scale `w ∈ {0, 1, 3}` (sweep at eval time).

### Score-based land guidance
- During sampling, add a correction term to the predicted score:
  `∇_x log p_data(x_t) → ∇_x log p_data(x_t) + γ · ∇_x [−ReLU(SDF(x_t) − τ)²]`
- `γ` = guidance strength, `τ` = land threshold (=10 km, matching v10).
- This is the SAME differentiable SDF sampler as v10's L_land — reuse `src/land_mask.py::sample_km_torch`.
- Gradient flows through `F.grid_sample` (or our manual bilinear, depending on device).
- **Implementation note:** during DDIM sampling, the SDF is evaluated on the *predicted clean* `x_0_hat = (x_t - σ_t · ε̂) / α_t` (not on `x_t`), so the guidance signal is in the right space.

## Training

### Objective
Standard EDM loss:
```
L = E_{x, ε, σ} [w(σ) || D_θ(x + σε; σ, c) − x ||^2]
```
where `D_θ` is the denoiser implied by `ε_θ`, `w(σ) = (σ² + 0.5²) / (σ · 0.5)²` per EDM.

### Anchoring
At every training step:
- Sample x (clean trajectory).
- Set `x[0] = start_position`, `x[T-1] = end_position` (already true for our data).
- Sample σ_t, noise.
- **Replace** `x_t[0]` and `x_t[T-1]` with the clean anchors (not noisy versions). The denoiser learns to inpaint the middle conditioned on fixed endpoints.

### Optimization
- AdamW, lr=2e-4, weight_decay=1e-4, batch_size=64.
- Cosine schedule with 5% warmup, same as v10.
- 100 epochs (longer than v10's 60 — diffusion needs more steps).
- bf16 on A100. Gradient clip 1.0.
- AMP-safe (use `torch.amp.autocast(dtype=torch.bfloat16)`).

## Inference

### Sampler
DDIM 50 steps with `s_churn=0` (deterministic). At each step:
1. Predict `ε̂` via CFG (mix conditional + unconditional with guidance scale w).
2. Add land-guidance correction: `ε̂ ← ε̂ - γ · σ_t · ∇_x [ReLU(SDF(x_0_hat) − τ)²]`.
3. Standard DDIM update: `x_{t-1} = α_{t-1} · x_0_hat + σ_{t-1} · ε̂`.
4. Re-clamp anchors: `x_{t-1}[0] = start`, `x_{t-1}[T-1] = end`.

### Post-processing (optional)
- Apply `WaterRouter` snap-only on the final sample, same as v10 inference. Hopefully unnecessary if land guidance works, but kept as safety net.

## Risk register

1. **Land guidance may destabilize sampling** at high γ. Mitigation: sweep γ on a held-out batch; pick the largest γ that keeps ADE within 5% of γ=0. If no setting works, downgrade to soft-loss-only-at-training + post-hoc snap (mirrors v10).
2. **Diffusion may not beat v10.** v10 is well-tuned for this corpus. If v13 ADE > v10 ADE on val, the diffusion model still wins on uncertainty quantification, which v10 cannot provide. Frame as "first diffusion baseline on this task; competitive ADE; better calibration".
3. **Compute.** 50-step DDIM sampling is ~50× more expensive than v10's AR rollout. For batch=1 latency on A100, expect ~500 ms vs v10's ~10 ms. Worth measuring; reviewers will ask.
4. **Numerical instability at σ=σ_max.** Common diffusion gotcha. Use EDM's exact formulas (Karras 2022 eq. 8) to avoid this.
5. **Training divergence.** Diffusion is more fragile than supervised learning. First training run will probably need debugging — budget 1 week for stabilization vs initial estimate of 3 days.

## Implementation plan

### File layout (paper-track)
- `src/diffusion.py` — sampler, schedule, helpers (new module).
- `src/model_gen_v13.py` — DiT backbone + conditioning paths.
- `configs/paper/v13/trajdiff_base.yaml` — full HP config.
- `scripts/paper/train_gen_v13.py` — training loop.
- `scripts/paper/eval_gen_v13.py` — eval with sampling.
- `paper/notes/v13_experiment_log.md` — live experiment notes (failure modes, fixes, decisions).

### Reuse from existing codebase
- `src/model_gen_v10.PerStepRouteEncoder` — verbatim for conditioning encoder.
- `src/land_mask.LandMask.sample_km_torch` — verbatim for differentiable SDF.
- `src/water_router.WaterRouter` — verbatim for optional post-processing.
- `src/data_gen_retrieval.RetrievalTrajectoryDataset` — verbatim for data loading.
- `src/metrics_gen` — verbatim for evaluation.

### Schedule (weeks 2-9, parallel with v10 + TrAISformer track)
- Week 2: skeleton + minimal denoiser, single-batch overfit test.
- Week 3: full DiT backbone wired; small-scale training on 5k subset.
- Week 4-5: full-scale training; CFG implementation; debugging.
- Week 6: land guidance implementation and sweep.
- Week 7: full evaluation on US val/test.
- Week 8-9: cross-region DMA evaluation.

## Open questions

- **Discrete vs continuous time embedding?** EDM uses continuous σ; DDPM uses discrete step indices. Recommend continuous (matches EDM, easier to vary at sampling).
- **Should we predict ε or v (velocity parameterization)?** v-parameterization (Salimans & Ho 2022) is more stable at low SNR. Recommend v-pred. Cheap to switch.
- **Sampler family.** DDIM (deterministic, fewer steps) vs EDM stochastic vs Heun 2nd-order? Recommend DDIM first; switch to Heun if quality is poor.
- **How to handle FDE?** Endpoint anchoring solves this trivially: `x[T-1] = end` is enforced exactly. Reviewers may ask if this is "cheating" — it's not, it's faithful to the task definition.

## Decision boundary

If v13 fails to beat v10 ADE by week 9, **freeze v13 results as-is, present it as a calibrated diffusion baseline (with uncertainty quantification), and headline with v10**. Don't try to over-tune v13 into beating v10 — the diffusion contribution is real even if ADE is comparable.
