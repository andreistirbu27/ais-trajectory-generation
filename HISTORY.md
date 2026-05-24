# Project History

> The work in this repo evolved through several distinct phases. This file
> is the consolidated memory of those phases: the pivots, lessons, and
> decisions that aren't reflected in the current code. **For the current
> production system, see [CLAUDE.md](CLAUDE.md). For the academic write-up,
> see [paper/paper.tex](paper/paper.tex). This file is read-only history.**

## Timeline at a glance

| Phase                                | Period           | What happened |
|--------------------------------------|------------------|---------------|
| **v1** — next-step prediction         | Feb–Apr 2026      | Causal Transformer encoder predicting $[d\lambda, d\phi]$ from a sliding window. Could not beat constant-velocity (CV) baseline on any configuration. |
| **v2–v5** — endpoint-conditioned gen  | Apr 18–21, 2026   | Pivot to encoder–decoder generating full 128-point routes from $(A, B, v)$. Three architectural rungs (encoder, bearing features, scheduled sampling). `val_delta` improved 0.086 → 0.074 but no real ADE gain — autoregressive accumulation swamped per-step delta wins. |
| **v6/v7** — obstacle conditioning     | Apr 19–28, 2026   | Soft obstacle penalty + augmented training. Did not learn avoidance: GT never crossed placed obstacles, so no gradient signal. Parked; see `docs/decisions.md#v6-v7`. |
| **v9** — retrieval (mean-pool)        | Apr 28–May 4, 2026| KNN retrieval of $K{=}5$ analogues. Mean-pool summarisation collapsed route shape on long voyages — retrieval-top-1 beat v9 on the >500 km bucket by 2.5×. |
| **v10** — per-timestep retrieval + land | May 4–14, 2026   | Each retrieved route tokenised into $t_{\text{retr}}{=}32$ phase-aware tokens. Soft SDF land loss + hard water-cell snap (WaterRouter). Production model. |
| **v11** — Halton-cone pointer         | May 14, 2026      | Candidate-pool min radius overshot the median GT step (3 km vs 0.34 km). Parked. |
| **v12** — water-cell pointer          | May 15–16, 2026   | $K{=}4$ ring silently dropped 10% of long GT transitions. Abandoned at ADE 27 km. |
| **E1** — clean-corpus filter          | May 16, 2026      | `filter_trajgen_npz.py` drops coastal/inland GT tracks. 207k → 61k tracks. v10+router land% and cross% collapse to 0% with no measurable ADE penalty. This was the upstream win, not the post-processor. |
| **paper-track** — ICLR submission     | May 17 → present  | Active work lives under `paper/`. v13 (TrajDiff) diffusion variant trained; K-anchor benchmark protocol introduced for TrAISformer comparison; DMA cross-basin transfer demonstrated. |

## The v1 → v2 pivot

v1 was a causal Transformer encoder predicting next-step displacement
$[d\lambda, d\phi]$ given a sliding window of past AIS positions, on
turn-segmented tracks at $\sim$3-min ping intervals.

**Result:** v1 ADE matched the constant-velocity (CV) baseline to within
seed variance on every configuration:

| Run                              | CV ADE | v1 ADE |
|----------------------------------|-------:|-------:|
| full dataset (12mo_seq120)       | 50.7 m | 50.8 m |
| open-sea only                    | 47.0 m | 47.5 m |
| open-sea + lighthouse features   | 47.0 m | 48.6 m |

**Root causes:**

1. **Turn segmentation removed the curves the model could have learned.**
   After splitting on heading change $>150°$, the remaining sub-tracks were
   near-linear. CV is the optimal predictor on linear data.
2. **Single-step ($\sim$3 min) prediction horizon is too short** to contain
   useful signal beyond local momentum.
3. **No destination awareness** — the model could not anticipate turns or
   course changes; it could only extrapolate local velocity.

**The pivot:** reformulate as endpoint-conditioned generation. Given
$(A, B, v)$, generate a complete 128-point route. The great-circle becomes
the natural baseline. This addresses all three v1 failures: full voyages
preserve curves; the horizon is now the whole voyage; the destination is an
explicit input.

v2 was implementable in $\sim$1 week and immediately yielded a 17% ADE
improvement over the great-circle baseline.

## Lessons (transferable beyond this project)

1. **Problem formulation matters more than model complexity.** v1 was
   well-engineered and extensively tuned, but the formulation made CV
   near-optimal. Pivoting to a different formulation produced a 17%
   improvement in a week.
2. **Data preprocessing can destroy signal.** Turn segmentation was
   intended to produce clean directionally-consistent training data. It
   removed exactly the structure the model needed.
3. **Baselines must be computed first.** The CV baseline was not computed
   until eight weeks into the project. Earlier phases optimised MSE in
   normalised space without knowing whether the model was actually better
   than the trivial predictor. Compute baselines first; pivot when they win.
4. **Loss-function design has sharp trade-offs.** v2 with
   $\lambda_{\text{end}}{=}10$ reached ADE $6.8$ km with end-point error
   $2.5$ km; with $\lambda_{\text{end}}{=}50$ the end-point error fell but
   ADE rose to $8.2$ km — the loss collapsed the route to a straight line.
   Post-processing (linear endpoint correction + WaterRouter) proved more
   effective than training-time endpoint weight.
5. **Exposure bias is real but hard to fix.** Scheduled sampling addresses
   the train/inference distribution mismatch theoretically. The naive
   step-by-step implementation OOM'd on MPS; a two-pass parallel-TF
   reformulation brought the cost back to $\sim$2$\times$ TF but still did
   not break the `val_delta` plateau. The plateau was a capacity/signal
   limitation, not exposure bias.
6. **Most maritime tracks are boring.** $\sim$85% of tracks are
   near-straight open-sea passages. The model's primary job is predicting
   constant deltas, which it learns in $\sim$4 epochs. The remaining 15% of
   curvy tracks contain the actual learning opportunity but are a minority
   of the loss; curriculum or weighted sampling would help.
7. **Generalisation has fundamental limits.** v10 is a route-memory /
   interpolation system for known waters. Given $(A, B)$ similar to
   training, it generates a plausible route. It cannot navigate unseen
   geography without the retrieval bank.
8. **Mid-step val_mse is misleading.** v1/v2's mid-step val MSE was
   computed on the first 30 val batches and is biased low. The first
   reliable signal is end-of-epoch full-val ADE on the actual rollout.
   Several phases chased mid-step gains that didn't survive AR accumulation.
9. **Per-step accuracy gains don't survive 128-step AR rollout.** The
   v2→v5 ladder (bigger model, bearing features, encoder, scheduled
   sampling) improved `val_delta` $0.086 \to 0.074$ but produced zero real
   ADE improvement on rollout. Only end-to-end rollout improvements matter.
10. **Modular code pays dividends.** The March refactor into
    `src/{model,data,metrics}.py` made every subsequent experimentation
    (configs, new scripts, v2 pivot, retrieval, land mask, diffusion
    variant) dramatically easier. Each new direction was an additive
    sibling (`*_gen.py`, `*_retrieval.py`, `model_gen_v10.py`,
    `model_gen_v13.py` + `diffusion.py`) rather than a destructive edit.

## E1 — the upstream data-filter win

After v10's headline win on the dirty corpus, a corpus audit revealed that
the ground-truth tracks themselves crossed land at 10.83% of positions
under the $10$ km clearance threshold. The problem was that
`prepare_trajgen.py` had been run without the `filter_inland.py` step from
the v1 pipeline. `filter_trajgen_npz.py` (E1) drops coastal/inland GT
tracks from the NPZ — 61k of 207k tracks survive.

Re-evaluating the v10 ep52 checkpoint on the clean corpus collapsed the
water-strict ADE penalty of WaterRouter from $+1.36$ km (dirty) to
$\sim$$0$ km (clean). The land-crossing rate dropped from 1.2% to 0.0%.
**The corpus-level filter was a larger win than any architectural change
between v5 and v10.** The lesson: when a model fails on a structural
property, check whether the training data itself violates that property
before adding model capacity to enforce it.

## What this file replaces

The pre-paper-track snapshot of the repo had four overlapping doc files at
the root, all of which were superseded:

- `project_report.txt` — chronological phase log (S4 course report,
  Feb–Apr 2026). Phase narratives are summarised in the timeline above;
  per-phase numbers are stale (predate the clean corpus and the May 2026
  production result).
- `trajectory_generation_report.txt` — technical narrative on v1→v2+. Why
  we pivoted is captured above; architectural specs and command inventory
  are now authoritative in [CLAUDE.md](CLAUDE.md) and
  [paper/paper.tex](paper/paper.tex).
- `report_v2_review_plan.md` — audit of issues in `report/report_v2.tex`
  (the school-report `.tex` file). Every C/M item was either resolved
  during the report rewrite or rendered moot by the paper-track pivot.
- `v10_progress_report.md` — April 2026 v9/v10 snapshot. v9 epoch-38
  numbers and v10 architecture rationale; both subsumed by the current
  production results in `results/seed_variance.csv` and
  [paper/paper.tex §5.1](paper/paper.tex) respectively.

These four files were deleted on 2026-05-23 in favour of this consolidated
record.
