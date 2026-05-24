# The K-Anchor Unified Protocol

This is the paper's headline contribution from §3 of the strategic plan
(`~/.claude/plans/i-need-strategic-research-elegant-shamir.md`). The protocol
unifies *next-step prediction* (TrAISformer's task) and *endpoint-conditioned
generation* (v10's task) under one evaluation metric, so any method that
produces a 128-waypoint trajectory can be compared apples-to-apples.

## Setup

For each ground-truth trajectory $y \in \mathbb{R}^{T \times 2}$ with
$T = 128$ waypoints, sample $K$ *anchor* indices

$$ \mathcal{A}_K = \{a_0, a_1, \ldots, a_{K-1}\} \subseteq \{0, 1, \ldots, T-1\} $$

The method is given the anchor positions $\{y_{a_k}\}_{k=0}^{K-1}$ and
the vessel type $v$, and must produce a complete trajectory
$\hat{y} \in \mathbb{R}^{T \times 2}$ satisfying $\hat{y}_{a_k} = y_{a_k}$
for all $k$. Anchor indices are evenly spaced:

$$ a_k = \mathrm{round}\!\left( k \cdot \frac{T-1}{K-1} \right), \quad k = 0, 1, \ldots, K-1 $$

so $a_0 = 0$ and $a_{K-1} = T-1$ in every setting.

## Special cases

| $K$ | Anchor set | Equivalent task |
|---|---|---|
| 2  | $\{0, T-1\}$ | **Endpoint-conditioned generation** (v10's setting) |
| 18 | $\{0, \lceil (T-1)/17 \rceil, \ldots, T-1\}$ | Approximation of TrAISformer's 18-step history protocol with the future-72 tail rolled out |
| 64 | every other waypoint | Soft middle ground; sparse history with mid-route checkpoints |
| 128 | all indices | Trivial — ADE $= 0$, an upper-bound sanity check |

## Metrics (computed *only* on non-anchor indices)

For each predicted trajectory $\hat{y}$, the score excludes anchor
positions to avoid trivially scoring them as zero error:

$$ \mathrm{ADE}_K(\hat{y}, y) = \frac{1}{T - K} \sum_{t \notin \mathcal{A}_K} \|\hat{y}_t - y_t\|_{\mathrm{haversine}} $$

Report:
1. **ADE**$_K$ (mean haversine error in metres over non-anchor indices)
2. **FDE**$_K$ (final-point error; equal to anchor-2 error when $K=2$, else the error at the largest non-anchor index)
3. **land\%@10km** (fraction of non-anchor predictions $\hat{y}_t$ with SDF$(\hat{y}_t) > 10$ km)
4. **cross\%@10km** (fraction of trajectories where any non-anchor segment crosses land at $> 10$ km penetration)
5. **sample-time per trajectory** (ms, batch=1 and batch=32, on RTX 3090)

The headline paper figure is **ADE$_K$ vs $K$** for each method, plotted on a
log-K x-axis, evaluated over the same 500-row val anchor set used by
`scripts/paper/eval_shared_protocol.py`.

## What each method does at K

| Method | $K=2$ | $K=18$ | $K=64$ | $K=128$ |
|---|---|---|---|---|
| **great_circle** | geodesic from $y_0$ to $y_{T-1}$ | piecewise geodesics between anchors | piecewise geodesics | $y$ (trivial) |
| **retr_top1** | nearest train-corpus route by (start, end, vtype) | nearest by (anchor_0…anchor_{K-1}) | nearest by all anchors | trivial |
| **v10_router** | run v10's encoder with (start, end, vtype); apply router post-hoc | wire anchor positions as extra encoder tokens; run v10 | same | trivial |
| **v13** | EDM sample with start+end anchors clamped each denoising step | EDM sample with all $K$ anchors clamped each step (uses the existing `anchor_mask` API in `src/diffusion.py:make_endpoint_anchor`, generalised) | same | trivial |
| **TrAISformer** | replicate $y_0$ to fill the 18-step history with zero velocity; autoregress 128 steps | use the actual first 18 anchors as history; autoregress the rest | re-condition every $\lceil (T-1)/(K-1) \rceil$ steps on the next anchor (teacher-forcing in their token space, then resume free-run) | trivial |

The TrAISformer-at-$K=2$ row is the **predicted-to-degenerate** case from
Protocol B (§3 of the strategic plan): it has no signal to extrapolate from
a single point, so its ADE will explode. That degeneration is the empirical
evidence that endpoint-conditioned generation is a *distinct* task, not just
a particular history length.

## Implementation status (2026-05-21)

- $K=2$ implemented: `scripts/paper/eval_shared_protocol.py` (v10_router, v13, retr_top1, great_circle).
- General $K$: needs the eval harness extended to take a `--k_anchor` flag and to mask anchor indices out of the scoring loop.
- TrAISformer adapter: blocked on training TrAISformer on US data; the wrapper is `scripts/paper/train_traisformer.py` (DMA reproduction sanity-runs without retraining).
- Headline figure (ADE-vs-K curves): blocked on the above.

## Why this is the paper's headline contribution

Reviewers in the AIS literature evaluate on next-step prediction; reviewers
in the trajectory-generation literature evaluate on endpoint completion.
Neither evaluation alone is fair to both sides. The $K$-anchor protocol is
the smallest generalisation that lets the same metric work for both
communities — and the $K=2$ corner is where v10/v13 have a real edge
that the next-step framing makes invisible.

The framing also lets us answer one of the strategic plan's "claims that need new
experiments": *"Our method is competitive with prior AIS work."* By evaluating
TrAISformer at $K=18$ (its native setting) AND $K=2$ (our setting), we get
either side of the picture without rigging the comparison.
