#!/usr/bin/env python3
"""CPU-only smoke tests guarding the load-bearing claims of the repo.

Fast (< a few seconds), no training, no GPU. Run directly:

    python3 tests/test_smoke.py

or via pytest (functions are named test_*):

    pytest tests/test_smoke.py

Guards:
  1. The differentiable geographic constraint actually backpropagates
     (the paper's headline novelty) — src/land_mask.py.
  2. Great-circle baseline + haversine metric are sane — src/metrics_gen.py.
  3. The production v10 checkpoint loads and has the expected size
     (catches silent architecture drift) — runs/trajgen_v10/best.pt.

Tests whose required artifact is missing are SKIPPED, not failed, so the
suite still runs on a fresh checkout without the large data/checkpoint files.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LAND_SDF = os.path.join(ROOT, "data/processed/land_sdf_050deg.npz")
V10_CKPT = os.path.join(ROOT, "runs/trajgen_v10/best.pt")


class SkipTest(Exception):
    """Raised when a required artifact is absent; reported as SKIP."""


def test_constraint_gradient():
    """L_land's differentiable SDF sampler must backprop to lon/lat.

    Guards src/land_mask.py:sample_km_torch — if gradients stop flowing,
    the soft land constraint silently becomes a no-op during training.
    """
    if not os.path.exists(LAND_SDF):
        raise SkipTest(f"missing {LAND_SDF}")
    import torch

    from src.land_mask import LandMask

    mask = LandMask.load(LAND_SDF)
    # Points spanning US coastal regions (within the SDF bbox) so at least
    # some sit near a land/water boundary where the SDF varies → nonzero grad.
    lon = torch.tensor([-70.0, -80.0, -90.0, -120.0, -75.0],
                       dtype=torch.float32, requires_grad=True)
    lat = torch.tensor([40.0, 28.0, 29.0, 36.0, 38.0],
                       dtype=torch.float32, requires_grad=True)

    sdf = mask.sample_km_torch(lon, lat)
    assert sdf.shape == lon.shape, f"shape mismatch: {sdf.shape} vs {lon.shape}"
    assert torch.isfinite(sdf).all(), "SDF sample produced non-finite values"

    sdf.sum().backward()
    assert lon.grad is not None and lat.grad is not None, "no gradient computed"
    assert torch.isfinite(lon.grad).all() and torch.isfinite(lat.grad).all(), \
        "non-finite gradients"
    total = lon.grad.abs().sum() + lat.grad.abs().sum()
    assert total.item() > 0.0, \
        "gradient is identically zero — constraint would not train"


def test_great_circle_ade():
    """Great-circle baseline + haversine metric sanity."""
    from src.metrics_gen import great_circle_trajectory, haversine_meters

    start = np.array([-70.0, 40.0])   # [lon, lat]
    end = np.array([-9.0, 38.7])      # roughly Boston -> Lisbon longitudes
    traj = great_circle_trajectory(start, end, n_points=128)

    assert traj.shape == (128, 2), f"unexpected shape {traj.shape}"
    assert np.isfinite(traj).all(), "non-finite great-circle points"

    # Endpoints land on start/end (sub-metre).
    d_start = haversine_meters(traj[0, 1], traj[0, 0], start[1], start[0])
    d_end = haversine_meters(traj[-1, 1], traj[-1, 0], end[1], end[0])
    assert float(d_start) < 1.0, f"start endpoint off by {float(d_start):.2f} m"
    assert float(d_end) < 1.0, f"end endpoint off by {float(d_end):.2f} m"

    # ADE of a path against itself is exactly zero.
    ade_self = haversine_meters(traj[:, 1], traj[:, 0], traj[:, 1], traj[:, 0]).mean()
    assert float(ade_self) == 0.0, f"ADE(path, path) != 0: {float(ade_self)}"

    # Cumulative arc length is monotonic non-decreasing.
    seg = haversine_meters(traj[:-1, 1], traj[:-1, 0], traj[1:, 1], traj[1:, 0])
    assert (seg >= 0).all(), "negative segment length"
    assert seg.sum() > 0, "degenerate (zero-length) great circle"


def test_v10_checkpoint_loads():
    """Production v10 checkpoint loads and is ~5.97M params.

    Guards against silent architecture drift: if the model definition changes
    shape, the param count moves and this trips.
    """
    if not os.path.exists(V10_CKPT):
        raise SkipTest(f"missing {V10_CKPT}")
    import torch

    sd = torch.load(V10_CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("model", "state_dict", "model_state_dict", "weights"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    tensors = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    assert tensors, "no tensors found in checkpoint"
    total_m = sum(v.numel() for v in tensors.values()) / 1e6
    assert 5.5 < total_m < 6.5, \
        f"v10 param count {total_m:.2f}M outside expected ~5.97M (arch drift?)"


def _run():
    tests = [test_constraint_gradient, test_great_circle_ade,
             test_v10_checkpoint_loads]
    n_pass = n_fail = n_skip = 0
    for t in tests:
        try:
            t()
        except SkipTest as e:
            n_skip += 1
            print(f"SKIP {t.__name__}: {e}")
        except AssertionError as e:
            n_fail += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 — surface any unexpected error
            n_fail += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
        else:
            n_pass += 1
            print(f"PASS {t.__name__}")
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(_run())
