#!/usr/bin/env python3
"""
sanity_check_env.py
-------------------
Smoke-test the paper-track environment. Run this immediately after creating
`paper/.venv` and pip-installing `paper/requirements.txt`.

Checks performed (each independent; failures isolated):
  1. Core ML imports (torch, numpy, pandas, scipy, sklearn, einops).
  2. Geo imports (geopandas, rasterio, shapely, contextily).
  3. CUDA availability + GPU info if present.
  4. Project-local imports from src/ (model_gen_v10, model_gen_v13, diffusion,
     data_gen_retrieval).
  5. Load TrAISformer's shipped DMA sample and verify the pickle schema.
  6. Instantiate v13 (TrajDiff) backbone and run one forward pass on random
     tensors to verify shape contracts.
  7. Instantiate the EDM training loss + sampler and run one forward+backward.

Exit code 0 iff all checks pass.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Allow `from src...` imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _section(title: str):
    print()
    print(title)
    print("-" * len(title))


def check(label: str, fn):
    try:
        fn()
        print(f"  [OK]   {label}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        return False


def main() -> int:
    results = []

    _section("1. Core ML imports")
    def _core():
        import torch, numpy, pandas, scipy, sklearn, einops  # noqa: F401
        print(f"     torch {torch.__version__}, numpy {numpy.__version__}, "
              f"pandas {pandas.__version__}")
    results.append(check("torch/numpy/pandas/scipy/sklearn/einops", _core))

    _section("2. Geo imports")
    def _geo():
        import geopandas, rasterio, shapely, contextily  # noqa: F401
        print(f"     geopandas {geopandas.__version__}, "
              f"rasterio {rasterio.__version__}, "
              f"shapely {shapely.__version__}")
    results.append(check("geopandas/rasterio/shapely/contextily", _geo))

    _section("3. CUDA / GPU")
    def _gpu():
        import torch
        if not torch.cuda.is_available():
            print("     CUDA NOT available — paper-track training will fall "
                  "back to CPU (very slow). Acceptable for sanity checks; "
                  "switch to a GPU node for real training.")
            return
        n = torch.cuda.device_count()
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            print(f"     GPU {i}: {p.name}  "
                  f"({total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free, "
                  f"compute {p.major}.{p.minor})")
    results.append(check("CUDA / GPU info", _gpu))

    _section("4. Project-local imports")
    def _src():
        from src.model_gen_v10 import TrajectoryGeneratorV10, PerStepRouteEncoder  # noqa: F401
        from src.model_gen_v13 import TrajDiff, TrajDiffConfig                      # noqa: F401
        from src.diffusion import EDMSchedule, edm_training_loss, edm_sample        # noqa: F401
        from src.data_gen_retrieval import build_and_cache_knn                       # noqa: F401
        from src.land_mask import LandMask                                            # noqa: F401
        from src.water_router import WaterRouter                                      # noqa: F401
        print("     all imports OK")
    results.append(check("project src/ imports", _src))

    _section("5. TrAISformer shipped DMA sample")
    def _dma():
        import pickle
        sample_dir = REPO_ROOT / "paper" / "external" / "TrAISformer" / "data" / "ct_dma"
        if not sample_dir.exists():
            print(f"     [skip] {sample_dir} not present "
                  "(submodule may need `git submodule update --init`)")
            return
        for fname in ("ct_dma_train.pkl", "ct_dma_valid.pkl", "ct_dma_test.pkl"):
            p = sample_dir / fname
            if not p.exists():
                print(f"     [skip] {p.name} not in shipped sample (not all splits ship)")
                continue
            with open(p, "rb") as f:
                data = pickle.load(f)
            assert isinstance(data, list) and len(data) > 0, f"{p.name} empty or wrong type"
            entry = data[0]
            assert "mmsi" in entry and "traj" in entry, \
                f"{p.name}: expected keys mmsi/traj, got {list(entry.keys())}"
            t = entry["traj"]
            # README says (N, 5) but shipped data is actually (N, 6) — column 5
            # is a per-row MMSI repeat (their datasets.py only reads [:, :4]
            # for features and [0, 4] for time, so column 5 is ignored).
            assert t.ndim == 2 and t.shape[1] in (5, 6), \
                f"{p.name}: expected (N, 5) or (N, 6), got {t.shape}"
            assert (t[:, :4] >= 0).all() and (t[:, :4] <= 1).all(), \
                f"{p.name}: lat/lon/sog/cog must be normalized to [0, 1]"
            print(f"     {p.name}: {len(data):,} tracks, "
                  f"first traj shape {t.shape}")
    results.append(check("DMA pickle schema", _dma))

    _section("6. v13 (TrajDiff) forward pass")
    def _v13():
        import torch
        from src.model_gen_v13 import TrajDiff, TrajDiffConfig
        cfg = TrajDiffConfig(seq_len=128, d_model=64, n_layer=2, n_heads=4,
                             max_retrieved=3, t_retr=8)
        model = TrajDiff(cfg)
        params = model.count_parameters()
        B, T, D, K = 2, 128, 2, 3
        x = torch.randn(B, T, D)
        c_noise = torch.zeros(B)
        cond = {
            "start": torch.randn(B, 2),
            "end": torch.randn(B, 2),
            "vessel_type": torch.zeros(B, dtype=torch.long),
            "retrieved": torch.randn(B, K, T, D),
        }
        out = model(x, c_noise, cond)
        assert out.shape == x.shape, f"output shape {out.shape} != input {x.shape}"
        print(f"     forward OK; small-config params = {params:,}")
    results.append(check("v13 forward pass", _v13))

    _section("7. EDM training-loss + 5-step sample")
    def _edm():
        import torch
        from src.model_gen_v13 import TrajDiff, TrajDiffConfig, make_cfg_conds, make_uncond
        from src.diffusion import EDMSchedule, edm_training_loss, edm_sample, make_endpoint_anchor

        cfg = TrajDiffConfig(seq_len=32, d_model=64, n_layer=2, n_heads=4,
                             max_retrieved=2, t_retr=4)
        model = TrajDiff(cfg)
        schedule = EDMSchedule(sigma_max=2.0)

        B, T, D, K = 2, cfg.seq_len, 2, 2
        x = torch.randn(B, T, D)
        cond = {
            "start": x[:, 0, :].clone(),
            "end": x[:, -1, :].clone(),
            "vessel_type": torch.zeros(B, dtype=torch.long),
            "retrieved": torch.randn(B, K, T, D),
        }
        cond_train = make_cfg_conds(cond, p_uncond=0.5, device=x.device)
        anchor_mask, _ = make_endpoint_anchor(cond["start"], cond["end"], T)

        loss = edm_training_loss(model, x, cond_train, schedule, anchor_mask=anchor_mask)
        loss.backward()
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        print(f"     training step OK; loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")

        model.eval()
        with torch.no_grad():
            anchor_values = torch.zeros(B, T, D)
            anchor_values[:, 0, :] = cond["start"]
            anchor_values[:, -1, :] = cond["end"]
            sampled = edm_sample(
                model, (B, T, D), cond, schedule, n_steps=5,
                anchor_mask=anchor_mask, anchor_values=anchor_values,
                guidance_scale=1.0, uncond_cond=make_uncond(cond, device=x.device),
            )
        assert sampled.shape == (B, T, D), f"sampler output {sampled.shape}"
        endpoint_err = (sampled[:, [0, -1]] - anchor_values[:, [0, -1]]).abs().max().item()
        print(f"     sampling OK; endpoint clamp error = {endpoint_err:.2e}")
    results.append(check("EDM loss + DDIM-style sampler", _edm))

    print()
    n_ok = sum(results)
    n_total = len(results)
    print(f"=== {n_ok}/{n_total} checks passed ===")
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
