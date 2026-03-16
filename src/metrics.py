"""
Evaluation metrics for AIS trajectory prediction.

All distance metrics are in metres (haversine).
Predictions are normalized displacements; absolute next positions are reconstructed
as: pred_pos = input_pos + disp_scaler.inverse(pred).
"""

import numpy as np
import torch
import torch.nn as nn

from .data import Scalers, VelScalers


def haversine_meters(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Haversine distance in metres between two sets of (lat, lon) in degrees."""
    R = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


@torch.no_grad()
def evaluate(model, loader, device, scalers: Scalers, pred_mode: str) -> dict:
    """
    Returns mse (normalised displacement space), ADE and FDE in metres.

    causal mode: ADE averages over all seq positions; FDE uses the last.
    single mode: ADE == FDE == single-step error.
    """
    model.eval()
    mse_fn = nn.MSELoss()
    total_mse, ade_sum, fde_sum, n = 0.0, 0.0, 0.0, 0

    for x, y, gap_mask, vtype in loader:
        x, y = x.to(device), y.to(device)
        gap_mask = gap_mask.to(device)
        vtype    = vtype.to(device)
        pred = model(x, gap_mask=gap_mask, vessel_type=vtype)

        total_mse += mse_fn(pred, y).item()
        n += 1

        x_np = x.cpu().numpy()[:, :, :2]   # normalized input positions

        if pred_mode == "causal":
            S, B = pred.shape[0], pred.shape[1]
            input_pos_geo = scalers.pos.inverse(x_np.reshape(-1, 2)).reshape(S, B, 2)
            pred_disp_geo = scalers.disp.inverse(
                pred.cpu().numpy().reshape(-1, 2)).reshape(S, B, 2)
            true_disp_geo = scalers.disp.inverse(
                y.cpu().numpy().reshape(-1, 2)).reshape(S, B, 2)
            pred_geo = input_pos_geo + pred_disp_geo
            y_geo    = input_pos_geo + true_disp_geo

            dist = haversine_meters(y_geo[:, :, 1], y_geo[:, :, 0],
                                    pred_geo[:, :, 1], pred_geo[:, :, 0])
            ade_sum += float(dist.mean())
            fde_sum += float(dist[-1].mean())
        else:
            last_pos_geo  = scalers.pos.inverse(x_np[-1])
            pred_disp_geo = scalers.disp.inverse(pred.cpu().numpy())
            true_disp_geo = scalers.disp.inverse(y.cpu().numpy())
            pred_geo = last_pos_geo + pred_disp_geo
            y_geo    = last_pos_geo + true_disp_geo
            dist = haversine_meters(y_geo[:, 1], y_geo[:, 0],
                                    pred_geo[:, 1], pred_geo[:, 0])
            ade_sum += float(dist.mean())
            fde_sum += float(dist.mean())

    if n == 0:
        return {"mse": float("nan"), "ade_m": float("nan"), "fde_m": float("nan")}
    return {"mse": total_mse / n, "ade_m": ade_sum / n, "fde_m": fde_sum / n}


@torch.no_grad()
def evaluate_constant_velocity_baseline(loader, device, scalers: Scalers,
                                        pred_mode: str) -> dict:
    """
    Constant-velocity baseline: predicted displacement = previous step displacement.
    Reconstructs absolute positions the same way as evaluate().
    """
    mse_fn = nn.MSELoss()
    total_mse, ade_sum, fde_sum, n = 0.0, 0.0, 0.0, 0

    for x, y, _gap, _vtype in loader:
        x, y = x.to(device), y.to(device)
        pos  = x[:, :, :2]   # (seq_len, B, 2) normalized lon/lat

        if pred_mode == "causal":
            vel_norm = torch.diff(pos, dim=0, prepend=pos[:1])   # (seq_len, B, 2)
            total_mse += mse_fn(vel_norm, y).item()
            n += 1

            S, B = vel_norm.shape[0], vel_norm.shape[1]
            x_np          = x.cpu().numpy()[:, :, :2]
            input_pos_geo = scalers.pos.inverse(x_np.reshape(-1, 2)).reshape(S, B, 2)
            pred_disp_geo = scalers.disp.inverse(
                vel_norm.cpu().numpy().reshape(-1, 2)).reshape(S, B, 2)
            true_disp_geo = scalers.disp.inverse(
                y.cpu().numpy().reshape(-1, 2)).reshape(S, B, 2)
            pred_geo = input_pos_geo + pred_disp_geo
            y_geo    = input_pos_geo + true_disp_geo
            dist = haversine_meters(y_geo[:, :, 1], y_geo[:, :, 0],
                                    pred_geo[:, :, 1], pred_geo[:, :, 0])
            ade_sum += float(dist.mean())
            fde_sum += float(dist[-1].mean())
        else:
            vel_norm = pos[-1] - pos[-2]   # (B, 2)
            total_mse += mse_fn(vel_norm, y).item()
            n += 1

            x_np          = x.cpu().numpy()[:, :, :2]
            last_pos_geo  = scalers.pos.inverse(x_np[-1])
            pred_disp_geo = scalers.disp.inverse(vel_norm.cpu().numpy())
            true_disp_geo = scalers.disp.inverse(y.cpu().numpy())
            pred_geo = last_pos_geo + pred_disp_geo
            y_geo    = last_pos_geo + true_disp_geo
            dist = haversine_meters(y_geo[:, 1], y_geo[:, 0],
                                    pred_geo[:, 1], pred_geo[:, 0])
            ade_sum += float(dist.mean())
            fde_sum += float(dist.mean())

    if n == 0:
        return {"mse": float("nan"), "ade_m": float("nan"), "fde_m": float("nan")}
    return {"mse": total_mse / n, "ade_m": ade_sum / n, "fde_m": fde_sum / n}


@torch.no_grad()
def sanity_check(model, loader, device, scalers: Scalers, pred_mode: str):
    """Print a few predicted vs true next positions to verify sensible output."""
    model.eval()
    x, y, gap_mask, vtype = next(iter(loader))
    x, y = x.to(device), y.to(device)
    gap_mask = gap_mask.to(device)
    vtype    = vtype.to(device)
    pred = model(x, gap_mask=gap_mask, vessel_type=vtype)

    p_batch = pred[-1] if pred_mode == "causal" else pred   # (B, 2)
    t_batch = y[-1]    if pred_mode == "causal" else y

    last_input = x[-1]   # (B, input_dim)

    for i in range(min(3, p_batch.shape[0])):
        inp_pos = scalers.pos.inverse(last_input[i, :2].cpu().numpy().reshape(1, 2))[0]
        p_disp  = scalers.disp.inverse(p_batch[i].cpu().numpy().reshape(1, 2))[0]
        t_disp  = scalers.disp.inverse(t_batch[i].cpu().numpy().reshape(1, 2))[0]
        p = inp_pos + p_disp
        t = inp_pos + t_disp
        err = haversine_meters(t[1], t[0], p[1], p[0])
        print(f"    sample {i}: pred ({p[0]:.3f}°, {p[1]:.3f}°)  "
              f"true ({t[0]:.3f}°, {t[1]:.3f}°)  err {err:.0f}m")


# ─── Velocity evaluation ──────────────────────────────────────────────────────

def _recover_positions(pred_vel_norm, y_vel_norm, x_np, dt_next_np, scalers: VelScalers):
    """
    Convert normalised velocity predictions to lon/lat positions.

    pred_vel_norm, y_vel_norm : (S, B, 2) normalised velocity
    x_np                      : (S, B, D) normalised input features
    dt_next_np                : (S, B)    seconds per step
    Returns pred_pos, true_pos : (S, B, 2) lon/lat degrees
    """
    input_pos    = scalers.pos.inverse(x_np[:, :, :2])        # (S, B, 2)
    pred_vel_raw = scalers.vel.inverse(pred_vel_norm)          # (S, B, 2) deg/s
    y_vel_raw    = scalers.vel.inverse(y_vel_norm)             # (S, B, 2) deg/s
    dt           = dt_next_np[:, :, None]                      # (S, B, 1)
    pred_pos     = input_pos + pred_vel_raw * dt               # (S, B, 2)
    true_pos     = input_pos + y_vel_raw    * dt               # (S, B, 2)
    return pred_pos, true_pos


@torch.no_grad()
def evaluate_vel(model, loader, device, scalers: VelScalers) -> dict:
    model.eval()
    ade_list, fde_list, mse_list = [], [], []
    for x, y, gap_mask, vtype, dt_next in loader:
        x, y     = x.to(device), y.to(device)
        gap_mask = gap_mask.to(device)
        vtype    = vtype.to(device)
        pred     = model(x, gap_mask=gap_mask, vessel_type=vtype)  # (S, B, 2)
        mse_list.append(nn.functional.mse_loss(pred[1:], y[1:]).item())

        pred_pos, true_pos = _recover_positions(
            pred.cpu().numpy(), y.cpu().numpy(),
            x.cpu().numpy(), dt_next.numpy(), scalers)

        S, B = pred_pos.shape[:2]
        for b in range(B):
            dists = [haversine_meters(true_pos[t, b, 1], true_pos[t, b, 0],
                                     pred_pos[t, b, 1], pred_pos[t, b, 0])
                     for t in range(1, S)]
            ade_list.append(np.mean(dists))
            fde_list.append(dists[-1])

    if not mse_list:
        return {"mse": float("nan"), "ade_m": float("nan"), "fde_m": float("nan")}
    return {"mse": np.mean(mse_list),
            "ade_m": np.mean(ade_list),
            "fde_m": np.mean(fde_list)}


def evaluate_constvel_baseline_vel(loader, device, scalers: VelScalers) -> dict:
    """
    Const-vel baseline for velocity mode.
    Predicts vel[t] = vel[t-1]; recovers pred_pos = input_pos + pred_vel * dt_next.
    """
    ade_list, fde_list, mse_list = [], [], []
    for x, y, gap_mask, vtype, dt_next in loader:
        x_np  = x.numpy()       # (S, B, D)
        y_np  = y.numpy()       # (S, B, 2)
        dt_np = dt_next.numpy() # (S, B)

        pos_raw  = scalers.pos.inverse(x_np[:, :, :2])            # (S, B, 2)
        disp_raw = np.concatenate([pos_raw[:1] * 0,
                                   np.diff(pos_raw, axis=0)], axis=0)  # (S, B, 2)
        logdt_raw = scalers.logdt.inverse(x_np[:, :, 2:3])        # (S, B, 1)
        dt_input  = np.expm1(logdt_raw[:, :, 0]).clip(min=1.0)    # (S, B)
        vel_raw   = disp_raw / dt_input[:, :, None]                # (S, B, 2) deg/s

        pred_vel_raw  = np.concatenate([vel_raw[:1], vel_raw[:-1]], axis=0)  # (S, B, 2)
        pred_vel_norm = scalers.vel.transform(
            pred_vel_raw.reshape(-1, 2)).reshape(pred_vel_raw.shape)
        mse_list.append(((pred_vel_norm[1:] - y_np[1:]) ** 2).mean())

        pred_pos = pos_raw + pred_vel_raw * dt_np[:, :, None]      # (S, B, 2)
        y_vel_raw = scalers.vel.inverse(y_np)
        true_pos  = pos_raw + y_vel_raw   * dt_np[:, :, None]      # (S, B, 2)

        S, B = pred_pos.shape[:2]
        for b in range(B):
            dists = [haversine_meters(true_pos[t, b, 1], true_pos[t, b, 0],
                                     pred_pos[t, b, 1], pred_pos[t, b, 0])
                     for t in range(1, S)]
            ade_list.append(np.mean(dists))
            fde_list.append(dists[-1])

    if not mse_list:
        return {"mse": float("nan"), "ade_m": float("nan"), "fde_m": float("nan")}
    return {"mse": np.mean(mse_list),
            "ade_m": np.mean(ade_list),
            "fde_m": np.mean(fde_list)}


@torch.no_grad()
def sanity_check_vel(model, loader, device, scalers: VelScalers):
    """Print a few predicted vs true next positions to verify sensible vel output."""
    model.eval()
    x, y, gap_mask, vtype, dt_next = next(iter(loader))
    x, y     = x.to(device), y.to(device)
    gap_mask = gap_mask.to(device)
    vtype    = vtype.to(device)
    pred     = model(x, gap_mask=gap_mask, vessel_type=vtype)

    pred_pos, true_pos = _recover_positions(
        pred.cpu().numpy(), y.cpu().numpy(),
        x.cpu().numpy(), dt_next.numpy(), scalers)

    t = pred_pos.shape[0] - 1
    for i in range(min(3, pred_pos.shape[1])):
        err = haversine_meters(true_pos[t, i, 1], true_pos[t, i, 0],
                               pred_pos[t, i, 1], pred_pos[t, i, 0])
        print(f"    sample {i}: pred ({pred_pos[t,i,0]:.3f}°, {pred_pos[t,i,1]:.3f}°)  "
              f"true ({true_pos[t,i,0]:.3f}°, {true_pos[t,i,1]:.3f}°)  err {err:.0f}m")
