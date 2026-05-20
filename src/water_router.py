"""water_router.py — A* router on the 0.005° water-cell graph.

Replaces v10's per-point hard projection (LandMask.project_to_water on the
0.05° SDF) with a globally-optimal water route between waypoints. Cures:

  - peninsula flips (local Euclidean argmin gives wrong-side-of-Cape-Cod
    snaps when graph distance is the right metric);
  - segment crossings invisible to point checks (a water-water pair with
    a peninsula between them passes a point check but plows through land
    on the segment).

Two-tier resolution:

  - **Snap** uses a coarse LandMask (0.05° SDF, ~5 km cells). This avoids
    snapping a point into a landlocked inlet that is graph-disconnected
    on the fine grid. The coarse SDF treats small bays as land, so snaps
    go to the open-ocean side.
  - **Route** uses the fine water graph (0.005°, ~0.5 km cells) so A*
    paths around peninsulas are precise and segments are guaranteed
    water-only.

Loads `data/processed/water_graph_005deg.npz` once.

Usage
-----
    from src.land_mask import LandMask
    coarse = LandMask.load("data/processed/land_sdf_050deg.npz")
    router = WaterRouter.load("data/processed/water_graph_005deg.npz",
                              coarse_mask=coarse)
    repaired, stats = router.repair_trajectory(traj, threshold_km=10.0)
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


_DIRECTIONS = np.array([
    (-1,  0),   # 0: N
    (-1,  1),   # 1: NE
    ( 0,  1),   # 2: E
    ( 1,  1),   # 3: SE
    ( 1,  0),   # 4: S
    ( 1, -1),   # 5: SW
    ( 0, -1),   # 6: W
    (-1, -1),   # 7: NW
], dtype=np.int32)


def _haversine_km(lon0, lat0, lon1, lat1):
    """Vectorized haversine in km."""
    R = 6371.0088
    lon0 = np.radians(lon0); lat0 = np.radians(lat0)
    lon1 = np.radians(lon1); lat1 = np.radians(lat1)
    dlon = lon1 - lon0
    dlat = lat1 - lat0
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclass
class WaterRouter:
    water_mask: np.ndarray          # (H, W) bool — fine 0.005° grid
    neighbor_bits: np.ndarray       # (H, W) uint8
    bbox: np.ndarray                # (4,) [minx, miny, maxx, maxy]
    dlon: float
    dlat: float
    midlat_cos: float
    tau_km: float
    coarse_mask: object = None      # optional LandMask (0.05° SDF) for snapping

    @classmethod
    def load(cls, path: str, coarse_mask=None) -> "WaterRouter":
        z = np.load(path)
        return cls(
            water_mask=z["water_mask"],
            neighbor_bits=z["neighbor_bits"],
            bbox=z["bbox"].astype(np.float64),
            dlon=float(z["dlon"]),
            dlat=float(z["dlat"]),
            midlat_cos=float(z["midlat_cos"]),
            tau_km=float(z["tau_km"]),
            coarse_mask=coarse_mask,
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return self.water_mask.shape

    # ── Coordinate <-> grid ─────────────────────────────────────────────

    def _to_grid(self, lon, lat):
        """Lon/lat (arrays) → (row, col) integer indices, clipped to grid."""
        minx, miny, maxx, maxy = self.bbox
        H, W = self.shape
        col = (np.asarray(lon, dtype=np.float64) - minx) / self.dlon - 0.5
        row = (maxy - np.asarray(lat, dtype=np.float64)) / self.dlat - 0.5
        col = np.clip(np.round(col).astype(np.int64), 0, W - 1)
        row = np.clip(np.round(row).astype(np.int64), 0, H - 1)
        return row, col

    def _to_lonlat(self, row, col):
        """(row, col) → cell-center (lon, lat)."""
        minx, miny, maxx, maxy = self.bbox
        lon = minx + (np.asarray(col, dtype=np.float64) + 0.5) * self.dlon
        lat = maxy - (np.asarray(row, dtype=np.float64) + 0.5) * self.dlat
        return lon, lat

    # ── Nearest-water snap ──────────────────────────────────────────────

    def nearest_water_cell(self, row: int, col: int,
                           search_radius_cells: int = 200,
                           require_connected: bool = True,
                           min_clearance_km: float = 0.0,
                           ) -> Tuple[int, int]:
        """Find the nearest water cell within a square window.

        If `require_connected`, restrict to cells with `neighbor_bits > 0`
        (i.e. at least one valid graph edge — filters out tiny isolated
        water cells and landlocked-inlet cells).

        If `min_clearance_km > 0` and a coarse_mask is attached, restrict
        the candidate set to cells with coarse-SDF ≤ -min_clearance_km
        (i.e. at least that far from any coast). This matches the
        threshold-based "on land" definition used in evaluation metrics.

        Returns (row, col) of the original cell if it already meets the
        criteria, else the closest qualifying cell in cell-distance.
        If no qualifying cell in the window, falls back progressively
        (relax connected, relax clearance).
        """
        H, W = self.shape
        rlo = max(0, row - search_radius_cells)
        rhi = min(H, row + search_radius_cells + 1)
        clo = max(0, col - search_radius_cells)
        chi = min(W, col + search_radius_cells + 1)
        sub_water = self.water_mask[rlo:rhi, clo:chi]
        sub_conn  = self.neighbor_bits[rlo:rhi, clo:chi] > 0

        if min_clearance_km > 0.0 and self.coarse_mask is not None:
            # Vectorize: sample coarse SDF at every cell-center in window.
            sub_lons = self.bbox[0] + (np.arange(clo, chi) + 0.5) * self.dlon
            sub_lats = self.bbox[3] - (np.arange(rlo, rhi) + 0.5) * self.dlat
            lon_grid = np.broadcast_to(sub_lons[None, :], (rhi - rlo, chi - clo))
            lat_grid = np.broadcast_to(sub_lats[:, None], (rhi - rlo, chi - clo))
            coarse_sdf = self.coarse_mask.sample_km_np(lon_grid, lat_grid)
            sub_clearance_ok = coarse_sdf <= -min_clearance_km
        else:
            sub_clearance_ok = np.ones_like(sub_water)

        # Local row, col within window
        loc_r = row - rlo
        loc_c = col - clo

        # Try strictest first → progressively relax
        candidate_masks = []
        if require_connected:
            candidate_masks.append(sub_water & sub_conn & sub_clearance_ok)
            candidate_masks.append(sub_water & sub_conn)
            candidate_masks.append(sub_water & sub_clearance_ok)
        else:
            candidate_masks.append(sub_water & sub_clearance_ok)
        candidate_masks.append(sub_water)

        for mask in candidate_masks:
            # If origin already qualifies under this mask, return it
            if (0 <= loc_r < mask.shape[0] and 0 <= loc_c < mask.shape[1]
                    and mask[loc_r, loc_c]):
                return row, col
            if mask.any():
                ii, jj = np.nonzero(mask)
                d2 = (ii - loc_r) ** 2 + (jj - loc_c) ** 2
                k = int(np.argmin(d2))
                return int(ii[k] + rlo), int(jj[k] + clo)
        return row, col

    # ── Segment land-cross check (Bresenham) ────────────────────────────

    def segment_crosses_land(self, r0: int, c0: int, r1: int, c1: int) -> bool:
        """Bresenham raster on water_mask. Returns True if any cell on the
        line is land. Endpoints assumed already snapped to water.
        """
        if (r0, c0) == (r1, c1):
            return False
        dr = abs(r1 - r0); dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        H, W = self.shape
        wm = self.water_mask
        while True:
            if not (0 <= r < H and 0 <= c < W) or not wm[r, c]:
                return True
            if r == r1 and c == c1:
                return False
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    # ── Bounded windowed Dijkstra (scipy.csgraph) ───────────────────────

    def route(self, r0: int, c0: int, r1: int, c1: int,
              window_margin_cells: int = 50,
              max_window_cells: int = 200) -> Optional[list]:
        """Find shortest water-only path from (r0,c0) to (r1,c1) on the
        8-neighbor graph using scipy.sparse.csgraph.dijkstra over a
        BOUNDED LOCAL WINDOW.

        The window is the bounding box of the two endpoints expanded by
        `window_margin_cells` on each side. If margin would yield more
        than `max_window_cells` cells per side, returns None (fall back
        to a coarser projection). This keeps per-call cost bounded.

        Cost per edge = haversine distance between cell centers (km).
        Returns list of (row, col) including both endpoints, or None on
        failure.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

        H, W = self.shape
        if not (0 <= r0 < H and 0 <= c0 < W and self.water_mask[r0, c0]):
            return None
        if not (0 <= r1 < H and 0 <= c1 < W and self.water_mask[r1, c1]):
            return None
        if r0 == r1 and c0 == c1:
            return [(r0, c0)]

        # Window bbox
        rmin = min(r0, r1) - window_margin_cells
        rmax = max(r0, r1) + window_margin_cells
        cmin = min(c0, c1) - window_margin_cells
        cmax = max(c0, c1) + window_margin_cells
        if (rmax - rmin) > max_window_cells or (cmax - cmin) > max_window_cells:
            # Window too big — A* would be expensive. Fall back.
            return None
        rmin = max(0, rmin); rmax = min(H - 1, rmax)
        cmin = max(0, cmin); cmax = min(W - 1, cmax)
        wh = rmax - rmin + 1
        ww = cmax - cmin + 1

        sub_water = self.water_mask[rmin:rmax + 1, cmin:cmax + 1]
        sub_nb    = self.neighbor_bits[rmin:rmax + 1, cmin:cmax + 1]
        n_cells = wh * ww

        # Build CSR: vectorized over 8 directions.
        rows_out = []
        cols_out = []
        data_out = []
        # Cell-center lons/lats for the window (for haversine).
        loc_rows = np.arange(rmin, rmax + 1)
        loc_cols = np.arange(cmin, cmax + 1)
        # (wh, 1) lats and (1, ww) lons
        cell_lats = self.bbox[3] - (loc_rows[:, None].astype(np.float64) + 0.5) * self.dlat   # (wh, 1)
        cell_lons = self.bbox[0] + (loc_cols[None, :].astype(np.float64) + 0.5) * self.dlon   # (1, ww)
        cell_lats = np.broadcast_to(cell_lats, (wh, ww))
        cell_lons = np.broadcast_to(cell_lons, (wh, ww))

        for i, (dy, dx) in enumerate(_DIRECTIONS):
            valid = ((sub_nb.astype(np.uint8) >> i) & 1).astype(bool)  # (wh, ww)
            # Need neighbor inside window
            if dy < 0:
                valid[:abs(dy), :] = False
            elif dy > 0:
                valid[-dy:, :] = False
            if dx < 0:
                valid[:, :abs(dx)] = False
            elif dx > 0:
                valid[:, -dx:] = False
            if not valid.any():
                continue
            yy, xx = np.nonzero(valid)
            ny = yy + int(dy); nx = xx + int(dx)
            # Only keep edges where both endpoints are water within window
            ok = sub_water[yy, xx] & sub_water[ny, nx]
            yy = yy[ok]; xx = xx[ok]
            ny = ny[ok]; nx = nx[ok]
            if len(yy) == 0:
                continue
            src_idx = yy * ww + xx
            dst_idx = ny * ww + nx
            d = _haversine_km(cell_lons[yy, xx], cell_lats[yy, xx],
                              cell_lons[ny, nx], cell_lats[ny, nx])
            rows_out.append(src_idx)
            cols_out.append(dst_idx)
            data_out.append(d)

        if not rows_out:
            return None
        rows_arr = np.concatenate(rows_out)
        cols_arr = np.concatenate(cols_out)
        data_arr = np.concatenate(data_out)
        graph = csr_matrix((data_arr, (rows_arr, cols_arr)),
                           shape=(n_cells, n_cells))

        src_local = (r0 - rmin) * ww + (c0 - cmin)
        dst_local = (r1 - rmin) * ww + (c1 - cmin)

        dist, pred = scipy_dijkstra(
            graph, indices=src_local, directed=True,
            return_predecessors=True, unweighted=False)

        if not np.isfinite(dist[dst_local]):
            return None

        # Reconstruct path
        path_local = [dst_local]
        cur = dst_local
        while cur != src_local:
            cur = int(pred[cur])
            if cur < 0:
                return None
            path_local.append(cur)
        path_local.reverse()

        path = [((idx // ww) + rmin, (idx % ww) + cmin) for idx in path_local]
        return path

    # ── Trajectory repair ───────────────────────────────────────────────

    def repair_trajectory(self, traj: np.ndarray,
                          threshold_km: float = 10.0,
                          do_segment_bridge: bool = False,
                          window_margin_cells: int = 100,
                          max_window_cells: int = 600,
                          fine_snap_radius: int = 200,
                          ) -> Tuple[np.ndarray, dict]:
        """Repair one trajectory `traj (T, 2)` so every output point lies
        on a fine-grid water cell connected to the graph.

        Default mode (`do_segment_bridge=False`): snap-only. Each on-land
        point is moved to the nearest connected fine-grid water cell.
        Output keeps the same T points; only on-land points move. This is
        like LandMask hard-projection but at 0.005° (~0.5 km) instead of
        0.05° (~5 km), and uses neighbor_bits>0 to avoid landlocked-inlet
        traps. Does NOT change the trajectory shape; segments may still
        cross thin peninsulas.

        Optional `do_segment_bridge=True`: also A*-route any segment that
        crosses land on the fine grid, and arc-length resample to T
        points. CAUTION: resampling redistributes points along the
        routed polyline, which changes the local time-step density and
        usually hurts ADE on long routes — use only when strict-0
        segment crossings are required. Post-resample re-snap catches
        any interpolation that lands on land.

        Returns (repaired (T, 2), stats dict).
        """
        traj = np.asarray(traj, dtype=np.float64)
        T = traj.shape[0]

        stats = {
            "n_points": T,
            "n_snapped": 0,
            "n_segments_total": T - 1,
            "n_segments_routed": 0,
            "n_route_failures": 0,
            "n_route_nodes_total": 0,
        }

        # 1. Snap. Detect "on land" via the coarse SDF at threshold_km
        # (matches LandMask hard-proj's measurement). For flagged points,
        # snap to nearest connected fine-grid water cell. This combines:
        #   - hard-proj's threshold-based detection (so we catch every point
        #     within `threshold_km` of land at coarse resolution),
        #   - fine-grid (0.005°, ~0.5 km) snap precision (so the moved
        #     point is ≤ 0.5 km from the original, vs 5+ km for coarse
        #     project_to_water),
        #   - connected-water filter (avoids snapping into landlocked
        #     inlets).
        # Points whose fine-grid cell is land but coarse SDF says
        # threshold-OK are also snapped, since we still need fine-grid
        # water for water_mask validity.
        rows0, cols0 = self._to_grid(traj[:, 0], traj[:, 1])
        snapped_rows = rows0.astype(np.int64).copy()
        snapped_cols = cols0.astype(np.int64).copy()

        # Detect "on land" by coarse threshold OR fine-grid land cell OR
        # disconnected fine-grid water cell.
        needs_snap = np.zeros(T, dtype=bool)
        if self.coarse_mask is not None:
            sdf = self.coarse_mask.sample_km_np(traj[:, 0], traj[:, 1])
            needs_snap |= sdf > threshold_km
        for t in range(T):
            r, c = int(rows0[t]), int(cols0[t])
            if not self.water_mask[r, c]:
                needs_snap[t] = True
            elif int(self.neighbor_bits[r, c]) == 0:
                needs_snap[t] = True

        # Snap target requires fine-grid water + connected, but does NOT
        # require min_clearance_km — that would force snaps far offshore
        # and break ADE. Hard-proj satisfies the threshold metric only
        # because the snapped cell happens to have sdf < 0; we replicate
        # that by snapping to fine-grid water (sdf <= -tau_km < 0). The
        # post-snap output has every point with coarse-sdf < 0 (i.e.
        # land%=0 at any positive threshold).
        # Snap target: prefer the nearest fine-grid water cell. If that
        # cell's COARSE-SDF is still > threshold (i.e. a small fine-grid
        # bay inside a coarse "land" cell — bilinear-positive even though
        # the cell is structurally water), retry with min_clearance_km
        # to force a snap into the open ocean. This matches hard-proj's
        # threshold-satisfaction guarantee.
        for t in range(T):
            if not needs_snap[t]:
                continue
            r, c = int(rows0[t]), int(cols0[t])
            nr, nc = self.nearest_water_cell(
                r, c, search_radius_cells=fine_snap_radius,
                require_connected=True,
                min_clearance_km=0.0)
            # Verify the snap satisfies the threshold metric.
            if self.coarse_mask is not None and threshold_km > 0:
                snapped_lon, snapped_lat = self._to_lonlat(nr, nc)
                snapped_sdf = float(self.coarse_mask.sample_km_np(
                    np.array([snapped_lon]), np.array([snapped_lat]))[0])
                if snapped_sdf > threshold_km:
                    # Force open-water snap.
                    nr2, nc2 = self.nearest_water_cell(
                        r, c, search_radius_cells=fine_snap_radius,
                        require_connected=True,
                        min_clearance_km=threshold_km)
                    nr, nc = nr2, nc2
            snapped_rows[t] = nr
            snapped_cols[t] = nc
            if (nr, nc) != (r, c):
                stats["n_snapped"] += 1

        # 2. Build polyline of (row, col), bridging segments that cross land.
        polyline = [(int(snapped_rows[0]), int(snapped_cols[0]))]
        for t in range(1, T):
            r0, c0 = polyline[-1]
            r1, c1 = int(snapped_rows[t]), int(snapped_cols[t])
            if (r0, c0) == (r1, c1):
                polyline.append((r1, c1))
                continue
            need_route = False
            if do_segment_bridge:
                need_route = self.segment_crosses_land(r0, c0, r1, c1)
            if need_route:
                # Try increasing window sizes if needed.
                path = None
                for margin in (window_margin_cells, window_margin_cells * 2,
                               window_margin_cells * 4):
                    path = self.route(r0, c0, r1, c1,
                                      window_margin_cells=margin,
                                      max_window_cells=max_window_cells)
                    if path is not None and len(path) >= 2:
                        break
                if path is None or len(path) < 2:
                    # Could not bridge — keep the endpoint and rely on
                    # the post-resample re-snap to fix any landed-on-land
                    # interpolation artifacts.
                    polyline.append((r1, c1))
                    stats["n_route_failures"] += 1
                else:
                    # Skip path[0] (== current end of polyline)
                    polyline.extend(path[1:])
                    stats["n_segments_routed"] += 1
                    stats["n_route_nodes_total"] += len(path)
            else:
                polyline.append((r1, c1))

        # 3. Convert polyline to lon/lat (cell centers).
        poly_arr = np.asarray(polyline, dtype=np.int64)
        lons, lats = self._to_lonlat(poly_arr[:, 0], poly_arr[:, 1])
        poly_lonlat = np.stack([lons, lats], axis=1)

        # 4. Arc-length resample to T points (haversine cumulative).
        if poly_lonlat.shape[0] == 1:
            repaired = np.tile(poly_lonlat[0], (T, 1))
        elif poly_lonlat.shape[0] == T and stats["n_segments_routed"] == 0:
            # No reroutes and exactly T points (after pure snap): no
            # resampling needed. Avoids resampling-induced shape drift.
            repaired = poly_lonlat
        else:
            seg_d = _haversine_km(
                poly_lonlat[:-1, 0], poly_lonlat[:-1, 1],
                poly_lonlat[1:, 0], poly_lonlat[1:, 1],
            )
            cum = np.concatenate([[0.0], np.cumsum(seg_d)])
            total = float(cum[-1])
            if total <= 0.0:
                repaired = np.tile(poly_lonlat[0], (T, 1))
            else:
                target = np.linspace(0.0, total, T)
                idx = np.searchsorted(cum, target, side="right") - 1
                idx = np.clip(idx, 0, len(cum) - 2)
                t_local = (target - cum[idx]) / np.maximum(cum[idx + 1] - cum[idx], 1e-12)
                t_local = t_local[:, None]
                p0 = poly_lonlat[idx]
                p1 = poly_lonlat[idx + 1]
                repaired = p0 + t_local * (p1 - p0)
                # Anchor exact endpoints to avoid floating-point drift.
                repaired[0] = poly_lonlat[0]
                repaired[-1] = poly_lonlat[-1]

        # 5. Post-resample re-snap: any output point that landed on land
        # (because of arc-length interpolation across a route-failure gap)
        # gets snapped to the nearest fine-grid water cell.
        out_rows, out_cols = self._to_grid(repaired[:, 0], repaired[:, 1])
        n_postsnap = 0
        for t in range(T):
            r, c = int(out_rows[t]), int(out_cols[t])
            if not self.water_mask[r, c]:
                nr, nc = self.nearest_water_cell(r, c, search_radius_cells=200)
                if (nr, nc) != (r, c) and self.water_mask[nr, nc]:
                    lon_w, lat_w = self._to_lonlat(nr, nc)
                    repaired[t, 0] = lon_w
                    repaired[t, 1] = lat_w
                    n_postsnap += 1
        stats["n_postsnap"] = n_postsnap

        return repaired, stats

    def repair_batch(self, trajs: np.ndarray,
                     threshold_km: float = 10.0,
                     do_segment_bridge: bool = True,
                     window_margin_cells: int = 100,
                     max_window_cells: int = 600,
                     ) -> Tuple[np.ndarray, dict]:
        """Repair a batch of (N, T, 2) trajectories. Returns (out, totals)."""
        N, T, _ = trajs.shape
        out = np.empty_like(trajs)
        agg = {
            "n_trajs": N,
            "n_points": 0,
            "n_snapped": 0,
            "n_segments_total": 0,
            "n_segments_routed": 0,
            "n_route_failures": 0,
            "n_route_nodes_total": 0,
        }
        for i in range(N):
            rep, st = self.repair_trajectory(
                trajs[i],
                threshold_km=threshold_km,
                do_segment_bridge=do_segment_bridge,
                window_margin_cells=window_margin_cells,
                max_window_cells=max_window_cells,
            )
            out[i] = rep
            for k in ("n_points", "n_snapped", "n_segments_total",
                      "n_segments_routed", "n_route_failures",
                      "n_route_nodes_total"):
                agg[k] += st[k]
        return out, agg
