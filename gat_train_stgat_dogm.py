#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST-GAT Uncertainty Trainer + Full-Point Inference Exporter (for DOGM)

What this script provides
-------------------------
1) Training: learns per-point uncertainty for LiDAR position (sigma_x, sigma_y) and Radar radial-velocity (sigma_v).
   - Uses a spatiotemporal graph built over a 4-frame window (WINDOW=4).
   - Supervision: compares nodes at the last window frame ("current") against the next frame GT (one-step).
     This yields uncertainty attached to CURRENT-frame measurements, which you can inject into DOGM at the same time t.

2) Inference (export): runs the trained network on ALL points (no downsampling) and exports per-frame sigma:
   - LiDAR: [x, y, sigma_x, sigma_y] for every LiDAR point at frame t
   - Radar1/Radar2: [x, y, v_r, sigma_v, snr] for every radar point at frame t
   Output is saved as NPZ with flattened arrays + per-frame offsets (DOGM-friendly).

Graph acceleration
------------------
- If torch_cluster is available (PyTorch Geometric component), kNN edges are built using torch_cluster (fast).
- Otherwise, falls back to torch.cdist (slower, O(N^2)).

DOGM injection (high-level)
---------------------------
For each frame t:
- LiDAR position measurement noise:
    R_pos = diag([sigma_x^2, sigma_y^2]) per point (or map to cell-level noise via aggregation).
- Radar velocity measurement noise:
    R_v = sigma_v^2 per radar point (or per associated track / cell).

You usually aggregate per-point noise into a cell-level measurement noise by:
- taking min/median over points that fall into the same occupancy grid cell,
- or using inverse-variance pooling: sigma_cell^{-2} = sum_i sigma_i^{-2}.

Files expected per run v under --data_root:
  LiDARMap_BaseScan_v{v}.txt  : [t, x, y, intensity]  (4 cols)
  Radar1Map_BaseScan_v{v}.txt : [t, x, y, v_r, snr]    (5 cols)
  Radar2Map_BaseScan_v{v}.txt : [t, x, y, v_r, snr]    (5 cols)
  odom_filtered_v{v}.txt      : [t, x, y, yaw, v, w]   (6 cols)

Example usage
-------------
# (A) Train (v1~v3 train, v4 val):
python gat_train_stgat_dogm.py --task train --data_root /path/to/dataset --mode train --batch 2 --num_workers 4

# (B) Debug overfit (v1 only):
python gat_train_stgat_dogm.py --task train --data_root /path/to/dataset --mode debug --batch 2 --max_windows_per_run 800

# (C) Inference export (full points) on run v1 using best_ckpt.pt:
python gat_train_stgat_dogm.py --task infer --data_root /path/to/dataset --ckpt best_ckpt.pt --infer_version 1 --infer_out sigma_v1.npz --infer_full_points

Notes
-----
- Your current ModelConfig modifications are reflected as defaults (K_LIDAR_SPATIAL=32, CROSS_RADIUS=0.5, ASSOC_GATE_* tightened).
- You can override caps/epochs/batch via CLI.

"""

from __future__ import annotations

import argparse
import math
import os
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional fast kNN backend
_HAS_TORCH_CLUSTER = False
try:
    from torch_cluster import knn_graph, knn  # type: ignore
    _HAS_TORCH_CLUSTER = True
except Exception:
    _HAS_TORCH_CLUSTER = False

# Optional scatter softmax backend
_HAS_SCATTER = False
try:
    from torch_scatter import scatter_max, scatter_sum  # type: ignore
    _HAS_SCATTER = True
except Exception:
    _HAS_SCATTER = False


# =============================================================================
# Utils
# =============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Edge-wise softmax over incoming edges per destination node (dst).

    scores: (E, H) or (E,)
    index : (E,) dst indices
    """
    if scores.numel() == 0:
        return scores

    if scores.dim() == 1:
        scores = scores.unsqueeze(-1)

    if _HAS_SCATTER:
        m, _ = scatter_max(scores, index, dim=0, dim_size=num_nodes)
        scores2 = scores - m[index]
        exp = torch.exp(scores2)
        denom = scatter_sum(exp, index, dim=0, dim_size=num_nodes)
        out = exp / (denom[index] + 1e-12)
        return out.squeeze(-1) if out.size(1) == 1 else out

    out = torch.empty_like(scores)
    for n in torch.unique(index).tolist():
        mask = (index == n)
        s = scores[mask]
        s = s - s.max(dim=0, keepdim=True).values
        e = torch.exp(s)
        out[mask] = e / (e.sum(dim=0, keepdim=True) + 1e-12)
    return out.squeeze(-1) if out.size(1) == 1 else out


def se2_inv(xytheta: torch.Tensor) -> torch.Tensor:
    x, y, th = xytheta
    c, s = torch.cos(th), torch.sin(th)
    xi = -(c * x + s * y)
    yi = -(-s * x + c * y)
    thi = -th
    return torch.stack([xi, yi, thi])


def se2_apply(xytheta: torch.Tensor, pts_xy: torch.Tensor) -> torch.Tensor:
    x, y, th = xytheta
    c, s = torch.cos(th), torch.sin(th)
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])  # (2,2)
    return (pts_xy @ R.T) + torch.stack([x, y])


def warp_points_to_frame(
    pts_xy: torch.Tensor,
    pose_src: torch.Tensor,
    pose_dst: torch.Tensor,
) -> torch.Tensor:
    """pts_xy in src base frame -> dst base frame, using base->world poses."""
    world = se2_apply(pose_src, pts_xy)
    dst_inv = se2_inv(pose_dst)
    return se2_apply(dst_inv, world)


def _edge_attr(src_xy: torch.Tensor, dst_xy: torch.Tensor, dt_edge: torch.Tensor) -> torch.Tensor:
    dxy = src_xy - dst_xy
    dist = torch.sqrt((dxy ** 2).sum(dim=1) + 1e-12)
    return torch.cat([dxy, dist.unsqueeze(1), dt_edge.unsqueeze(1)], dim=1)


def maybe_dropout_edges(edge_index: torch.Tensor, edge_attr: torch.Tensor, p: float, training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if (not training) or p <= 0.0 or edge_index.numel() == 0:
        return edge_index, edge_attr
    E = edge_index.size(1)
    keep = torch.rand(E, device=edge_index.device) > p
    return edge_index[:, keep], edge_attr[keep]


def _extract_zip_if_needed(data_zip: Optional[str], data_root: str) -> str:
    if data_zip is None:
        return data_root
    data_root = str(Path(data_root).resolve())
    p = Path(data_root)
    if p.exists() and any(p.glob("*.txt")):
        return data_root
    p.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data_zip, "r") as z:
        z.extractall(p)
    subs = [d for d in p.iterdir() if d.is_dir()]
    if len(subs) == 1:
        return str(subs[0].resolve())
    return data_root


def _read_txt_np(path: Path, expected_cols: int, dtype=np.float32) -> np.ndarray:
    arr = np.loadtxt(str(path), dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != expected_cols:
        raise ValueError(f"Unexpected columns in {path.name}: got {arr.shape[1]}, expected {expected_cols}")
    return arr


def _assign_to_odom_frames(t_pts: np.ndarray, t_odom: np.ndarray) -> np.ndarray:
    F = t_odom.shape[0]
    idx = np.searchsorted(t_odom, t_pts, side="left")
    idx1 = np.clip(idx, 0, F - 1)
    idx0 = np.clip(idx - 1, 0, F - 1)
    d0 = np.abs(t_pts - t_odom[idx0])
    d1 = np.abs(t_pts - t_odom[idx1])
    use0 = d0 <= d1
    out = np.where(use0, idx0, idx1)
    return out.astype(np.int64)


def _sort_by_frame(frame_idx: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(frame_idx, kind="mergesort")
    frame_sorted = frame_idx[order]
    data_sorted = data[order]
    F = int(frame_sorted.max()) + 1 if frame_sorted.size > 0 else 0
    counts = np.bincount(frame_sorted, minlength=F)
    offsets = np.zeros(F + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return data_sorted, offsets


# =============================================================================
# Config (user-modified defaults)
# =============================================================================

@dataclass
class ModelConfig:
    # Unified raw vector spec (11-D)
    INPUT_DIM: int = 11
    IDX_POS: Tuple[int, int] = (0, 1)
    IDX_DT: int = 2
    IDX_EGO: Tuple[int, int] = (3, 4)
    IDX_INTENSITY: int = 5
    IDX_VR: int = 6
    IDX_SNR: int = 7
    IDX_SID: Tuple[int, int, int] = (8, 9, 10)

    # Window / frames
    WINDOW: int = 4
    DT_DEFAULT: float = 0.1

    # Downsample caps (TRAINING); inference can disable
    LIDAR_CAP_PER_FRAME: int = 512
    RADAR_CAP_PER_FRAME: int = 128

    # Graph construction
    K_LIDAR_SPATIAL: int = 32
    K_RADAR_SPATIAL: int = 8
    K_TEMPORAL: int = 8
    TEMPORAL_ADJ_ONLY: bool = True

    # Cross edges
    CROSS_RADIUS: float = 0.5
    K_CROSS_L2R: int = 4
    K_CROSS_R2L: int = 8
    CROSS_DROPOUT: float = 0.2

    # Model
    HIDDEN_DIM: int = 64
    NUM_HEADS: int = 4
    DROPOUT: float = 0.1
    EDGE_DIM: int = 4

    # Sigma constraints
    MIN_LOG_SIGMA: float = -5.0
    MAX_LOG_SIGMA: float = 2.0

    # Loss
    REG_LAMBDA: float = 1e-3
    RADAR_LOSS_WEIGHT: float = 5.0
    ASSOC_TOPK: int = 5
    ASSOC_TAU: float = 0.5
    ASSOC_GATE_LIDAR: float = 0.15
    ASSOC_GATE_RADAR: float = 0.3

    # Cross residual schedule
    CROSS_ALPHA_MAX: float = 0.3
    CROSS_WARMUP_STEPS: int = 2000
    CROSS_RAMP_STEPS: int = 8000

    # Training defaults
    BATCH_SIZE: int = 2
    LR: float = 3e-4
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 30
    GRAD_CLIP: float = 1.0
    AMP: bool = False


# =============================================================================
# Graph Builder (torch_cluster if available)
# =============================================================================

class GraphBuilder:
    """
    Build edges for flattened nodes across a batch.

    Inputs per batch:
      x:        (N, 11)
      frame_id: (N,) in [0..WINDOW-1]
      batch_id: (N,)
      sensor_id:(N,) {0:LiDAR, 1:Radar1, 2:Radar2}
      pose_by_frame: (B, WINDOW or WINDOW+1, 3) base->world poses for window frames (and optional next)
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    @torch.no_grad()
    def _knn_within(self, pts: torch.Tensor, k: int) -> torch.Tensor:
        """Return edge_index (2,E) for kNN within one set."""
        M = pts.size(0)
        if M == 0 or k <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=pts.device)

        kk = min(k, max(M - 1, 1))
        if _HAS_TORCH_CLUSTER:
            # knn_graph returns edges as [src, dst] (neighbor -> query) for flow='source_to_target'
            return knn_graph(pts, k=kk, loop=False, flow='source_to_target')
        # fallback: cdist
        d = torch.cdist(pts, pts)
        d.fill_diagonal_(1e9)
        nn_idx = torch.topk(d, k=kk, largest=False, dim=1).indices
        dst = torch.arange(M, device=pts.device).unsqueeze(1).expand(M, kk).reshape(-1)
        src = nn_idx.reshape(-1)
        return torch.stack([src, dst], dim=0)

    @torch.no_grad()
    def _knn_cross(self, src_pts: torch.Tensor, dst_pts: torch.Tensor, k_per_dst: int) -> torch.Tensor:
        """
        Return pairs for kNN from src_pts to each dst_pt:
          edge_index (2,E) with indices in src/dst LOCAL coordinates: [src_idx, dst_idx]
        """
        Ns = src_pts.size(0)
        Nd = dst_pts.size(0)
        if Ns == 0 or Nd == 0 or k_per_dst <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=dst_pts.device)

        kk = min(k_per_dst, Ns)
        if _HAS_TORCH_CLUSTER:
            # torch_cluster.knn(x, y, k) returns [y_index, x_index] (query indices, neighbor indices)
            pair = knn(src_pts, dst_pts, k=kk)  # (2,E) : [dst, src]
            dst = pair[0]
            src = pair[1]
            return torch.stack([src, dst], dim=0)

        # fallback: cdist
        d = torch.cdist(dst_pts, src_pts)  # (Nd, Ns)
        nn = torch.topk(d, k=kk, largest=False, dim=1).indices  # (Nd,kk)
        dst = torch.arange(Nd, device=dst_pts.device).unsqueeze(1).expand(Nd, kk).reshape(-1)
        src = nn.reshape(-1)
        return torch.stack([src, dst], dim=0)

    @torch.no_grad()
    def build(
        self,
        x: torch.Tensor,
        frame_id: torch.Tensor,
        batch_id: torch.Tensor,
        sensor_id: torch.Tensor,
        pose_by_frame: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        device = x.device
        cfg = self.cfg

        edges: Dict[str, Dict[str, torch.Tensor]] = {}
        def init_rel(rel: str):
            edges[rel] = {
                "edge_index": torch.empty((2, 0), dtype=torch.long, device=device),
                "edge_attr": torch.empty((0, cfg.EDGE_DIM), dtype=torch.float32, device=device),
            }
        for rel in ["LL", "R1R1", "R2R2", "L2R1", "R12L", "L2R2", "R22L"]:
            init_rel(rel)

        if batch_id.numel() == 0:
            return edges
        B = int(batch_id.max().item()) + 1

        for b in range(B):
            bmask = (batch_id == b)
            if not bmask.any():
                continue
            idx_b = torch.where(bmask)[0]
            xb = x[idx_b]
            fb = frame_id[idx_b]
            sb = sensor_id[idx_b]

            # Spatial edges within each frame, per sensor
            for f in range(cfg.WINDOW):
                fmask = (fb == f)
                if not fmask.any():
                    continue
                idx_f = torch.where(fmask)[0]  # local indices in idx_b

                for sid, rel, k in [(0, "LL", cfg.K_LIDAR_SPATIAL), (1, "R1R1", cfg.K_RADAR_SPATIAL), (2, "R2R2", cfg.K_RADAR_SPATIAL)]:
                    sm = (sb[idx_f] == sid)
                    if not sm.any():
                        continue
                    loc = idx_f[sm]
                    pts = xb[loc][:, list(cfg.IDX_POS)]
                    ei_local = self._knn_within(pts, k)
                    if ei_local.numel() == 0:
                        continue
                    src_loc = loc[ei_local[0]]
                    dst_loc = loc[ei_local[1]]
                    dt_edge = torch.zeros(src_loc.size(0), device=device)
                    eattr = _edge_attr(xb[src_loc][:, list(cfg.IDX_POS)], xb[dst_loc][:, list(cfg.IDX_POS)], dt_edge)
                    edges[rel]["edge_index"] = torch.cat([edges[rel]["edge_index"], torch.stack([idx_b[src_loc], idx_b[dst_loc]], dim=0)], dim=1)
                    edges[rel]["edge_attr"] = torch.cat([edges[rel]["edge_attr"], eattr], dim=0)

            # Temporal edges between adjacent frames only: (0->1), (1->2), (2->3)
            if cfg.TEMPORAL_ADJ_ONLY:
                pairs = [(0, 1), (1, 2), (2, 3)]
            else:
                pairs = [(i, j) for i in range(cfg.WINDOW) for j in range(cfg.WINDOW) if i < j]

            # Pose availability check
            has_pose = pose_by_frame is not None and pose_by_frame.dim() == 3 and pose_by_frame.size(0) > b and pose_by_frame.size(1) >= cfg.WINDOW

            for f0, f1 in pairs:
                idx0 = torch.where(fb == f0)[0]
                idx1 = torch.where(fb == f1)[0]
                if idx0.numel() == 0 or idx1.numel() == 0:
                    continue

                pose0 = pose1 = None
                if has_pose:
                    pose0 = pose_by_frame[b, f0]
                    pose1 = pose_by_frame[b, f1]

                for sid, rel in [(0, "LL"), (1, "R1R1"), (2, "R2R2")]:
                    a0 = idx0[sb[idx0] == sid]
                    a1 = idx1[sb[idx1] == sid]
                    if a0.numel() == 0 or a1.numel() == 0:
                        continue

                    pts0 = xb[a0][:, list(cfg.IDX_POS)]
                    pts1 = xb[a1][:, list(cfg.IDX_POS)]

                    # Warp src (f0) points into dst (f1) frame before kNN
                    pts0_warp = pts0
                    if pose0 is not None and pose1 is not None:
                        pts0_warp = warp_points_to_frame(pts0, pose_src=pose0, pose_dst=pose1)

                    # kNN from src-set to each dst point: gives edges src->dst
                    ei = self._knn_cross(pts0_warp, pts1, cfg.K_TEMPORAL)  # local src in a0, local dst in a1
                    if ei.numel() == 0:
                        continue

                    src = a0[ei[0]]
                    dst = a1[ei[1]]

                    # dt_edge: |dt_src - dt_dst| is the frame-to-frame time gap
                    dt_edge = (xb[src, cfg.IDX_DT] - xb[dst, cfg.IDX_DT]).abs()

                    src_xy = pts0_warp[ei[0]]
                    dst_xy = pts1[ei[1]]
                    eattr = _edge_attr(src_xy, dst_xy, dt_edge)

                    edges[rel]["edge_index"] = torch.cat([edges[rel]["edge_index"], torch.stack([idx_b[src], idx_b[dst]], dim=0)], dim=1)
                    edges[rel]["edge_attr"] = torch.cat([edges[rel]["edge_attr"], eattr], dim=0)

            # Cross edges: only on last frame in the window (current)
            f = cfg.WINDOW - 1
            idx_f = torch.where(fb == f)[0]
            if idx_f.numel() > 0:
                l = idx_f[sb[idx_f] == 0]
                r1 = idx_f[sb[idx_f] == 1]
                r2 = idx_f[sb[idx_f] == 2]

                def add_cross(rel: str, src_loc: torch.Tensor, dst_loc: torch.Tensor, k: int):
                    if src_loc.numel() == 0 or dst_loc.numel() == 0 or k <= 0:
                        return
                    ptsS = xb[src_loc][:, list(cfg.IDX_POS)]
                    ptsD = xb[dst_loc][:, list(cfg.IDX_POS)]
                    ei = self._knn_cross(ptsS, ptsD, k)  # [src, dst] local indices
                    if ei.numel() == 0:
                        return
                    # radius gate (post-filter)
                    src_xy = ptsS[ei[0]]
                    dst_xy = ptsD[ei[1]]
                    dist = torch.sqrt(((src_xy - dst_xy) ** 2).sum(dim=1) + 1e-12)
                    keep = dist <= cfg.CROSS_RADIUS
                    if not keep.any():
                        return
                    ei = ei[:, keep]
                    src_xy = src_xy[keep]
                    dst_xy = dst_xy[keep]

                    src = src_loc[ei[0]]
                    dst = dst_loc[ei[1]]
                    dt_edge = torch.zeros(src.size(0), device=device)
                    eattr = _edge_attr(src_xy, dst_xy, dt_edge)
                    edges[rel]["edge_index"] = torch.cat([edges[rel]["edge_index"], torch.stack([idx_b[src], idx_b[dst]], dim=0)], dim=1)
                    edges[rel]["edge_attr"] = torch.cat([edges[rel]["edge_attr"], eattr], dim=0)

                # LiDAR <-> Radar1
                add_cross("L2R1", l, r1, cfg.K_CROSS_L2R)
                add_cross("R12L", r1, l, cfg.K_CROSS_R2L)
                # LiDAR <-> Radar2
                add_cross("L2R2", l, r2, cfg.K_CROSS_L2R)
                add_cross("R22L", r2, l, cfg.K_CROSS_R2L)

        return edges


# =============================================================================
# Edge-aware Relational GAT
# =============================================================================

class EdgeAwareGAT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, heads: int, dropout: float, concat: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.lin_node = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.lin_edge = nn.Linear(edge_dim, out_dim * heads, bias=False)

        self.att = nn.Parameter(torch.empty((heads, 3 * out_dim)))
        nn.init.xavier_uniform_(self.att)

        self.leaky = nn.LeakyReLU(0.2)
        self.bn = nn.LayerNorm(out_dim * heads if concat else out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        if edge_index.numel() == 0:
            h = self.lin_node(x).view(N, self.heads, self.out_dim)
            out = h.reshape(N, self.heads * self.out_dim) if self.concat else h.mean(dim=1)
            return self.bn(out)

        src, dst = edge_index[0], edge_index[1]
        h = self.lin_node(x).view(N, self.heads, self.out_dim)
        e = self.lin_edge(edge_attr).view(-1, self.heads, self.out_dim)

        hs = h[src]
        hd = h[dst]
        cat = torch.cat([hs, hd, e], dim=-1)
        logits = (cat * self.att.unsqueeze(0)).sum(dim=-1)
        logits = self.leaky(logits)

        alpha = safe_softmax(logits, dst, num_nodes=N)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = (hs + e) * alpha.unsqueeze(-1)

        out = torch.zeros((N, self.heads, self.out_dim), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)

        if self.concat:
            out = out.reshape(N, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)
        return self.bn(out)


class RelationalGATBlock(nn.Module):
    def __init__(self, dim: int, edge_dim: int, heads: int, dropout: float):
        super().__init__()
        self.gat1 = EdgeAwareGAT(dim, dim, edge_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = EdgeAwareGAT(dim * heads, dim, edge_dim, heads=heads, dropout=dropout, concat=True)
        self.proj = nn.Linear(dim, dim * heads)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x0 = self.proj(x)
        h = self.gat1(x, edge_index, edge_attr)
        h = F.elu(h)
        h = self.drop(h)
        h2 = self.gat2(h, edge_index, edge_attr)
        h2 = self.drop(h2)
        return x0 + h2


# =============================================================================
# Model
# =============================================================================

class DOGMSTGATUncertaintyNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.enc_lidar = nn.Sequential(nn.Linear(cfg.INPUT_DIM, cfg.HIDDEN_DIM), nn.ReLU(), nn.LayerNorm(cfg.HIDDEN_DIM))
        self.enc_r1    = nn.Sequential(nn.Linear(cfg.INPUT_DIM, cfg.HIDDEN_DIM), nn.ReLU(), nn.LayerNorm(cfg.HIDDEN_DIM))
        self.enc_r2    = nn.Sequential(nn.Linear(cfg.INPUT_DIM, cfg.HIDDEN_DIM), nn.ReLU(), nn.LayerNorm(cfg.HIDDEN_DIM))

        self.st1_LL = RelationalGATBlock(cfg.HIDDEN_DIM, cfg.EDGE_DIM, cfg.NUM_HEADS, cfg.DROPOUT)
        self.st1_R1 = RelationalGATBlock(cfg.HIDDEN_DIM, cfg.EDGE_DIM, cfg.NUM_HEADS, cfg.DROPOUT)
        self.st1_R2 = RelationalGATBlock(cfg.HIDDEN_DIM, cfg.EDGE_DIM, cfg.NUM_HEADS, cfg.DROPOUT)

        self.st2_L2R1 = RelationalGATBlock(cfg.HIDDEN_DIM, cfg.EDGE_DIM, cfg.NUM_HEADS, cfg.DROPOUT)
        self.st2_R12L = RelationalGATBlock(cfg.HIDDEN_DIM, cfg.EDGE_DIM, cfg.NUM_HEADS, cfg.DROPOUT)
        self.st2_L2R2 = RelationalGATBlock(cfg.HIDDEN_DIM, cfg.EDGE_DIM, cfg.NUM_HEADS, cfg.DROPOUT)
        self.st2_R22L = RelationalGATBlock(cfg.HIDDEN_DIM, cfg.EDGE_DIM, cfg.NUM_HEADS, cfg.DROPOUT)

        self.adap_back = nn.Linear(cfg.HIDDEN_DIM * cfg.NUM_HEADS, cfg.HIDDEN_DIM)

        out_dim = cfg.HIDDEN_DIM * cfg.NUM_HEADS
        self.head_lidar = nn.Sequential(nn.Linear(out_dim, 128), nn.ReLU(), nn.Dropout(cfg.DROPOUT), nn.Linear(128, 4))
        self.head_r1    = nn.Sequential(nn.Linear(out_dim, 128), nn.ReLU(), nn.Dropout(cfg.DROPOUT), nn.Linear(128, 2))
        self.head_r2    = nn.Sequential(nn.Linear(out_dim, 128), nn.ReLU(), nn.Dropout(cfg.DROPOUT), nn.Linear(128, 2))

    def cross_alpha(self, global_step: int) -> float:
        cfg = self.cfg
        if global_step < cfg.CROSS_WARMUP_STEPS:
            return 0.0
        t = global_step - cfg.CROSS_WARMUP_STEPS
        if t >= cfg.CROSS_RAMP_STEPS:
            return cfg.CROSS_ALPHA_MAX
        return cfg.CROSS_ALPHA_MAX * (t / max(cfg.CROSS_RAMP_STEPS, 1))

    def forward(
        self,
        x: torch.Tensor,
        frame_id: torch.Tensor,
        batch_id: torch.Tensor,
        sensor_id: torch.Tensor,
        edges: Dict[str, Dict[str, torch.Tensor]],
        global_step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        N = x.size(0)
        device = x.device

        h = torch.zeros((N, cfg.HIDDEN_DIM), device=device, dtype=x.dtype)
        mL  = (sensor_id == 0)
        mR1 = (sensor_id == 1)
        mR2 = (sensor_id == 2)
        if mL.any():
            h[mL]  = self.enc_lidar(x[mL])
        if mR1.any():
            h[mR1] = self.enc_r1(x[mR1])
        if mR2.any():
            h[mR2] = self.enc_r2(x[mR2])

        # Stage1: intra-sensor ST graph
        h_st1 = torch.zeros((N, cfg.HIDDEN_DIM * cfg.NUM_HEADS), device=device, dtype=x.dtype)
        hL = self.st1_LL(h, edges["LL"]["edge_index"], edges["LL"]["edge_attr"]) if mL.any() else h_st1
        h1 = self.st1_R1(h, edges["R1R1"]["edge_index"], edges["R1R1"]["edge_attr"]) if mR1.any() else h_st1
        h2 = self.st1_R2(h, edges["R2R2"]["edge_index"], edges["R2R2"]["edge_attr"]) if mR2.any() else h_st1
        h_st1[mL]  = hL[mL]
        h_st1[mR1] = h1[mR1]
        h_st1[mR2] = h2[mR2]

        # Stage2: cross-sensor residual (only affects last window frame nodes through cross edges)
        alpha = self.cross_alpha(global_step)

        eL2R1_i, eL2R1_a = maybe_dropout_edges(edges["L2R1"]["edge_index"], edges["L2R1"]["edge_attr"], cfg.CROSS_DROPOUT, self.training)
        eR12L_i, eR12L_a = maybe_dropout_edges(edges["R12L"]["edge_index"], edges["R12L"]["edge_attr"], cfg.CROSS_DROPOUT, self.training)
        eL2R2_i, eL2R2_a = maybe_dropout_edges(edges["L2R2"]["edge_index"], edges["L2R2"]["edge_attr"], cfg.CROSS_DROPOUT, self.training)
        eR22L_i, eR22L_a = maybe_dropout_edges(edges["R22L"]["edge_index"], edges["R22L"]["edge_attr"], cfg.CROSS_DROPOUT, self.training)

        h_base = self.adap_back(h_st1)
        h_cross = torch.zeros_like(h_st1)

        if alpha > 0.0:
            if eL2R1_i.numel() > 0:
                upd = self.st2_L2R1(h_base, eL2R1_i, eL2R1_a)
                h_cross[mR1] += upd[mR1]
            if eR12L_i.numel() > 0:
                upd = self.st2_R12L(h_base, eR12L_i, eR12L_a)
                h_cross[mL] += upd[mL]
            if eL2R2_i.numel() > 0:
                upd = self.st2_L2R2(h_base, eL2R2_i, eL2R2_a)
                h_cross[mR2] += upd[mR2]
            if eR22L_i.numel() > 0:
                upd = self.st2_R22L(h_base, eR22L_i, eR22L_a)
                h_cross[mL] += upd[mL]

        h_final = h_st1 + (alpha * h_cross)

        # Heads
        lidar_out = torch.zeros((N, 4), device=device, dtype=h_final.dtype)
        r1_out    = torch.zeros((N, 2), device=device, dtype=h_final.dtype)
        r2_out    = torch.zeros((N, 2), device=device, dtype=h_final.dtype)
        if mL.any():
            lidar_out[mL] = self.head_lidar(h_final[mL])
        if mR1.any():
            r1_out[mR1] = self.head_r1(h_final[mR1])
        if mR2.any():
            r2_out[mR2] = self.head_r2(h_final[mR2])

        lidar_mu = lidar_out[:, 0:2]
        lidar_log_sigma = torch.clamp(lidar_out[:, 2:4], cfg.MIN_LOG_SIGMA, cfg.MAX_LOG_SIGMA)

        r1_mu = r1_out[:, 0:1]
        r1_log_sigma = torch.clamp(r1_out[:, 1:2], cfg.MIN_LOG_SIGMA, cfg.MAX_LOG_SIGMA)

        r2_mu = r2_out[:, 0:1]
        r2_log_sigma = torch.clamp(r2_out[:, 1:2], cfg.MIN_LOG_SIGMA, cfg.MAX_LOG_SIGMA)

        return {
            "h_final": h_final,
            "lidar_mu": lidar_mu,
            "lidar_log_sigma": lidar_log_sigma,
            "r1_mu": r1_mu,
            "r1_log_sigma": r1_log_sigma,
            "r2_mu": r2_mu,
            "r2_log_sigma": r2_log_sigma,
            "frame_id": frame_id,
            "batch_id": batch_id,
            "sensor_id": sensor_id,
            "cross_alpha": torch.tensor(alpha, device=device, dtype=x.dtype),
        }


# =============================================================================
# Loss (one-step association)
# =============================================================================

class UncertaintyLoss(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

    def gaussian_nll_vec2(self, mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2.0 * log_sigma)
        nll = 0.5 * (2.0 * log_sigma + (target - mu) ** 2 / (var + 1e-12))
        return nll.sum(dim=1)

    def gaussian_nll_scalar(self, mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2.0 * log_sigma)
        nll = 0.5 * (2.0 * log_sigma + (target - mu) ** 2 / (var + 1e-12))
        return nll.squeeze(1)

    @torch.no_grad()
    def _soft_assoc(self, pred_xy: torch.Tensor, gt_xy: torch.Tensor, topk: int, tau: float, gate: float) -> Tuple[torch.Tensor, torch.Tensor]:
        M = pred_xy.size(0)
        G = gt_xy.size(0)
        device = pred_xy.device
        if M == 0 or G == 0:
            return torch.empty((0, 2), device=device), torch.empty((0,), dtype=torch.bool, device=device)

        # NOTE: For training, M and G are typically <= 512; cdist here is acceptable.
        d = torch.cdist(pred_xy, gt_xy)
        k = min(topk, G)
        nn_d, nn_idx = torch.topk(d, k=k, largest=False, dim=1)
        min_d = nn_d[:, 0]
        valid = min_d <= gate
        w = torch.softmax(-(nn_d ** 2) / max(tau ** 2, 1e-6), dim=1)
        cand = gt_xy[nn_idx]
        gt_expect = (w.unsqueeze(-1) * cand).sum(dim=1)
        return gt_expect, valid

    def forward(
        self,
        out_t: Dict[str, torch.Tensor],
        x_t: torch.Tensor,
        frame_id_t: torch.Tensor,
        batch_id_t: torch.Tensor,
        sensor_id_t: torch.Tensor,
        x_tp1: torch.Tensor,
        batch_id_tp1: torch.Tensor,
        sensor_id_tp1: torch.Tensor,
        pose_by_frame: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        device = x_t.device

        t_idx = cfg.WINDOW - 1
        is_t = (frame_id_t == t_idx)

        mL  = (sensor_id_t == 0) & is_t
        mR1 = (sensor_id_t == 1) & is_t
        mR2 = (sensor_id_t == 2) & is_t

        B = int(batch_id_t.max().item()) + 1 if batch_id_t.numel() > 0 else 0

        lidar_losses: List[torch.Tensor] = []
        r1_losses: List[torch.Tensor] = []
        r2_losses: List[torch.Tensor] = []

        for b in range(B):
            idxL  = torch.where((batch_id_t == b) & mL)[0]
            idxR1 = torch.where((batch_id_t == b) & mR1)[0]
            idxR2 = torch.where((batch_id_t == b) & mR2)[0]

            idxL_gt  = torch.where((batch_id_tp1 == b) & (sensor_id_tp1 == 0))[0]
            idxR1_gt = torch.where((batch_id_tp1 == b) & (sensor_id_tp1 == 1))[0]
            idxR2_gt = torch.where((batch_id_tp1 == b) & (sensor_id_tp1 == 2))[0]

            pose_t = pose_tp1 = None
            if pose_by_frame is not None and pose_by_frame.size(0) > b and pose_by_frame.size(1) >= cfg.WINDOW + 1:
                pose_t = pose_by_frame[b, t_idx]
                pose_tp1 = pose_by_frame[b, t_idx + 1]

            # LiDAR position loss
            if idxL.numel() > 0 and idxL_gt.numel() > 0:
                pred_mu = out_t["lidar_mu"][idxL]
                pred_logsig = out_t["lidar_log_sigma"][idxL]
                gt_xy = x_tp1[idxL_gt][:, list(cfg.IDX_POS)]
                if pose_t is not None and pose_tp1 is not None:
                    gt_xy = warp_points_to_frame(gt_xy, pose_src=pose_tp1, pose_dst=pose_t)

                gt_expect, valid = self._soft_assoc(pred_mu, gt_xy, cfg.ASSOC_TOPK, cfg.ASSOC_TAU, cfg.ASSOC_GATE_LIDAR)
                if valid.any():
                    nll = self.gaussian_nll_vec2(pred_mu[valid], pred_logsig[valid], gt_expect[valid])
                    lidar_losses.append(nll.mean())

            # Radar1 vr loss (associate by position; supervise vr)
            if idxR1.numel() > 0 and idxR1_gt.numel() > 0:
                pred_mu = out_t["r1_mu"][idxR1]
                pred_logsig = out_t["r1_log_sigma"][idxR1]
                pred_xy = x_t[idxR1][:, list(cfg.IDX_POS)]
                gt_xy = x_tp1[idxR1_gt][:, list(cfg.IDX_POS)]
                if pose_t is not None and pose_tp1 is not None:
                    gt_xy = warp_points_to_frame(gt_xy, pose_src=pose_tp1, pose_dst=pose_t)

                _, valid = self._soft_assoc(pred_xy, gt_xy, cfg.ASSOC_TOPK, cfg.ASSOC_TAU, cfg.ASSOC_GATE_RADAR)
                if valid.any():
                    d = torch.cdist(pred_xy[valid], gt_xy)
                    nn = d.argmin(dim=1)
                    gt_vr = x_tp1[idxR1_gt][:, cfg.IDX_VR].unsqueeze(1)[nn]
                    nll = self.gaussian_nll_scalar(pred_mu[valid], pred_logsig[valid], gt_vr)
                    r1_losses.append(nll.mean())

            # Radar2 vr
            if idxR2.numel() > 0 and idxR2_gt.numel() > 0:
                pred_mu = out_t["r2_mu"][idxR2]
                pred_logsig = out_t["r2_log_sigma"][idxR2]
                pred_xy = x_t[idxR2][:, list(cfg.IDX_POS)]
                gt_xy = x_tp1[idxR2_gt][:, list(cfg.IDX_POS)]
                if pose_t is not None and pose_tp1 is not None:
                    gt_xy = warp_points_to_frame(gt_xy, pose_src=pose_tp1, pose_dst=pose_t)

                _, valid = self._soft_assoc(pred_xy, gt_xy, cfg.ASSOC_TOPK, cfg.ASSOC_TAU, cfg.ASSOC_GATE_RADAR)
                if valid.any():
                    d = torch.cdist(pred_xy[valid], gt_xy)
                    nn = d.argmin(dim=1)
                    gt_vr = x_tp1[idxR2_gt][:, cfg.IDX_VR].unsqueeze(1)[nn]
                    nll = self.gaussian_nll_scalar(pred_mu[valid], pred_logsig[valid], gt_vr)
                    r2_losses.append(nll.mean())

        loss_l = torch.stack(lidar_losses).mean() if len(lidar_losses) else torch.tensor(0.0, device=device)
        loss_r1 = torch.stack(r1_losses).mean() if len(r1_losses) else torch.tensor(0.0, device=device)
        loss_r2 = torch.stack(r2_losses).mean() if len(r2_losses) else torch.tensor(0.0, device=device)

        reg = (
            (out_t["lidar_log_sigma"] ** 2).mean()
            + (out_t["r1_log_sigma"] ** 2).mean()
            + (out_t["r2_log_sigma"] ** 2).mean()
        )

        total = loss_l + cfg.RADAR_LOSS_WEIGHT * (loss_r1 + loss_r2) + cfg.REG_LAMBDA * reg
        return {
            "total": total,
            "loss_lidar": loss_l.detach(),
            "loss_r1": loss_r1.detach(),
            "loss_r2": loss_r2.detach(),
            "reg": reg.detach(),
        }


# =============================================================================
# Dataset: TXT -> NPZ cache -> window samples (TRAIN)
# =============================================================================

class MultiRunTextWindowDataset(torch.utils.data.Dataset):
    """
    Training dataset yielding:
      sample_t  : nodes for frames [t-3..t]   (WINDOW frames, last is current)
      sample_tp1: nodes for frame  [t+1]      (GT)

    Caches per v are built under cache_dir as cache_v{v}.npz.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        data_root: str,
        versions: List[int],
        cache_dir: Optional[str] = None,
        max_windows_per_run: Optional[int] = None,
        seed: int = 42,
    ):
        self.cfg = cfg
        self.data_root = Path(data_root)
        self.versions = list(versions)
        self.seed = seed

        self.cache_dir = Path(cache_dir) if cache_dir else (self.data_root / "_cache_npz")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.runs: List[Dict[str, Any]] = []
        self.index: List[Tuple[int, int]] = []  # (run_i, t_end)

        for vi, v in enumerate(self.versions):
            run = self._load_or_build_run_cache(v)
            self.runs.append(run)

            F = int(run["F"])
            # valid t_end: need [t_end-3..t_end] within [0..F-1] AND t_end+1 exists
            t_ends = list(range(cfg.WINDOW - 1, F - 1))
            if max_windows_per_run is not None:
                rng = np.random.RandomState(seed + v)
                if len(t_ends) > max_windows_per_run:
                    t_ends = rng.choice(t_ends, size=max_windows_per_run, replace=False).tolist()
                    t_ends.sort()
            for t_end in t_ends:
                self.index.append((vi, t_end))

        if len(self.index) == 0:
            raise RuntimeError("No valid windows found. Check your data_root and versions.")

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _pad_offsets(off: np.ndarray, F: int) -> np.ndarray:
        if off.shape[0] == F + 1:
            return off
        if off.shape[0] < F + 1:
            last = off[-1] if off.size > 0 else 0
            pad = np.full((F + 1 - off.shape[0],), last, dtype=np.int64)
            return np.concatenate([off, pad], axis=0)
        return off[:F+1]

    def _load_or_build_run_cache(self, v: int) -> Dict[str, Any]:
        cache_path = self.cache_dir / f"cache_v{v}.npz"
        if cache_path.exists():
            z = np.load(str(cache_path), allow_pickle=False)
            return {
                "v": v,
                "F": int(z["F"]),
                "t_odom": z["t_odom"].astype(np.float32),
                "pose": z["pose"].astype(np.float32),
                "twist": z["twist"].astype(np.float32),
                "lidar": z["lidar"].astype(np.float32),
                "lidar_off": z["lidar_off"].astype(np.int64),
                "r1": z["r1"].astype(np.float32),
                "r1_off": z["r1_off"].astype(np.int64),
                "r2": z["r2"].astype(np.float32),
                "r2_off": z["r2_off"].astype(np.int64),
            }

        lidar_path = self.data_root / f"LiDARMap_BaseScan_v{v}.txt"
        r1_path = self.data_root / f"Radar1Map_BaseScan_v{v}.txt"
        r2_path = self.data_root / f"Radar2Map_BaseScan_v{v}.txt"
        odom_path = self.data_root / f"odom_filtered_v{v}.txt"

        if not (lidar_path.exists() and r1_path.exists() and r2_path.exists() and odom_path.exists()):
            raise FileNotFoundError(f"Missing files for v{v} under {self.data_root}")

        odom = _read_txt_np(odom_path, expected_cols=6, dtype=np.float32)
        t_odom = odom[:, 0].astype(np.float32)
        pose = odom[:, 1:4].astype(np.float32)   # x,y,yaw
        twist = odom[:, 4:6].astype(np.float32)  # v,w
        F_ = t_odom.shape[0]

        # LiDAR: [t,x,y,intensity] -> store [x,y,intensity]
        lidar_raw = _read_txt_np(lidar_path, expected_cols=4, dtype=np.float32)
        fidx = _assign_to_odom_frames(lidar_raw[:, 0], t_odom)
        lidar_data = np.stack([lidar_raw[:, 1], lidar_raw[:, 2], lidar_raw[:, 3]], axis=1).astype(np.float32)
        lidar_sorted, lidar_off = _sort_by_frame(fidx, lidar_data)
        lidar_off = self._pad_offsets(lidar_off, F_)

        # Radar1: [t,x,y,vr,snr] -> store [x,y,vr,snr]
        r1_raw = _read_txt_np(r1_path, expected_cols=5, dtype=np.float32)
        fidx = _assign_to_odom_frames(r1_raw[:, 0], t_odom)
        r1_data = np.stack([r1_raw[:, 1], r1_raw[:, 2], r1_raw[:, 3], r1_raw[:, 4]], axis=1).astype(np.float32)
        r1_sorted, r1_off = _sort_by_frame(fidx, r1_data)
        r1_off = self._pad_offsets(r1_off, F_)

        # Radar2
        r2_raw = _read_txt_np(r2_path, expected_cols=5, dtype=np.float32)
        fidx = _assign_to_odom_frames(r2_raw[:, 0], t_odom)
        r2_data = np.stack([r2_raw[:, 1], r2_raw[:, 2], r2_raw[:, 3], r2_raw[:, 4]], axis=1).astype(np.float32)
        r2_sorted, r2_off = _sort_by_frame(fidx, r2_data)
        r2_off = self._pad_offsets(r2_off, F_)

        np.savez_compressed(
            str(cache_path),
            F=np.int64(F_),
            t_odom=t_odom,
            pose=pose,
            twist=twist,
            lidar=lidar_sorted,
            lidar_off=lidar_off,
            r1=r1_sorted,
            r1_off=r1_off,
            r2=r2_sorted,
            r2_off=r2_off,
        )

        return {
            "v": v, "F": int(F_), "t_odom": t_odom, "pose": pose, "twist": twist,
            "lidar": lidar_sorted, "lidar_off": lidar_off,
            "r1": r1_sorted, "r1_off": r1_off,
            "r2": r2_sorted, "r2_off": r2_off,
        }

    def _slice_frame(self, arr: np.ndarray, off: np.ndarray, f: int) -> np.ndarray:
        s = int(off[f]); e = int(off[f+1])
        if e <= s:
            return arr[0:0]
        return arr[s:e]

    def _downsample(self, pts: np.ndarray, cap: int, rng: np.random.RandomState) -> np.ndarray:
        if cap <= 0 or pts.shape[0] <= cap:
            return pts
        idx = rng.choice(pts.shape[0], size=cap, replace=False)
        return pts[idx]

    def __getitem__(self, idx: int):
        cfg = self.cfg
        run_i, t_end = self.index[idx]
        run = self.runs[run_i]

        F = int(run["F"])
        t_odom = run["t_odom"]
        pose = run["pose"]
        twist = run["twist"]

        frames = [t_end - 3, t_end - 2, t_end - 1, t_end]
        f_next = t_end + 1

        rng = np.random.RandomState(self.seed + idx * 1337 + int(run["v"]) * 17)
        x_list: List[torch.Tensor] = []
        fid_list: List[torch.Tensor] = []
        sid_list: List[torch.Tensor] = []

        t_cur = float(t_odom[t_end])
        pose_win = np.stack([pose[f] for f in frames] + [pose[f_next]], axis=0).astype(np.float32)

        for local_f, f in enumerate(frames):
            dt = float(t_cur - float(t_odom[f]))
            v, w = twist[f].astype(np.float32)

            # LiDAR
            L = self._slice_frame(run["lidar"], run["lidar_off"], f)
            L = self._downsample(L, cfg.LIDAR_CAP_PER_FRAME, rng)
            if L.shape[0] > 0:
                x = np.zeros((L.shape[0], cfg.INPUT_DIM), dtype=np.float32)
                x[:, 0] = L[:, 0]; x[:, 1] = L[:, 1]
                x[:, cfg.IDX_DT] = dt
                x[:, cfg.IDX_EGO[0]] = v; x[:, cfg.IDX_EGO[1]] = w
                x[:, cfg.IDX_INTENSITY] = L[:, 2]
                x[:, cfg.IDX_SID[0]] = 1.0
                x_list.append(torch.from_numpy(x))
                fid_list.append(torch.full((L.shape[0],), local_f, dtype=torch.long))
                sid_list.append(torch.full((L.shape[0],), 0, dtype=torch.long))

            # Radar1
            R1 = self._slice_frame(run["r1"], run["r1_off"], f)
            R1 = self._downsample(R1, cfg.RADAR_CAP_PER_FRAME, rng)
            if R1.shape[0] > 0:
                x = np.zeros((R1.shape[0], cfg.INPUT_DIM), dtype=np.float32)
                x[:, 0] = R1[:, 0]; x[:, 1] = R1[:, 1]
                x[:, cfg.IDX_DT] = dt
                x[:, cfg.IDX_EGO[0]] = v; x[:, cfg.IDX_EGO[1]] = w
                x[:, cfg.IDX_VR] = R1[:, 2]; x[:, cfg.IDX_SNR] = R1[:, 3]
                x[:, cfg.IDX_SID[1]] = 1.0
                x_list.append(torch.from_numpy(x))
                fid_list.append(torch.full((R1.shape[0],), local_f, dtype=torch.long))
                sid_list.append(torch.full((R1.shape[0],), 1, dtype=torch.long))

            # Radar2
            R2 = self._slice_frame(run["r2"], run["r2_off"], f)
            R2 = self._downsample(R2, cfg.RADAR_CAP_PER_FRAME, rng)
            if R2.shape[0] > 0:
                x = np.zeros((R2.shape[0], cfg.INPUT_DIM), dtype=np.float32)
                x[:, 0] = R2[:, 0]; x[:, 1] = R2[:, 1]
                x[:, cfg.IDX_DT] = dt
                x[:, cfg.IDX_EGO[0]] = v; x[:, cfg.IDX_EGO[1]] = w
                x[:, cfg.IDX_VR] = R2[:, 2]; x[:, cfg.IDX_SNR] = R2[:, 3]
                x[:, cfg.IDX_SID[2]] = 1.0
                x_list.append(torch.from_numpy(x))
                fid_list.append(torch.full((R2.shape[0],), local_f, dtype=torch.long))
                sid_list.append(torch.full((R2.shape[0],), 2, dtype=torch.long))

        if len(x_list) == 0:
            return self.__getitem__((idx + 1) % len(self))

        x_t = torch.cat(x_list, dim=0).float()
        frame_id_t = torch.cat(fid_list, dim=0)
        sensor_id_t = torch.cat(sid_list, dim=0)

        # t+1 GT nodes (dt=0)
        x2_list: List[torch.Tensor] = []
        sid2_list: List[torch.Tensor] = []
        v, w = twist[f_next].astype(np.float32)

        L = self._slice_frame(run["lidar"], run["lidar_off"], f_next)
        L = self._downsample(L, cfg.LIDAR_CAP_PER_FRAME, rng)
        if L.shape[0] > 0:
            x = np.zeros((L.shape[0], cfg.INPUT_DIM), dtype=np.float32)
            x[:, 0] = L[:, 0]; x[:, 1] = L[:, 1]
            x[:, cfg.IDX_DT] = 0.0
            x[:, cfg.IDX_EGO[0]] = v; x[:, cfg.IDX_EGO[1]] = w
            x[:, cfg.IDX_INTENSITY] = L[:, 2]
            x[:, cfg.IDX_SID[0]] = 1.0
            x2_list.append(torch.from_numpy(x))
            sid2_list.append(torch.full((L.shape[0],), 0, dtype=torch.long))

        R1 = self._slice_frame(run["r1"], run["r1_off"], f_next)
        R1 = self._downsample(R1, cfg.RADAR_CAP_PER_FRAME, rng)
        if R1.shape[0] > 0:
            x = np.zeros((R1.shape[0], cfg.INPUT_DIM), dtype=np.float32)
            x[:, 0] = R1[:, 0]; x[:, 1] = R1[:, 1]
            x[:, cfg.IDX_DT] = 0.0
            x[:, cfg.IDX_EGO[0]] = v; x[:, cfg.IDX_EGO[1]] = w
            x[:, cfg.IDX_VR] = R1[:, 2]; x[:, cfg.IDX_SNR] = R1[:, 3]
            x[:, cfg.IDX_SID[1]] = 1.0
            x2_list.append(torch.from_numpy(x))
            sid2_list.append(torch.full((R1.shape[0],), 1, dtype=torch.long))

        R2 = self._slice_frame(run["r2"], run["r2_off"], f_next)
        R2 = self._downsample(R2, cfg.RADAR_CAP_PER_FRAME, rng)
        if R2.shape[0] > 0:
            x = np.zeros((R2.shape[0], cfg.INPUT_DIM), dtype=np.float32)
            x[:, 0] = R2[:, 0]; x[:, 1] = R2[:, 1]
            x[:, cfg.IDX_DT] = 0.0
            x[:, cfg.IDX_EGO[0]] = v; x[:, cfg.IDX_EGO[1]] = w
            x[:, cfg.IDX_VR] = R2[:, 2]; x[:, cfg.IDX_SNR] = R2[:, 3]
            x[:, cfg.IDX_SID[2]] = 1.0
            x2_list.append(torch.from_numpy(x))
            sid2_list.append(torch.full((R2.shape[0],), 2, dtype=torch.long))

        if len(x2_list) == 0:
            x_tp1 = torch.zeros((0, cfg.INPUT_DIM), dtype=torch.float32)
            sensor_id_tp1 = torch.zeros((0,), dtype=torch.long)
        else:
            x_tp1 = torch.cat(x2_list, dim=0).float()
            sensor_id_tp1 = torch.cat(sid2_list, dim=0)

        sample_t = {
            "x": x_t,
            "frame_id": frame_id_t,
            "sensor_id": sensor_id_t,
            "pose_by_frame": torch.from_numpy(pose_win).float(),  # (WINDOW+1,3)
        }
        sample_tp1 = {
            "x": x_tp1,
            "sensor_id": sensor_id_tp1,
        }
        return sample_t, sample_tp1


def collate_fn(batch):
    sample_ts, sample_tp1s = zip(*batch)

    x_list, f_list, sid_list, bid_list = [], [], [], []
    pose_list = []
    x2_list, sid2_list, bid2_list = [], [], []

    for b, (st, sp) in enumerate(zip(sample_ts, sample_tp1s)):
        x = st["x"]
        f = st["frame_id"]
        sid = st["sensor_id"]
        x_list.append(x)
        f_list.append(f)
        sid_list.append(sid)
        bid_list.append(torch.full((x.size(0),), b, dtype=torch.long))
        pose_list.append(st["pose_by_frame"])

        x2 = sp["x"]
        sid2 = sp["sensor_id"]
        x2_list.append(x2)
        sid2_list.append(sid2)
        bid2_list.append(torch.full((x2.size(0),), b, dtype=torch.long))

    x = torch.cat(x_list, dim=0)
    frame_id = torch.cat(f_list, dim=0)
    sensor_id = torch.cat(sid_list, dim=0)
    batch_id = torch.cat(bid_list, dim=0)

    x2 = torch.cat(x2_list, dim=0) if len(x2_list) else torch.zeros((0, 11), dtype=torch.float32)
    sensor_id2 = torch.cat(sid2_list, dim=0) if len(sid2_list) else torch.zeros((0,), dtype=torch.long)
    batch_id2 = torch.cat(bid2_list, dim=0) if len(bid2_list) else torch.zeros((0,), dtype=torch.long)

    pose_by_frame = torch.stack(pose_list, dim=0)  # (B, WINDOW+1, 3)

    return (x, frame_id, batch_id, sensor_id, pose_by_frame), (x2, batch_id2, sensor_id2)


# =============================================================================
# Train/Eval
# =============================================================================

def train_one_epoch(
    cfg: ModelConfig,
    model: DOGMSTGATUncertaintyNet,
    graph_builder: GraphBuilder,
    loss_fn: UncertaintyLoss,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    accum_steps: int = 1,
) -> int:
    model.train()
    scaler = torch.amp.GradScaler(device="cuda", enabled=(cfg.AMP and device.type == "cuda"))

    for it, (bt, bt1) in enumerate(loader):
        (x, frame_id, batch_id, sensor_id, pose_by_frame) = bt
        (x2, batch_id2, sensor_id2) = bt1

        x = x.to(device); frame_id = frame_id.to(device); batch_id = batch_id.to(device); sensor_id = sensor_id.to(device)
        x2 = x2.to(device); batch_id2 = batch_id2.to(device); sensor_id2 = sensor_id2.to(device)
        pose_by_frame = pose_by_frame.to(device)

        edges = graph_builder.build(x, frame_id, batch_id, sensor_id, pose_by_frame=pose_by_frame)

        if (it % max(accum_steps, 1)) == 0:
            optimizer.zero_grad(set_to_none=True)

        amp_enabled = (cfg.AMP and device.type == "cuda")
        amp_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(amp_device, enabled=amp_enabled):
            out = model(x, frame_id, batch_id, sensor_id, edges, global_step=global_step)
            losses = loss_fn(out, x, frame_id, batch_id, sensor_id, x2, batch_id2, sensor_id2, pose_by_frame=pose_by_frame)
            total = losses["total"]
            loss_scaled = total / float(max(accum_steps, 1))

        scaler.scale(loss_scaled).backward()

        do_step = ((it + 1) % max(accum_steps, 1) == 0) or ((it + 1) == len(loader))
        if do_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

        if it % 20 == 0:
            print(
                f"[Train E{epoch:02d} I{it:04d}] "
                f"loss={total.item():.4f} "
                f"L={losses['loss_lidar'].item():.4f} "
                f"R1={losses['loss_r1'].item():.4f} "
                f"R2={losses['loss_r2'].item():.4f} "
                f"alpha={out['cross_alpha'].item():.3f} "
                f"knn_backend={'torch_cluster' if _HAS_TORCH_CLUSTER else 'cdist'}"
            )

    return global_step


@torch.no_grad()
def eval_one_epoch(
    cfg: ModelConfig,
    model: DOGMSTGATUncertaintyNet,
    graph_builder: GraphBuilder,
    loss_fn: UncertaintyLoss,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals = []
    Ls, R1s, R2s = [], [], []
    for bt, bt1 in loader:
        (x, frame_id, batch_id, sensor_id, pose_by_frame) = bt
        (x2, batch_id2, sensor_id2) = bt1
        x = x.to(device); frame_id = frame_id.to(device); batch_id = batch_id.to(device); sensor_id = sensor_id.to(device)
        x2 = x2.to(device); batch_id2 = batch_id2.to(device); sensor_id2 = sensor_id2.to(device)
        pose_by_frame = pose_by_frame.to(device)

        edges = graph_builder.build(x, frame_id, batch_id, sensor_id, pose_by_frame=pose_by_frame)
        out = model(x, frame_id, batch_id, sensor_id, edges, global_step=10**9)  # alpha=max
        losses = loss_fn(out, x, frame_id, batch_id, sensor_id, x2, batch_id2, sensor_id2, pose_by_frame=pose_by_frame)

        totals.append(losses["total"].item())
        Ls.append(losses["loss_lidar"].item())
        R1s.append(losses["loss_r1"].item())
        R2s.append(losses["loss_r2"].item())

    def mean(xs): return float(np.mean(xs)) if len(xs) else 0.0
    return {"total": mean(totals), "lidar": mean(Ls), "r1": mean(R1s), "r2": mean(R2s)}


# =============================================================================
# Inference export (full points, DOGM-friendly NPZ)
# =============================================================================

@torch.no_grad()
def infer_export(
    cfg: ModelConfig,
    model: DOGMSTGATUncertaintyNet,
    graph_builder: GraphBuilder,
    data_root: str,
    version: int,
    out_path: str,
    cache_dir: Optional[str],
    device: torch.device,
    full_points: bool = True,
    max_frames: Optional[int] = None,
):
    """
    Exports sigma for every point at every frame (starting from WINDOW-1) as flattened arrays + offsets.

    NPZ keys:
      t_odom, pose, twist, F
      lidar_out, lidar_off   where lidar_out rows = [x,y,sigx,sigy]
      r1_out, r1_off         rows = [x,y,vr,sigv,snr]
      r2_out, r2_off         rows = [x,y,vr,sigv,snr]
    """
    # Load cache via the dataset helper
    ds = MultiRunTextWindowDataset(cfg, data_root=data_root, versions=[version], cache_dir=cache_dir, max_windows_per_run=1, seed=1)
    run = ds.runs[0]
    F = int(run["F"])
    t_odom = run["t_odom"]
    pose = run["pose"]
    twist = run["twist"]
    lidar = run["lidar"]; lidar_off = run["lidar_off"]
    r1 = run["r1"]; r1_off = run["r1_off"]
    r2 = run["r2"]; r2_off = run["r2_off"]

    def slice_frame(arr: np.ndarray, off: np.ndarray, f: int) -> np.ndarray:
        s = int(off[f]); e = int(off[f+1])
        if e <= s: return arr[0:0]
        return arr[s:e]

    # Prepare output containers (flattened with offsets)
    lidar_rows: List[np.ndarray] = []
    r1_rows: List[np.ndarray] = []
    r2_rows: List[np.ndarray] = []
    lidar_offsets = np.zeros(F + 1, dtype=np.int64)
    r1_offsets = np.zeros(F + 1, dtype=np.int64)
    r2_offsets = np.zeros(F + 1, dtype=np.int64)

    model.eval()
    use_amp = (cfg.AMP and device.type == "cuda")
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    f_start = cfg.WINDOW - 1
    f_end = F
    if max_frames is not None:
        f_end = min(F, f_start + max_frames)

    # For each frame f_cur, build window [f_cur-3..f_cur] and run forward once
    for f_cur in range(f_start, f_end):
        frames = [f_cur - 3, f_cur - 2, f_cur - 1, f_cur]
        t_cur = float(t_odom[f_cur])

        x_list: List[torch.Tensor] = []
        fid_list: List[torch.Tensor] = []
        sid_list: List[torch.Tensor] = []

        # track index mapping for CURRENT frame outputs per sensor
        cur_indices: Dict[str, np.ndarray] = {"L": np.array([], dtype=np.int64), "R1": np.array([], dtype=np.int64), "R2": np.array([], dtype=np.int64)}
        cur_xy_vr_snr: Dict[str, np.ndarray] = {}  # store raw for current frame

        node_cursor = 0
        for local_f, f in enumerate(frames):
            dt = float(t_cur - float(t_odom[f]))
            v, w = twist[f].astype(np.float32)

            L = slice_frame(lidar, lidar_off, f)
            R1 = slice_frame(r1, r1_off, f)
            R2 = slice_frame(r2, r2_off, f)

            # Inference: optionally keep all points
            if not full_points:
                # Reuse training caps (random sampling would be undesirable for export, so take first cap for determinism)
                if L.shape[0] > cfg.LIDAR_CAP_PER_FRAME > 0:
                    L = L[:cfg.LIDAR_CAP_PER_FRAME]
                if R1.shape[0] > cfg.RADAR_CAP_PER_FRAME > 0:
                    R1 = R1[:cfg.RADAR_CAP_PER_FRAME]
                if R2.shape[0] > cfg.RADAR_CAP_PER_FRAME > 0:
                    R2 = R2[:cfg.RADAR_CAP_PER_FRAME]

            # LiDAR nodes
            if L.shape[0] > 0:
                x = np.zeros((L.shape[0], cfg.INPUT_DIM), dtype=np.float32)
                x[:, 0] = L[:, 0]; x[:, 1] = L[:, 1]
                x[:, cfg.IDX_DT] = dt
                x[:, cfg.IDX_EGO[0]] = v; x[:, cfg.IDX_EGO[1]] = w
                x[:, cfg.IDX_INTENSITY] = L[:, 2]
                x[:, cfg.IDX_SID[0]] = 1.0
                x_list.append(torch.from_numpy(x))
                fid_list.append(torch.full((L.shape[0],), local_f, dtype=torch.long))
                sid_list.append(torch.full((L.shape[0],), 0, dtype=torch.long))
                if local_f == cfg.WINDOW - 1:
                    cur_indices["L"] = np.arange(node_cursor, node_cursor + L.shape[0], dtype=np.int64)
                    cur_xy_vr_snr["L"] = L[:, 0:2].copy()
                node_cursor += L.shape[0]

            # Radar1 nodes
            if R1.shape[0] > 0:
                x = np.zeros((R1.shape[0], cfg.INPUT_DIM), dtype=np.float32)
                x[:, 0] = R1[:, 0]; x[:, 1] = R1[:, 1]
                x[:, cfg.IDX_DT] = dt
                x[:, cfg.IDX_EGO[0]] = v; x[:, cfg.IDX_EGO[1]] = w
                x[:, cfg.IDX_VR] = R1[:, 2]; x[:, cfg.IDX_SNR] = R1[:, 3]
                x[:, cfg.IDX_SID[1]] = 1.0
                x_list.append(torch.from_numpy(x))
                fid_list.append(torch.full((R1.shape[0],), local_f, dtype=torch.long))
                sid_list.append(torch.full((R1.shape[0],), 1, dtype=torch.long))
                if local_f == cfg.WINDOW - 1:
                    cur_indices["R1"] = np.arange(node_cursor, node_cursor + R1.shape[0], dtype=np.int64)
                    cur_xy_vr_snr["R1"] = R1.copy()  # [x,y,vr,snr]
                node_cursor += R1.shape[0]

            # Radar2 nodes
            if R2.shape[0] > 0:
                x = np.zeros((R2.shape[0], cfg.INPUT_DIM), dtype=np.float32)
                x[:, 0] = R2[:, 0]; x[:, 1] = R2[:, 1]
                x[:, cfg.IDX_DT] = dt
                x[:, cfg.IDX_EGO[0]] = v; x[:, cfg.IDX_EGO[1]] = w
                x[:, cfg.IDX_VR] = R2[:, 2]; x[:, cfg.IDX_SNR] = R2[:, 3]
                x[:, cfg.IDX_SID[2]] = 1.0
                x_list.append(torch.from_numpy(x))
                fid_list.append(torch.full((R2.shape[0],), local_f, dtype=torch.long))
                sid_list.append(torch.full((R2.shape[0],), 2, dtype=torch.long))
                if local_f == cfg.WINDOW - 1:
                    cur_indices["R2"] = np.arange(node_cursor, node_cursor + R2.shape[0], dtype=np.int64)
                    cur_xy_vr_snr["R2"] = R2.copy()
                node_cursor += R2.shape[0]

        if len(x_list) == 0:
            # no nodes at this frame; keep offsets unchanged
            lidar_offsets[f_cur + 1] = lidar_offsets[f_cur]
            r1_offsets[f_cur + 1] = r1_offsets[f_cur]
            r2_offsets[f_cur + 1] = r2_offsets[f_cur]
            continue

        x_t = torch.cat(x_list, dim=0).float().to(device)
        frame_id_t = torch.cat(fid_list, dim=0).to(device)
        sensor_id_t = torch.cat(sid_list, dim=0).to(device)
        batch_id_t = torch.zeros((x_t.size(0),), dtype=torch.long, device=device)

        # pose_by_frame: (1, WINDOW, 3) is enough for edge warping among window frames
        pose_win = np.stack([pose[f] for f in frames], axis=0).astype(np.float32)
        pose_by_frame = torch.from_numpy(pose_win).unsqueeze(0).to(device)

        edges = graph_builder.build(x_t, frame_id_t, batch_id_t, sensor_id_t, pose_by_frame=pose_by_frame)

        with torch.amp.autocast(amp_device, enabled=use_amp):
            out = model(x_t, frame_id_t, batch_id_t, sensor_id_t, edges, global_step=10**9)

        # Convert log_sigma to sigma
        lidar_sig = torch.exp(out["lidar_log_sigma"]).detach().cpu().numpy()  # (N,2) but only valid for lidar nodes
        r1_sig = torch.exp(out["r1_log_sigma"]).detach().cpu().numpy()        # (N,1)
        r2_sig = torch.exp(out["r2_log_sigma"]).detach().cpu().numpy()

        # Extract current-frame outputs per sensor and append to flattened outputs
        # LiDAR: [x,y,sigx,sigy]
        L_idx = cur_indices["L"]
        if L_idx.size > 0:
            xy = cur_xy_vr_snr["L"]  # (NL,2)
            sig = lidar_sig[L_idx]   # (NL,2)
            rows = np.concatenate([xy, sig], axis=1).astype(np.float32)
            lidar_rows.append(rows)
            lidar_offsets[f_cur + 1] = lidar_offsets[f_cur] + rows.shape[0]
        else:
            lidar_offsets[f_cur + 1] = lidar_offsets[f_cur]

        # Radar1: [x,y,vr,sigv,snr]
        R1_idx = cur_indices["R1"]
        if R1_idx.size > 0:
            raw = cur_xy_vr_snr["R1"]  # (N,4) [x,y,vr,snr]
            sigv = r1_sig[R1_idx, 0:1]
            rows = np.concatenate([raw[:, 0:3], sigv, raw[:, 3:4]], axis=1).astype(np.float32)
            r1_rows.append(rows)
            r1_offsets[f_cur + 1] = r1_offsets[f_cur] + rows.shape[0]
        else:
            r1_offsets[f_cur + 1] = r1_offsets[f_cur]

        # Radar2
        R2_idx = cur_indices["R2"]
        if R2_idx.size > 0:
            raw = cur_xy_vr_snr["R2"]
            sigv = r2_sig[R2_idx, 0:1]
            rows = np.concatenate([raw[:, 0:3], sigv, raw[:, 3:4]], axis=1).astype(np.float32)
            r2_rows.append(rows)
            r2_offsets[f_cur + 1] = r2_offsets[f_cur] + rows.shape[0]
        else:
            r2_offsets[f_cur + 1] = r2_offsets[f_cur]

        if (f_cur - f_start) % 10 == 0:
            print(f"[Infer v{version}] frame {f_cur}/{f_end-1} | L={L_idx.size} R1={R1_idx.size} R2={R2_idx.size}")

    # Fill offsets for frames before f_start (no outputs)
    for f in range(0, f_start + 1):
        if f == 0:
            continue
        lidar_offsets[f] = 0
        r1_offsets[f] = 0
        r2_offsets[f] = 0

    lidar_out = np.concatenate(lidar_rows, axis=0) if len(lidar_rows) else np.zeros((0, 4), dtype=np.float32)
    r1_out = np.concatenate(r1_rows, axis=0) if len(r1_rows) else np.zeros((0, 5), dtype=np.float32)
    r2_out = np.concatenate(r2_rows, axis=0) if len(r2_rows) else np.zeros((0, 5), dtype=np.float32)

    np.savez_compressed(
        out_path,
        F=np.int64(F),
        t_odom=t_odom.astype(np.float32),
        pose=pose.astype(np.float32),
        twist=twist.astype(np.float32),
        lidar_out=lidar_out,
        lidar_off=lidar_offsets,
        r1_out=r1_out,
        r1_off=r1_offsets,
        r2_out=r2_out,
        r2_off=r2_offsets,
        meta=np.array([f"torch_cluster={_HAS_TORCH_CLUSTER}", f"full_points={full_points}"], dtype=object),
    )

    print(f"Saved inference export: {out_path}")
    print("NPZ layout:")
    print("  lidar_out rows: [x,y,sigma_x,sigma_y]")
    print("  r1_out/r2_out rows: [x,y,v_r,sigma_v,snr]")
    print("  *_off: offsets per frame (length F+1), slice f is [off[f]:off[f+1])")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["train", "infer"])
    parser.add_argument("--data_root", type=str, default="", help="Directory containing *_v#.txt files (or extracted subdir).")
    parser.add_argument("--data_zip", type=str, default=None, help="Optional: zip path to extract.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional cache directory for .npz")
    parser.add_argument("--mode", type=str, default="train", choices=["debug", "train"], help="train task only: debug=v1 only, train=v1~v3+val v4")

    # Training overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lidar_cap", type=int, default=None)
    parser.add_argument("--radar_cap", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_windows_per_run", type=int, default=None, help="Limit windows per run for quick tests.")

    # Inference
    parser.add_argument("--ckpt", type=str, default="best_ckpt.pt", help="Checkpoint path for inference.")
    parser.add_argument("--infer_version", type=int, default=1, help="Run version v# to export.")
    parser.add_argument("--infer_out", type=str, default="sigma_export.npz", help="Output NPZ path.")
    parser.add_argument("--infer_full_points", action="store_true", help="Export sigmas for all points (no downsampling).")
    parser.add_argument("--infer_max_frames", type=int, default=None, help="Optional: export only first N frames after warmup.")

    args = parser.parse_args()

    cfg = ModelConfig()
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.batch is not None:
        cfg.BATCH_SIZE = args.batch
    if args.lidar_cap is not None:
        cfg.LIDAR_CAP_PER_FRAME = args.lidar_cap
    if args.radar_cap is not None:
        cfg.RADAR_CAP_PER_FRAME = args.radar_cap

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | torch_cluster={_HAS_TORCH_CLUSTER} | torch_scatter={_HAS_SCATTER}")

    if args.data_root == "":
        raise ValueError("Set --data_root to the extracted dataset directory.")
    data_root = _extract_zip_if_needed(args.data_zip, args.data_root)

    model = DOGMSTGATUncertaintyNet(cfg).to(device)
    graph_builder = GraphBuilder(cfg)
    loss_fn = UncertaintyLoss(cfg)

    if args.task == "infer":
        ckpt = torch.load(args.ckpt, map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)

        infer_export(
            cfg=cfg,
            model=model,
            graph_builder=graph_builder,
            data_root=data_root,
            version=args.infer_version,
            out_path=args.infer_out,
            cache_dir=args.cache_dir,
            device=device,
            full_points=args.infer_full_points,
            max_frames=args.infer_max_frames,
        )
        return

    # Training task
    if args.mode == "debug":
        train_versions = [1]
        val_versions = None
        if args.epochs is None:
            cfg.EPOCHS = 10
        if args.batch is None:
            cfg.BATCH_SIZE = 2
        if args.max_windows_per_run is None:
            args.max_windows_per_run = 800
    else:
        train_versions = [1, 2, 3]
        val_versions = [4]

    train_ds = MultiRunTextWindowDataset(
        cfg, data_root=data_root, versions=train_versions,
        cache_dir=args.cache_dir, max_windows_per_run=args.max_windows_per_run, seed=42
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn, drop_last=True
    )

    val_loader = None
    if val_versions is not None:
        val_ds = MultiRunTextWindowDataset(
            cfg, data_root=data_root, versions=val_versions,
            cache_dir=args.cache_dir, max_windows_per_run=args.max_windows_per_run, seed=123
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn, drop_last=False
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    global_step = 0
    best_val = float("inf")

    for epoch in range(cfg.EPOCHS):
        global_step = train_one_epoch(
            cfg, model, graph_builder, loss_fn, train_loader, optimizer,
            device, epoch, global_step, accum_steps=max(args.accum, 1)
        )

        if val_loader is not None:
            metrics = eval_one_epoch(cfg, model, graph_builder, loss_fn, val_loader, device)
            print(f"[Val   E{epoch:02d}] loss={metrics['total']:.4f} L={metrics['lidar']:.4f} R1={metrics['r1']:.4f} R2={metrics['r2']:.4f}")
            if metrics["total"] < best_val:
                best_val = metrics["total"]
                torch.save({"cfg": cfg.__dict__, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "global_step": global_step}, "best_ckpt.pt")
                print("Saved best_ckpt.pt")

    torch.save({"cfg": cfg.__dict__, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "global_step": global_step}, "last_ckpt.pt")
    print("Saved last_ckpt.pt")


if __name__ == "__main__":
    main()
