# sigma_diagnostics.py
import argparse
import numpy as np
import torch

from gat_train_stgat_dogm import (
    ModelConfig, MultiRunTextWindowDataset, collate_fn,
    GraphBuilder, DOGMSTGATUncertaintyNet, UncertaintyLoss
)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--version", type=int, default=4)  # val v4
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_windows_per_run", type=int, default=None)
    args = ap.parse_args()

    cfg = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = DOGMSTGATUncertaintyNet(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    gb = GraphBuilder(cfg)
    loss_obj = UncertaintyLoss(cfg)

    ds = MultiRunTextWindowDataset(
        cfg, data_root=args.data_root, versions=[args.version],
        cache_dir=None, max_windows_per_run=args.max_windows_per_run, seed=123
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), collate_fn=collate_fn, drop_last=False
    )

    # Accumulators
    clamp = {
        "L_min": 0, "L_max": 0, "L_n": 0,
        "R1_min": 0, "R1_max": 0, "R1_n": 0,
        "R2_min": 0, "R2_max": 0, "R2_n": 0,
    }
    zstats = {
        "L_zx": [], "L_zy": [], "R1_z": [], "R2_z": [],
        "L_valid": [], "R1_valid": [], "R2_valid": []
    }

    for (bt, bt1) in loader:
        (x, frame_id, batch_id, sensor_id, pose_by_frame) = bt
        (x2, batch_id2, sensor_id2) = bt1

        x = x.to(device); frame_id = frame_id.to(device)
        batch_id = batch_id.to(device); sensor_id = sensor_id.to(device)
        pose_by_frame = pose_by_frame.to(device)
        x2 = x2.to(device); batch_id2 = batch_id2.to(device); sensor_id2 = sensor_id2.to(device)

        edges = gb.build(x, frame_id, batch_id, sensor_id, pose_by_frame=pose_by_frame)
        out = model(x, frame_id, batch_id, sensor_id, edges, global_step=10**9)

        # ----- clamp rate (t-frame only) -----
        t_idx = cfg.WINDOW - 1
        is_t = (frame_id == t_idx)

        mL  = (sensor_id == 0) & is_t
        mR1 = (sensor_id == 1) & is_t
        mR2 = (sensor_id == 2) & is_t

        def clamp_count(logsig, key_prefix):
            if logsig.numel() == 0:
                return
            clamp[f"{key_prefix}_n"] += int(logsig.numel())
            clamp[f"{key_prefix}_min"] += int((logsig <= cfg.MIN_LOG_SIGMA + 1e-6).sum().item())
            clamp[f"{key_prefix}_max"] += int((logsig >= cfg.MAX_LOG_SIGMA - 1e-6).sum().item())

        clamp_count(out["lidar_log_sigma"][mL], "L")
        clamp_count(out["r1_log_sigma"][mR1], "R1")
        clamp_count(out["r2_log_sigma"][mR2], "R2")

        # ----- z-score calibration (association-aware) -----
        B = int(batch_id.max().item()) + 1 if batch_id.numel() else 0

        for b in range(B):
            # indices in t-frame
            idxL  = torch.where((batch_id == b) & mL)[0]
            idxR1 = torch.where((batch_id == b) & mR1)[0]
            idxR2 = torch.where((batch_id == b) & mR2)[0]

            # indices in tp1 (GT)
            idxL_gt  = torch.where((batch_id2 == b) & (sensor_id2 == 0))[0]
            idxR1_gt = torch.where((batch_id2 == b) & (sensor_id2 == 1))[0]
            idxR2_gt = torch.where((batch_id2 == b) & (sensor_id2 == 2))[0]

            pose_t = pose_tp1 = None
            pose_t = pose_by_frame[b, t_idx]
            pose_tp1 = pose_by_frame[b, t_idx + 1]

            # LiDAR: z_x, z_y
            if idxL.numel() and idxL_gt.numel():
                mu = out["lidar_mu"][idxL]                # (M,2)
                ls = out["lidar_log_sigma"][idxL]         # (M,2)
                sigma = torch.exp(ls).clamp_min(1e-6)

                gt_xy = x2[idxL_gt][:, list(cfg.IDX_POS)]
                # warp GT (tp1 frame) -> t frame
                from gat_train_stgat_dogm import warp_points_to_frame
                gt_xy = warp_points_to_frame(gt_xy, pose_src=pose_tp1, pose_dst=pose_t)

                gt_expect, valid = loss_obj._soft_assoc(mu, gt_xy, cfg.ASSOC_TOPK, cfg.ASSOC_TAU, cfg.ASSOC_GATE_LIDAR)
                zstats["L_valid"].append(valid.float().mean().item())
                if valid.any():
                    res = (gt_expect[valid] - mu[valid])
                    z = res / sigma[valid]
                    zstats["L_zx"].append(z[:, 0].detach().cpu().numpy())
                    zstats["L_zy"].append(z[:, 1].detach().cpu().numpy())

            # Radar1: z_v
            if idxR1.numel() and idxR1_gt.numel():
                mu = out["r1_mu"][idxR1]                  # (M,1)
                ls = out["r1_log_sigma"][idxR1]           # (M,1)
                sigma = torch.exp(ls).clamp_min(1e-6)

                pred_xy = x[idxR1][:, list(cfg.IDX_POS)]
                gt_xy = x2[idxR1_gt][:, list(cfg.IDX_POS)]
                from gat_train_stgat_dogm import warp_points_to_frame
                gt_xy = warp_points_to_frame(gt_xy, pose_src=pose_tp1, pose_dst=pose_t)

                _, valid = loss_obj._soft_assoc(pred_xy, gt_xy, cfg.ASSOC_TOPK, cfg.ASSOC_TAU, cfg.ASSOC_GATE_RADAR)
                zstats["R1_valid"].append(valid.float().mean().item())
                if valid.any():
                    d = torch.cdist(pred_xy[valid], gt_xy)
                    nn = d.argmin(dim=1)
                    gt_vr = x2[idxR1_gt][:, cfg.IDX_VR].unsqueeze(1)[nn]
                    res = (gt_vr - mu[valid])
                    z = (res / sigma[valid]).squeeze(1)
                    zstats["R1_z"].append(z.detach().cpu().numpy())

            # Radar2: z_v
            if idxR2.numel() and idxR2_gt.numel():
                mu = out["r2_mu"][idxR2]
                ls = out["r2_log_sigma"][idxR2]
                sigma = torch.exp(ls).clamp_min(1e-6)

                pred_xy = x[idxR2][:, list(cfg.IDX_POS)]
                gt_xy = x2[idxR2_gt][:, list(cfg.IDX_POS)]
                from gat_train_stgat_dogm import warp_points_to_frame
                gt_xy = warp_points_to_frame(gt_xy, pose_src=pose_tp1, pose_dst=pose_t)

                _, valid = loss_obj._soft_assoc(pred_xy, gt_xy, cfg.ASSOC_TOPK, cfg.ASSOC_TAU, cfg.ASSOC_GATE_RADAR)
                zstats["R2_valid"].append(valid.float().mean().item())
                if valid.any():
                    d = torch.cdist(pred_xy[valid], gt_xy)
                    nn = d.argmin(dim=1)
                    gt_vr = x2[idxR2_gt][:, cfg.IDX_VR].unsqueeze(1)[nn]
                    res = (gt_vr - mu[valid])
                    z = (res / sigma[valid]).squeeze(1)
                    zstats["R2_z"].append(z.detach().cpu().numpy())

    # ---- summarize ----
    def pct(a, b): return 100.0 * a / max(b, 1)

    print("\n[Clamp rates on t-frame nodes]")
    for key in ["L", "R1", "R2"]:
        print(f"{key}: n={clamp[key+'_n']:,}  min%={pct(clamp[key+'_min'], clamp[key+'_n']):.2f}  max%={pct(clamp[key+'_max'], clamp[key+'_n']):.2f}")

    def cat_list(xs):
        if len(xs) == 0: return np.array([], dtype=np.float32)
        return np.concatenate(xs, axis=0)

    Lzx = cat_list(zstats["L_zx"])
    Lzy = cat_list(zstats["L_zy"])
    R1z = cat_list(zstats["R1_z"])
    R2z = cat_list(zstats["R2_z"])

    def z_report(name, z):
        if z.size == 0:
            print(f"{name}: (no valid associations)")
            return
        mean = float(z.mean())
        std = float(z.std())
        cov1 = float((np.abs(z) < 1.0).mean())
        cov2 = float((np.abs(z) < 2.0).mean())
        cov196 = float((np.abs(z) < 1.96).mean())
        print(f"{name}: mean={mean:.3f} std={std:.3f}  |z|<1:{cov1:.3f}  |z|<1.96:{cov196:.3f}  |z|<2:{cov2:.3f}")

    print("\n[Z-score calibration (want mean~0, std~1)]")
    z_report("LiDAR zx", Lzx)
    z_report("LiDAR zy", Lzy)
    z_report("Radar1 z", R1z)
    z_report("Radar2 z", R2z)

    def valid_report(name, xs):
        if len(xs) == 0:
            print(f"{name}_valid: (none)")
            return
        print(f"{name}_valid: mean={np.mean(xs):.3f}  (fraction of nodes that found a GT match under gate)")

    print("\n[Association valid ratio]")
    valid_report("LiDAR", zstats["L_valid"])
    valid_report("Radar1", zstats["R1_valid"])
    valid_report("Radar2", zstats["R2_valid"])

if __name__ == "__main__":
    main()
