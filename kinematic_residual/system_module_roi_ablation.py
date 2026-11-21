"""Real 3-seed System Module Optimality & ROI Ablation Suite.
Tests each pipeline module of our Lagrangian Kinematic Residual framework on the exact
24-h HURDAT2 splits (train/val 2004-2021, test 2022-2024, out-of-sample 2025; seeds 42, 123, 2024):
  0. Full Optimal Compact Pipeline (L=12, Tangent-Plane km, PosEnc, Standardized MSE, Head B, d=64)
  1. w/o Tangent-Plane Metric Scaling (Raw Spherical Lat/Lon Degree Regression)
  2. Truncated Lookback L=4 (last 4 synoptic steps, 18 h)
  3. Truncated Lookback L=8 (last 8 synoptic steps, 42 h)
  4. w/o Temporal Positional Encoding
  5. Alternative Objective: L1 (MAE) Loss
  6. Alternative Objective: Huber (Smooth-L1) Loss
"""
import concurrent.futures as cf
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
J_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from controlled_heads import ControlledNumericModel
from vendor.common import denormalize_disp, disp_to_latlon, haversine_km, normalize_disp, seed_everything


def load_split_fast(root: Path, split: str):
    paths = sorted((root / split).glob("*.npy"))
    rows = [np.load(p, allow_pickle=True).item() for p in paths]
    tensors = {
        "x": torch.from_numpy(np.stack([r["input_num"] for r in rows]).astype(np.float32)),
        "y": torch.from_numpy(np.stack([r["target_disp"] for r in rows]).astype(np.float32)),
        "history": torch.from_numpy(np.stack([r["history_latlon"] for r in rows]).astype(np.float32)),
        "origin": torch.from_numpy(np.stack([r["origin"] for r in rows]).astype(np.float32)),
        "future": torch.from_numpy(np.stack([r["future_latlon"] for r in rows]).astype(np.float32)),
    }
    keys = [(str(r["storm_id"]), int(r["window"])) for r in rows]
    return {"tensors": tensors, "keys": keys}


def wrap_lon_diff_deg(lon2: torch.Tensor, lon1: torch.Tensor) -> torch.Tensor:
    d = lon2 - lon1
    return (d + 180.0) % 360.0 - 180.0


class SphericalDegreeResidualModel(ControlledNumericModel):
    """Ablation without cos(phi) metric tangent-plane scaling: operates in raw (dlat, dlon) degrees."""
    def __init__(self, input_dim, input_len, lead_steps, deg_mean, deg_std, d_model=64, num_layers=1):
        super().__init__(input_dim, input_len, lead_steps, deg_mean, deg_std,
                         head="residual", d_model=d_model, num_heads=4, num_layers=num_layers)

    def forward(self, x_num, history_latlon, force_cv=False):
        context = self.encode(x_num, None, None, None)["context"]
        prev, orig = history_latlon[:, -2], history_latlon[:, -1]
        dlat = orig[:, 0] - prev[:, 0]
        dlon = wrap_lon_diff_deg(orig[:, 1], prev[:, 1])
        step_deg = torch.stack([dlat, dlon], dim=-1)
        cv_deg = step_deg[:, None, :] * self.lead_steps[None, :, None]
        cv_norm = normalize_disp(cv_deg, self.target_mean, self.target_std)
        delta = self.residual_head(context).reshape_as(cv_norm)
        return {"pred": cv_norm + delta, "cv": cv_norm}


@torch.no_grad()
def eval_model(model, data, variant: str, seq_len: int, batch_size: int = 128):
    model.eval()
    errs = []
    n = len(data["keys"])
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        bx = data["tensors"]["x"][sl][:, -seq_len:, :]
        bh = data["tensors"]["history"][sl][:, -seq_len:, :]
        bo = data["tensors"]["origin"][sl]
        bf = data["tensors"]["future"][sl]
        out = model(bx, bh)
        raw = denormalize_disp(out["pred"], model.target_mean, model.target_std)
        if variant == "raw_latlon_deg":
            lat = bo[:, None, 0] + raw[..., 0]
            lon = bo[:, None, 1] + raw[..., 1]
            latlon = torch.stack([lat, lon], dim=-1)
        else:
            latlon = disp_to_latlon(raw, bo)
        ekm = haversine_km(latlon, bf).cpu().numpy().reshape(-1)
        errs.append(ekm)
    errors = np.concatenate(errs)
    ids = np.asarray([k[0] for k in data["keys"]])
    storm_means = np.array([errors[ids == sid].mean() for sid in sorted(set(ids))])
    return {
        "window_mean_km": float(errors.mean()),
        "storm_equal_mean_km": float(storm_means.mean()),
        "errors": errors,
        "storm_ids": ids,
    }


def run_job(job):
    variant, seed = job
    torch.set_num_threads(1)
    data_dir = J_ROOT / "data" / "controlled_12to1_lead4"
    ext_dir = J_ROOT / "data" / "extension2025_lead4"
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))

    train_d = load_split_fast(data_dir, "train")
    val_d = load_split_fast(data_dir, "val")
    test_d = load_split_fast(data_dir, "test")
    ext_d = load_split_fast(ext_dir, "test")

    seq_len = 4 if variant == "lookback_L4" else (8 if variant == "lookback_L8" else 12)
    input_dim = train_d["tensors"]["x"].shape[-1]

    seed_everything(seed)
    if variant == "raw_latlon_deg":
        # Compute training-only degree displacement moments
        orig_tr = train_d["tensors"]["origin"][:, None, :]
        fut_tr = train_d["tensors"]["future"]
        dlat_tr = fut_tr[..., 0] - orig_tr[..., 0]
        dlon_tr = wrap_lon_diff_deg(fut_tr[..., 1], orig_tr[..., 1])
        y_deg_tr = torch.stack([dlat_tr, dlon_tr], dim=-1)
        deg_mean = y_deg_tr.mean(dim=0)
        deg_std = y_deg_tr.std(dim=0).clamp_min(1e-6)
        y_train_target = normalize_disp(y_deg_tr, deg_mean, deg_std)
        model = SphericalDegreeResidualModel(input_dim, seq_len, meta["lead_steps"], deg_mean, deg_std, d_model=64, num_layers=1)
    else:
        y_train_target = train_d["tensors"]["y"]
        model = ControlledNumericModel(
            input_dim=input_dim,
            input_len=seq_len,
            lead_steps=meta["lead_steps"],
            target_mean=meta["stats"]["target_mean"],
            target_std=meta["stats"]["target_std"],
            head="residual",
            d_model=64,
            num_heads=4,
            num_layers=1,
            dropout=0.1,
        )
        if variant == "no_pos_enc":
            for pname, p in model.named_parameters():
                if "pos" in pname.lower():
                    p.data.zero_()
                    p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)

    best_val = math.inf
    best_state = None
    best_epoch = 0
    t0 = time.perf_counter()
    n_train = len(train_d["keys"])

    for epoch in range(40):
        model.train()
        gen = torch.Generator().manual_seed(seed + epoch + 1)
        perm = torch.randperm(n_train, generator=gen)
        for ids in perm.split(32):
            bx = train_d["tensors"]["x"][ids][:, -seq_len:, :]
            bh = train_d["tensors"]["history"][ids][:, -seq_len:, :]
            by = y_train_target[ids]
            opt.zero_grad(set_to_none=True)
            pred = model(bx, bh)["pred"]
            if variant == "loss_l1":
                loss = F.l1_loss(pred, by)
            elif variant == "loss_huber":
                loss = F.smooth_l1_loss(pred, by, beta=0.5)
            else:
                loss = F.mse_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_res = eval_model(model, val_d, variant, seq_len)
        if val_res["window_mean_km"] < best_val:
            best_val = val_res["window_mean_km"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.perf_counter() - t0
    model.load_state_dict(best_state)
    test_res = eval_model(model, test_d, variant, seq_len)
    ext_res = eval_model(model, ext_d, variant, seq_len)

    return {
        "variant": variant,
        "seed": seed,
        "seq_len": seq_len,
        "parameters": n_params,
        "best_epoch": best_epoch,
        "val_window_km": best_val,
        "test_window_km": test_res["window_mean_km"],
        "test_storm_km": test_res["storm_equal_mean_km"],
        "ext_window_km": ext_res["window_mean_km"],
        "ext_storm_km": ext_res["storm_equal_mean_km"],
        "elapsed_seconds": elapsed,
    }


def main():
    variants = [
        "optimal_compact_ref",
        "raw_latlon_deg",
        "lookback_L4",
        "lookback_L8",
        "no_pos_enc",
        "loss_l1",
        "loss_huber",
    ]
    seeds = [42, 123, 2024]
    jobs = [(v, s) for v in variants for s in seeds]

    t_start = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=min(12, len(jobs))) as ex:
        runs = list(ex.map(run_job, jobs))
    wall_s = time.perf_counter() - t_start

    agg = {}
    for v in variants:
        sub = [r for r in runs if r["variant"] == v]
        agg[v] = {
            "parameters": sub[0]["parameters"],
            "seq_len": sub[0]["seq_len"],
            "val_window_mean": float(np.mean([r["val_window_km"] for r in sub])),
            "test_window_mean": float(np.mean([r["test_window_km"] for r in sub])),
            "test_window_std": float(np.std([r["test_window_km"] for r in sub], ddof=1)),
            "test_storm_mean": float(np.mean([r["test_storm_km"] for r in sub])),
            "ext_window_mean": float(np.mean([r["ext_window_km"] for r in sub])),
            "ext_window_std": float(np.std([r["ext_window_km"] for r in sub], ddof=1)),
            "ext_storm_mean": float(np.mean([r["ext_storm_km"] for r in sub])),
            "mean_fit_seconds": float(np.mean([r["elapsed_seconds"] for r in sub])),
        }
        print(f"{v:22s} | Params={agg[v]['parameters']:7d} | 2022-24 W/S={agg[v]['test_window_mean']:.2f}/{agg[v]['test_storm_mean']:.2f} | 2025 W/S={agg[v]['ext_window_mean']:.2f}/{agg[v]['ext_storm_mean']:.2f} | {agg[v]['mean_fit_seconds']:.1f}s")

    out_dir = J_ROOT / "results" / "system_module_roi"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"wall_clock_seconds": wall_s, "aggregated": agg, "runs": runs}, indent=2),
        encoding="utf-8",
    )
    print(f"Completed 21 real module-ROI runs in {wall_s:.1f}s!")


if __name__ == "__main__":
    main()
