"""Full-Pipeline Module & Hyperparameter ROI Audit:
Tests every remaining pipeline design choice on North Atlantic 24-h HURDAT2 (seeds 42, 123, 2024)
relative to our Compact Residual reference (B64, 40 ch, AdamW lr=1e-4, wd=1e-4, dropout=0.1):
  1. Feature Subset: Kinematics + Coords + Clock only (16 ch, dropping wind/pressure/radii/status)
  2. Feature Subset: Dropping 12 quadrant wind-radii channels (28 ch)
  3. LR Schedule: CosineAnnealingLR (1e-4 -> 1e-6)
  4. Higher Learning Rate: AdamW lr=3e-4
  5. Stronger Weight Decay: AdamW wd=1e-3
  6. Regularization: Dropout = 0.0
  7. Regularization: Dropout = 0.2
"""
import concurrent.futures as cf
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from controlled_heads import ControlledNumericModel
from sota_sequence_baselines import load_split_fast
from vendor.common import denormalize_disp, disp_to_latlon, haversine_km, seed_everything

SPEC = importlib.util.spec_from_file_location("original_prepare", EXP_ROOT / "data" / "prepare_frozen.py")
LIB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIB)

OUT_DIR = EXP_ROOT / "results" / "module_hyperparam_roi"


@torch.no_grad()
def eval_split(model, data, feat_idx, batch_size=128):
    model.eval()
    errs = []
    n = len(data["keys"])
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        bx = data["tensors"]["x"][sl][..., feat_idx]
        bh = data["tensors"]["history"][sl]
        bo = data["tensors"]["origin"][sl]
        bf = data["tensors"]["future"][sl]
        out = model(bx, bh)
        raw = denormalize_disp(out["pred"], model.target_mean, model.target_std)
        latlon = disp_to_latlon(raw, bo)
        errs.append(haversine_km(latlon, bf).cpu().numpy().reshape(-1))
    errors = np.concatenate(errs)
    return float(errors.mean())


def run_variant_job(job):
    name, cfg, seed = job
    torch.set_num_threads(1)
    d4 = EXP_ROOT / "data" / "controlled_12to1_lead4"
    e4 = EXP_ROOT / "data" / "extension2025_lead4"
    meta = json.loads((d4 / "metadata.json").read_text(encoding="utf-8"))
    tr = load_split_fast(d4, "train")
    va = load_split_fast(d4, "val")
    te = load_split_fast(d4, "test")
    ex = load_split_fast(e4, "test")

    feat_idx = cfg["feat_idx"]
    seed_everything(seed)
    model = ControlledNumericModel(
        input_dim=len(feat_idx),
        input_len=12,
        lead_steps=meta["lead_steps"],
        target_mean=meta["stats"]["target_mean"],
        target_std=meta["stats"]["target_std"],
        head="residual",
        d_model=64,
        num_heads=4,
        num_layers=1,
        dropout=cfg["dropout"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-6) if cfg["cosine"] else None

    best_val = math.inf
    best_state = None
    n_tr = len(tr["keys"])
    for epoch in range(40):
        model.train()
        gen = torch.Generator().manual_seed(seed + epoch + 1)
        perm = torch.randperm(n_tr, generator=gen)
        for ids in perm.split(32):
            bx = tr["tensors"]["x"][ids][..., feat_idx]
            bh = tr["tensors"]["history"][ids]
            by = tr["tensors"]["y"][ids]
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(bx, bh)["pred"], by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if sched is not None:
            sched.step()
        vk = eval_split(model, va, feat_idx)
        if vk < best_val:
            best_val = vk
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return {
        "variant": name,
        "seed": seed,
        "input_dim": len(feat_idx),
        "val_km": round(best_val, 2),
        "test_22_24_km": round(eval_split(model, te, feat_idx), 2),
        "ext_25_km": round(eval_split(model, ex, feat_idx), 2),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    fnames = LIB.FEATURE_NAMES
    all_idx = list(range(len(fnames)))
    kin_names = {
        "lat", "lon", "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
        "hour_sin", "hour_cos", "dx6", "dy6", "speed6", "heading_sin", "heading_cos",
        "accel6", "turn_sin", "turn_cos",
    }
    kin_idx = [i for i, f in enumerate(fnames) if f in kin_names]
    no_rad_idx = [i for i, f in enumerate(fnames) if f not in set(LIB.RADIUS_COLS)]

    variants = {
        "ref_40ch_lr1e4_wd1e4_do01": dict(feat_idx=all_idx, lr=1e-4, wd=1e-4, dropout=0.1, cosine=False),
        "feat_kinematics_only_16ch": dict(feat_idx=kin_idx, lr=1e-4, wd=1e-4, dropout=0.1, cosine=False),
        "feat_no_wind_radii_28ch":   dict(feat_idx=no_rad_idx, lr=1e-4, wd=1e-4, dropout=0.1, cosine=False),
        "sched_cosine_annealing":    dict(feat_idx=all_idx, lr=1e-4, wd=1e-4, dropout=0.1, cosine=True),
        "lr_3e4_higher":             dict(feat_idx=all_idx, lr=3e-4, wd=1e-4, dropout=0.1, cosine=False),
        "wd_1e3_stronger":           dict(feat_idx=all_idx, lr=1e-4, wd=1e-3, dropout=0.1, cosine=False),
        "dropout_00":                dict(feat_idx=all_idx, lr=1e-4, wd=1e-4, dropout=0.0, cosine=False),
        "dropout_02":                dict(feat_idx=all_idx, lr=1e-4, wd=1e-4, dropout=0.2, cosine=False),
    }

    jobs = [(name, cfg, seed) for name, cfg in variants.items() for seed in (42, 123, 2024)]
    runs = []
    with cf.ProcessPoolExecutor(max_workers=12) as ex:
        for r in ex.map(run_variant_job, jobs):
            runs.append(r)

    agg = {}
    for name in variants:
        rs = [r for r in runs if r["variant"] == name]
        agg[name] = {
            "input_dim": rs[0]["input_dim"],
            "val_mean": round(float(np.mean([r["val_km"] for r in rs])), 2),
            "test_22_24_mean": round(float(np.mean([r["test_22_24_km"] for r in rs])), 2),
            "test_22_24_sd": round(float(np.std([r["test_22_24_km"] for r in rs], ddof=1)), 2),
            "ext_25_mean": round(float(np.mean([r["ext_25_km"] for r in rs])), 2),
            "ext_25_sd": round(float(np.std([r["ext_25_km"] for r in rs], ddof=1)), 2),
        }
        a = agg[name]
        print(f"  {name:28s} (d={a['input_dim']:2d}) | val={a['val_mean']:.2f} | 22-24={a['test_22_24_mean']:.2f}+/-{a['test_22_24_sd']:.2f} | 25={a['ext_25_mean']:.2f}+/-{a['ext_25_sd']:.2f}")

    summary = {"wall_seconds": round(time.perf_counter() - t0, 2), "aggregated": agg, "runs": runs}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDONE in {summary['wall_seconds']:.1f}s -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
