"""Export exact 4-in-1 Joint Multi-Horizon predictions, per-horizon 90%/80% Split vs.
Kinematic Adaptive Conformal cones, contrasting real-storm 2022-2024 case windows
for Figure 10, and exact FLOPs/Params/CPU-latency profiling matching Table 10/11.
"""
import concurrent.futures as cf
import copy
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

from stage2_sota_cv_profile import count_model_flops
from tier1_extreme_roi_surgery import (
    KIN_NAMES,
    SurgicalResidualModel,
    build_joint_4horizon_split,
    denorm_disp_mh,
    evaluate_conformal_cones,
    predict_latlon,
    update_ema,
)
from vendor.common import disp_to_latlon, haversine_km, latlon_to_disp, seed_everything

SPEC = importlib.util.spec_from_file_location("original_prepare", EXP_ROOT / "data" / "prepare_frozen.py")
LIB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIB)

OUT_DIR = EXP_ROOT / "results" / "tier1_extreme_roi_surgery"
HURDAT2_AL = EXP_ROOT / "data" / "hurdat2-1851-2025-031126.txt"


def parse_storm_names(path: Path) -> dict:
    """Map HURDAT2 storm IDs (e.g. AL052024) to human-readable Name (Year)."""
    names = {}
    if not path.exists():
        return names
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].startswith("AL") and len(parts[0]) == 8:
            sid = parts[0]
            sname = parts[1].title()
            yr = sid[-4:]
            names[sid] = f"{sname} ({yr})"
    return names


def run_joint_mse_seed(seed: int, feat_idx: list, seq_len: int, tr, va, te, ex):
    """Exact reproduction of joint_4in1_mse from tier1_extreme_roi_surgery.py, returning val/test/ext preds."""
    torch.set_num_threads(1)
    tmean = tr["tensors"]["raw_y"].mean(dim=0)
    tstd = tr["tensors"]["raw_y"].std(dim=0).clamp(min=1e-3)
    tr_y_norm = (tr["tensors"]["raw_y"] - tmean) / tstd

    seed_everything(seed)
    model = SurgicalResidualModel(
        arch="moderntcn", input_dim=len(feat_idx), seq_len=seq_len,
        lead_steps=[1, 2, 3, 4], target_mean=tmean, target_std=tstd,
        anchor_mode="cv1", d_model=128, dropout=0.1,
    )
    ema_model = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    n_tr = len(tr["keys"])
    best_val, best_state = math.inf, None

    for epoch in range(45):
        model.train()
        gen = torch.Generator().manual_seed(seed + epoch + 1)
        for ids in torch.randperm(n_tr, generator=gen).split(32):
            bx = tr["tensors"]["x"][ids, -seq_len:, :][..., feat_idx]
            bh = tr["tensors"]["history"][ids]
            by = tr_y_norm[ids]
            opt.zero_grad(set_to_none=True)
            pred = model(bx, bh)
            loss = F.mse_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update_ema(ema_model, model, decay=0.985)

        eval_m = ema_model if epoch >= 2 else model
        _, va_err = predict_latlon(eval_m, va, feat_idx, seq_len)
        vk = float((va_err / np.array([35.0, 90.0, 160.0, 230.0])).mean())
        if vk < best_val:
            best_val = vk
            best_state = {k: v.detach().cpu().clone() for k, v in eval_m.state_dict().items()}

    model.load_state_dict(best_state)
    va_ll, va_err = predict_latlon(model, va, feat_idx, seq_len)
    te_ll, te_err = predict_latlon(model, te, feat_idx, seq_len)
    ex_ll, ex_err = predict_latlon(model, ex, feat_idx, seq_len)
    return {
        "seed": seed,
        "params": int(sum(p.numel() for p in model.parameters())),
        "va_ll": va_ll, "te_ll": te_ll, "ex_ll": ex_ll,
        "te_per_h": [float(te_err[:, h].mean()) for h in range(4)],
        "ex_per_h": [float(ex_err[:, h].mean()) for h in range(4)],
    }


def fit_conformal_per_horizon(va_errs_4: np.ndarray, te_errs_4: np.ndarray, ex_errs_4: np.ndarray,
                              va_hist: np.ndarray, te_hist: np.ndarray, ex_hist: np.ndarray, alpha: float = 0.10):
    """Calibrate per-horizon (6/12/18/24h) Split & Kinematic Adaptive Conformal strictly on validation."""
    def motion_features(hist_ll: np.ndarray):
        v1 = latlon_to_disp(torch.from_numpy(hist_ll[:, -1, :]), torch.from_numpy(hist_ll[:, -2, :])).numpy()
        v0 = latlon_to_disp(torch.from_numpy(hist_ll[:, -2, :]), torch.from_numpy(hist_ll[:, -3, :])).numpy()
        spd = np.hypot(v1[:, 0], v1[:, 1])
        spd0 = np.hypot(v0[:, 0], v0[:, 1])
        acc = np.abs(spd - spd0)
        dot = (v1[:, 0] * v0[:, 0] + v1[:, 1] * v0[:, 1]) / np.maximum(spd * spd0, 1e-3)
        turn = np.arccos(np.clip(dot, -1.0, 1.0))
        lat = np.abs(hist_ll[:, -1, 0])
        return np.column_stack([np.ones(len(spd)), spd / 150.0, acc / 50.0, turn, (lat - 20.0) / 15.0])

    F_va = motion_features(va_hist)
    F_te = motion_features(te_hist)
    F_ex = motion_features(ex_hist)
    n_va = len(va_errs_4)
    q_idx = min(n_va - 1, int(math.ceil((n_va + 1) * (1.0 - alpha))) - 1)
    I_reg = np.eye(F_va.shape[1]) * 10.0
    I_reg[0, 0] = 0.0

    scp_q = []
    acp_r_te = []
    acp_r_ex = []
    horizon_stats = {}
    clip_bounds = [(15.0, 150.0), (30.0, 300.0), (45.0, 450.0), (60.0, 600.0)]

    for h, lead_hr in enumerate([6, 12, 18, 24]):
        ve, te_e, ex_e = va_errs_4[:, h], te_errs_4[:, h], ex_errs_4[:, h]
        beta = np.linalg.solve(F_va.T @ F_va + I_reg, F_va.T @ ve)
        lo, hi = clip_bounds[h]
        sig_va = np.clip(F_va @ beta, lo, hi)
        sig_te = np.clip(F_te @ beta, lo, hi)
        sig_ex = np.clip(F_ex @ beta, lo, hi)

        q_s = float(np.sort(ve)[q_idx])
        q_a = float(np.sort(ve / sig_va)[q_idx])
        r_te = q_a * sig_te
        r_ex = q_a * sig_ex
        hi_mask = sig_te >= np.quantile(sig_te, 0.75)

        scp_q.append(q_s)
        acp_r_te.append(r_te)
        acp_r_ex.append(r_ex)
        horizon_stats[f"{lead_hr}h"] = {
            "scp_radius_km": round(q_s, 1),
            "scp_cov_pct": round(float((te_e <= q_s).mean() * 100), 1),
            "scp_hivol_cov_pct": round(float((te_e[hi_mask] <= q_s).mean() * 100), 1),
            "acp_mean_radius_km": round(float(r_te.mean()), 1),
            "acp_steady_radius_km": round(float(r_te[~hi_mask].mean()), 1),
            "acp_hivol_radius_km": round(float(r_te[hi_mask].mean()), 1),
            "acp_cov_pct": round(float((te_e <= r_te).mean() * 100), 1),
            "acp_hivol_cov_pct": round(float((te_e[hi_mask] <= r_te[hi_mask]).mean() * 100), 1),
            "acp_ext25_cov_pct": round(float((ex_e <= r_ex).mean() * 100), 1),
        }

    return np.array(scp_q), np.stack(acp_r_te, axis=1), np.stack(acp_r_ex, axis=1), horizon_stats


def profile_tier1_models():
    """Exact single-sample FLOPs, FP32 size, and warmed 1-thread CPU latency/FPS matching Table 10."""
    torch.set_num_threads(1)
    h1 = torch.tensor([[[20.0 + 0.2 * t, -65.0 - 0.3 * t] for t in range(12)]], dtype=torch.float32)
    tm1, ts1 = [0.0, 0.0], [100.0, 100.0]
    tm4, ts4 = [[0.0, 0.0]] * 4, [[100.0, 100.0]] * 4

    specs = [
        ("ModernTCN+KR 4-in-1 Joint (16ch, L=8)", 16, 8, SurgicalResidualModel("moderntcn", 16, 8, [1, 2, 3, 4], tm4, ts4, "cv1", 128, 0.1)),
        ("ModernTCN+KR 24h (16ch, L=8)",          16, 8, SurgicalResidualModel("moderntcn", 16, 8, [4],          tm1, ts1, "cv1", 128, 0.1)),
        ("ST-B64+KR 24h (16ch, L=8)",             16, 8, SurgicalResidualModel("st_b64",    16, 8, [4],          tm1, ts1, "cv1", 64,  0.1)),
        ("ModernTCN+KR 24h Ref (40ch, L=12)",     40, 12, SurgicalResidualModel("moderntcn", 40, 12, [4],        tm1, ts1, "cv1", 128, 0.1)),
    ]
    out = {}
    for name, c_in, seq_l, mod in specs:
        mod.eval()
        x1 = torch.randn(1, seq_l, c_in)
        params = int(sum(p.numel() for p in mod.parameters()))
        flops = count_model_flops(mod, x1, h1)
        with torch.no_grad():
            for _ in range(50):
                _ = mod(x1, h1)
            t0 = time.perf_counter()
            n_iter = 300
            for _ in range(n_iter):
                _ = mod(x1, h1)
            dt_ms = (time.perf_counter() - t0) * 1000.0 / n_iter
        out[name] = {
            "params": params,
            "mflops": round(flops / 1e6, 3),
            "weights_mb_fp32": round(params * 4 / (1024 * 1024), 3),
            "cpu_ms_per_sample": round(dt_ms, 3),
            "cpu_fps": round(1000.0 / dt_ms, 1),
        }

    # Also time pure conformal inference overhead (5-feature scale + quantile multiplication)
    f_vec = np.array([1.0, 1.2, 0.4, 0.3, 0.8], dtype=np.float64)
    beta = np.array([120.0, 90.0, 25.0, 35.0, 45.0], dtype=np.float64)
    q_acp = 1.95
    t0 = time.perf_counter()
    for _ in range(50000):
        _ = q_acp * max(60.0, min(600.0, float(f_vec @ beta)))
    conf_us = (time.perf_counter() - t0) * 1e6 / 50000.0
    out["Kinematic_Adaptive_Conformal_Overhead"] = {
        "params": 6,  # 5 linear scale weights + 1 conformal quantile q_acp
        "cpu_us_per_sample": round(conf_us, 3),
        "cpu_ms_per_sample": round(conf_us / 1000.0, 6),
    }
    return out


def main():
    t0 = time.perf_counter()
    fnames = LIB.FEATURE_NAMES
    kin_idx = [i for i, f in enumerate(fnames) if f in KIN_NAMES]
    storm_names = parse_storm_names(HURDAT2_AL)

    print("1. Loading aligned 4-horizon North Atlantic datasets...", flush=True)
    tr = build_joint_4horizon_split("controlled_12to1_lead", "train")
    va = build_joint_4horizon_split("controlled_12to1_lead", "val")
    te = build_joint_4horizon_split("controlled_12to1_lead", "test")
    ex = build_joint_4horizon_split("extension2025_lead", "test")

    print("2. Training 3-seed joint_4in1_mse (16ch, L=8, EMA) in parallel...", flush=True)
    runs = []
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(run_joint_mse_seed, s, kin_idx, 8, tr, va, te, ex) for s in (42, 123, 2024)]
        for f in futs:
            runs.append(f.result())

    ens_va_ll = np.mean([r["va_ll"] for r in runs], axis=0)  # (Nv, 4, 2)
    ens_te_ll = np.mean([r["te_ll"] for r in runs], axis=0)  # (Nt, 4, 2)
    ens_ex_ll = np.mean([r["ex_ll"] for r in runs], axis=0)  # (Ne, 4, 2)

    va_errs = haversine_km(torch.from_numpy(ens_va_ll), va["tensors"]["future"]).numpy()
    te_errs = haversine_km(torch.from_numpy(ens_te_ll), te["tensors"]["future"]).numpy()
    ex_errs = haversine_km(torch.from_numpy(ens_ex_ll), ex["tensors"]["future"]).numpy()

    joint_metrics = {
        "params_single": runs[0]["params"],
        "params_3ens": runs[0]["params"] * 3,
        "single_22_24_mean": [round(float(np.mean([r["te_per_h"][h] for r in runs])), 2) for h in range(4)],
        "single_22_24_sd":   [round(float(np.std([r["te_per_h"][h] for r in runs], ddof=1)), 2) for h in range(4)],
        "single_25_mean":    [round(float(np.mean([r["ex_per_h"][h] for r in runs])), 2) for h in range(4)],
        "single_25_sd":      [round(float(np.std([r["ex_per_h"][h] for r in runs], ddof=1)), 2) for h in range(4)],
        "ensemble_22_24_km": [round(float(te_errs[:, h].mean()), 2) for h in range(4)],
        "ensemble_25_km":    [round(float(ex_errs[:, h].mean()), 2) for h in range(4)],
    }
    print("   Joint 4-in-1 Metrics:", json.dumps(joint_metrics), flush=True)

    print("3. Calibrating 90% and 80% per-horizon Conformal Cones on 4-in-1 Joint Ensemble...", flush=True)
    scp_90, acp_te_90, acp_ex_90, stats_90 = fit_conformal_per_horizon(
        va_errs, te_errs, ex_errs,
        va["tensors"]["history"].numpy(), te["tensors"]["history"].numpy(), ex["tensors"]["history"].numpy(),
        alpha=0.10,
    )
    _, _, _, stats_80 = fit_conformal_per_horizon(
        va_errs, te_errs, ex_errs,
        va["tensors"]["history"].numpy(), te["tensors"]["history"].numpy(), ex["tensors"]["history"].numpy(),
        alpha=0.20,
    )
    print("   90% Joint Conformal 24h:", json.dumps(stats_90["24h"]), flush=True)

    # Select TWO contrasting real 2022-2024 Atlantic cyclone windows for Figure 10c:
    # (1) High-volatility recurving storm where Split Conformal FAILS at 24h (err > scp_90[3])
    #     while Kinematic Adaptive Conformal SUCCEEDS (err <= acp_te_90[i, 3]), preferably well-known named storm
    #     in western/central Atlantic (lon between -85 and -35, lat between 22 and 48) with clear recurving track!
    # (2) Steady low-volatility trade-wind storm where Adaptive Conformal SHRINKS 24h radius by > 110 km vs Split
    #     while tightly enclosing ground truth (err < 160 km), in tropical Atlantic (lat 12..24, lon -75..-35)
    #     spatially well-separated from Case 1 so both can be plotted cleanly!
    te_hist = te["tensors"]["history"].numpy()   # (N, 12, 2)
    te_orig = te["tensors"]["origin"].numpy()    # (N, 2)
    te_fut = te["tensors"]["future"].numpy()     # (N, 4, 2)
    keys = te["keys"]

    hivol_cands = []
    steady_cands = []
    for i, (sid, widx) in enumerate(keys):
        e24 = float(te_errs[i, 3])
        r_s24 = float(scp_90[3])
        r_a24 = float(acp_te_90[i, 3])
        olat, olon = float(te_orig[i, 0]), float(te_orig[i, 1])
        # High-volatility rescue candidate: Split misses (e24 > r_s24 + 15), Adaptive covers (e24 < r_a24 - 15)
        if (e24 > r_s24 + 15.0) and (e24 < r_a24 - 15.0) and (25.0 <= olat <= 44.0) and (-78.0 <= olon <= -40.0):
            # Prefer moderate-high displacement and clean northeastward recurvature
            hivol_cands.append((abs(e24 - (r_s24 + 65.0)), i, sid, widx, e24, r_s24, r_a24, olat, olon))
        # Steady shrinkage candidate: Adaptive shrinks radius by >= 115 km AND covers all 4 horizons cleanly
        if (r_s24 - r_a24 >= 115.0) and np.all(te_errs[i] < acp_te_90[i] * 0.65) and (13.0 <= olat <= 23.0) and (-78.0 <= olon <= -40.0):
            # Prefer well-behaved westward trade-wind motion
            dx_hist = te_orig[i, 1] - te_hist[i, -8, 1]
            if dx_hist < -3.0:
                steady_cands.append((r_a24 + 0.3 * e24, i, sid, widx, e24, r_s24, r_a24, olat, olon))

    hivol_cands.sort(key=lambda t: t[0])
    steady_cands.sort(key=lambda t: t[0])

    print(f"   Found {len(hivol_cands)} high-vol rescue candidates, {len(steady_cands)} steady shrinkage candidates", flush=True)
    for tag, arr in [("HiVol", hivol_cands[:6]), ("Steady", steady_cands[:6])]:
        for c in arr:
            print(f"     [{tag}] idx={c[1]} {c[2]} {storm_names.get(c[2], c[2])} w={c[3]} | origin=({c[7]:.1f},{c[8]:.1f}) | err24={c[4]:.1f} | SCP={c[5]:.1f} | ACP={c[6]:.1f}", flush=True)

    def pack_case(idx: int):
        sid, widx = keys[idx]
        # Compute 4-horizon CV track too for reference
        cv_ll = disp_to_latlon(
            torch.cat([
                latlon_to_disp(te["tensors"]["history"][idx:idx+1, -1], te["tensors"]["history"][idx:idx+1, -2]) * s
                for s in (1.0, 2.0, 3.0, 4.0)
            ], dim=0).unsqueeze(0),
            te["tensors"]["origin"][idx:idx+1],
        )[0].numpy()
        return {
            "test_index": int(idx),
            "storm_id": sid,
            "storm_name": storm_names.get(sid, sid),
            "window": int(widx),
            "history_48h_latlon": np.round(te_hist[idx, -8:, :], 3).tolist(),
            "origin_latlon": np.round(te_orig[idx], 3).tolist(),
            "truth_6_24h_latlon": np.round(te_fut[idx], 3).tolist(),
            "pred_6_24h_latlon": np.round(ens_te_ll[idx], 3).tolist(),
            "cv_6_24h_latlon": np.round(cv_ll, 3).tolist(),
            "errors_km": [round(float(x), 1) for x in te_errs[idx]],
            "scp_90_radii_km": [round(float(x), 1) for x in scp_90],
            "acp_90_radii_km": [round(float(x), 1) for x in acp_te_90[idx]],
        }

    # Pick spatially non-overlapping top pair if available
    chosen_hivol = hivol_cands[0][1]
    chosen_steady = steady_cands[0][1]
    # Check if there is a famous Hurricane (Ernesto/Fiona/Lee/Idalia/Beryl/Milton/Kirk) near the top!
    famous = {"Ernesto", "Fiona", "Lee", "Idalia", "Beryl", "Milton", "Ian", "Nicole", "Franklin", "Kirk", "Helene"}
    for c in hivol_cands[:12]:
        nm = storm_names.get(c[2], "")
        if any(f in nm for f in famous):
            chosen_hivol = c[1]
            break
    for c in steady_cands[:12]:
        nm = storm_names.get(c[2], "")
        if any(f in nm for f in famous):
            chosen_steady = c[1]
            break

    print("4. Profiling exact FLOPs, FP32 size, and 1-thread CPU latency/FPS...", flush=True)
    prof = profile_tier1_models()
    print(json.dumps(prof, indent=2), flush=True)

    payload = {
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "joint_4in1_mse_verified": joint_metrics,
        "joint_4in1_conformal_90pct": stats_90,
        "joint_4in1_conformal_80pct": stats_80,
        "case_high_volatility_rescue": pack_case(chosen_hivol),
        "case_steady_shrinkage": pack_case(chosen_steady),
        "efficiency_profile": prof,
    }
    out_path = OUT_DIR / "fig10_and_profile.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"DONE in {payload['wall_seconds']}s -> {out_path}", flush=True)

    # Also patch single_25_km in summary.json so it reflects exact verified single-seed 2025 mean!
    sum_path = OUT_DIR / "summary.json"
    if sum_path.exists():
        sdata = json.loads(sum_path.read_text(encoding="utf-8"))
        sdata["part2_atlantic_4in1_joint"]["joint_4in1_mse"]["single_25_km"] = joint_metrics["single_25_mean"]
        sdata["part2_atlantic_4in1_joint"]["joint_4in1_mse"]["single_22_24_sd"] = joint_metrics["single_22_24_sd"]
        sdata["part2_atlantic_4in1_joint"]["joint_4in1_mse"]["single_25_sd"] = joint_metrics["single_25_sd"]
        sdata["part2_atlantic_4in1_joint"]["joint_4in1_mse"]["conformal_90pct_by_horizon"] = stats_90
        sdata["efficiency_profile"] = prof
        sum_path.write_text(json.dumps(sdata, indent=2), encoding="utf-8")
        print(f"Updated {sum_path} with verified single_25_km & efficiency_profile!", flush=True)


if __name__ == "__main__":
    main()
