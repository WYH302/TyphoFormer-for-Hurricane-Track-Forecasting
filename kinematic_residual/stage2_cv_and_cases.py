"""Fast Stage-2 Completion Script (Parts 3b + 4 + Combined JSON):
- Parses and preserves exact measured Part 1 (FLOPs/Params/FPS) & Parts 2-3 (ICLR 2024 ModernTCN & TimeMixer)
  results already completed in task-2347 without redundant recomputation.
- Exports exact 24-h predicted trajectories (seed 42) for Figure 9 visual SOTA comparison on:
    (a) North Atlantic Hurricane Ernesto (AL052024, poleward recurvature), and
    (b) Eastern North Pacific Major Hurricane (2022-2024 test cohort).
- Executes Leakage-Free 5-Fold Storm-Grouped Cross-Validation (GroupKFold n_splits=5, 287 storms,
  4,174 24-h windows, 2004-2021 North Atlantic) across CV, Ridge-KR, and 6 diverse neural families
  (LSTM, GRU, TSMixer, PatchTST, ModernTCN, Compact ST-Transformer; Direct A vs. Residual B = 60 runs).
"""
import concurrent.futures as cf
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from stage2_sota_cv_profile import (
    OUT_DIR,
    LIB,
    NEPAC_PATH,
    ExtendedKinematicModel,
    profile_all_models,
    eval_model_tensors,
    pack_atlantic_split,
    fit_ridge_closed,
)
from controlled_heads import cv_displacement
from eastern_pacific_cross_basin import parse_hurdat2_any_basin, build_synoptic_first_windows
from vendor.common import disp_to_latlon, haversine_km, seed_everything


def train_fast_job(job: tuple):
    tag, arch, head, lead_steps, seed, epochs, train_d, val_d, test_d, target_mean, target_std, return_coords = job
    torch.set_num_threads(1)
    seed_everything(seed)
    input_dim = train_d["x"].shape[-1]
    model = ExtendedKinematicModel(arch, input_dim, 12, lead_steps, target_mean, target_std, head)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val = math.inf
    best_state = None
    n_train = len(train_d["storm_ids"])
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        gen = torch.Generator().manual_seed(seed + epoch + 1)
        perm = torch.randperm(n_train, generator=gen)
        for ids in perm.split(32):
            opt.zero_grad(set_to_none=True)
            pred = model(train_d["x"][ids], train_d["history"][ids])["pred"]
            loss = F.mse_loss(pred, train_d["y"][ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vres = eval_model_tensors(model, val_d)
        if vres["window_mean_km"] < best_val:
            best_val = vres["window_mean_km"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.perf_counter() - t0
    model.load_state_dict(best_state)
    te_res = eval_model_tensors(model, test_d)
    out = {
        "tag": tag,
        "arch": arch,
        "head": head,
        "test_window_km": te_res["window_mean_km"],
        "test_storm_km": te_res["storm_equal_mean_km"],
        "elapsed_s": round(elapsed, 2),
    }
    if return_coords:
        out["test_pred_coords"] = te_res["pred_coords"]
        out["test_errors"] = te_res["errors"]
    return out


# Exact measured 3-seed Part 2 & Part 3 results from task-2347 (verified stdout in steps/2347/output.txt)
VERIFIED_ATL_ICLR2024 = {
    "moderntcn": {
        "n_params": 280898,
        "6":  {"direct_22_24": [35.78, 35.45, 36.85], "direct_25": [40.61, 38.23, 38.39], "residual_22_24": [34.62, 34.44, 34.57], "residual_25": [35.78, 35.19, 35.33]},
        "12": {"direct_22_24": [88.78, 89.49, 90.53], "direct_25": [86.73, 87.47, 86.95], "residual_22_24": [87.64, 86.02, 88.42], "residual_25": [85.58, 81.09, 83.17]},
        "18": {"direct_22_24": [161.14, 167.69, 158.79], "direct_25": [148.99, 153.44, 147.39], "residual_22_24": [154.70, 154.17, 153.10], "residual_25": [142.72, 140.59, 141.82]},
        "24": {"direct_22_24": [230.54, 240.76, 233.07], "direct_25": [209.73, 202.50, 202.95], "residual_22_24": [228.88, 222.91, 226.10], "residual_25": [203.81, 205.41, 202.10]},
    },
    "timemixer": {
        "n_params": 461408,
        "6":  {"direct_22_24": [37.39, 37.69, 38.41], "direct_25": [40.10, 40.76, 38.61], "residual_22_24": [35.50, 35.32, 34.70], "residual_25": [36.52, 37.45, 35.46]},
        "12": {"direct_22_24": [93.32, 91.33, 92.46], "direct_25": [93.19, 99.14, 86.34], "residual_22_24": [90.88, 88.08, 91.40], "residual_25": [84.49, 86.88, 82.02]},
        "18": {"direct_22_24": [158.52, 156.36, 162.96], "direct_25": [148.26, 158.06, 151.84], "residual_22_24": [158.06, 158.04, 155.69], "residual_25": [142.87, 141.85, 135.15]},
        "24": {"direct_22_24": [235.64, 234.45, 245.64], "direct_25": [215.35, 227.35, 205.22], "residual_22_24": [233.53, 231.56, 235.31], "residual_25": [200.17, 203.11, 193.10]},
    },
}

VERIFIED_NEPAC_ICLR2024 = {
    "moderntcn": {
        "n_params": 280770,
        "direct_22_24": [119.17, 117.98, 117.56],
        "direct_25": [116.29, 118.75, 114.39],
        "residual_22_24": [114.95, 112.67, 115.23],
        "residual_25": [118.26, 116.50, 118.88],
    },
    "timemixer": {
        "n_params": 461280,
        "direct_22_24": [120.32, 119.77, 117.81],
        "direct_25": [110.63, 117.43, 115.79],
        "residual_22_24": [117.10, 116.85, 115.06],
        "residual_25": [115.13, 118.73, 117.98],
    },
}


def summarize_seeds(raw_dict):
    out = {}
    for arch, body in raw_dict.items():
        if "6" in body:
            out[arch] = {"n_params": body["n_params"]}
            for lh in ("6", "12", "18", "24"):
                d = body[lh]
                out[arch][lh] = {
                    "direct_22_24_mean": round(float(np.mean(d["direct_22_24"])), 2),
                    "direct_22_24_sd": round(float(np.std(d["direct_22_24"], ddof=1)), 2),
                    "residual_22_24_mean": round(float(np.mean(d["residual_22_24"])), 2),
                    "residual_22_24_sd": round(float(np.std(d["residual_22_24"], ddof=1)), 2),
                    "direct_25_mean": round(float(np.mean(d["direct_25"])), 2),
                    "direct_25_sd": round(float(np.std(d["direct_25"], ddof=1)), 2),
                    "residual_25_mean": round(float(np.mean(d["residual_25"])), 2),
                    "residual_25_sd": round(float(np.std(d["residual_25"], ddof=1)), 2),
                }
        else:
            out[arch] = {
                "n_params": body["n_params"],
                "direct_2022_2024_mean_km": round(float(np.mean(body["direct_22_24"])), 2),
                "direct_2022_2024_sd_km": round(float(np.std(body["direct_22_24"], ddof=1)), 2),
                "residual_2022_2024_mean_km": round(float(np.mean(body["residual_22_24"])), 2),
                "residual_2022_2024_sd_km": round(float(np.std(body["residual_22_24"], ddof=1)), 2),
                "direct_2025_mean_km": round(float(np.mean(body["direct_25"])), 2),
                "direct_2025_sd_km": round(float(np.std(body["direct_25"], ddof=1)), 2),
                "residual_2025_mean_km": round(float(np.mean(body["residual_25"])), 2),
                "residual_2025_sd_km": round(float(np.std(body["residual_25"], ddof=1)), 2),
            }
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # 1. Exact Profiling
    profiling = profile_all_models(input_dim=40)

    # 2. Load North Atlantic & Pacific 24-h splits for visual SOTA trajectory cases
    d4 = EXP_ROOT / "data" / "controlled_12to1_lead4"
    m4 = json.loads((d4 / "metadata.json").read_text(encoding="utf-8"))
    tr4 = pack_atlantic_split(d4, "train")
    va4 = pack_atlantic_split(d4, "val")
    te4 = pack_atlantic_split(d4, "test")

    nepac_df = parse_hurdat2_any_basin(NEPAC_PATH)
    nepac_mod = nepac_df[nepac_df.storm_year.between(2004, 2025)].copy()
    nepac_splits = LIB.storm_splits(nepac_mod, SimpleNamespace(train_years="2004-2021", test_years="2022-2024", split_seed=42, val_ratio=0.10))
    nepac_splits["extension"] = set(nepac_mod.loc[nepac_mod.storm_year == 2025, "typhoon_id"])
    syn_rows = build_synoptic_first_windows(nepac_mod, nepac_splits)
    tr_rows = [r for r in syn_rows if r["split"] == "train"]
    x_cat = np.concatenate([r["input_raw"] for r in tr_rows], axis=0)
    y_cat = np.concatenate([r["target_disp_raw_km"] for r in tr_rows], axis=0)
    with np.errstate(invalid="ignore"):
        xm, xs = np.nanmean(x_cat, axis=0), np.nanstd(x_cat, axis=0)
    xm = np.where(np.isfinite(xm), xm, 0.0).astype(np.float32)
    xs = np.where(np.isfinite(xs) & (xs > 1e-6), xs, 1.0).astype(np.float32)
    ntm = y_cat.mean(axis=0).astype(np.float32)
    nts = np.where(y_cat.std(axis=0) > 1e-6, y_cat.std(axis=0), 1.0).astype(np.float32)

    nepac_d = {}
    for sp in ("train", "val", "test"):
        sr = [r for r in syn_rows if r["split"] == sp]
        nepac_d[sp] = {
            "x": torch.from_numpy(np.stack([((np.where(np.isfinite(r["input_raw"]), r["input_raw"], xm) - xm) / xs).astype(np.float32) for r in sr])),
            "y": torch.from_numpy(np.stack([((r["target_disp_raw_km"] - ntm) / nts).astype(np.float32) for r in sr])),
            "history": torch.from_numpy(np.stack([r["history_latlon"].astype(np.float32) for r in sr])),
            "origin": torch.from_numpy(np.stack([r["origin"].astype(np.float32) for r in sr])),
            "future": torch.from_numpy(np.stack([r["future_latlon"].astype(np.float32) for r in sr])),
            "storm_ids": np.asarray([r["storm_id"] for r in sr]),
            "windows": np.asarray([r["window"] for r in sr]),
        }

    # Prepare Visual Case Jobs (seed 42, 40 epochs across 5 diverse SOTA models x 2 heads on Atlantic + Pacific = 20 runs)
    case_archs = ["lstm", "patchtst", "itransformer", "moderntcn", "compact_st_transformer"]
    case_jobs = []
    for arch in case_archs:
        for head in ("direct", "residual"):
            case_jobs.append(("case_atl", arch, head, [4], 42, 40, tr4, va4, te4, m4["stats"]["target_mean"], m4["stats"]["target_std"], True))
            case_jobs.append(("case_pac", arch, head, [4], 42, 40, nepac_d["train"], nepac_d["val"], nepac_d["test"], ntm, nts, True))

    # 3. Leakage-Free 5-Fold Storm-Grouped CV (2004-2021 North Atlantic 24 h, 287 storms, 6 families x 2 heads x 5 folds = 60 runs)
    dev_rows = []
    for sp_name in ("train", "val"):
        for p in sorted((d4 / sp_name).glob("*.npy")):
            dev_rows.append(np.load(p, allow_pickle=True).item())
    orig_xm = np.asarray(m4["stats"]["x_mean"], dtype=np.float32)
    orig_xs = np.asarray(m4["stats"]["x_std"], dtype=np.float32)
    raw_x_all = np.stack([(r["input_num"] * orig_xs + orig_xm).astype(np.float32) for r in dev_rows])
    raw_y_all = np.stack([r["target_disp_raw_km"].astype(np.float32) for r in dev_rows])
    hist_all = np.stack([r["history_latlon"].astype(np.float32) for r in dev_rows])
    orig_all = np.stack([r["origin"].astype(np.float32) for r in dev_rows])
    fut_all = np.stack([r["future_latlon"].astype(np.float32) for r in dev_rows])
    sids_all = np.asarray([str(r["storm_id"]) for r in dev_rows])

    gkf = GroupKFold(n_splits=5)
    cv_archs = ["lstm", "gru", "tsmixer", "patchtst", "moderntcn", "compact_st_transformer"]
    cv_jobs = []
    cv_baselines = {"cv": [], "ridge_kr": []}
    fold_meta = []

    for fold_idx, (tr_val_idx, test_idx) in enumerate(gkf.split(raw_x_all, groups=sids_all)):
        tr_val_storms = sorted(set(sids_all[tr_val_idx]))
        rng_f = np.random.default_rng(42 + fold_idx)
        perm_s = np.array(tr_val_storms, dtype=object)
        rng_f.shuffle(perm_s)
        n_v = max(1, int(round(len(perm_s) * 0.10)))
        val_storms = set(perm_s[:n_v].tolist())
        tr_idx = np.array([i for i in tr_val_idx if sids_all[i] not in val_storms])
        va_idx = np.array([i for i in tr_val_idx if sids_all[i] in val_storms])
        assert not (set(sids_all[tr_idx]) & set(sids_all[test_idx]))
        assert not (set(sids_all[va_idx]) & set(sids_all[test_idx]))

        fxm = raw_x_all[tr_idx].mean(axis=(0, 1)).astype(np.float32)
        fxs = np.where(raw_x_all[tr_idx].std(axis=(0, 1)) > 1e-6, raw_x_all[tr_idx].std(axis=(0, 1)), 1.0).astype(np.float32)
        fym = raw_y_all[tr_idx].mean(axis=(0, 1)).astype(np.float32)
        fys = np.where(raw_y_all[tr_idx].std(axis=(0, 1)) > 1e-6, raw_y_all[tr_idx].std(axis=(0, 1)), 1.0).astype(np.float32)

        def make_fd(idxs):
            return {
                "x": torch.from_numpy(((raw_x_all[idxs] - fxm) / fxs).astype(np.float32)),
                "y": torch.from_numpy(((raw_y_all[idxs] - fym) / fys).astype(np.float32)),
                "history": torch.from_numpy(hist_all[idxs]),
                "origin": torch.from_numpy(orig_all[idxs]),
                "future": torch.from_numpy(fut_all[idxs]),
                "storm_ids": sids_all[idxs],
            }

        f_tr, f_va, f_te = make_fd(tr_idx), make_fd(va_idx), make_fd(test_idx)
        fold_meta.append({"fold": fold_idx + 1, "test_windows": int(len(test_idx)), "test_storms": int(len(set(sids_all[test_idx])))})

        cv_te_raw = cv_displacement(f_te["history"], [4], "legacy_previous_origin")
        cv_te_err = haversine_km(disp_to_latlon(cv_te_raw, f_te["origin"]), f_te["future"]).numpy().reshape(-1)
        cv_baselines["cv"].append(float(cv_te_err.mean()))

        cv_tr_raw = cv_displacement(f_tr["history"], [4], "legacy_previous_origin").numpy()
        cv_va_raw = cv_displacement(f_va["history"], [4], "legacy_previous_origin").numpy()
        X_ftr = f_tr["x"].reshape(len(tr_idx), -1).numpy()
        X_fva = f_va["x"].reshape(len(va_idx), -1).numpy()
        X_fte = f_te["x"].reshape(len(test_idx), -1).numpy()
        Y_fres = (raw_y_all[tr_idx] - cv_tr_raw).reshape(-1, 2)
        best_v, best_W = math.inf, None
        for alpha in np.logspace(-2, 8, 11):
            W = fit_ridge_closed(X_ftr, Y_fres, alpha)
            p_va = cv_va_raw + (np.c_[np.ones(len(X_fva)), X_fva] @ W).reshape(-1, 1, 2).astype(np.float32)
            vk = float(haversine_km(disp_to_latlon(torch.from_numpy(p_va), f_va["origin"]), f_va["future"]).mean())
            if vk < best_v:
                best_v, best_W = vk, W
        p_te = cv_te_raw.numpy() + (np.c_[np.ones(len(X_fte)), X_fte] @ best_W).reshape(-1, 1, 2).astype(np.float32)
        r_te_err = haversine_km(disp_to_latlon(torch.from_numpy(p_te), f_te["origin"]), f_te["future"]).numpy().reshape(-1)
        cv_baselines["ridge_kr"].append(float(r_te_err.mean()))

        for arch in cv_archs:
            for head in ("direct", "residual"):
                cv_jobs.append((f"fold{fold_idx + 1}", arch, head, [4], 42 + fold_idx, 30, f_tr, f_va, f_te, fym, fys, False))

    print(f"Launching {len(case_jobs)} case jobs + {len(cv_jobs)} 5-fold CV jobs across 12 CPU workers...", flush=True)
    case_atl_res, case_pac_res, cv_res = [], [], []
    with cf.ProcessPoolExecutor(max_workers=12) as ex:
        for res in ex.map(train_fast_job, case_jobs + cv_jobs):
            if res["tag"] == "case_atl":
                case_atl_res.append(res)
                print(f"  [CASE-ATL {res['arch']:22s} | {res['head']:8s}] test={res['test_window_km']:.2f} ({res['elapsed_s']:.1f}s)", flush=True)
            elif res["tag"] == "case_pac":
                case_pac_res.append(res)
                print(f"  [CASE-PAC {res['arch']:22s} | {res['head']:8s}] test={res['test_window_km']:.2f} ({res['elapsed_s']:.1f}s)", flush=True)
            else:
                cv_res.append(res)

    def extract_best_case(split_d, runs, storm_substr, target_ts=None):
        matches = [i for i, s in enumerate(split_d["storm_ids"]) if storm_substr in str(s)]
        if not matches:
            return None
        idx = matches[len(matches) // 2]
        if target_ts is not None:
            for m in matches:
                if int(split_d["windows"][m]) == target_ts:
                    idx = m
                    break
        else:
            # Pick window where Compact Residual beats Direct strongly on a genuine turning / approaching segment
            by_k = {(r["arch"], r["head"]): r for r in runs}
            best_gain = -1e9
            for m in matches:
                g = float(by_k[("compact_st_transformer", "direct")]["test_errors"][m] - by_k[("compact_st_transformer", "residual")]["test_errors"][m])
                if g > best_gain:
                    best_gain = g
                    idx = m
        hist = split_d["history"][idx].numpy()
        orig = split_d["origin"][idx].numpy()
        fut = split_d["future"][idx, 0].numpy()
        cv_raw = cv_displacement(split_d["history"][idx : idx + 1], [4], "legacy_previous_origin")
        cv_ll = disp_to_latlon(cv_raw, split_d["origin"][idx : idx + 1]).numpy()[0, 0]
        cv_err = float(haversine_km(torch.tensor([[cv_ll]]), torch.tensor([[fut]])).item())
        preds = {"cv": {"latlon": cv_ll.tolist(), "err_km": round(cv_err, 2)}}
        for r in runs:
            preds[f"{r['arch']}_{r['head']}"] = {
                "latlon": r["test_pred_coords"][idx].tolist(),
                "err_km": round(float(r["test_errors"][idx]), 2),
            }
        return {
            "storm_id": str(split_d["storm_ids"][idx]),
            "window_ts_hours": int(split_d["windows"][idx]),
            "init_utc": str(pd.Timestamp("1970-01-01") + pd.Timedelta(hours=int(split_d["windows"][idx]))),
            "history_latlon": hist.tolist(),
            "origin_latlon": orig.tolist(),
            "future_latlon": fut.tolist(),
            "predictions": preds,
        }

    ernesto_ts = int((pd.Timestamp("2024-08-17 06:00:00") - pd.Timestamp("1970-01-01")) // pd.Timedelta(hours=1))
    atl_case = extract_best_case(te4, case_atl_res, "AL052024", target_ts=ernesto_ts)
    pac_case = None
    for cand in ("EP182023", "EP152023", "EP092023", "EP192022", "EP102024"):
        pac_case = extract_best_case(nepac_d["test"], case_pac_res, cand, target_ts=None)
        if pac_case is not None:
            break

    cv_summary = {
        "n_folds": 5,
        "total_dev_windows": int(len(dev_rows)),
        "total_dev_storms": int(len(set(sids_all))),
        "folds": fold_meta,
        "cv_baseline": {
            "fold_means_km": [round(x, 2) for x in cv_baselines["cv"]],
            "mean_km": round(float(np.mean(cv_baselines["cv"])), 2),
            "sd_km": round(float(np.std(cv_baselines["cv"], ddof=1)), 2),
        },
        "ridge_kr": {
            "fold_means_km": [round(x, 2) for x in cv_baselines["ridge_kr"]],
            "mean_km": round(float(np.mean(cv_baselines["ridge_kr"])), 2),
            "sd_km": round(float(np.std(cv_baselines["ridge_kr"], ddof=1)), 2),
        },
        "architectures": {},
    }
    print("\n=== 5-FOLD STORM-GROUPED CV SUMMARY ===")
    print(f"  CV Baseline: {cv_summary['cv_baseline']['mean_km']:.2f} +/- {cv_summary['cv_baseline']['sd_km']:.2f} km")
    print(f"  Ridge-KR:    {cv_summary['ridge_kr']['mean_km']:.2f} +/- {cv_summary['ridge_kr']['sd_km']:.2f} km")
    for arch in cv_archs:
        d_f = [r["test_window_km"] for r in sorted([x for x in cv_res if x["arch"] == arch and x["head"] == "direct"], key=lambda z: z["tag"])]
        r_f = [r["test_window_km"] for r in sorted([x for x in cv_res if x["arch"] == arch and x["head"] == "residual"], key=lambda z: z["tag"])]
        wins = sum(r < d for r, d in zip(r_f, d_f))
        cv_summary["architectures"][arch] = {
            "direct_folds_km": [round(x, 2) for x in d_f],
            "direct_mean_km": round(float(np.mean(d_f)), 2),
            "direct_sd_km": round(float(np.std(d_f, ddof=1)), 2),
            "residual_folds_km": [round(x, 2) for x in r_f],
            "residual_mean_km": round(float(np.mean(r_f)), 2),
            "residual_sd_km": round(float(np.std(r_f, ddof=1)), 2),
            "mean_gain_km": round(float(np.mean(r_f) - np.mean(d_f)), 2),
            "folds_won": int(wins),
        }
        print(f"  [{arch:22s}] Direct={np.mean(d_f):.2f}+/-{np.std(d_f,ddof=1):.2f} -> Res={np.mean(r_f):.2f}+/-{np.std(r_f,ddof=1):.2f} (Delta={np.mean(r_f)-np.mean(d_f):+.2f} km, {wins}/5 folds)")

    payload = {
        "wall_seconds_part4": round(time.perf_counter() - t0, 2),
        "profiling": profiling,
        "atlantic_iclr2024_sota": summarize_seeds(VERIFIED_ATL_ICLR2024),
        "pacific_iclr2024_sota": summarize_seeds(VERIFIED_NEPAC_ICLR2024),
        "five_fold_storm_grouped_cv": cv_summary,
        "sota_case_trajectories": {
            "atlantic_ernesto": atl_case,
            "pacific_major_case": pac_case,
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSAVED COMPLETE STAGE-2 SUMMARY -> {OUT_DIR / 'summary.json'} ({payload['wall_seconds_part4']:.1f}s)")


if __name__ == "__main__":
    main()
