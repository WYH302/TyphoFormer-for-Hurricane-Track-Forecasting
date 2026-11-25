"""Evaluate Unified 16-ch KinResTC (ModernTCN+KR & ST-B64, L=8, EMA) on the
Synoptic-First Reconstructed North Atlantic HURDAT2 Archive (2004-2025) on GPU.

Strictly adheres to storm-disjoint chronological splits (2004-2021 train/val,
2022-2024 primary test [905 windows, 40 storms, 101 future-L], and 2025
out-of-sample season [187 windows, 8 storms, 10 Hurricane Melissa future-L]),
training-only z-score normalization, seeds {42, 123, 2024}, and 10,000-draw
paired whole-storm cluster bootstrap 95% confidence intervals.
"""
import concurrent.futures as cf
import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP_ROOT / "synoptic_reconstruction"))

import prepare_revision as prep
from controlled_heads import cv_displacement
from sota_sequence_baselines import load_split_fast
from tier1_extreme_roi_surgery import (
    KIN_NAMES,
    SurgicalResidualModel,
    denorm_disp_mh,
    norm_disp_mh,
    update_ema,
)
from vendor.common import disp_to_latlon, haversine_km, seed_everything

OUT_DIR = EXP_ROOT / "results" / "kinrestc_synoptic_landfall"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 2024]
BOOTSTRAP_SEED = 20260913


class Direct16ChModel(nn.Module):
    """Matched Eulerian Direct baseline (same ModernTCN 16-ch L=8 trunk, zero CV anchor)."""

    def __init__(self, input_dim: int, seq_len: int, target_mean, target_std, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        from stage2_sota_cv_profile import ModernTCNEncoder
        self.backbone = ModernTCNEncoder(input_dim, seq_len=seq_len, d_model=d_model, num_blocks=2, dropout=dropout)
        d = self.backbone.out_dim
        self.head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 2))
        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.as_tensor(target_std, dtype=torch.float32))

    def forward(self, x: torch.Tensor, history: torch.Tensor):
        ctx = self.backbone(x)
        return self.head(ctx).unsqueeze(1)


def summarize_subset(err_km: np.ndarray, storm_ids: np.ndarray, mask: np.ndarray):
    e = err_km[mask]
    sids = storm_ids[mask]
    uniq = sorted(set(sids.tolist()))
    storm_means = np.array([e[sids == s].mean() for s in uniq], dtype=np.float64)
    return {
        "windows": int(mask.sum()),
        "storms": int(len(uniq)),
        "window_mean_km": float(e.mean()),
        "equal_storm_km": float(storm_means.mean()),
    }


def paired_storm_bootstrap(err_a: np.ndarray, err_b: np.ndarray, storm_ids: np.ndarray, mask: np.ndarray, draws: int = 10000):
    """Paired storm-cluster bootstrap of (err_a - err_b)."""
    diff = (err_a - err_b)[mask]
    sids = storm_ids[mask]
    uniq = sorted(set(sids.tolist()))
    sums = np.array([diff[sids == s].sum() for s in uniq], dtype=np.float64)
    cnts = np.array([(sids == s).sum() for s in uniq], dtype=np.float64)
    smeans = sums / cnts
    res = {
        "windows": int(mask.sum()),
        "storms": int(len(uniq)),
        "win_diff_km": float(diff.mean()),
        "storm_diff_km": float(smeans.mean()),
    }
    if len(uniq) >= 2:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        idx = rng.integers(0, len(uniq), size=(draws, len(uniq)))
        w_draws = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
        s_draws = smeans[idx].mean(axis=1)
        res["win_ci95"] = [float(np.quantile(w_draws, 0.025)), float(np.quantile(w_draws, 0.975))]
        res["storm_ci95"] = [float(np.quantile(s_draws, 0.025)), float(np.quantile(s_draws, 0.975))]
    return res


def pack_rows(rows, x_mean, x_std, t_mean, t_std):
    x_raw = np.stack([r["input_raw"] for r in rows]).astype(np.float32)
    x_imp = np.where(np.isfinite(x_raw), x_raw, x_mean[None, None, :])
    x_norm = ((x_imp - x_mean[None, None, :]) / x_std[None, None, :]).astype(np.float32)
    hist = np.stack([r["history_latlon"] for r in rows]).astype(np.float32)
    orig = np.stack([r["origin"] for r in rows]).astype(np.float32)
    fut = np.stack([r["future_latlon"] for r in rows]).astype(np.float32)
    y_raw = np.stack([r["target_disp_raw_km"] for r in rows]).astype(np.float32)
    y_norm = ((y_raw - t_mean[None, None, :]) / t_std[None, None, :]).astype(np.float32)
    sids = np.array([str(r["storm_id"]) for r in rows])
    wins = np.array([int(r["window"]) for r in rows], dtype=np.int64)
    fut_l = np.array([bool(r["recorded_future_landfall"]) for r in rows], dtype=bool)
    return {
        "x": torch.from_numpy(x_norm),
        "history": torch.from_numpy(hist),
        "origin": torch.from_numpy(orig),
        "future": torch.from_numpy(fut),
        "y": torch.from_numpy(y_norm),
        "y_raw": y_raw,
        "storm_ids": sids,
        "windows": wins,
        "keys": list(zip(sids.tolist(), wins.tolist())),
        "future_landfall": fut_l,
    }


@torch.no_grad()
def predict_gpu(model, data, feat_idx, seq_len: int, batch_size: int = 256):
    model.eval()
    preds_ll, errs = [], []
    n = len(data["keys"])
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        bx = data["x"][sl, -seq_len:, :][..., feat_idx].to(DEVICE)
        bh = data["history"][sl].to(DEVICE)
        bo = data["origin"][sl].to(DEVICE)
        bf = data["future"][sl].to(DEVICE)
        pred_norm = model(bx, bh)
        raw = denorm_disp_mh(pred_norm, model.target_mean, model.target_std)
        ll = disp_to_latlon(raw, bo)
        preds_ll.append(ll.cpu().numpy())
        errs.append(haversine_km(ll, bf).cpu().numpy())
    return np.concatenate(preds_ll, axis=0), np.concatenate(errs, axis=0).reshape(-1)


def train_one_gpu(mode: str, arch: str, seed: int, tr, va, feat_idx, seq_len: int, t_mean, t_std, epochs: int = 40):
    seed_everything(seed)
    if mode == "direct":
        model = Direct16ChModel(len(feat_idx), seq_len, t_mean, t_std, d_model=128, dropout=0.1).to(DEVICE)
    else:
        model = SurgicalResidualModel(
            arch=arch, input_dim=len(feat_idx), seq_len=seq_len, lead_steps=[4],
            target_mean=t_mean, target_std=t_std, anchor_mode="cv1",
            d_model=128 if arch == "moderntcn" else 64, dropout=0.1,
        ).to(DEVICE)
    ema_model = copy.deepcopy(model).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    x_tr = tr["x"][:, -seq_len:, :][..., feat_idx].to(DEVICE)
    h_tr = tr["history"].to(DEVICE)
    y_tr = tr["y"].to(DEVICE)
    n_tr = len(tr["keys"])
    best_val, best_state = math.inf, None

    # Pre-move validation tensors to GPU once for sub-millisecond epoch evaluation
    va_x = va["x"][:, -seq_len:, :][..., feat_idx].to(DEVICE)
    va_h = va["history"].to(DEVICE)
    va_o = va["origin"].to(DEVICE)
    va_f = va["future"].to(DEVICE)

    for epoch in range(epochs):
        model.train()
        gen = torch.Generator().manual_seed(seed + epoch + 1)
        perm = torch.randperm(n_tr, generator=gen).to(DEVICE)
        for ids_gpu in perm.split(32):
            opt.zero_grad(set_to_none=True)
            pred_norm = model(x_tr[ids_gpu], h_tr[ids_gpu])
            loss = F.mse_loss(pred_norm, y_tr[ids_gpu])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update_ema(ema_model, model, decay=0.985)

        eval_m = ema_model if epoch >= 2 else model
        eval_m.eval()
        with torch.no_grad():
            p_va = eval_m(va_x, va_h)
            raw_va = denorm_disp_mh(p_va, eval_m.target_mean, eval_m.target_std)
            ll_va = disp_to_latlon(raw_va, va_o)
            vk = float(haversine_km(ll_va, va_f).mean().item())
        if vk < best_val:
            best_val = vk
            best_state = {k: v.detach().cpu().clone() for k, v in eval_m.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val


def eval_suite(models_by_seed, te_data, masks_22, ex_data, masks_25, feat_idx, seq_len: int):
    seed_errs_22, seed_lls_22 = [], []
    seed_errs_25, seed_lls_25 = [], []
    for m in models_by_seed:
        ll22, e22 = predict_gpu(m, te_data, feat_idx, seq_len)
        ll25, e25 = predict_gpu(m, ex_data, feat_idx, seq_len)
        seed_errs_22.append(e22)
        seed_lls_22.append(ll22)
        seed_errs_25.append(e25)
        seed_lls_25.append(ll25)

    # Mean error across 3 seeds per window (matches Table XIII/XIV/XV protocol) + 3-seed ensemble latlon mean!
    mean_e22 = np.mean(seed_errs_22, axis=0)
    mean_e25 = np.mean(seed_errs_25, axis=0)
    ens_ll22 = np.mean(seed_lls_22, axis=0)
    ens_ll25 = np.mean(seed_lls_25, axis=0)
    ens_e22 = haversine_km(torch.from_numpy(ens_ll22), te_data["future"]).numpy().reshape(-1)
    ens_e25 = haversine_km(torch.from_numpy(ens_ll25), ex_data["future"]).numpy().reshape(-1)

    out = {"test_2022_2024": {}, "ext_2025": {}, "test_2022_2024_ens": {}, "ext_2025_ens": {}}
    for k, msk in masks_22.items():
        out["test_2022_2024"][k] = summarize_subset(mean_e22, te_data["storm_ids"], msk)
        out["test_2022_2024_ens"][k] = summarize_subset(ens_e22, te_data["storm_ids"], msk)
    for k, msk in masks_25.items():
        out["ext_2025"][k] = summarize_subset(mean_e25, ex_data["storm_ids"], msk)
        out["ext_2025_ens"][k] = summarize_subset(ens_e25, ex_data["storm_ids"], msk)
    return out, mean_e22, ens_e22, mean_e25, ens_e25


def main():
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    print(f"Running KinResTC Synoptic-First Landfall GPU Verification on {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Build EXACT synoptic-first Atlantic 2004-2025 rows via prepare_revision
    frame = prep.validate(prep.LIB.parse_hurdat2(str(prep.SOURCE)))
    frame = frame[frame.storm_year.between(2004, 2025)].copy()
    splits = prep.LIB.storm_splits(frame, SimpleNamespace(train_years="2004-2021", test_years="2022-2024", split_seed=42, val_ratio=0.1))
    splits["extension"] = set(frame.loc[frame.storm_year == 2025, "typhoon_id"])
    rows = prep.make_rows(frame, splits)

    syn_tr_rows = [r for r in rows if r["split"] == "train"]
    syn_va_rows = [r for r in rows if r["split"] == "val"]
    syn_te_rows = [r for r in rows if r["split"] == "test"]
    syn_ex_rows = [r for r in rows if r["split"] == "extension"]
    print(f"Synoptic-first counts: train={len(syn_tr_rows)}, val={len(syn_va_rows)}, test={len(syn_te_rows)}, ext25={len(syn_ex_rows)}")
    assert len(syn_tr_rows) == 4643 and len(syn_va_rows) == 410 and len(syn_te_rows) == 905 and len(syn_ex_rows) == 187

    # Identify exact original 630 & 172 windows from baseline dataset timestamps
    base_meta = json.loads((EXP_ROOT / "data/controlled_12to1_lead4/metadata.json").read_text(encoding="utf-8"))
    feat_names = base_meta["feature_names"]
    kin_idx = [i for i, n in enumerate(feat_names) if n in KIN_NAMES]
    assert len(kin_idx) == 16

    base_te_files = sorted((EXP_ROOT / "data/controlled_12to1_lead4/test").glob("*.npy"))
    base_ex_files = sorted((EXP_ROOT / "data/extension2025_lead4/test").glob("*.npy"))
    orig_keys_22 = {(str(d["storm_id"]), int(d["history_timestamp_hours"][-1])) for p in base_te_files for d in [np.load(p, allow_pickle=True).item()]}
    orig_keys_25 = {(str(d["storm_id"]), int(d["history_timestamp_hours"][-1])) for p in base_ex_files for d in [np.load(p, allow_pickle=True).item()]}
    assert len(orig_keys_22) == 630 and len(orig_keys_25) == 172

    # Pack synoptic-first dataset with its OWN training-only stats (Protocol B: end-to-end retrained)
    syn_stats = prep.fit_stats(rows)
    syn_tr = pack_rows(syn_tr_rows, syn_stats["x_mean"], syn_stats["x_std"], syn_stats["target_mean"], syn_stats["target_std"])
    syn_va = pack_rows(syn_va_rows, syn_stats["x_mean"], syn_stats["x_std"], syn_stats["target_mean"], syn_stats["target_std"])
    syn_te = pack_rows(syn_te_rows, syn_stats["x_mean"], syn_stats["x_std"], syn_stats["target_mean"], syn_stats["target_std"])
    syn_ex = pack_rows(syn_ex_rows, syn_stats["x_mean"], syn_stats["x_std"], syn_stats["target_mean"], syn_stats["target_std"])

    # Also pack baseline-trained splits + synoptic test normalized with baseline training stats (Protocol A: zero-shot baseline transfer)
    base_tr_d = load_split_fast(EXP_ROOT / "data/controlled_12to1_lead4", "train")
    base_va_d = load_split_fast(EXP_ROOT / "data/controlled_12to1_lead4", "val")
    base_tr = {
        "x": base_tr_d["tensors"]["x"], "history": base_tr_d["tensors"]["history"],
        "origin": base_tr_d["tensors"]["origin"], "future": base_tr_d["tensors"]["future"],
        "y": base_tr_d["tensors"]["y"], "keys": base_tr_d["keys"],
    }
    base_va = {
        "x": base_va_d["tensors"]["x"], "history": base_va_d["tensors"]["history"],
        "origin": base_va_d["tensors"]["origin"], "future": base_va_d["tensors"]["future"],
        "y": base_va_d["tensors"]["y"], "keys": base_va_d["keys"],
    }
    bx_mean = np.asarray(base_meta["stats"]["x_mean"], dtype=np.float32)
    bx_std = np.asarray(base_meta["stats"]["x_std"], dtype=np.float32)
    bt_mean = np.asarray(base_meta["stats"]["target_mean"], dtype=np.float32)
    bt_std = np.asarray(base_meta["stats"]["target_std"], dtype=np.float32)
    syn_te_base_norm = pack_rows(syn_te_rows, bx_mean, bx_std, bt_mean, bt_std)
    syn_ex_base_norm = pack_rows(syn_ex_rows, bx_mean, bx_std, bt_mean, bt_std)

    # Exact subset masks
    orig_m22 = np.array([k in orig_keys_22 for k in syn_te["keys"]], dtype=bool)
    rest_m22 = ~orig_m22
    comb_m22 = np.ones(len(syn_te["keys"]), dtype=bool)
    futl_m22 = syn_te["future_landfall"]
    assert orig_m22.sum() == 630 and rest_m22.sum() == 275 and futl_m22.sum() == 101

    orig_m25 = np.array([k in orig_keys_25 for k in syn_ex["keys"]], dtype=bool)
    rest_m25 = ~orig_m25
    comb_m25 = np.ones(len(syn_ex["keys"]), dtype=bool)
    futl_m25 = syn_ex["future_landfall"]
    assert orig_m25.sum() == 172 and rest_m25.sum() == 15 and futl_m25.sum() == 10

    masks_22 = {"orig_630": orig_m22, "restored_275": rest_m22, "combined_905": comb_m22, "future_L_101": futl_m22}
    masks_25 = {"orig_172": orig_m25, "restored_15": rest_m25, "combined_187": comb_m25, "future_L_10": futl_m25}

    # CV baseline check (sanity match with Table XIII/XIV/XV!)
    cv22_raw = cv_displacement(syn_te["history"], [4], "legacy_previous_origin")
    cv22_err = haversine_km(disp_to_latlon(cv22_raw, syn_te["origin"]), syn_te["future"]).numpy().reshape(-1)
    cv25_raw = cv_displacement(syn_ex["history"], [4], "legacy_previous_origin")
    cv25_err = haversine_km(disp_to_latlon(cv25_raw, syn_ex["origin"]), syn_ex["future"]).numpy().reshape(-1)
    print(f"CV sanity check: Comb905={cv22_err.mean():.2f} / storm={summarize_subset(cv22_err, syn_te['storm_ids'], comb_m22)['equal_storm_km']:.2f}, FutL101={summarize_subset(cv22_err, syn_te['storm_ids'], futl_m22)['window_mean_km']:.2f}/{summarize_subset(cv22_err, syn_te['storm_ids'], futl_m22)['equal_storm_km']:.2f}")

    results = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "counts": {
            "syn_train": 4643, "syn_val": 410,
            "test_2022_2024": {k: int(v.sum()) for k, v in masks_22.items()},
            "ext_2025": {k: int(v.sum()) for k, v in masks_25.items()},
        },
        "cv_baseline": {
            "test_2022_2024": {k: summarize_subset(cv22_err, syn_te["storm_ids"], m) for k, m in masks_22.items()},
            "ext_2025": {k: summarize_subset(cv25_err, syn_ex["storm_ids"], m) for k, m in masks_25.items()},
        },
    }

    # Train all 12 lightweight models sequentially with 4 CPU threads for 100% deterministic seeding & zero GPU contention
    print("\n--- Training 12 deterministic models (4 CPU threads, zero GPU contention) ---", flush=True)
    job_specs = (
        [("base_res", "residual", "moderntcn", s, base_tr, base_va, bt_mean, bt_std) for s in SEEDS]
        + [("syn_dir", "direct", "moderntcn", s, syn_tr, syn_va, syn_stats["target_mean"], syn_stats["target_std"]) for s in SEEDS]
        + [("syn_res", "residual", "moderntcn", s, syn_tr, syn_va, syn_stats["target_mean"], syn_stats["target_std"]) for s in SEEDS]
        + [("syn_stb", "residual", "st_b64", s, syn_tr, syn_va, syn_stats["target_mean"], syn_stats["target_std"]) for s in SEEDS]
    )
    trained = {k: [None] * len(SEEDS) for k in ("base_res", "syn_dir", "syn_res", "syn_stb")}
    for tag, mode, arch, s, tr_d, va_d, tm, ts in job_specs:
        tm0 = time.perf_counter()
        model_i, bval = train_one_gpu(mode, arch, s, tr_d, va_d, kin_idx, 8, tm, ts)
        trained[tag][SEEDS.index(s)] = model_i
        print(f"  [DONE] {tag:8s} seed={s:4d} best_val={bval:6.2f} km ({time.perf_counter()-tm0:4.1f}s | total {time.perf_counter()-t0:5.1f}s)", flush=True)

    res_base_eval, be22, be22_ens, be25, be25_ens = eval_suite(trained["base_res"], syn_te_base_norm, masks_22, syn_ex_base_norm, masks_25, kin_idx, 8)
    results["baseline_trained_kinrestc_16ch"] = res_base_eval
    res_syn_dir, sd22, sd22_ens, sd25, sd25_ens = eval_suite(trained["syn_dir"], syn_te, masks_22, syn_ex, masks_25, kin_idx, 8)
    results["synoptic_retrained_direct_16ch"] = res_syn_dir
    res_syn_res, sr22, sr22_ens, sr25, sr25_ens = eval_suite(trained["syn_res"], syn_te, masks_22, syn_ex, masks_25, kin_idx, 8)
    results["synoptic_retrained_kinrestc_16ch"] = res_syn_res
    res_syn_stb, sb22, sb22_ens, sb25, sb25_ens = eval_suite(trained["syn_stb"], syn_te, masks_22, syn_ex, masks_25, kin_idx, 8)
    results["synoptic_retrained_st_b64_16ch"] = res_syn_stb

    # Paired bootstrap CIs for Retrained KinResTC vs Retrained Direct & vs CV
    results["paired_bootstrap_retrained"] = {
        "kinrestc_minus_direct_comb905": paired_storm_bootstrap(sr22, sd22, syn_te["storm_ids"], comb_m22),
        "kinrestc_minus_direct_futL101": paired_storm_bootstrap(sr22, sd22, syn_te["storm_ids"], futl_m22),
        "kinrestc_minus_cv_comb905": paired_storm_bootstrap(sr22, cv22_err, syn_te["storm_ids"], comb_m22),
        "kinrestc_minus_cv_futL101": paired_storm_bootstrap(sr22, cv22_err, syn_te["storm_ids"], futl_m22),
        "kinrestc_minus_direct_ext187": paired_storm_bootstrap(sr25, sd25, syn_ex["storm_ids"], comb_m25),
        "kinrestc_minus_cv_ext187": paired_storm_bootstrap(sr25, cv25_err, syn_ex["storm_ids"], comb_m25),
    }

    elapsed = time.perf_counter() - t0
    results["runtime_seconds"] = round(elapsed, 2)
    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n================ FINAL GPU SUMMARY ================")
    for label, d in [
        ("Retrained Direct 16ch (Eulerian)", res_syn_dir),
        ("Retrained KinResTC 16ch (ModernTCN+KR)", res_syn_res),
        ("Retrained ST-B64 16ch (53.9K)", res_syn_stb),
    ]:
        c22 = d["test_2022_2024"]["combined_905"]
        f22 = d["test_2022_2024"]["future_L_101"]
        c25 = d["ext_2025"]["combined_187"]
        f25 = d["ext_2025"]["future_L_10"]
        print(f"{label:40s} | 22-24 Comb905: {c22['window_mean_km']:6.2f}/{c22['equal_storm_km']:6.2f} | FutL101: {f22['window_mean_km']:6.2f}/{f22['equal_storm_km']:6.2f} | 25 Comb187: {c25['window_mean_km']:6.2f}/{c25['equal_storm_km']:6.2f} | Melissa10: {f25['window_mean_km']:6.2f}")
    print(f"Done in {elapsed:.1f}s -> saved {out_path}")


if __name__ == "__main__":
    main()
