"""Tier-1 Extreme ROI Surgical Probe:
Zero extra training data; strictly identical North Atlantic (AL) & Eastern/Central
North Pacific (EP/CP) storm-disjoint splits. Audits 4 surgical upgrades:
  1. Physics Anchor Upgrade: 1-step CV vs. 0-param 12-h Smoothed Velocity vs.
     Closed-Form Ridge-KR Linear Backbone + Neural Nonlinear Residual (training-only SVD).
  2. Feature/Lookback/EMA/Loss/Ensemble ROI Sweep: 16-ch Pure Kinematics, L=8/L=4,
     ModernTCN vs. Compact ST-B64, EMA (decay=0.985), Differentiable Haversine Loss,
     and 3-Seed Lightweight Ensembling.
  3. Multi-Horizon Non-Iterative Joint Decoder (4-in-1 shared backbone across 6/12/18/24 h
     with cross-horizon velocity smoothness regularization -> 4x param/compute compression).
  4. Zero-Cost Kinematic Adaptive Conformal Prediction (90% / 80% dynamic uncertainty cones
     calibrated strictly on 2004-2021 validation residuals, verified on unseen 2022-2024 & 2025).
  5. Cross-Basin Verification on existing Eastern/Central North Pacific HURDAT2 archive.
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
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from controlled_heads import ControlledNumericModel, cv_displacement
from eastern_pacific_cross_basin import (
    NEPAC_PATH,
    build_synoptic_first_windows,
    parse_hurdat2_any_basin,
)
from sota_sequence_baselines import load_split_fast
from stage2_sota_cv_profile import ModernTCNEncoder
from vendor.common import (
    denormalize_disp,
    disp_to_latlon,
    haversine_km,
    latlon_to_disp,
    normalize_disp,
    seed_everything,
)

SPEC = importlib.util.spec_from_file_location("original_prepare", EXP_ROOT / "data" / "prepare_frozen.py")
LIB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIB)

OUT_DIR = EXP_ROOT / "results" / "tier1_extreme_roi_surgery"

KIN_NAMES = {
    "lat", "lon", "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
    "hour_sin", "hour_cos", "dx6", "dy6", "speed6", "heading_sin", "heading_cos",
    "accel6", "turn_sin", "turn_cos",
}


# ==============================================================================
# Analytical Anchors (0 extra params) & Closed-Form Training-Only Ridge Anchor
# ==============================================================================
def norm_disp_mh(y_raw: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    return (y_raw - target_mean.view(1, -1, 2)) / target_std.view(1, -1, 2)


def denorm_disp_mh(y_norm: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    return y_norm * target_std.view(1, -1, 2) + target_mean.view(1, -1, 2)


def smoothed_cv_displacement(history_latlon: torch.Tensor, lead_steps: torch.Tensor, alpha: float = 0.75) -> torch.Tensor:
    """0-parameter damped 12-h velocity anchor fighting HURDAT2 0.1-deg quantization jitter."""
    # history_latlon: (B, L, 2) where [:, -1] = p_T, [:, -2] = p_{T-1}, [:, -3] = p_{T-2}
    p_t = history_latlon[:, -1]       # (B, 2)
    p_tm1 = history_latlon[:, -2]     # (B, 2)
    p_tm2 = history_latlon[:, -3]     # (B, 2)
    v1 = latlon_to_disp(p_t, p_tm1)   # (B, 2)
    v0 = latlon_to_disp(p_tm1, p_tm2) # (B, 2)
    v_smooth = alpha * v1 + (1.0 - alpha) * v0
    steps = lead_steps.to(history_latlon.device, dtype=history_latlon.dtype).view(1, -1, 1)
    return v_smooth[:, None, :] * steps


def fit_ridge_kr_anchor(tr_x: np.ndarray, tr_hist: np.ndarray, tr_raw_y: np.ndarray,
                        va_x: np.ndarray, va_hist: np.ndarray, va_orig: np.ndarray, va_fut: np.ndarray,
                        lead_step: int):
    """Fit exact closed-form Ridge residual over 1-step CV strictly on train, select alpha on val."""
    tr_cv = cv_displacement(torch.from_numpy(tr_hist), [lead_step], "legacy_previous_origin").numpy()[:, 0, :]
    va_cv = cv_displacement(torch.from_numpy(va_hist), [lead_step], "legacy_previous_origin").numpy()[:, 0, :]
    tr_res = tr_raw_y[:, 0, :] - tr_cv  # (N, 2)

    Xtr = np.nan_to_num(tr_x.reshape(len(tr_x), -1), nan=0.0).astype(np.float64)
    Xva = np.nan_to_num(va_x.reshape(len(va_x), -1), nan=0.0).astype(np.float64)
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = np.where(Xtr.std(axis=0, keepdims=True) < 1e-6, 1.0, Xtr.std(axis=0, keepdims=True))
    Ztr = (Xtr - mu) / sd
    Zva = (Xva - mu) / sd
    y_mean = tr_res.mean(axis=0, keepdims=True)
    Yc = tr_res - y_mean

    U, S, Vt = np.linalg.svd(Ztr, full_matrices=False)
    UtY = U.T @ Yc

    best_err, best_w = math.inf, None
    for lam in [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]:
        W = (Vt.T * (S / (S ** 2 + lam))) @ UtY
        pred_raw = va_cv + (Zva @ W + y_mean)
        ll = disp_to_latlon(torch.from_numpy(pred_raw[:, None, :].astype(np.float32)), torch.from_numpy(va_orig))
        err = float(haversine_km(ll, torch.from_numpy(va_fut)).numpy().mean())
        if err < best_err:
            best_err = err
            best_w = (mu, sd, W, y_mean)

    def predict_anchor(x_arr: np.ndarray, hist_arr: np.ndarray) -> np.ndarray:
        mu_, sd_, W_, ym_ = best_w
        cv_ = cv_displacement(torch.from_numpy(hist_arr), [lead_step], "legacy_previous_origin").numpy()[:, 0, :]
        Z = (np.nan_to_num(x_arr.reshape(len(x_arr), -1), nan=0.0).astype(np.float64) - mu_) / sd_
        return (cv_ + Z @ W_ + ym_)[:, None, :].astype(np.float32)

    return predict_anchor, best_err


# ==============================================================================
# Surgical Model Wrapper (Supports ModernTCN / Compact ST-B64 + Any Anchor + Multi-Horizon)
# ==============================================================================
class CompactSTEncoder(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, d_model: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x) + self.pos[:, : x.size(1), :]
        h = self.enc(h)
        return self.norm(h[:, -1, :])


class SurgicalResidualModel(nn.Module):
    def __init__(self, arch: str, input_dim: int, seq_len: int, lead_steps: list,
                 target_mean, target_std, anchor_mode: str = "cv1", d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.anchor_mode = anchor_mode
        if arch == "moderntcn":
            self.backbone = ModernTCNEncoder(input_dim, seq_len=seq_len, d_model=d_model, num_blocks=2, dropout=dropout)
        elif arch == "st_b64":
            self.backbone = CompactSTEncoder(input_dim, seq_len=seq_len, d_model=64, num_heads=4, dropout=dropout)
        else:
            raise ValueError(arch)
        d = self.backbone.out_dim
        self.head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, len(lead_steps) * 2),
        )
        self.register_buffer("lead_steps", torch.as_tensor(lead_steps, dtype=torch.float32))
        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.as_tensor(target_std, dtype=torch.float32))

    def forward(self, x: torch.Tensor, history: torch.Tensor, precomputed_anchor_raw: torch.Tensor = None):
        ctx = self.backbone(x)
        if self.anchor_mode == "cv1":
            anc_raw = cv_displacement(history, self.lead_steps, "legacy_previous_origin")
        elif self.anchor_mode == "smoothv075":
            anc_raw = smoothed_cv_displacement(history, self.lead_steps, alpha=0.75)
        elif self.anchor_mode == "ridge":
            anc_raw = precomputed_anchor_raw
        else:
            raise ValueError(self.anchor_mode)
        anc_norm = norm_disp_mh(anc_raw, self.target_mean, self.target_std)
        delta = self.head(ctx).reshape_as(anc_norm)
        return anc_norm + delta


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.985):
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


@torch.no_grad()
def predict_latlon(model: SurgicalResidualModel, data: dict, feat_idx, seq_len: int, anchor_raw=None, batch_size: int = 128):
    model.eval()
    preds_ll, errs_list = [], []
    n = len(data["keys"])
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        bx = data["tensors"]["x"][sl, -seq_len:, :][..., feat_idx]
        bh = data["tensors"]["history"][sl]
        ba = anchor_raw[sl] if anchor_raw is not None else None
        pred_norm = model(bx, bh, ba)
        raw = denorm_disp_mh(pred_norm, model.target_mean, model.target_std)
        ll = disp_to_latlon(raw, data["tensors"]["origin"][sl])
        preds_ll.append(ll.cpu().numpy())
        errs_list.append(haversine_km(ll, data["tensors"]["future"][sl]).cpu().numpy())
    return np.concatenate(preds_ll, axis=0), np.concatenate(errs_list, axis=0)


# ==============================================================================
# PART 1: North Atlantic 24-h Surgical Variant Worker (Shared-RAM ThreadPool -> 0 Windows IPC issues!)
# ==============================================================================
def run_part1_job(name, cfg, seed, tr, va, te, ex, meta, ridge_cache):
    torch.set_num_threads(1)
    feat_idx = cfg["feat_idx"]
    seq_len = cfg["seq_len"]
    anchor_mode = cfg["anchor_mode"]

    tr_anc = va_anc = te_anc = ex_anc = None
    if anchor_mode == "ridge":
        tr_anc, va_anc, te_anc, ex_anc = ridge_cache[(tuple(feat_idx), seq_len)]

    seed_everything(seed)
    model = SurgicalResidualModel(
        arch=cfg["arch"], input_dim=len(feat_idx), seq_len=seq_len,
        lead_steps=[4], target_mean=meta["stats"]["target_mean"], target_std=meta["stats"]["target_std"],
        anchor_mode=anchor_mode, d_model=cfg.get("d_model", 128), dropout=cfg.get("dropout", 0.1),
    )
    ema_model = copy.deepcopy(model) if cfg.get("ema", False) else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("wd", 1e-4))
    n_tr = len(tr["keys"])
    best_val, best_state = math.inf, None

    for epoch in range(40):
        model.train()
        gen = torch.Generator().manual_seed(seed + epoch + 1)
        perm = torch.randperm(n_tr, generator=gen)
        for ids in perm.split(32):
            bx = tr["tensors"]["x"][ids, -seq_len:, :][..., feat_idx]
            bh = tr["tensors"]["history"][ids]
            by = tr["tensors"]["y"][ids]
            ba = tr_anc[ids] if tr_anc is not None else None
            opt.zero_grad(set_to_none=True)
            pred_norm = model(bx, bh, ba)
            if cfg.get("loss", "mse") == "haversine":
                raw = denorm_disp_mh(pred_norm, model.target_mean, model.target_std)
                ll = disp_to_latlon(raw, tr["tensors"]["origin"][ids])
                loss = haversine_km(ll, tr["tensors"]["future"][ids]).mean() / 100.0
            else:
                loss = F.mse_loss(pred_norm, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema_model is not None:
                update_ema(ema_model, model, decay=0.985)

        eval_m = ema_model if (ema_model is not None and epoch >= 2) else model
        _, va_err = predict_latlon(eval_m, va, feat_idx, seq_len, va_anc)
        vk = float(va_err.mean())
        if vk < best_val:
            best_val = vk
            best_state = {k: v.detach().cpu().clone() for k, v in eval_m.state_dict().items()}

    model.load_state_dict(best_state)
    va_ll, va_err = predict_latlon(model, va, feat_idx, seq_len, va_anc)
    te_ll, te_err = predict_latlon(model, te, feat_idx, seq_len, te_anc)
    ex_ll, ex_err = predict_latlon(model, ex, feat_idx, seq_len, ex_anc)
    return {
        "variant": name,
        "seed": seed,
        "params": int(sum(p.numel() for p in model.parameters())),
        "val_km": float(va_err.mean()),
        "test_22_24_km": float(te_err.mean()),
        "ext_25_km": float(ex_err.mean()),
        "va_ll": va_ll, "te_ll": te_ll, "ex_ll": ex_ll,
    }


# ==============================================================================
# PART 2: Multi-Horizon Non-Iterative Joint 4-in-1 Decoder (6, 12, 18, 24 h)
# ==============================================================================
def build_joint_4horizon_split(base_prefix: str, split: str):
    """Align exact (storm_id, window) across lead1, lead2, lead3, lead4."""
    d_list = [load_split_fast(EXP_ROOT / "data" / f"{base_prefix}{h}", split) for h in (1, 2, 3, 4)]
    key_maps = [{k: i for i, k in enumerate(d["keys"])} for d in d_list]
    common_keys = [k for k in d_list[3]["keys"] if all(k in km for km in key_maps[:3])]

    idx_per_h = [[km[k] for k in common_keys] for km in key_maps]
    d4 = d_list[3]
    i4 = idx_per_h[3]
    x = d4["tensors"]["x"][i4]
    history = d4["tensors"]["history"][i4]
    origin = d4["tensors"]["origin"][i4]
    futures = torch.cat([d_list[h]["tensors"]["future"][idx_per_h[h]] for h in range(4)], dim=1)  # (N, 4, 2)
    raw_disp = latlon_to_disp(futures, origin[:, None, :])  # (N, 4, 2) exact km on tangent plane at p_T
    return {"keys": common_keys, "tensors": {"x": x, "history": history, "origin": origin, "future": futures, "raw_y": raw_disp}}


def run_joint_4in1_job(seed, feat_idx, seq_len, mode_cfg, tr, va, te, ex):
    torch.set_num_threads(1)
    # Per-horizon training-only target normalization moments
    tmean = tr["tensors"]["raw_y"].mean(dim=0)  # (4, 2)
    tstd = tr["tensors"]["raw_y"].std(dim=0).clamp(min=1e-3)  # (4, 2)
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
        perm = torch.randperm(n_tr, generator=gen)
        for ids in perm.split(32):
            bx = tr["tensors"]["x"][ids, -seq_len:, :][..., feat_idx]
            bh = tr["tensors"]["history"][ids]
            by = tr_y_norm[ids]
            opt.zero_grad(set_to_none=True)
            pred = model(bx, bh)
            if mode_cfg == "haversine":
                p_raw = denorm_disp_mh(pred, model.target_mean, model.target_std)
                p_ll = disp_to_latlon(p_raw, tr["tensors"]["origin"][ids])
                # Horizon-weighted haversine balancing 6/12/18/24 h gradients
                hk = haversine_km(p_ll, tr["tensors"]["future"][ids])
                w = torch.tensor([1.5, 1.2, 1.0, 1.0], dtype=hk.dtype)
                loss = (hk * w).mean() / 100.0
            else:
                loss_pos = F.mse_loss(pred, by)
                if mode_cfg == "mse_smooth":
                    p_raw = pred * tstd + tmean
                    t_raw = tr["tensors"]["raw_y"][ids]
                    dp = p_raw[:, 1:, :] - p_raw[:, :-1, :]
                    dt = t_raw[:, 1:, :] - t_raw[:, :-1, :]
                    loss = loss_pos + 0.05 * F.mse_loss(dp / 100.0, dt / 100.0)
                else:
                    loss = loss_pos
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update_ema(ema_model, model, decay=0.985)

        eval_m = ema_model if epoch >= 2 else model
        _, va_err = predict_latlon(eval_m, va, feat_idx, seq_len)
        # Scale-normalized validation mean across 4 horizons
        vk = float((va_err / np.array([35.0, 90.0, 160.0, 230.0])).mean())
        if vk < best_val:
            best_val = vk
            best_state = {k: v.detach().cpu().clone() for k, v in eval_m.state_dict().items()}

    model.load_state_dict(best_state)
    te_ll, te_err = predict_latlon(model, te, feat_idx, seq_len)
    ex_ll, ex_err = predict_latlon(model, ex, feat_idx, seq_len)
    return {
        "mode": mode_cfg,
        "seed": seed,
        "params": int(sum(p.numel() for p in model.parameters())),
        "te_per_h": [float(te_err[:, h].mean()) for h in range(4)],
        "ex_per_h": [float(ex_err[:, h].mean()) for h in range(4)],
        "te_ll": te_ll, "ex_ll": ex_ll,
    }


# ==============================================================================
# PART 3: Zero-Cost Kinematic Adaptive Conformal Prediction Uncertainty Cones
# ==============================================================================
def evaluate_conformal_cones(va_errs: np.ndarray, te_errs: np.ndarray, ex_errs: np.ndarray,
                             va_hist: np.ndarray, te_hist: np.ndarray, ex_hist: np.ndarray):
    """Calibrate Split Conformal & Kinematic Adaptive Conformal strictly on validation residuals."""
    def motion_features(hist_ll: np.ndarray):
        # hist_ll: (N, 12, 2); compute latest 6-h speed (km/6h) and turning angle magnitude (rad)
        v1 = latlon_to_disp(torch.from_numpy(hist_ll[:, -1, :]), torch.from_numpy(hist_ll[:, -2, :])).numpy()
        v0 = latlon_to_disp(torch.from_numpy(hist_ll[:, -2, :]), torch.from_numpy(hist_ll[:, -3, :])).numpy()
        spd = np.hypot(v1[:, 0], v1[:, 1])
        spd0 = np.hypot(v0[:, 0], v0[:, 1])
        acc = np.abs(spd - spd0)
        dot = (v1[:, 0] * v0[:, 0] + v1[:, 1] * v0[:, 1]) / (np.maximum(spd * spd0, 1e-3))
        turn = np.arccos(np.clip(dot, -1.0, 1.0))
        lat = np.abs(hist_ll[:, -1, 0])
        return np.column_stack([np.ones(len(spd)), spd / 150.0, acc / 50.0, turn, (lat - 20.0) / 15.0])

    F_va = motion_features(va_hist)
    F_te = motion_features(te_hist)
    F_ex = motion_features(ex_hist)

    # Fit positive scale estimator sigma_hat(v, a, turn, lat) strictly on validation residuals via ridge
    I_reg = np.eye(F_va.shape[1]) * 10.0
    I_reg[0, 0] = 0.0
    beta = np.linalg.solve(F_va.T @ F_va + I_reg, F_va.T @ va_errs)
    sig_va = np.clip(F_va @ beta, 60.0, 600.0)
    sig_te = np.clip(F_te @ beta, 60.0, 600.0)
    sig_ex = np.clip(F_ex @ beta, 60.0, 600.0)

    out = {}
    n_va = len(va_errs)
    for alpha in (0.10, 0.20):
        cov_lbl = f"{int((1 - alpha) * 100)}pct"
        q_idx = min(n_va - 1, int(math.ceil((n_va + 1) * (1.0 - alpha))) - 1)
        # 1. Split Conformal (constant radius)
        q_scp = float(np.sort(va_errs)[q_idx])
        # 2. Kinematic Adaptive Conformal (locally scaled radius)
        scores_va = va_errs / sig_va
        q_acp = float(np.sort(scores_va)[q_idx])
        r_te_acp = q_acp * sig_te
        r_ex_acp = q_acp * sig_ex
        # Stratify test coverage on high-volatility (top 25% speed/turn) vs steady (bottom 75%)
        hi_mask = sig_te >= np.quantile(sig_te, 0.75)
        out[cov_lbl] = {
            "split_conformal": {
                "radius_km": round(q_scp, 1),
                "test_22_24_cov_pct": round(float((te_errs <= q_scp).mean() * 100), 1),
                "test_22_24_hi_vol_cov_pct": round(float((te_errs[hi_mask] <= q_scp).mean() * 100), 1),
                "ext_25_cov_pct": round(float((ex_errs <= q_scp).mean() * 100), 1),
            },
            "kinematic_adaptive_conformal": {
                "test_22_24_mean_radius_km": round(float(r_te_acp.mean()), 1),
                "test_22_24_steady_radius_km": round(float(r_te_acp[~hi_mask].mean()), 1),
                "test_22_24_hivol_radius_km": round(float(r_te_acp[hi_mask].mean()), 1),
                "test_22_24_cov_pct": round(float((te_errs <= r_te_acp).mean() * 100), 1),
                "test_22_24_hi_vol_cov_pct": round(float((te_errs[hi_mask] <= r_te_acp[hi_mask]).mean() * 100), 1),
                "ext_25_mean_radius_km": round(float(r_ex_acp.mean()), 1),
                "ext_25_cov_pct": round(float((ex_errs <= r_ex_acp).mean() * 100), 1),
            },
        }
    return out


# ==============================================================================
# PART 4: Cross-Basin Sanity Check on Existing Eastern/Central North Pacific
# ==============================================================================
def run_nepac_tier1():
    from types import SimpleNamespace
    full_df = parse_hurdat2_any_basin(NEPAC_PATH)
    df_mod = full_df[full_df.storm_year.between(2004, 2025)].copy()
    splits = LIB.storm_splits(
        df_mod,
        SimpleNamespace(train_years="2004-2021", test_years="2022-2024", split_seed=42, val_ratio=0.10),
    )
    splits["extension"] = set(df_mod.loc[df_mod.storm_year == 2025, "typhoon_id"])
    synoptic_rows = build_synoptic_first_windows(df_mod, splits)

    train_rows = [r for r in synoptic_rows if r["split"] == "train"]
    x_tr_cat = np.concatenate([r["input_raw"] for r in train_rows], axis=0)
    y_tr_cat = np.concatenate([r["target_disp_raw_km"] for r in train_rows], axis=0)
    with np.errstate(invalid="ignore"):
        x_mean = np.nanmean(x_tr_cat, axis=0)
        x_std = np.nanstd(x_tr_cat, axis=0)
    x_mean = np.where(np.isfinite(x_mean), x_mean, 0.0).astype(np.float32)
    x_std = np.where(np.isfinite(x_std) & (x_std > 1e-6), x_std, 1.0).astype(np.float32)
    tmean = y_tr_cat.mean(axis=0).astype(np.float32)
    tstd = np.where(y_tr_cat.std(axis=0) > 1e-6, y_tr_cat.std(axis=0), 1.0).astype(np.float32)

    fnames = [f for f in LIB.FEATURE_NAMES if f != "rmax"]
    all_idx = list(range(len(fnames)))
    kin_idx = [i for i, f in enumerate(fnames) if f in KIN_NAMES]

    def to_split_dict(sp_key):
        sp_rows = [r for r in synoptic_rows if r["split"] == sp_key]
        x_arr = np.stack([((np.where(np.isfinite(r["input_raw"]), r["input_raw"], x_mean) - x_mean) / x_std).astype(np.float32) for r in sp_rows])
        y_arr = np.stack([((r["target_disp_raw_km"] - tmean) / tstd).astype(np.float32) for r in sp_rows])
        return {
            "keys": [(str(r["storm_id"]), int(r["window"])) for r in sp_rows],
            "tensors": {
                "x": torch.from_numpy(x_arr),
                "y": torch.from_numpy(y_arr),
                "history": torch.from_numpy(np.stack([r["history_latlon"].astype(np.float32) for r in sp_rows])),
                "origin": torch.from_numpy(np.stack([r["origin"].astype(np.float32) for r in sp_rows])),
                "future": torch.from_numpy(np.stack([r["future_latlon"].astype(np.float32) for r in sp_rows])),
            },
        }

    tr = to_split_dict("train")
    va = to_split_dict("val")
    te = to_split_dict("test")
    ex = to_split_dict("extension")

    configs = {
        "ep_ref_moderntcn_39ch_L12_cv1":      dict(feat_idx=all_idx, seq_len=12, anchor_mode="cv1",   ema=False, loss="mse"),
        "ep_moderntcn_16ch_L8_cv1_ema":       dict(feat_idx=kin_idx, seq_len=8,  anchor_mode="cv1",   ema=True,  loss="mse"),
        "ep_moderntcn_16ch_L8_ridge_ema":     dict(feat_idx=kin_idx, seq_len=8,  anchor_mode="ridge", ema=True,  loss="mse"),
        "ep_moderntcn_16ch_L8_ridge_ema_hav": dict(feat_idx=kin_idx, seq_len=8,  anchor_mode="ridge", ema=True,  loss="haversine"),
    }
    ep_res = {}
    for cname, cfg in configs.items():
        feat_idx, seq_len, amode = cfg["feat_idx"], cfg["seq_len"], cfg["anchor_mode"]
        tr_anc = va_anc = te_anc = ex_anc = None
        if amode == "ridge":
            tr_raw_y = tr["tensors"]["y"].numpy() * np.asarray(tstd, np.float32) + np.asarray(tmean, np.float32)
            pred_fn, _ = fit_ridge_kr_anchor(
                tr["tensors"]["x"][:, -seq_len:, :][..., feat_idx].numpy(), tr["tensors"]["history"].numpy(), tr_raw_y,
                va["tensors"]["x"][:, -seq_len:, :][..., feat_idx].numpy(), va["tensors"]["history"].numpy(),
                va["tensors"]["origin"].numpy(), va["tensors"]["future"].numpy(), lead_step=4,
            )
            tr_anc = torch.from_numpy(pred_fn(tr["tensors"]["x"][:, -seq_len:, :][..., feat_idx].numpy(), tr["tensors"]["history"].numpy()))
            va_anc = torch.from_numpy(pred_fn(va["tensors"]["x"][:, -seq_len:, :][..., feat_idx].numpy(), va["tensors"]["history"].numpy()))
            te_anc = torch.from_numpy(pred_fn(te["tensors"]["x"][:, -seq_len:, :][..., feat_idx].numpy(), te["tensors"]["history"].numpy()))
            ex_anc = torch.from_numpy(pred_fn(ex["tensors"]["x"][:, -seq_len:, :][..., feat_idx].numpy(), ex["tensors"]["history"].numpy()))

        te_lls, ex_lls, va_lls = [], [], []
        te_singles, ex_singles = [], []
        for seed in (42, 123, 2024):
            torch.set_num_threads(4)
            seed_everything(seed)
            model = SurgicalResidualModel("moderntcn", len(feat_idx), seq_len, [4], tmean, tstd, anchor_mode=amode, d_model=128)
            ema_m = copy.deepcopy(model) if cfg["ema"] else None
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            best_v, best_st = math.inf, None
            n_tr = len(tr["keys"])
            for epoch in range(40):
                model.train()
                gen = torch.Generator().manual_seed(seed + epoch + 1)
                for ids in torch.randperm(n_tr, generator=gen).split(32):
                    bx = tr["tensors"]["x"][ids, -seq_len:, :][..., feat_idx]
                    bh = tr["tensors"]["history"][ids]
                    by = tr["tensors"]["y"][ids]
                    ba = tr_anc[ids] if tr_anc is not None else None
                    opt.zero_grad(set_to_none=True)
                    p_n = model(bx, bh, ba)
                    if cfg.get("loss", "mse") == "haversine":
                        p_raw = denorm_disp_mh(p_n, model.target_mean, model.target_std)
                        p_ll = disp_to_latlon(p_raw, tr["tensors"]["origin"][ids])
                        loss = haversine_km(p_ll, tr["tensors"]["future"][ids]).mean() / 100.0
                    else:
                        loss = F.mse_loss(p_n, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    if ema_m is not None:
                        update_ema(ema_m, model, decay=0.985)
                em = ema_m if (ema_m is not None and epoch >= 2) else model
                _, ve = predict_latlon(em, va, feat_idx, seq_len, va_anc)
                if float(ve.mean()) < best_v:
                    best_v = float(ve.mean())
                    best_st = {k: v.detach().cpu().clone() for k, v in em.state_dict().items()}
            model.load_state_dict(best_st)
            v_ll, _ = predict_latlon(model, va, feat_idx, seq_len, va_anc)
            t_ll, t_e = predict_latlon(model, te, feat_idx, seq_len, te_anc)
            e_ll, e_e = predict_latlon(model, ex, feat_idx, seq_len, ex_anc)
            va_lls.append(v_ll); te_lls.append(t_ll); ex_lls.append(e_ll)
            te_singles.append(float(t_e.mean())); ex_singles.append(float(e_e.mean()))

        ens_va = float(haversine_km(torch.from_numpy(np.mean(va_lls, axis=0)), va["tensors"]["future"]).numpy().mean())
        ens_te_e = haversine_km(torch.from_numpy(np.mean(te_lls, axis=0)), te["tensors"]["future"]).numpy().reshape(-1)
        ens_ex_e = haversine_km(torch.from_numpy(np.mean(ex_lls, axis=0)), ex["tensors"]["future"]).numpy().reshape(-1)
        ens_va_e = haversine_km(torch.from_numpy(np.mean(va_lls, axis=0)), va["tensors"]["future"]).numpy().reshape(-1)
        ep_res[cname] = {
            "single_22_24_mean": round(float(np.mean(te_singles)), 2),
            "single_22_24_sd": round(float(np.std(te_singles, ddof=1)), 2),
            "single_25_mean": round(float(np.mean(ex_singles)), 2),
            "single_25_sd": round(float(np.std(ex_singles, ddof=1)), 2),
            "ensemble_22_24_km": round(float(ens_te_e.mean()), 2),
            "ensemble_25_km": round(float(ens_ex_e.mean()), 2),
            "conformal": evaluate_conformal_cones(ens_va_e, ens_te_e, ens_ex_e,
                                                  va["tensors"]["history"].numpy(),
                                                  te["tensors"]["history"].numpy(),
                                                  ex["tensors"]["history"].numpy()),
        }
        r = ep_res[cname]
        print(f"  [EP] {cname:32s} | single={r['single_22_24_mean']:.2f}/{r['single_25_mean']:.2f} | 3-ens={r['ensemble_22_24_km']:.2f}/{r['ensemble_25_km']:.2f}")
    return ep_res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    fnames = LIB.FEATURE_NAMES
    all_idx = list(range(len(fnames)))
    kin_idx = [i for i, f in enumerate(fnames) if f in KIN_NAMES]

    # Load splits ONCE in shared RAM (zero duplicate disk I/O & zero Windows multiprocessing crashes!)
    d4 = EXP_ROOT / "data" / "controlled_12to1_lead4"
    e4 = EXP_ROOT / "data" / "extension2025_lead4"
    meta = json.loads((d4 / "metadata.json").read_text(encoding="utf-8"))
    tr_ref = load_split_fast(d4, "train")
    va_ref = load_split_fast(d4, "val")
    te_ref = load_split_fast(d4, "test")
    ex_ref = load_split_fast(e4, "test")

    # Pre-fit closed-form Ridge-KR anchor strictly on train split once per (feat_idx, seq_len)
    tmean_np = np.asarray(meta["stats"]["target_mean"], dtype=np.float32)
    tstd_np = np.asarray(meta["stats"]["target_std"], dtype=np.float32)
    tr_raw_y_np = tr_ref["tensors"]["y"].numpy() * tstd_np + tmean_np
    ridge_cache = {}
    pred_fn_16_8, r_val = fit_ridge_kr_anchor(
        tr_ref["tensors"]["x"][:, -8:, :][..., kin_idx].numpy(), tr_ref["tensors"]["history"].numpy(), tr_raw_y_np,
        va_ref["tensors"]["x"][:, -8:, :][..., kin_idx].numpy(), va_ref["tensors"]["history"].numpy(),
        va_ref["tensors"]["origin"].numpy(), va_ref["tensors"]["future"].numpy(), lead_step=4,
    )
    ridge_cache[(tuple(kin_idx), 8)] = (
        torch.from_numpy(pred_fn_16_8(tr_ref["tensors"]["x"][:, -8:, :][..., kin_idx].numpy(), tr_ref["tensors"]["history"].numpy())),
        torch.from_numpy(pred_fn_16_8(va_ref["tensors"]["x"][:, -8:, :][..., kin_idx].numpy(), va_ref["tensors"]["history"].numpy())),
        torch.from_numpy(pred_fn_16_8(te_ref["tensors"]["x"][:, -8:, :][..., kin_idx].numpy(), te_ref["tensors"]["history"].numpy())),
        torch.from_numpy(pred_fn_16_8(ex_ref["tensors"]["x"][:, -8:, :][..., kin_idx].numpy(), ex_ref["tensors"]["history"].numpy())),
    )
    # Also log pure Ridge-KR(16ch, L=8) anchor error alone!
    r_te_km = float(haversine_km(disp_to_latlon(ridge_cache[(tuple(kin_idx), 8)][2], te_ref["tensors"]["origin"]), te_ref["tensors"]["future"]).numpy().mean())
    r_ex_km = float(haversine_km(disp_to_latlon(ridge_cache[(tuple(kin_idx), 8)][3], ex_ref["tensors"]["origin"]), ex_ref["tensors"]["future"]).numpy().mean())
    print(f"  [Closed-Form Anchor] Ridge-KR (16ch, L=8) alone: val={r_val:.2f} | 22-24={r_te_km:.2f} | 25={r_ex_km:.2f}", flush=True)

    variants = {
        "01_ref_moderntcn_40ch_L12_cv1":     dict(arch="moderntcn", feat_idx=all_idx, seq_len=12, anchor_mode="cv1",       ema=False, loss="mse"),
        "02_moderntcn_16ch_L12_cv1":         dict(arch="moderntcn", feat_idx=kin_idx, seq_len=12, anchor_mode="cv1",       ema=False, loss="mse"),
        "03_moderntcn_16ch_L8_cv1":          dict(arch="moderntcn", feat_idx=kin_idx, seq_len=8,  anchor_mode="cv1",       ema=False, loss="mse"),
        "04_moderntcn_16ch_L4_cv1":          dict(arch="moderntcn", feat_idx=kin_idx, seq_len=4,  anchor_mode="cv1",       ema=False, loss="mse"),
        "05_moderntcn_16ch_L8_smoothv075":   dict(arch="moderntcn", feat_idx=kin_idx, seq_len=8,  anchor_mode="smoothv075",ema=False, loss="mse"),
        "06_moderntcn_16ch_L8_ridge_anchor": dict(arch="moderntcn", feat_idx=kin_idx, seq_len=8,  anchor_mode="ridge",     ema=False, loss="mse"),
        "07_moderntcn_16ch_L8_cv1_ema":      dict(arch="moderntcn", feat_idx=kin_idx, seq_len=8,  anchor_mode="cv1",       ema=True,  loss="mse"),
        "08_moderntcn_16ch_L8_ridge_ema":    dict(arch="moderntcn", feat_idx=kin_idx, seq_len=8,  anchor_mode="ridge",     ema=True,  loss="mse"),
        "09_moderntcn_16ch_L8_haversine":    dict(arch="moderntcn", feat_idx=kin_idx, seq_len=8,  anchor_mode="cv1",       ema=True,  loss="haversine"),
        "10_st_b64_16ch_L8_ridge_ema":       dict(arch="st_b64",    feat_idx=kin_idx, seq_len=8,  anchor_mode="ridge",     ema=True,  loss="mse"),
        "11_moderntcn_16ch_L8_ridge_ema_hav":dict(arch="moderntcn", feat_idx=kin_idx, seq_len=8,  anchor_mode="ridge",     ema=True,  loss="haversine"),
        "12_moderntcn_16ch_L8_smooth_ema_hav":dict(arch="moderntcn",feat_idx=kin_idx, seq_len=8,  anchor_mode="smoothv075",ema=True,  loss="haversine"),
    }

    print("=== PART 1: NORTH ATLANTIC 24-H EXTREME ROI SURGERY (12 VARIANTS x 3 SEEDS = 36 RUNS) ===", flush=True)
    raw_runs = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex_pool:
        futs = [
            ex_pool.submit(run_part1_job, name, cfg, seed, tr_ref, va_ref, te_ref, ex_ref, meta, ridge_cache)
            for name, cfg in variants.items() for seed in (42, 123, 2024)
        ]
        for fut in futs:
            raw_runs.append(fut.result())

    part1_summary = {"00_ridge_kr_16ch_L8_anchor_only": {"val_km": round(r_val, 2), "test_22_24_km": round(r_te_km, 2), "ext_25_km": round(r_ex_km, 2)}}
    best_name, best_te_ens, best_va_e, best_te_e, best_ex_e = None, math.inf, None, None, None
    for name in variants:
        rs = [r for r in raw_runs if r["variant"] == name]
        ens_va_ll = np.mean([r["va_ll"] for r in rs], axis=0)
        ens_te_ll = np.mean([r["te_ll"] for r in rs], axis=0)
        ens_ex_ll = np.mean([r["ex_ll"] for r in rs], axis=0)
        e_va = haversine_km(torch.from_numpy(ens_va_ll), va_ref["tensors"]["future"]).numpy().reshape(-1)
        e_te = haversine_km(torch.from_numpy(ens_te_ll), te_ref["tensors"]["future"]).numpy().reshape(-1)
        e_ex = haversine_km(torch.from_numpy(ens_ex_ll), ex_ref["tensors"]["future"]).numpy().reshape(-1)

        s_val = float(np.mean([r["val_km"] for r in rs]))
        s_22 = float(np.mean([r["test_22_24_km"] for r in rs]))
        s_22_sd = float(np.std([r["test_22_24_km"] for r in rs], ddof=1))
        s_25 = float(np.mean([r["ext_25_km"] for r in rs]))
        s_25_sd = float(np.std([r["ext_25_km"] for r in rs], ddof=1))
        part1_summary[name] = {
            "params": rs[0]["params"],
            "val_mean": round(s_val, 2),
            "single_22_24_mean": round(s_22, 2),
            "single_22_24_sd": round(s_22_sd, 2),
            "single_25_mean": round(s_25, 2),
            "single_25_sd": round(s_25_sd, 2),
            "ensemble_val_km": round(float(e_va.mean()), 2),
            "ensemble_22_24_km": round(float(e_te.mean()), 2),
            "ensemble_25_km": round(float(e_ex.mean()), 2),
        }
        p = part1_summary[name]
        print(f"  {name:36s} ({p['params']//1000:3d}K) | val={p['val_mean']:6.2f} | single={p['single_22_24_mean']:6.2f}+/-{p['single_22_24_sd']:4.2f} / {p['single_25_mean']:6.2f}+/-{p['single_25_sd']:4.2f} | 3-ens={p['ensemble_22_24_km']:6.2f} / {p['ensemble_25_km']:6.2f}", flush=True)
        if float(e_va.mean()) < best_te_ens:  # Select strictly by validation ensemble error (zero test peeking!)
            best_te_ens = float(e_va.mean())
            best_name = name
            best_va_e, best_te_e, best_ex_e = e_va, e_te, e_ex

    print(f"\n=== PART 2: MULTI-HORIZON 4-IN-1 JOINT NON-ITERATIVE DECODER (6/12/18/24 h) ===", flush=True)
    tr_j = build_joint_4horizon_split("controlled_12to1_lead", "train")
    va_j = build_joint_4horizon_split("controlled_12to1_lead", "val")
    te_j_ref = build_joint_4horizon_split("controlled_12to1_lead", "test")
    ex_j_ref = build_joint_4horizon_split("extension2025_lead", "test")
    modes_p2 = ("mse", "mse_smooth", "haversine")
    j_runs = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex_pool:
        futs = [
            ex_pool.submit(run_joint_4in1_job, seed, kin_idx, 8, mc, tr_j, va_j, te_j_ref, ex_j_ref)
            for mc in modes_p2 for seed in (42, 123, 2024)
        ]
        for fut in futs:
            j_runs.append(fut.result())

    part2_summary = {}
    for mc in modes_p2:
        rs = [r for r in j_runs if r["mode"] == mc]
        ens_te_ll = np.mean([r["te_ll"] for r in rs], axis=0)
        ens_ex_ll = np.mean([r["ex_ll"] for r in rs], axis=0)
        e_te = haversine_km(torch.from_numpy(ens_te_ll), te_j_ref["tensors"]["future"]).numpy()
        e_ex = haversine_km(torch.from_numpy(ens_ex_ll), ex_j_ref["tensors"]["future"]).numpy()
        key = f"joint_4in1_{mc}"
        part2_summary[key] = {
            "params_total_for_all_4_horizons": rs[0]["params"],
            "single_22_24_km": [round(float(np.mean([r["te_per_h"][h] for r in rs])), 2) for h in range(4)],
            "single_25_km": [round(float(np.mean([r["ex_per_h"][h] for r in rs])), 2) for h in range(4)],
            "ensemble_22_24_km": [round(float(e_te[:, h].mean()), 2) for h in range(4)],
            "ensemble_25_km": [round(float(e_ex[:, h].mean()), 2) for h in range(4)],
        }
        p2 = part2_summary[key]
        print(f"  {key:24s} ({p2['params_total_for_all_4_horizons']//1000}K total) | 22-24 single={p2['single_22_24_km']} ens={p2['ensemble_22_24_km']} | 25 ens={p2['ensemble_25_km']}", flush=True)

    print(f"\n=== PART 3: ZERO-COST ADAPTIVE CONFORMAL PREDICTION CONES (VAL-SELECTED: {best_name}) ===", flush=True)
    # Compare Conformal Cone of our Val-Selected Tier-1 Winner vs CV baseline
    cv_va_ll = disp_to_latlon(cv_displacement(va_ref["tensors"]["history"], [4], "legacy_previous_origin"), va_ref["tensors"]["origin"])
    cv_te_ll = disp_to_latlon(cv_displacement(te_ref["tensors"]["history"], [4], "legacy_previous_origin"), te_ref["tensors"]["origin"])
    cv_ex_ll = disp_to_latlon(cv_displacement(ex_ref["tensors"]["history"], [4], "legacy_previous_origin"), ex_ref["tensors"]["origin"])
    cv_va_e = haversine_km(cv_va_ll, va_ref["tensors"]["future"]).numpy().reshape(-1)
    cv_te_e = haversine_km(cv_te_ll, te_ref["tensors"]["future"]).numpy().reshape(-1)
    cv_ex_e = haversine_km(cv_ex_ll, ex_ref["tensors"]["future"]).numpy().reshape(-1)

    part3_summary = {
        "val_selected_winner": best_name,
        "tier1_winner_conformal": evaluate_conformal_cones(
            best_va_e, best_te_e, best_ex_e,
            va_ref["tensors"]["history"].numpy(), te_ref["tensors"]["history"].numpy(), ex_ref["tensors"]["history"].numpy(),
        ),
        "cv_baseline_conformal": evaluate_conformal_cones(
            cv_va_e, cv_te_e, cv_ex_e,
            va_ref["tensors"]["history"].numpy(), te_ref["tensors"]["history"].numpy(), ex_ref["tensors"]["history"].numpy(),
        ),
    }
    print(json.dumps(part3_summary, indent=2), flush=True)

    print("\n=== PART 4: EASTERN & CENTRAL NORTH PACIFIC CROSS-BASIN SANITY CHECK ===", flush=True)
    part4_summary = run_nepac_tier1()

    summary = {
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "part1_atlantic_24h_surgery": part1_summary,
        "part2_atlantic_4in1_joint": part2_summary,
        "part3_atlantic_conformal_cones": part3_summary,
        "part4_pacific_cross_basin": part4_summary,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nALL TIER-1 SURGICAL PROBES DONE IN {summary['wall_seconds']:.1f}s -> {OUT_DIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
