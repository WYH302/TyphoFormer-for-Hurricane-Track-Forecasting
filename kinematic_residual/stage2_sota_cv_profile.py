"""Stage 2 Comprehensive Experimental Suite:
1. Hardware Complexity & Efficiency Profiling:
   Exact FLOPs (2*MACs), Params, FP32 Weight Footprint (MB), Peak Activation/VRAM Estimate,
   and Single-CPU Latency (ms) & Throughput (FPS) across all 12 linear/neural backbones.
2. 2024-2025 ICLR SOTA Sequence Expansion:
   - ModernTCN (Luo & Wang, ICLR 2024 Spotlight): depthwise large-kernel conv + decoupled ConvFFN.
   - TimeMixer (Wang et al., ICLR 2024): multi-scale seasonal-trend Past-Decomposable-Mixing (PDM).
   Evaluated strictly under Direct (A) vs. Kinematic Residual (B) across:
   (a) North Atlantic HURDAT2 (6, 12, 18, 24 h; 2022-2024 test & 2025 season; 3 seeds = 48 runs),
   (b) Eastern & Central North Pacific HURDAT2 (24 h; 2022-2024 test & 2025 season; 3 seeds = 12 runs).
3. Leakage-Free 5-Fold Storm-Grouped Cross-Validation (2004-2021 North Atlantic, 24 h, 287 storms):
   GroupKFold(n_splits=5) strictly grouped by storm_id (zero storm overlap across folds; training-only
   z-score normalization recomputed inside every fold) comparing CV, Ridge-KR, and Direct (A) vs.
   Kinematic Residual (B) across 8 neural families (80 runs total).
4. SOTA Visual Trajectory Case Predictions:
   Saves exact 24-h predicted coordinates for Hurricane Ernesto (AL052024, Atlantic recurvature) and
   Category-5 Hurricane Otis (EP182023, Pacific Acapulco landfall) across CV, Ridge-KR, LSTM, PatchTST,
   iTransformer, ModernTCN, TimeMixer, and Compact ST-Transformer (Direct A vs. Residual B) for Figure 9.
"""
import concurrent.futures as cf
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
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from controlled_heads import ControlledNumericModel, cv_displacement
from sota_sequence_baselines import (
    RecurrentEncoder,
    TSMixerEncoder,
    PatchTSTEncoder,
    ITransformerEncoder,
    WrappedKinematicModel,
    load_split_fast,
)
from eastern_pacific_cross_basin import (
    NEPAC_PATH,
    parse_hurdat2_any_basin,
    build_synoptic_first_windows,
    paired_storm_bootstrap,
)
from vendor.common import (
    denormalize_disp,
    disp_to_latlon,
    haversine_km,
    normalize_disp,
    seed_everything,
)

SPEC = importlib.util.spec_from_file_location("original_prepare", EXP_ROOT / "data" / "prepare_frozen.py")
LIB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIB)

OUT_DIR = EXP_ROOT / "results" / "stage2_sota_cv_profile"


# ==============================================================================
# 1. Two ICLR 2024 SOTA Backbones: ModernTCN (Spotlight) & TimeMixer
# ==============================================================================
class ModernTCNBlock(nn.Module):
    """ModernTCN block (Luo & Wang, ICLR 2024): large-kernel depthwise temporal conv + decoupled FFN."""
    def __init__(self, d_model: int, kernel_size: int = 7, ff_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=pad, groups=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.pw_ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * ff_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_ratio, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.pw_ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * ff_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        y = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.pw_ffn1(self.norm1(y))
        x = x + self.pw_ffn2(self.norm2(x))
        return x


class ModernTCNEncoder(nn.Module):
    """ModernTCN (Luo & Wang, ICLR 2024 Spotlight) adapted for 12-step multivariate track sequences."""
    def __init__(self, input_dim: int, seq_len: int = 12, d_model: int = 128, num_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList([ModernTCNBlock(d_model, kernel_size=7, ff_ratio=2, dropout=dropout) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h[:, -1, :])


class SeriesDecomp(nn.Module):
    """Moving-average seasonal-trend decomposition used in TimeMixer (ICLR 2024)."""
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        # x: (B, L, D)
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_pad.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class PastDecomposableMixingBlock(nn.Module):
    """Bottom-up seasonal + top-down trend multiscale mixing across scales L=12, 6, 3."""
    def __init__(self, d_model: int, scales=(12, 6, 3), dropout: float = 0.1):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size=3)
        self.scales = scales
        self.s_down = nn.ModuleList([
            nn.Sequential(nn.Linear(scales[i], scales[i + 1]), nn.GELU(), nn.Dropout(dropout))
            for i in range(len(scales) - 1)
        ])
        self.t_up = nn.ModuleList([
            nn.Sequential(nn.Linear(scales[i + 1], scales[i]), nn.GELU(), nn.Dropout(dropout))
            for i in range(len(scales) - 1)
        ])
        self.out_mix = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout),
            )
            for _ in scales
        ])

    def forward(self, x_list):
        seasonals, trends = [], []
        for x in x_list:
            s, t = self.decomp(x)
            seasonals.append(s)
            trends.append(t)
        # Bottom-up seasonal mixing (fine -> coarse)
        for i in range(len(self.scales) - 1):
            delta = self.s_down[i](seasonals[i].transpose(1, 2)).transpose(1, 2)
            seasonals[i + 1] = seasonals[i + 1] + delta
        # Top-down trend mixing (coarse -> fine)
        for i in reversed(range(len(self.scales) - 1)):
            delta = self.t_up[i](trends[i + 1].transpose(1, 2)).transpose(1, 2)
            trends[i] = trends[i] + delta
        return [x_list[i] + self.out_mix[i](seasonals[i] + trends[i]) for i in range(len(self.scales))]


class TimeMixerEncoder(nn.Module):
    """TimeMixer (Wang et al., ICLR 2024): multi-scale decomposable past mixing."""
    def __init__(self, input_dim: int, seq_len: int = 12, d_model: int = 128, num_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        assert seq_len == 12
        self.scales = (12, 6, 3)
        self.embed = nn.Linear(input_dim, d_model)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.blocks = nn.ModuleList([PastDecomposableMixingBlock(d_model, self.scales, dropout) for _ in range(num_blocks)])
        self.scale_proj = nn.Sequential(
            nn.LayerNorm(d_model * len(self.scales)),
            nn.Linear(d_model * len(self.scales), d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.embed(x)
        x1 = self.pool(x0.transpose(1, 2)).transpose(1, 2)
        x2 = self.pool(x1.transpose(1, 2)).transpose(1, 2)
        xs = [x0, x1, x2]
        for blk in self.blocks:
            xs = blk(xs)
        cat = torch.cat([s[:, -1, :] for s in xs], dim=-1)
        return self.scale_proj(cat)


class ExtendedKinematicModel(nn.Module):
    """Unified wrapper supporting all 8 neural backbones under identical Direct (A) or Residual (B) heads."""
    def __init__(self, arch: str, input_dim: int, seq_len: int, lead_steps, target_mean, target_std, head: str):
        super().__init__()
        assert head in {"direct", "residual"}
        self.arch = arch
        self.head = head
        if arch == "compact_st_transformer":
            self.inner = ControlledNumericModel(
                input_dim=input_dim,
                input_len=seq_len,
                lead_steps=lead_steps,
                target_mean=target_mean,
                target_std=target_std,
                head=head,
                d_model=64,
                num_heads=4,
                num_layers=1,
                dropout=0.1,
            )
            self.target_mean = self.inner.target_mean
            self.target_std = self.inner.target_std
            return
        self.inner = None
        if arch in {"lstm", "gru"}:
            self.backbone = RecurrentEncoder(arch, input_dim, d_model=128, num_layers=2, dropout=0.1)
        elif arch == "tsmixer":
            self.backbone = TSMixerEncoder(input_dim, seq_len=seq_len, d_model=128, num_blocks=3, dropout=0.1)
        elif arch == "patchtst":
            self.backbone = PatchTSTEncoder(input_dim, seq_len=seq_len, patch_len=3, d_model=128, num_heads=4, num_layers=2, dropout=0.1)
        elif arch == "itransformer":
            self.backbone = ITransformerEncoder(input_dim, seq_len=seq_len, d_model=128, num_heads=4, num_layers=2, dropout=0.1)
        elif arch == "moderntcn":
            self.backbone = ModernTCNEncoder(input_dim, seq_len=seq_len, d_model=128, num_blocks=2, dropout=0.1)
        elif arch == "timemixer":
            self.backbone = TimeMixerEncoder(input_dim, seq_len=seq_len, d_model=128, num_blocks=2, dropout=0.1)
        else:
            raise ValueError(f"Unknown arch: {arch}")
        d = self.backbone.out_dim
        self.proj = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, len(lead_steps) * 2),
        )
        self.register_buffer("lead_steps", torch.as_tensor(lead_steps, dtype=torch.float32))
        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.as_tensor(target_std, dtype=torch.float32))

    def forward(self, x_num: torch.Tensor, history_latlon: torch.Tensor):
        if self.inner is not None:
            return self.inner(x_num, history_latlon)
        ctx = self.backbone(x_num)
        cv_raw = cv_displacement(history_latlon, self.lead_steps, "legacy_previous_origin")
        cv_norm = normalize_disp(cv_raw, self.target_mean, self.target_std)
        delta = self.proj(ctx).reshape_as(cv_norm)
        pred = delta if self.head == "direct" else (cv_norm + delta)
        return {"pred": pred, "cv": cv_norm}


# ==============================================================================
# 2. Exact FLOPs / MACs, Params, Memory & CPU Latency/FPS Profiler
# ==============================================================================
def count_model_flops(model: nn.Module, x_in: torch.Tensor, h_in: torch.Tensor) -> int:
    """Measure exact forward FLOPs (2 * MACs) for a single-sample pass via PyTorch module hooks."""
    total_flops = 0
    hooks = []

    def linear_hook(mod, inp, out):
        nonlocal total_flops
        x = inp[0]
        n_vecs = x.numel() // mod.in_features
        total_flops += 2 * n_vecs * mod.in_features * mod.out_features

    def conv1d_hook(mod, inp, out):
        nonlocal total_flops
        out_len = out.shape[-1]
        bs = out.shape[0]
        macs_per_pos = (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.out_channels
        total_flops += 2 * bs * out_len * macs_per_pos

    def rnn_hook(mod, inp, out):
        nonlocal total_flops
        x = inp[0]
        bs, seq_len, in_sz = x.shape
        h_sz = mod.hidden_size
        gates = 4 if isinstance(mod, nn.LSTM) else 3
        for layer in range(mod.num_layers):
            dim_in = in_sz if layer == 0 else h_sz
            macs_step = gates * (dim_in * h_sz + h_sz * h_sz)
            total_flops += 2 * bs * seq_len * macs_step

    def mha_hook(mod, inp, out):
        nonlocal total_flops
        q, k = inp[0], inp[1]
        # PyTorch MHA may be batch_first or seq_first; total tokens * d_model
        if mod.batch_first:
            bs, lq, d = q.shape
            lk = k.shape[1]
        else:
            lq, bs, d = q.shape
            lk = k.shape[0]
        # Q, K, V, Out projections + QK^T + Attn*V
        proj_macs = 4 * bs * lq * d * d
        attn_macs = 2 * bs * lq * lk * d
        total_flops += 2 * (proj_macs + attn_macs)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv1d_hook))
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            hooks.append(m.register_forward_hook(rnn_hook))
        elif isinstance(m, nn.MultiheadAttention):
            hooks.append(m.register_forward_hook(mha_hook))

    model.eval()
    with torch.no_grad():
        _ = model(x_in, h_in)
    for hk in hooks:
        hk.remove()
    return int(total_flops)


def profile_all_models(input_dim: int = 40):
    """Profile exact Params, MFLOPs, FP32 checkpoint size (MB), estimated training VRAM, and 1-thread CPU latency/FPS."""
    torch.set_num_threads(1)
    x1 = torch.randn(1, 12, input_dim)
    h1 = torch.tensor([[[20.0 + 0.2 * t, -65.0 - 0.3 * t] for t in range(12)]], dtype=torch.float32)
    tm, ts = [0.0, 0.0], [100.0, 100.0]

    # Define MLP baseline identically to mlp_residual_baseline.py (12*40=480 -> 128 -> 128 -> 2)
    class MLPRes(nn.Module):
        def __init__(self, flat_dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(flat_dim, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2))
        def forward(self, x, h):
            return self.net(x.reshape(x.shape[0], -1))

    specs = [
        ("Ridge Residual", None),
        ("MLP Residual", MLPRes(12 * input_dim)),
        ("LSTM (+Res B)", ExtendedKinematicModel("lstm", input_dim, 12, [4], tm, ts, "residual")),
        ("GRU (+Res B)", ExtendedKinematicModel("gru", input_dim, 12, [4], tm, ts, "residual")),
        ("TSMixer (+Res B)", ExtendedKinematicModel("tsmixer", input_dim, 12, [4], tm, ts, "residual")),
        ("PatchTST (+Res B)", ExtendedKinematicModel("patchtst", input_dim, 12, [4], tm, ts, "residual")),
        ("iTransformer (+Res B)", ExtendedKinematicModel("itransformer", input_dim, 12, [4], tm, ts, "residual")),
        ("ModernTCN (+Res B)", ExtendedKinematicModel("moderntcn", input_dim, 12, [4], tm, ts, "residual")),
        ("TimeMixer (+Res B)", ExtendedKinematicModel("timemixer", input_dim, 12, [4], tm, ts, "residual")),
        ("ST-Trans Full Direct (A)", ControlledNumericModel(input_dim, 12, [4], tm, ts, head="direct", d_model=256, num_heads=4, num_layers=3)),
        ("ST-Trans Full Residual (B)", ControlledNumericModel(input_dim, 12, [4], tm, ts, head="residual", d_model=256, num_heads=4, num_layers=3)),
        ("ST-Trans Compact Residual (B64)", ControlledNumericModel(input_dim, 12, [4], tm, ts, head="residual", d_model=64, num_heads=4, num_layers=1)),
        ("ST-Trans Compact Adaptive (D64)", ControlledNumericModel(input_dim, 12, [4], tm, ts, head="adaptive", d_model=64, num_heads=4, num_layers=1)),
    ]

    prof = {}
    for name, mod in specs:
        if mod is None:
            # Ridge linear map: (481 -> 2) => 962 params, 2 * 481 * 2 = 1,924 FLOPs
            w = np.zeros((12 * input_dim + 1, 2), dtype=np.float32)
            xb = np.ones((1, 12 * input_dim + 1), dtype=np.float32)
            for _ in range(30):
                _ = xb @ w
            t0 = time.perf_counter()
            for _ in range(500):
                _ = xb @ w
            ms = (time.perf_counter() - t0) * 1000.0 / 500.0
            prof[name] = {
                "params": 962,
                "mflops": round(1924 / 1e6, 4),
                "weight_mb": round(962 * 4 / (1024 ** 2), 4),
                "train_state_mb": round(962 * 4 / (1024 ** 2), 4),
                "cpu_b1_ms": round(ms, 3),
                "cpu_b1_fps": round(1000.0 / max(ms, 1e-6), 1),
            }
            continue
        mod.eval()
        n_params = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        flops = count_model_flops(mod, x1, h1)
        # Warmup & benchmark 1-thread CPU latency (100 reps)
        with torch.no_grad():
            for _ in range(20):
                _ = mod(x1, h1)
            t0 = time.perf_counter()
            for _ in range(100):
                _ = mod(x1, h1)
            ms = (time.perf_counter() - t0) * 1000.0 / 100.0
        weight_mb = n_params * 4.0 / (1024 ** 2)
        # AdamW FP32 training state = weights + grads + m + v = 4x weight memory
        prof[name] = {
            "params": int(n_params),
            "mflops": round(flops / 1e6, 3),
            "weight_mb": round(weight_mb, 2),
            "adamw_state_mb": round(weight_mb * 4.0, 2),
            "cpu_b1_ms": round(ms, 3),
            "cpu_b1_fps": round(1000.0 / max(ms, 1e-6), 1),
        }
    return prof


# ==============================================================================
# 3. Unified Training & Evaluation Helpers
# ==============================================================================
@torch.no_grad()
def eval_model_tensors(model, split_d: dict, batch_size: int = 128):
    model.eval()
    errs, preds_ll = [], []
    n = len(split_d["storm_ids"])
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        out = model(split_d["x"][sl], split_d["history"][sl])
        raw = denormalize_disp(out["pred"], model.target_mean, model.target_std)
        latlon = disp_to_latlon(raw, split_d["origin"][sl])
        ekm = haversine_km(latlon, split_d["future"][sl]).cpu().numpy().reshape(-1)
        errs.append(ekm)
        preds_ll.append(latlon.cpu().numpy().reshape(-1, 2))
    errors = np.concatenate(errs)
    pred_coords = np.concatenate(preds_ll, axis=0)
    sids = split_d["storm_ids"]
    smeans = np.array([errors[sids == s].mean() for s in sorted(set(sids))])
    return {
        "window_mean_km": float(errors.mean()),
        "storm_equal_mean_km": float(smeans.mean()),
        "errors": errors,
        "pred_coords": pred_coords,
    }


def train_and_eval_job(job: tuple):
    tag, arch, head, lead_steps, seed, train_d, val_d, test_d, ext_d, target_mean, target_std, return_coords = job
    torch.set_num_threads(1)
    seed_everything(seed)
    input_dim = train_d["x"].shape[-1]
    model = ExtendedKinematicModel(arch, input_dim, 12, lead_steps, target_mean, target_std, head)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val = math.inf
    best_state = None
    best_epoch = 0
    n_train = len(train_d["storm_ids"])
    t0 = time.perf_counter()

    for epoch in range(40):
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
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.perf_counter() - t0
    model.load_state_dict(best_state)
    te_res = eval_model_tensors(model, test_d)
    ex_res = eval_model_tensors(model, ext_d) if ext_d is not None else None
    out = {
        "tag": tag,
        "arch": arch,
        "head": head,
        "lead_hours": int(lead_steps[0] * 6),
        "seed": seed,
        "n_params": int(n_params),
        "best_epoch": int(best_epoch),
        "val_km": float(best_val),
        "test_window_km": te_res["window_mean_km"],
        "test_storm_km": te_res["storm_equal_mean_km"],
        "test_errors": te_res["errors"],
        "ext_window_km": ex_res["window_mean_km"] if ex_res else None,
        "ext_storm_km": ex_res["storm_equal_mean_km"] if ex_res else None,
        "ext_errors": ex_res["errors"] if ex_res else None,
        "elapsed_s": round(elapsed, 2),
    }
    if return_coords:
        out["test_pred_coords"] = te_res["pred_coords"]
    return out


def pack_atlantic_split(root: Path, split: str):
    raw = load_split_fast(root, split)
    t = raw["tensors"]
    return {
        "x": t["x"],
        "y": t["y"],
        "history": t["history"],
        "origin": t["origin"],
        "future": t["future"],
        "storm_ids": np.asarray([k[0] for k in raw["keys"]]),
        "windows": np.asarray([k[1] for k in raw["keys"]]),
    }


def fit_ridge_closed(X_tr, Y_tr, alpha: float):
    Xb = np.c_[np.ones(len(X_tr), dtype=np.float64), X_tr.astype(np.float64)]
    reg = np.eye(Xb.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y_tr.astype(np.float64))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    # ------------------------------------------------------------------
    # PART 1: Hardware Complexity & Efficiency Profiling (FLOPs, FPS, MB)
    # ------------------------------------------------------------------
    print("\n=== PART 1: EXACT FLOPs / PARAMS / MEMORY / FPS PROFILING ===")
    profiling = profile_all_models(input_dim=40)
    for k, v in profiling.items():
        print(f"  {k:34s} | Params={v['params']:9,d} | MFLOPs={v['mflops']:7.3f} | W={v['weight_mb']:5.2f} MB | CPU={v['cpu_b1_ms']:6.3f} ms ({v['cpu_b1_fps']:7.1f} FPS)")

    # ------------------------------------------------------------------
    # PART 2: North Atlantic ICLR 2024 SOTA (ModernTCN & TimeMixer, 6-24 h)
    # ------------------------------------------------------------------
    print("\n=== PART 2: NORTH ATLANTIC ICLR 2024 SOTA (ModernTCN & TimeMixer, 48 runs) ===")
    atl_jobs = []
    atl_test_sids = {}
    atl_ext_sids = {}
    for lead_idx in (1, 2, 3, 4):
        d_dir = EXP_ROOT / "data" / f"controlled_12to1_lead{lead_idx}"
        e_dir = EXP_ROOT / "data" / f"extension2025_lead{lead_idx}"
        meta = json.loads((d_dir / "metadata.json").read_text(encoding="utf-8"))
        tr_d = pack_atlantic_split(d_dir, "train")
        va_d = pack_atlantic_split(d_dir, "val")
        te_d = pack_atlantic_split(d_dir, "test")
        ex_d = pack_atlantic_split(e_dir, "test")
        atl_test_sids[lead_idx * 6] = te_d["storm_ids"]
        atl_ext_sids[lead_idx * 6] = ex_d["storm_ids"]
        tm, ts = meta["stats"]["target_mean"], meta["stats"]["target_std"]
        for arch in ("moderntcn", "timemixer"):
            for head in ("direct", "residual"):
                for seed in (42, 123, 2024):
                    atl_jobs.append(("atl", arch, head, meta["lead_steps"], seed, tr_d, va_d, te_d, ex_d, tm, ts, (lead_idx == 4 and seed == 42)))

    # Also save Ernesto 24-h case coords for PatchTST, iTransformer, LSTM, Compact B64 (seed 42)
    d4 = EXP_ROOT / "data" / "controlled_12to1_lead4"
    e4 = EXP_ROOT / "data" / "extension2025_lead4"
    m4 = json.loads((d4 / "metadata.json").read_text(encoding="utf-8"))
    tr4 = pack_atlantic_split(d4, "train")
    va4 = pack_atlantic_split(d4, "val")
    te4 = pack_atlantic_split(d4, "test")
    ex4 = pack_atlantic_split(e4, "test")
    case_atl_jobs = [
        ("case_atl", arch, head, [4], 42, tr4, va4, te4, None, m4["stats"]["target_mean"], m4["stats"]["target_std"], True)
        for arch in ("lstm", "patchtst", "itransformer", "compact_st_transformer")
        for head in ("direct", "residual")
    ]

    atl_results = []
    case_atl_results = []
    with cf.ProcessPoolExecutor(max_workers=10) as ex:
        for res in ex.map(train_and_eval_job, atl_jobs + case_atl_jobs):
            if res["tag"] == "atl":
                atl_results.append(res)
                print(f"  [ATL {res['arch']:10s} | {res['head']:8s} | {res['lead_hours']:2d}h | s={res['seed']}] 22-24={res['test_window_km']:.2f} | 25={res['ext_window_km']:.2f} ({res['elapsed_s']:.1f}s)")
            else:
                case_atl_results.append(res)

    atl_summary = {}
    for arch in ("moderntcn", "timemixer"):
        atl_summary[arch] = {}
        for lh in (6, 12, 18, 24):
            dr = sorted([r for r in atl_results if r["arch"] == arch and r["head"] == "direct" and r["lead_hours"] == lh], key=lambda x: x["seed"])
            rr = sorted([r for r in atl_results if r["arch"] == arch and r["head"] == "residual" and r["lead_hours"] == lh], key=lambda x: x["seed"])
            d_te = np.mean([r["test_errors"] for r in dr], axis=0)
            r_te = np.mean([r["test_errors"] for r in rr], axis=0)
            d_ex = np.mean([r["ext_errors"] for r in dr], axis=0)
            r_ex = np.mean([r["ext_errors"] for r in rr], axis=0)
            atl_summary[arch][str(lh)] = {
                "n_params": rr[0]["n_params"],
                "direct_22_24": round(float(np.mean([r["test_window_km"] for r in dr])), 2),
                "direct_22_24_sd": round(float(np.std([r["test_window_km"] for r in dr], ddof=1)), 2),
                "residual_22_24": round(float(np.mean([r["test_window_km"] for r in rr])), 2),
                "residual_22_24_sd": round(float(np.std([r["test_window_km"] for r in rr], ddof=1)), 2),
                "direct_25": round(float(np.mean([r["ext_window_km"] for r in dr])), 2),
                "direct_25_sd": round(float(np.std([r["ext_window_km"] for r in dr], ddof=1)), 2),
                "residual_25": round(float(np.mean([r["ext_window_km"] for r in rr])), 2),
                "residual_25_sd": round(float(np.std([r["ext_window_km"] for r in rr], ddof=1)), 2),
                "boot_22_24": {k: round(v, 2) for k, v in paired_storm_bootstrap(r_te, d_te, atl_test_sids[lh]).items()},
                "boot_25": {k: round(v, 2) for k, v in paired_storm_bootstrap(r_ex, d_ex, atl_ext_sids[lh]).items()},
            }

    # ------------------------------------------------------------------
    # PART 3: Eastern & Central North Pacific ICLR 2024 SOTA (ModernTCN & TimeMixer) + Otis Case Coords
    # ------------------------------------------------------------------
    print("\n=== PART 3: EASTERN & CENTRAL NORTH PACIFIC ICLR 2024 SOTA (ModernTCN & TimeMixer) ===")
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

    nepac_splits_d = {}
    for sp in ("train", "val", "test", "extension"):
        sr = [r for r in syn_rows if r["split"] == sp]
        nepac_splits_d[sp] = {
            "x": torch.from_numpy(np.stack([((np.where(np.isfinite(r["input_raw"]), r["input_raw"], xm) - xm) / xs).astype(np.float32) for r in sr])),
            "y": torch.from_numpy(np.stack([((r["target_disp_raw_km"] - ntm) / nts).astype(np.float32) for r in sr])),
            "history": torch.from_numpy(np.stack([r["history_latlon"].astype(np.float32) for r in sr])),
            "origin": torch.from_numpy(np.stack([r["origin"].astype(np.float32) for r in sr])),
            "future": torch.from_numpy(np.stack([r["future_latlon"].astype(np.float32) for r in sr])),
            "storm_ids": np.asarray([r["storm_id"] for r in sr]),
            "windows": np.asarray([r["window"] for r in sr]),
        }

    nepac_jobs = [
        ("nepac", arch, head, [4], seed, nepac_splits_d["train"], nepac_splits_d["val"], nepac_splits_d["test"], nepac_splits_d["extension"], ntm, nts, (seed == 42))
        for arch in ("moderntcn", "timemixer")
        for head in ("direct", "residual")
        for seed in (42, 123, 2024)
    ]
    case_nepac_jobs = [
        ("case_nepac", arch, head, [4], 42, nepac_splits_d["train"], nepac_splits_d["val"], nepac_splits_d["test"], None, ntm, nts, True)
        for arch in ("lstm", "patchtst", "itransformer", "compact_st_transformer")
        for head in ("direct", "residual")
    ]

    nepac_results = []
    case_nepac_results = []
    with cf.ProcessPoolExecutor(max_workers=10) as ex:
        for res in ex.map(train_and_eval_job, nepac_jobs + case_nepac_jobs):
            if res["tag"] == "nepac":
                nepac_results.append(res)
                print(f"  [NEPAC {res['arch']:10s} | {res['head']:8s} | s={res['seed']}] 22-24={res['test_window_km']:.2f} | 25={res['ext_window_km']:.2f} ({res['elapsed_s']:.1f}s)")
            else:
                case_nepac_results.append(res)

    nepac_new_summary = {}
    for arch in ("moderntcn", "timemixer"):
        dr = sorted([r for r in nepac_results if r["arch"] == arch and r["head"] == "direct"], key=lambda x: x["seed"])
        rr = sorted([r for r in nepac_results if r["arch"] == arch and r["head"] == "residual"], key=lambda x: x["seed"])
        d_te = np.mean([r["test_errors"] for r in dr], axis=0)
        r_te = np.mean([r["test_errors"] for r in rr], axis=0)
        d_ex = np.mean([r["ext_errors"] for r in dr], axis=0)
        r_ex = np.mean([r["ext_errors"] for r in rr], axis=0)
        nepac_new_summary[arch] = {
            "n_params": rr[0]["n_params"],
            "direct_2022_2024_mean_km": round(float(np.mean([r["test_window_km"] for r in dr])), 2),
            "direct_2022_2024_sd_km": round(float(np.std([r["test_window_km"] for r in dr], ddof=1)), 2),
            "residual_2022_2024_mean_km": round(float(np.mean([r["test_window_km"] for r in rr])), 2),
            "residual_2022_2024_sd_km": round(float(np.std([r["test_window_km"] for r in rr], ddof=1)), 2),
            "direct_2025_mean_km": round(float(np.mean([r["ext_window_km"] for r in dr])), 2),
            "direct_2025_sd_km": round(float(np.std([r["ext_window_km"] for r in dr], ddof=1)), 2),
            "residual_2025_mean_km": round(float(np.mean([r["ext_window_km"] for r in rr])), 2),
            "residual_2025_sd_km": round(float(np.std([r["ext_window_km"] for r in rr], ddof=1)), 2),
            "bootstrap_B_minus_A_2022_2024": {k: round(v, 2) for k, v in paired_storm_bootstrap(r_te, d_te, nepac_splits_d["test"]["storm_ids"]).items()},
            "bootstrap_B_minus_A_2025": {k: round(v, 2) for k, v in paired_storm_bootstrap(r_ex, d_ex, nepac_splits_d["extension"]["storm_ids"]).items()},
        }

    # Save Visual Trajectory Comparison Data (Hurricane Ernesto AL052024 & Hurricane Otis EP182023)
    def extract_case(split_d, all_case_runs, target_storm_substr, preferred_idx=None):
        matches = [i for i, s in enumerate(split_d["storm_ids"]) if target_storm_substr in str(s)]
        if not matches:
            return None
        # Choose either preferred window (e.g. Ernesto median/recurvature or Otis pre-landfall)
        idx = matches[preferred_idx] if (preferred_idx is not None and preferred_idx < len(matches)) else matches[len(matches) // 2]
        hist = split_d["history"][idx].numpy()
        orig = split_d["origin"][idx].numpy()
        fut = split_d["future"][idx, 0].numpy()
        cv_raw = cv_displacement(split_d["history"][idx : idx + 1], [4], "legacy_previous_origin")
        cv_ll = disp_to_latlon(cv_raw, split_d["origin"][idx : idx + 1]).numpy()[0, 0]
        cv_err = float(haversine_km(torch.tensor([[cv_ll]]), torch.tensor([[fut]])).item())
        models_out = {"cv": {"latlon": cv_ll.tolist(), "err_km": round(cv_err, 2)}}
        for r in all_case_runs:
            if "test_pred_coords" in r:
                k = f"{r['arch']}_{r['head']}"
                ll = r["test_pred_coords"][idx]
                models_out[k] = {"latlon": ll.tolist(), "err_km": round(float(r["test_errors"][idx]), 2)}
        return {
            "storm_id": str(split_d["storm_ids"][idx]),
            "window_ts_hours": int(split_d["windows"][idx]),
            "history_latlon": hist.tolist(),
            "origin_latlon": orig.tolist(),
            "future_latlon": fut.tolist(),
            "predictions": models_out,
        }

    # Find exact Ernesto 0600 UTC 17 Aug 2024 window (if present in te4) and best Otis/Beryl pre-landfall window
    atl_case_runs = case_atl_results + [r for r in atl_results if r["lead_hours"] == 24 and r["seed"] == 42]
    nepac_case_runs = case_nepac_results + [r for r in nepac_results if r["seed"] == 42]
    # Check Ernesto window index matching 2024-08-17 0600 UTC
    ernesto_target_ts = int((pd.Timestamp("2024-08-17 06:00:00") - pd.Timestamp("1970-01-01")) // pd.Timedelta(hours=1))
    ernesto_idx = None
    for i, (s, w) in enumerate(zip(te4["storm_ids"], te4["windows"])):
        if "AL052024" in str(s) and int(w) == ernesto_target_ts:
            ernesto_idx = i
            break
    if ernesto_idx is not None:
        # Compute relative offset inside AL052024 matches
        ern_matches = [i for i, s in enumerate(te4["storm_ids"]) if "AL052024" in str(s)]
        ern_pref = ern_matches.index(ernesto_idx)
    else:
        ern_pref = None

    # Check major Pacific storms with 24-h synoptic windows (prefer Hurricane Roslyn / Norma / Hillary / Otis / Lidia / John)
    pac_case = None
    for cand in ("EP182023", "EP102024", "EP152023", "EP092023", "EP192022"):
        c_matches = [i for i, s in enumerate(nepac_splits_d["test"]["storm_ids"]) if cand in str(s)]
        if c_matches:
            # Pick last pre-landfall or peak-turn window
            pac_case = extract_case(nepac_splits_d["test"], nepac_case_runs, cand, preferred_idx=len(c_matches) - 1)
            break
    atl_case = extract_case(te4, atl_case_runs, "AL052024", preferred_idx=ern_pref)

    # ------------------------------------------------------------------
    # PART 4: Leakage-Free 5-Fold Storm-Grouped Cross-Validation (2004-2021 North Atlantic, 24 h)
    # ------------------------------------------------------------------
    print("\n=== PART 4: LEAKAGE-FREE 5-FOLD STORM-GROUPED CV (287 STORMS, 2004-2021, 80 RUNS) ===")
    # Reconstruct raw (unstandardized) 2004-2021 24-h development windows from train + val .npy files
    dev_rows = []
    for sp_name in ("train", "val"):
        for p in sorted((d4 / sp_name).glob("*.npy")):
            dev_rows.append(np.load(p, allow_pickle=True).item())
    # Invert original train z-score so we can refit z-score STRICTLY inside each of the 5 folds (zero fold leakage!)
    orig_xm = np.asarray(m4["stats"]["x_mean"], dtype=np.float32)
    orig_xs = np.asarray(m4["stats"]["x_std"], dtype=np.float32)
    raw_x_all = np.stack([(r["input_num"] * orig_xs + orig_xm).astype(np.float32) for r in dev_rows])
    raw_y_all = np.stack([r["target_disp_raw_km"].astype(np.float32) for r in dev_rows])
    hist_all = np.stack([r["history_latlon"].astype(np.float32) for r in dev_rows])
    orig_all = np.stack([r["origin"].astype(np.float32) for r in dev_rows])
    fut_all = np.stack([r["future_latlon"].astype(np.float32) for r in dev_rows])
    sids_all = np.asarray([str(r["storm_id"]) for r in dev_rows])

    gkf = GroupKFold(n_splits=5)
    cv_archs = ["lstm", "gru", "tsmixer", "patchtst", "itransformer", "moderntcn", "timemixer", "compact_st_transformer"]
    cv_jobs = []
    cv_baselines = {"cv": [], "ridge_kr": []}
    fold_meta = []

    for fold_idx, (tr_val_idx, test_idx) in enumerate(gkf.split(raw_x_all, groups=sids_all)):
        # Inside the 4-fold training pool, hold out 10% of storms strictly as validation for checkpointing
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

        # Fold-specific training-only z-score moments (ZERO leakage from val or held-out fold!)
        fxm = raw_x_all[tr_idx].mean(axis=(0, 1)).astype(np.float32)
        fxs = np.where(raw_x_all[tr_idx].std(axis=(0, 1)) > 1e-6, raw_x_all[tr_idx].std(axis=(0, 1)), 1.0).astype(np.float32)
        fym = raw_y_all[tr_idx].mean(axis=(0, 1)).astype(np.float32)
        fys = np.where(raw_y_all[tr_idx].std(axis=(0, 1)) > 1e-6, raw_y_all[tr_idx].std(axis=(0, 1)), 1.0).astype(np.float32)

        def make_fold_dict(idxs):
            return {
                "x": torch.from_numpy(((raw_x_all[idxs] - fxm) / fxs).astype(np.float32)),
                "y": torch.from_numpy(((raw_y_all[idxs] - fym) / fys).astype(np.float32)),
                "history": torch.from_numpy(hist_all[idxs]),
                "origin": torch.from_numpy(orig_all[idxs]),
                "future": torch.from_numpy(fut_all[idxs]),
                "storm_ids": sids_all[idxs],
            }

        f_tr, f_va, f_te = make_fold_dict(tr_idx), make_fold_dict(va_idx), make_fold_dict(test_idx)
        fold_meta.append({"fold": fold_idx + 1, "test_windows": int(len(test_idx)), "test_storms": int(len(set(sids_all[test_idx])))})

        # CV & Ridge-KR on this fold
        cv_te_raw = cv_displacement(f_te["history"], [4], "legacy_previous_origin")
        cv_te_err = haversine_km(disp_to_latlon(cv_te_raw, f_te["origin"]), f_te["future"]).numpy().reshape(-1)
        cv_baselines["cv"].append(float(cv_te_err.mean()))

        cv_tr_raw = cv_displacement(f_tr["history"], [4], "legacy_previous_origin").numpy()
        cv_va_raw = cv_displacement(f_va["history"], [4], "legacy_previous_origin").numpy()
        X_ftr = f_tr["x"].reshape(len(tr_idx), -1).numpy()
        X_fva = f_va["x"].reshape(len(va_idx), -1).numpy()
        X_fte = f_te["x"].reshape(len(test_idx), -1).numpy()
        Y_fres = (raw_y_all[tr_idx] - cv_tr_raw).reshape(-1, 2)
        best_a, best_v, best_W = None, math.inf, None
        for alpha in np.logspace(-2, 8, 11):
            W = fit_ridge_closed(X_ftr, Y_fres, alpha)
            p_va = cv_va_raw + (np.c_[np.ones(len(X_fva)), X_fva] @ W).reshape(-1, 1, 2).astype(np.float32)
            vk = float(haversine_km(disp_to_latlon(torch.from_numpy(p_va), f_va["origin"]), f_va["future"]).mean())
            if vk < best_v:
                best_v, best_a, best_W = vk, alpha, W
        p_te = cv_te_raw.numpy() + (np.c_[np.ones(len(X_fte)), X_fte] @ best_W).reshape(-1, 1, 2).astype(np.float32)
        r_te_err = haversine_km(disp_to_latlon(torch.from_numpy(p_te), f_te["origin"]), f_te["future"]).numpy().reshape(-1)
        cv_baselines["ridge_kr"].append(float(r_te_err.mean()))

        for arch in cv_archs:
            for head in ("direct", "residual"):
                cv_jobs.append((f"fold{fold_idx + 1}", arch, head, [4], 42 + fold_idx, f_tr, f_va, f_te, None, fym, fys, False))

    cv_runs = []
    with cf.ProcessPoolExecutor(max_workers=10) as ex:
        for res in ex.map(train_and_eval_job, cv_jobs):
            cv_runs.append(res)

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
    for arch in cv_archs:
        d_f = [r["test_window_km"] for r in sorted([x for x in cv_runs if x["arch"] == arch and x["head"] == "direct"], key=lambda z: z["tag"])]
        r_f = [r["test_window_km"] for r in sorted([x for x in cv_runs if x["arch"] == arch and x["head"] == "residual"], key=lambda z: z["tag"])]
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
        print(f"  [5-Fold CV | {arch:22s}] Direct={np.mean(d_f):.2f}+/-{np.std(d_f,ddof=1):.2f} -> Res={np.mean(r_f):.2f}+/-{np.std(r_f,ddof=1):.2f} (Delta={np.mean(r_f)-np.mean(d_f):+.2f} km, {wins}/5 folds won)")

    final_payload = {
        "total_wall_seconds": round(time.perf_counter() - t_all, 2),
        "profiling": profiling,
        "atlantic_iclr2024_sota": atl_summary,
        "pacific_iclr2024_sota": nepac_new_summary,
        "five_fold_storm_grouped_cv": cv_summary,
        "sota_case_trajectories": {
            "atlantic_ernesto": atl_case,
            "pacific_major_case": pac_case,
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(final_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== ALL STAGE-2 EXPERIMENTS COMPLETE IN {final_payload['total_wall_seconds']:.1f}s -> {OUT_DIR / 'summary.json'} ===")


if __name__ == "__main__":
    main()
