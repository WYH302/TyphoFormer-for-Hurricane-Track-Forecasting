"""Strictly matched multi-architecture SOTA sequence baseline suite (LSTM, GRU, TSMixer,
PatchTST, iTransformer) comparing Direct Eulerian regression (A) vs. Lagrangian Kinematic
Residual parameterization (B) across 6, 12, 18, and 24 h on 2022-2024 and 2025 HURDAT2 cohorts.

Fairness & Integrity Contract:
- Uses the exact same train/val/test/extension2025 .npy splits and training-only z-score moments
  as run_controlled.py.
- Uses the exact same AdamW(lr=1e-4, weight_decay=1e-4), batch_size=32, 40 epochs, grad_clip=1.0,
  three seeds (42, 123, 2024), and minimum-validation-haversine checkpoint selection.
- Test (2022-2024) and Extension (2025) splits are evaluated strictly once after validation selection.
"""
import concurrent.futures as cf
import json
import math
import os
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

from controlled_heads import cv_displacement
from vendor.common import (
    denormalize_disp,
    disp_to_latlon,
    haversine_km,
    normalize_disp,
    seed_everything,
)


# ==============================================================================
# Five Canonical & SOTA Sequence Backbones
# ==============================================================================
class RecurrentEncoder(nn.Module):
    """2-layer LSTM (Gao et al., 2018) or GRU (Alemany et al., 2019)."""
    def __init__(self, kind: str, input_dim: int, d_model: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = cls(input_dim, d_model, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.norm(out[:, -1, :])


class TSMixerBlock(nn.Module):
    def __init__(self, seq_len: int, channels: int, ff_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.norm_t = nn.LayerNorm(channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm_c = nn.LayerNorm(channels)
        self.chan_mlp = nn.Sequential(
            nn.Linear(channels, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm_t(x).transpose(1, 2)
        x = x + self.time_mlp(y).transpose(1, 2)
        x = x + self.chan_mlp(self.norm_c(x))
        return x


class TSMixerEncoder(nn.Module):
    """TSMixer (Ekambaram et al., KDD 2023): alternating temporal & channel MLP-Mixer blocks."""
    def __init__(self, input_dim: int, seq_len: int = 12, d_model: int = 128, num_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([TSMixerBlock(seq_len, input_dim, d_model, dropout) for _ in range(num_blocks)])
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)


class PatchTSTEncoder(nn.Module):
    """PatchTST (Nie et al., ICLR 2023): non-overlapping temporal patch tokenization + Pre-LN Transformer."""
    def __init__(self, input_dim: int, seq_len: int = 12, patch_len: int = 3, d_model: int = 128,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert seq_len % patch_len == 0
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.patch_proj = nn.Linear(patch_len * input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        patches = x.reshape(b, self.n_patches, self.patch_len * c)
        h = self.patch_proj(patches) + self.pos_emb
        h = self.encoder(h)
        return self.norm(h[:, -1, :])


class ITransformerEncoder(nn.Module):
    """iTransformer (Liu et al., ICLR 2024): variate-as-token inverted self-attention."""
    def __init__(self, input_dim: int, seq_len: int = 12, d_model: int = 128,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.var_proj = nn.Linear(seq_len, d_model)
        self.var_emb = nn.Parameter(torch.zeros(1, input_dim, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> transpose to (B, C, L) so each physical channel is an attention token
        xt = x.transpose(1, 2)
        h = self.var_proj(xt) + self.var_emb
        h = self.encoder(h)
        return self.pool(h.mean(dim=1))


class WrappedKinematicModel(nn.Module):
    def __init__(self, arch: str, input_dim: int, seq_len: int, lead_steps, target_mean, target_std, head: str):
        super().__init__()
        assert head in {"direct", "residual"}
        self.arch = arch
        self.head = head
        if arch in {"lstm", "gru"}:
            self.backbone = RecurrentEncoder(arch, input_dim, d_model=128, num_layers=2, dropout=0.1)
        elif arch == "tsmixer":
            self.backbone = TSMixerEncoder(input_dim, seq_len=seq_len, d_model=128, num_blocks=3, dropout=0.1)
        elif arch == "patchtst":
            self.backbone = PatchTSTEncoder(input_dim, seq_len=seq_len, patch_len=3, d_model=128, num_heads=4, num_layers=2, dropout=0.1)
        elif arch == "itransformer":
            self.backbone = ITransformerEncoder(input_dim, seq_len=seq_len, d_model=128, num_heads=4, num_layers=2, dropout=0.1)
        else:
            raise ValueError(arch)

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
        ctx = self.backbone(x_num)
        cv_raw = cv_displacement(history_latlon, self.lead_steps, "legacy_previous_origin")
        cv_norm = normalize_disp(cv_raw, self.target_mean, self.target_std)
        delta = self.proj(ctx).reshape_as(cv_norm)
        pred = delta if self.head == "direct" else (cv_norm + delta)
        return {"pred": pred, "cv": cv_norm}


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


@torch.no_grad()
def eval_split(model: WrappedKinematicModel, data: dict, batch_size: int = 128):
    model.eval()
    errs = []
    n = len(data["keys"])
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        out = model(data["tensors"]["x"][sl], data["tensors"]["history"][sl])
        raw = denormalize_disp(out["pred"], model.target_mean, model.target_std)
        latlon = disp_to_latlon(raw, data["tensors"]["origin"][sl])
        ekm = haversine_km(latlon, data["tensors"]["future"][sl]).cpu().numpy().reshape(-1)
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


def run_single_job(job: tuple):
    arch, head, lead_idx, seed = job
    torch.set_num_threads(1)
    data_dir = J_ROOT / "data" / f"controlled_12to1_lead{lead_idx}"
    ext_dir = J_ROOT / "data" / f"extension2025_lead{lead_idx}"
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))

    train_d = load_split_fast(data_dir, "train")
    val_d = load_split_fast(data_dir, "val")
    test_d = load_split_fast(data_dir, "test")
    ext_d = load_split_fast(ext_dir, "test")

    seed_everything(seed)
    model = WrappedKinematicModel(
        arch=arch,
        input_dim=train_d["tensors"]["x"].shape[-1],
        seq_len=meta["input_len"],
        lead_steps=meta["lead_steps"],
        target_mean=meta["stats"]["target_mean"],
        target_std=meta["stats"]["target_std"],
        head=head,
    )
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

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
            bx = train_d["tensors"]["x"][ids]
            bh = train_d["tensors"]["history"][ids]
            by = train_d["tensors"]["y"][ids]
            opt.zero_grad(set_to_none=True)
            pred = model(bx, bh)["pred"]
            loss = F.mse_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_res = eval_split(model, val_d)
        if val_res["window_mean_km"] < best_val:
            best_val = val_res["window_mean_km"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed_s = time.perf_counter() - t0
    model.load_state_dict(best_state)
    test_res = eval_split(model, test_d)
    ext_res = eval_split(model, ext_d)

    out_root = J_ROOT / "results" / "sota_sequence_baselines" / f"lead{lead_idx}_{arch}_{head}_seed{seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_root / "predictions.npz",
        test_errors=test_res["errors"],
        test_storm_ids=test_res["storm_ids"],
        ext_errors=ext_res["errors"],
        ext_storm_ids=ext_res["storm_ids"],
    )
    rec = {
        "arch": arch,
        "head": head,
        "lead_hours": lead_idx * 6,
        "seed": seed,
        "parameters": n_params,
        "best_epoch": best_epoch,
        "val_window_km": best_val,
        "test_window_km": test_res["window_mean_km"],
        "test_storm_km": test_res["storm_equal_mean_km"],
        "ext_window_km": ext_res["window_mean_km"],
        "ext_storm_km": ext_res["storm_equal_mean_km"],
        "elapsed_seconds": elapsed_s,
    }
    (out_root / "metrics.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def paired_storm_bootstrap(err_res: np.ndarray, err_dir: np.ndarray, storm_ids: np.ndarray, n_boot: int = 10000, seed: int = 20260913):
    uniq = sorted(set(storm_ids.tolist()))
    storm_diffs = np.array([(err_res[storm_ids == s] - err_dir[storm_ids == s]).mean() for s in uniq])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot_means = storm_diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(storm_diffs.mean()), float(lo), float(hi)


def main():
    archs = ["lstm", "gru", "tsmixer", "patchtst", "itransformer"]
    heads = ["direct", "residual"]
    leads = [1, 2, 3, 4]
    seeds = [42, 123, 2024]

    jobs = [(a, h, l, s) for l in leads for a in archs for h in heads for s in seeds]
    print(f"Launching {len(jobs)} matched SOTA sequence runs across 12 CPU workers...")
    t_start = time.perf_counter()
    results = []
    with cf.ProcessPoolExecutor(max_workers=12) as ex:
        for i, rec in enumerate(ex.map(run_single_job, jobs), 1):
            results.append(rec)
            if i % 15 == 0 or i == len(jobs):
                print(f"  Completed {i}/{len(jobs)} runs ({time.perf_counter() - t_start:.1f}s elapsed)")

    # Aggregate 3-seed statistics and 10,000-resample paired whole-storm bootstrap (Residual - Direct)
    summary = {"runs": results, "aggregated": [], "paired_residual_minus_direct": [], "wall_seconds": time.perf_counter() - t_start}
    for l in leads:
        lh = l * 6
        for a in archs:
            for h in heads:
                sub = [r for r in results if r["lead_hours"] == lh and r["arch"] == a and r["head"] == h]
                tw = [r["test_window_km"] for r in sub]
                ts = [r["test_storm_km"] for r in sub]
                ew = [r["ext_window_km"] for r in sub]
                es = [r["ext_storm_km"] for r in sub]
                summary["aggregated"].append({
                    "arch": a, "head": h, "lead_hours": lh,
                    "parameters": sub[0]["parameters"],
                    "test_window_mean": float(np.mean(tw)), "test_window_sd": float(np.std(tw, ddof=1)),
                    "test_storm_mean": float(np.mean(ts)), "test_storm_sd": float(np.std(ts, ddof=1)),
                    "ext_window_mean": float(np.mean(ew)), "ext_window_sd": float(np.std(ew, ddof=1)),
                    "ext_storm_mean": float(np.mean(es)), "ext_storm_sd": float(np.std(es, ddof=1)),
                    "mean_train_seconds": float(np.mean([r["elapsed_seconds"] for r in sub])),
                })

            # Paired storm bootstrap across 3-seed per-window mean errors
            res_npzs = [np.load(J_ROOT / "results" / "sota_sequence_baselines" / f"lead{l}_{a}_residual_seed{s}" / "predictions.npz") for s in seeds]
            dir_npzs = [np.load(J_ROOT / "results" / "sota_sequence_baselines" / f"lead{l}_{a}_direct_seed{s}" / "predictions.npz") for s in seeds]

            test_res_bar = np.mean([z["test_errors"] for z in res_npzs], axis=0)
            test_dir_bar = np.mean([z["test_errors"] for z in dir_npzs], axis=0)
            test_ids = res_npzs[0]["test_storm_ids"]
            d_test, lo_test, hi_test = paired_storm_bootstrap(test_res_bar, test_dir_bar, test_ids)

            ext_res_bar = np.mean([z["ext_errors"] for z in res_npzs], axis=0)
            ext_dir_bar = np.mean([z["ext_errors"] for z in dir_npzs], axis=0)
            ext_ids = res_npzs[0]["ext_storm_ids"]
            d_ext, lo_ext, hi_ext = paired_storm_bootstrap(ext_res_bar, ext_dir_bar, ext_ids)

            summary["paired_residual_minus_direct"].append({
                "arch": a, "lead_hours": lh,
                "test_storm_diff_km": d_test, "test_ci_low": lo_test, "test_ci_high": hi_test,
                "ext_storm_diff_km": d_ext, "ext_ci_low": lo_ext, "ext_ci_high": hi_ext,
            })

    sum_path = J_ROOT / "results" / "sota_sequence_baselines" / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved complete 120-run SOTA sequence summary to {sum_path} in {summary['wall_seconds']:.1f}s!")


if __name__ == "__main__":
    main()
