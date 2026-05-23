import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset


EARTH_RADIUS_KM = 6371.0
LEADS = [1, 2, 3, 4]
SEEDS = [42, 123, 2024]
RIDGE_ALPHAS = np.logspace(-4, 4, 17)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wrap_lon_delta(lon, lon0):
    return (lon - lon0 + 180.0) % 360.0 - 180.0


def latlon_to_disp_np(latlon, origin):
    lat = np.deg2rad(latlon[..., 0])
    lon = latlon[..., 1]
    lat0 = np.deg2rad(origin[..., 0])
    lon0 = origin[..., 1]
    dlat = lat - lat0
    dlon = np.deg2rad(wrap_lon_delta(lon, lon0))
    dx = EARTH_RADIUS_KM * np.cos(lat0) * dlon
    dy = EARTH_RADIUS_KM * dlat
    return np.stack([dx, dy], axis=-1)


def disp_to_latlon_np(disp_km, origin):
    lat0_deg = origin[:, 0][:, None]
    lon0_deg = origin[:, 1][:, None]
    lat0_rad = np.deg2rad(lat0_deg)
    lat = lat0_deg + np.rad2deg(disp_km[..., 1] / EARTH_RADIUS_KM)
    denom = EARTH_RADIUS_KM * np.maximum(np.cos(lat0_rad), 1e-6)
    lon = lon0_deg + np.rad2deg(disp_km[..., 0] / denom)
    lon = (lon + 180.0) % 360.0 - 180.0
    return np.stack([lat, lon], axis=-1)


def haversine_np(pred_latlon, target_latlon):
    lat1 = np.deg2rad(pred_latlon[..., 0])
    lon1 = np.deg2rad(pred_latlon[..., 1])
    lat2 = np.deg2rad(target_latlon[..., 0])
    lon2 = np.deg2rad(target_latlon[..., 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return EARTH_RADIUS_KM * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def constant_velocity_disp(history_latlon, lead_steps):
    prev = history_latlon[:, -2]
    origin = history_latlon[:, -1]
    step_disp = latlon_to_disp_np(origin, prev)
    return step_disp[:, None, :] * np.asarray(lead_steps, dtype=np.float32)[None, :, None]


def load_records(split_dir, lead_steps):
    files = sorted(Path(split_dir).glob("*.npy"))
    rows = [np.load(path, allow_pickle=True).item() for path in files]
    x = np.stack([row["input_num"] for row in rows]).astype(np.float32)
    y_raw = np.stack([row["target_disp_raw_km"] for row in rows]).astype(np.float32)
    history = np.stack([row["history_latlon"] for row in rows]).astype(np.float32)
    origin = np.stack([row["origin"] for row in rows]).astype(np.float32)
    future = np.stack([row["future_latlon"] for row in rows]).astype(np.float32)
    storm_ids = [str(row["storm_id"]) for row in rows]
    storm_year = np.asarray([int(row["storm_year"]) for row in rows], dtype=np.int32)
    window = np.asarray([int(row["window"]) for row in rows], dtype=np.int32)
    cv_raw = constant_velocity_disp(history, lead_steps).astype(np.float32)
    return {
        "files": [str(path) for path in files],
        "x": x.reshape(x.shape[0], -1),
        "y_raw": y_raw,
        "cv_raw": cv_raw,
        "origin": origin,
        "future": future,
        "storm_ids": storm_ids,
        "storm_year": storm_year,
        "window": window,
    }


def standardize(train_x, *arrays):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return [(arr - mean) / std for arr in arrays], mean, std


def target_standardize(train_y, *arrays):
    mean = train_y.reshape(-1, 2).mean(axis=0).reshape(1, 1, 2)
    std = train_y.reshape(-1, 2).std(axis=0).reshape(1, 1, 2)
    std = np.where(std < 1e-6, 1.0, std)
    return [(arr - mean) / std for arr in arrays], mean.astype(np.float32), std.astype(np.float32)


def metrics_from_raw(pred_raw, data):
    pred_latlon = disp_to_latlon_np(pred_raw, data["origin"])
    err = haversine_np(pred_latlon, data["future"])
    mae = np.abs(pred_latlon - data["future"]).mean(axis=-1)
    storm_sums = {}
    for idx, storm_id in enumerate(data["storm_ids"]):
        storm_sums.setdefault(storm_id, []).append(float(err[idx, -1]))
    storm_means = {storm_id: float(np.mean(values)) for storm_id, values in storm_sums.items()}
    return {
        "ade": float(err.mean(axis=1).mean()),
        "fde": float(err[:, -1].mean()),
        "err": float(err[:, -1].mean()),
        "mae": float(mae[:, -1].mean()),
        "windows": int(err.shape[0]),
        "storms": int(len(storm_means)),
        "storm_mean": float(np.mean(list(storm_means.values()))),
        "storm_better_values": storm_means,
    }


def fit_ridge(train, val, test, model_type):
    y_train = train["y_raw"]
    if model_type == "ridge_cv_residual":
        y_train = train["y_raw"] - train["cv_raw"]
    ([x_train, x_val, x_test], _, _) = standardize(train["x"], train["x"], val["x"], test["x"])

    best = None
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd")
        model.fit(x_train, y_train.reshape(y_train.shape[0], -1))
        pred_val = model.predict(x_val).reshape(val["y_raw"].shape)
        if model_type == "ridge_cv_residual":
            pred_val = val["cv_raw"] + pred_val
        val_metrics = metrics_from_raw(pred_val, val)
        if best is None or val_metrics["fde"] < best["val"]["fde"]:
            pred_test = model.predict(x_test).reshape(test["y_raw"].shape)
            if model_type == "ridge_cv_residual":
                pred_test = test["cv_raw"] + pred_test
            best = {
                "alpha": float(alpha),
                "val": val_metrics,
                "test": metrics_from_raw(pred_test, test),
            }
    return best


class MLPResidual(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


class TinyDualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.05):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.direct = nn.Linear(hidden_dim, 2)
        self.residual = nn.Linear(hidden_dim, 2)
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.direct(h), self.residual(h), torch.sigmoid(self.gate(h))


def parameter_count(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def make_tensor_dataset(x, y, cv):
    return TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y[:, 0, :], dtype=torch.float32),
        torch.tensor(cv[:, 0, :], dtype=torch.float32),
    )


@torch.no_grad()
def eval_mlp_residual(model, x, cv, res_mean, res_std, data, device):
    model.eval()
    preds = []
    for start in range(0, x.shape[0], 512):
        xb = torch.tensor(x[start : start + 512], dtype=torch.float32, device=device)
        residual = model(xb).cpu().numpy()[:, None, :] * res_std + res_mean
        preds.append(cv[start : start + 512] + residual.astype(np.float32))
    return metrics_from_raw(np.concatenate(preds, axis=0), data)


@torch.no_grad()
def eval_tiny_dual(model, x, cv, y_mean, y_std, res_mean, res_std, data, device):
    model.eval()
    preds = []
    gates = []
    for start in range(0, x.shape[0], 512):
        xb = torch.tensor(x[start : start + 512], dtype=torch.float32, device=device)
        direct_z, residual_z, gate = model(xb)
        direct = direct_z.cpu().numpy()[:, None, :] * y_std + y_mean
        residual = residual_z.cpu().numpy()[:, None, :] * res_std + res_mean
        gate_np = gate.cpu().numpy()[:, None, :]
        pred = gate_np * (cv[start : start + 512] + residual) + (1.0 - gate_np) * direct
        preds.append(pred.astype(np.float32))
        gates.append(gate_np.reshape(-1))
    metrics = metrics_from_raw(np.concatenate(preds, axis=0), data)
    all_gates = np.concatenate(gates)
    metrics["mean_gate_cv_residual"] = float(all_gates.mean())
    metrics["p10_gate_cv_residual"] = float(np.percentile(all_gates, 10))
    metrics["p90_gate_cv_residual"] = float(np.percentile(all_gates, 90))
    return metrics


def train_mlp_residual(train, val, test, seed, args, device):
    seed_everything(seed)
    ([x_train, x_val, x_test], _, _) = standardize(train["x"], train["x"], val["x"], test["x"])
    res_train = train["y_raw"] - train["cv_raw"]
    ([res_train_z], res_mean, res_std) = target_standardize(res_train, res_train)
    train_ds = make_tensor_dataset(x_train, res_train_z, train["cv_raw"])
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = MLPResidual(x_train.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = {"score": math.inf, "epoch": 0, "state": None}
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb, _ in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_metrics = eval_mlp_residual(model, x_val, val["cv_raw"], res_mean, res_std, val, device)
        if val_metrics["fde"] < best["score"]:
            best = {
                "score": val_metrics["fde"],
                "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best["state"])
    return {
        "seed": seed,
        "epoch": best["epoch"],
        "params": parameter_count(model),
        "val": eval_mlp_residual(model, x_val, val["cv_raw"], res_mean, res_std, val, device),
        "test": eval_mlp_residual(model, x_test, test["cv_raw"], res_mean, res_std, test, device),
    }


def train_tiny_dual(train, val, test, seed, args, device):
    seed_everything(seed)
    ([x_train, x_val, x_test], _, _) = standardize(train["x"], train["x"], val["x"], test["x"])
    ([y_train_z], y_mean, y_std) = target_standardize(train["y_raw"], train["y_raw"])
    res_train = train["y_raw"] - train["cv_raw"]
    ([res_train_z], res_mean, res_std) = target_standardize(res_train, res_train)
    train_ds = make_tensor_dataset(x_train, y_train_z, train["cv_raw"])
    residual_target = torch.tensor(res_train_z[:, 0, :], dtype=torch.float32)
    cv_z_train = torch.tensor((train["cv_raw"] - y_mean)[:, 0, :] / y_std.reshape(1, 2), dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(train_ds.tensors[0], train_ds.tensors[1], residual_target, cv_z_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = TinyDualMLP(x_train.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    res_scale = torch.tensor((res_std / y_std).reshape(1, 2), dtype=torch.float32, device=device)
    res_offset = torch.tensor((res_mean / y_std).reshape(1, 2), dtype=torch.float32, device=device)
    best = {"score": math.inf, "epoch": 0, "state": None}
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb, rb, cvb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            rb = rb.to(device)
            cvb = cvb.to(device)
            optimizer.zero_grad(set_to_none=True)
            direct_z, residual_z, gate = model(xb)
            pred_z = gate * (cvb + residual_z * res_scale + res_offset) + (1.0 - gate) * direct_z
            loss = nn.functional.mse_loss(pred_z, yb)
            loss = loss + 0.1 * nn.functional.mse_loss(direct_z, yb)
            loss = loss + 0.1 * nn.functional.mse_loss(residual_z, rb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_metrics = eval_tiny_dual(model, x_val, val["cv_raw"], y_mean, y_std, res_mean, res_std, val, device)
        if val_metrics["fde"] < best["score"]:
            best = {
                "score": val_metrics["fde"],
                "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best["state"])
    return {
        "seed": seed,
        "epoch": best["epoch"],
        "params": parameter_count(model),
        "val": eval_tiny_dual(model, x_val, val["cv_raw"], y_mean, y_std, res_mean, res_std, val, device),
        "test": eval_tiny_dual(model, x_test, test["cv_raw"], y_mean, y_std, res_mean, res_std, test, device),
    }


def mean_std(values):
    values = [float(value) for value in values]
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}


def summarize_rows(rows):
    out = {}
    for model_name in sorted({row["model"] for row in rows}):
        out[model_name] = {}
        for lead in LEADS:
            vals = [row["test"]["err"] for row in rows if row["model"] == model_name and row["lead"] == lead]
            if vals:
                out[model_name][str(lead)] = mean_std(vals)
    return out


def write_markdown(path, summary):
    lines = [
        "# Official 12->1 Simple Residual Baselines",
        "",
        "Values are test DeltaR km under the strict-HURDAT2 safe-negative protocol. Ridge rows select alpha on validation; neural rows select epoch on validation.",
        "",
        "| Model | 6h | 12h | 18h | 24h |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "ridge_direct": "Direct ridge",
        "ridge_cv_residual": "Ridge residual-on-CV",
        "mlp_cv_residual": "MLP residual-on-CV",
        "tiny_dual_mlp": "Tiny dual MLP, no Transformer",
    }
    for model_name in ["ridge_direct", "ridge_cv_residual", "mlp_cv_residual", "tiny_dual_mlp"]:
        cells = []
        for lead in LEADS:
            stat = summary["aggregate"].get(model_name, {}).get(str(lead))
            if not stat:
                cells.append("--")
            elif stat["std"] == 0.0:
                cells.append(f"{stat['mean']:.3f}")
            else:
                cells.append(f"{stat['mean']:.3f} +/- {stat['std']:.3f}")
        lines.append(f"| {labels[model_name]} | " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="../local_artifacts/TyphoFormerPlus")
    parser.add_argument("--output-json", default="official_12to1_simple_residual_baselines.json")
    parser.add_argument("--output-md", default="official_12to1_simple_residual_baselines.md")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument("--leads", type=int, nargs="*", default=LEADS)
    return parser.parse_args()


def main():
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    started = time.time()
    for lead in args.leads:
        data_dir = artifact_root / f"data_official_pp_12to1_lead{lead}_safeneg"
        metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
        lead_steps = metadata.get("lead_steps", [lead])
        train = load_records(data_dir / "train", lead_steps)
        val = load_records(data_dir / "val", lead_steps)
        test = load_records(data_dir / "test", lead_steps)
        print(f"Lead {lead * 6}h: train={len(train['x'])} val={len(val['x'])} test={len(test['x'])}")

        for model_name in ["ridge_direct", "ridge_cv_residual"]:
            result = fit_ridge(train, val, test, model_name)
            rows.append({"model": model_name, "lead": lead, "lead_hours": lead * 6, "seed": None, **result})
            print(f"  {model_name}: val={result['val']['err']:.3f} test={result['test']['err']:.3f} alpha={result['alpha']}")

        for seed in args.seeds:
            result = train_mlp_residual(train, val, test, seed, args, device)
            rows.append({"model": "mlp_cv_residual", "lead": lead, "lead_hours": lead * 6, **result})
            print(f"  mlp_cv_residual s{seed}: epoch={result['epoch']} val={result['val']['err']:.3f} test={result['test']['err']:.3f}")

            result = train_tiny_dual(train, val, test, seed, args, device)
            rows.append({"model": "tiny_dual_mlp", "lead": lead, "lead_hours": lead * 6, **result})
            print(f"  tiny_dual_mlp s{seed}: epoch={result['epoch']} val={result['val']['err']:.3f} test={result['test']['err']:.3f}")

    summary = {
        "protocol": "Official HURDAT2 strict-6h lead-specific safe-negative 12->1 data. Numeric-only simple residual baselines use train windows for fitting and validation windows for alpha/epoch selection.",
        "device": str(device),
        "config": vars(args),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "rows": rows,
        "aggregate": summarize_rows(rows),
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(args.output_md, summary)
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
