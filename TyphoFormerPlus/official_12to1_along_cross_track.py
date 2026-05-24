import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_official_cliper_baselines import fit_cliper_ridge
from eval_typhoformerpp import deterministic_prediction, make_model as make_plus_model
from train_sequence_baseline import make_model as make_sequence_model
from typhoformerpp_common import (
    TyphoPlusDataset,
    denormalize_disp,
    disp_to_latlon,
    haversine_km,
    latlon_to_disp,
    load_metadata,
    metadata_lead_steps,
    move_to_device,
    stats_tensors,
)


LEADS = [1, 2, 3, 4]
SEEDS = [42, 123, 2024]
BEST_BASELINE_BY_LEAD = {1: "constant_velocity", 2: "cliper_ridge", 3: "informer", 4: "informer"}


def split_track_error(pred_raw, target_raw):
    pred = pred_raw[:, -1, :]
    target = target_raw[:, -1, :]
    err = pred - target
    norm = np.linalg.norm(target, axis=1, keepdims=True)
    unit = np.divide(target, norm, out=np.zeros_like(target), where=norm > 1e-6)
    cross = np.stack([-unit[:, 1], unit[:, 0]], axis=1)
    along = np.sum(err * unit, axis=1)
    cross_track = np.sum(err * cross, axis=1)
    return {
        "mean_abs_along": float(np.mean(np.abs(along))),
        "mean_abs_cross": float(np.mean(np.abs(cross_track))),
        "mean_signed_along": float(np.mean(along)),
        "mean_signed_cross": float(np.mean(cross_track)),
    }


def fde_from_raw(pred_raw, data):
    pred_latlon = disp_to_latlon(torch.tensor(pred_raw, dtype=torch.float32), torch.tensor(data["origin"], dtype=torch.float32))
    target_latlon = torch.tensor(data["future"], dtype=torch.float32)
    return float(haversine_km(pred_latlon, target_latlon)[:, -1].mean().item())


def load_test_arrays(data_dir):
    ds = TyphoPlusDataset(str(Path(data_dir) / "test"))
    rows = [ds[idx] for idx in range(len(ds))]
    return {
        "target_raw": np.stack([row["target_raw"].numpy() for row in rows]).astype(np.float32),
        "origin": np.stack([row["origin"].numpy() for row in rows]).astype(np.float32),
        "future": np.stack([row["future_latlon"].numpy() for row in rows]).astype(np.float32),
    }


@torch.no_grad()
def constant_velocity_raw(data_dir, metadata, batch_size, device):
    ds = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    lead_steps = metadata_lead_steps(metadata, device=device, dtype=torch.float32)
    out = []
    for batch in loader:
        batch = move_to_device(batch, device)
        prev = batch["history_latlon"][:, -2]
        origin = batch["origin"]
        step_disp = latlon_to_disp(origin, prev)
        out.append((step_disp.unsqueeze(1) * lead_steps.view(1, -1, 1)).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


@torch.no_grad()
def cliper_raw(data_dir, metadata, batch_size, device):
    model = fit_cliper_ridge(str(data_dir))
    ds = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    for batch in loader:
        batch = move_to_device(batch, device)
        bsz = batch["future_latlon"].shape[0]
        x_num = batch["x_num"].detach().cpu().numpy().reshape(bsz, -1)
        hist = batch["history_latlon"].detach().cpu()
        org = batch["origin"].detach().cpu()
        hist_disp = latlon_to_disp(hist, org.unsqueeze(1)).numpy().reshape(bsz, -1)
        x = np.concatenate([x_num, hist_disp / 500.0], axis=1)
        out.append(model.predict(x).reshape(bsz, metadata["pred_len"], 2).astype(np.float32))
    return np.concatenate(out, axis=0)


@torch.no_grad()
def plus_raw(checkpoint_path, data_dir, batch_size, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = Namespace(**checkpoint["args"])
    metadata = load_metadata(str(data_dir))
    model = make_plus_model(train_args, metadata, device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    target_mean, target_std = stats_tensors(metadata, device)
    ds = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    for batch in loader:
        batch = move_to_device(batch, device)
        features = model(batch["x_num"], batch["x_text"], batch["analog_pos"], batch["analog_neg"])
        pred_norm = deterministic_prediction(features, batch, metadata, train_args, device)
        out.append(denormalize_disp(pred_norm, target_mean, target_std).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


@torch.no_grad()
def sequence_raw(checkpoint_path, data_dir, batch_size, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = Namespace(**checkpoint["args"])
    metadata = load_metadata(str(data_dir))
    model = make_sequence_model(train_args, metadata, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    target_mean, target_std = stats_tensors(metadata, device)
    ds = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    for batch in loader:
        batch = move_to_device(batch, device)
        pred_norm = model(batch["x_num"])
        out.append(denormalize_disp(pred_norm, target_mean, target_std).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def row_for(model_name, lead, pred_raw, data):
    row = {
        "model": model_name,
        "lead": lead,
        "lead_hours": lead * 6,
        "fde": fde_from_raw(pred_raw, data),
    }
    row.update(split_track_error(pred_raw, data["target_raw"]))
    return row


def aggregate_metric_rows(model_name, lead, metric_rows):
    keys = ["fde", "mean_abs_along", "mean_abs_cross", "mean_signed_along", "mean_signed_cross"]
    out = {"model": model_name, "lead": lead, "lead_hours": lead * 6, "seed_count": len(metric_rows)}
    for key in keys:
        values = np.asarray([row[key] for row in metric_rows], dtype=np.float32)
        out[key] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return out


def write_markdown(path, rows):
    lines = [
        "# Official 12->1 Along/Cross-Track Diagnostics",
        "",
        "Values are km on the test split. Along/cross components are computed in local displacement space against the observed final displacement vector.",
        "",
        "| Lead | Model | FDE | Abs along | Abs cross | Signed along | Signed cross |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['lead_hours']}h | {row['model']} | {row['fde']:.3f} | {row['mean_abs_along']:.3f} | "
            f"{row['mean_abs_cross']:.3f} | {row['mean_signed_along']:.3f} | {row['mean_signed_cross']:.3f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="../local_artifacts/TyphoFormerPlus")
    parser.add_argument("--output-json", default="official_12to1_along_cross_track.json")
    parser.add_argument("--output-md", default="official_12to1_along_cross_track.md")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.artifact_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for lead in LEADS:
        data_dir = root / f"data_official_pp_12to1_lead{lead}_safeneg"
        metadata = load_metadata(str(data_dir))
        data = load_test_arrays(data_dir)
        cv = constant_velocity_raw(data_dir, metadata, args.batch_size, device)
        cliper = cliper_raw(data_dir, metadata, args.batch_size, device)
        informer_rows = []
        b4_rows = []
        for seed in SEEDS:
            informer_pred = sequence_raw(
                root / "checkpoints_official_12to1_sequence_baselines_safeneg" / f"informer_lead{lead}_s{seed}" / "best_model.pt",
                data_dir,
                args.batch_size,
                device,
            )
            informer_rows.append(row_for("Informer", lead, informer_pred, data))
            b4_pred = plus_raw(
                root / "checkpoints_official_12to1_leadspecific_safeneg" / f"b4_plus_dual_safeneg_lead{lead}_s{seed}" / "best_model.pt",
                data_dir,
                args.batch_size,
                device,
            )
            b4_rows.append(row_for("B4 TyphoFormer++", lead, b4_pred, data))
        baseline_name = BEST_BASELINE_BY_LEAD[lead]
        if baseline_name == "informer":
            baseline_row = aggregate_metric_rows(f"Best baseline ({baseline_name})", lead, informer_rows)
        else:
            baseline_raw = {"constant_velocity": cv, "cliper_ridge": cliper}[baseline_name]
            baseline_row = aggregate_metric_rows(f"Best baseline ({baseline_name})", lead, [row_for(baseline_name, lead, baseline_raw, data)])
        rows.append(aggregate_metric_rows("B4 TyphoFormer++", lead, b4_rows))
        rows.append(baseline_row)
        print(f"Lead {lead * 6}h done.")
    summary = {
        "protocol": "Official HURDAT2 strict-6h lead-specific safe-negative 12->1 test diagnostics.",
        "rows": rows,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(args.output_md, rows)
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
