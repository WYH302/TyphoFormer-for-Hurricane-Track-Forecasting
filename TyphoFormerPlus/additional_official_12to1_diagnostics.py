import argparse
import json
import statistics
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_official_cliper_baselines import fit_cliper_ridge
from eval_typhoformerpp import deterministic_raw_prediction, make_model as make_pp_model
from train_sequence_baseline import make_model as make_sequence_model
from typhoformerpp_common import (
    TyphoPlusDataset,
    constant_velocity_future_disp,
    disp_to_latlon,
    haversine_km,
    latlon_to_disp,
    load_metadata,
    metadata_lead_steps,
    move_to_device,
)


SEEDS = [42, 123, 2024]
LEADS = [1, 2, 3, 4]
SEQUENCE_BASELINES = ["gru", "lstm", "informer", "autoformer", "tsmixer"]
BEST_MAIN_BASELINE = {1: "constant_velocity", 2: "cliper_ridge", 3: "informer", 4: "informer"}


def sample_key(storm_id, window):
    return f"{storm_id}::{int(window)}"


def batch_keys(batch):
    storms = batch["storm_id"]
    windows = batch["window"].detach().cpu().tolist()
    return [sample_key(storm, window) for storm, window in zip(storms, windows)]


def mean_std(values):
    values = list(values)
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def heading_change_deg(history_latlon):
    if history_latlon.shape[0] < 3:
        return 0.0
    hist = torch.tensor(history_latlon, dtype=torch.float32)
    v_prev = latlon_to_disp(hist[-2].view(1, 2), hist[-3].view(1, 2))[0].numpy()
    v_last = latlon_to_disp(hist[-1].view(1, 2), hist[-2].view(1, 2))[0].numpy()
    n_prev = float(np.linalg.norm(v_prev))
    n_last = float(np.linalg.norm(v_last))
    if n_prev < 1e-6 or n_last < 1e-6:
        return 0.0
    cross = float(v_prev[0] * v_last[1] - v_prev[1] * v_last[0])
    dot = float(v_prev[0] * v_last[0] + v_prev[1] * v_last[1])
    return float(abs(np.degrees(np.arctan2(cross, dot))))


def dataset_info(data_dir, split="test"):
    ds = TyphoPlusDataset(str(Path(data_dir) / split))
    info = {}
    for idx in range(len(ds)):
        item = ds[idx]
        key = sample_key(item["storm_id"], item["window"])
        info[key] = {
            "storm_id": item["storm_id"],
            "window": int(item["window"]),
            "turn_deg": heading_change_deg(item["history_latlon"].numpy()),
        }
    return info


@torch.no_grad()
def predict_cv(data_dir, batch_size, device):
    metadata = load_metadata(data_dir)
    ds = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    lead_steps = metadata_lead_steps(metadata, device=device, dtype=torch.float32)
    out = {}
    for batch in loader:
        batch = move_to_device(batch, device)
        pred_raw = constant_velocity_future_disp(batch["history_latlon"], metadata["pred_len"], lead_steps)
        pred_latlon = disp_to_latlon(pred_raw, batch["origin"])
        errors = haversine_km(pred_latlon, batch["future_latlon"])[:, -1].detach().cpu().numpy()
        for key, err in zip(batch_keys(batch), errors):
            out[key] = float(err)
    return out


@torch.no_grad()
def predict_persistence(data_dir, batch_size, device):
    ds = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = {}
    for batch in loader:
        batch = move_to_device(batch, device)
        pred_latlon = batch["origin"].unsqueeze(1).expand_as(batch["future_latlon"])
        errors = haversine_km(pred_latlon, batch["future_latlon"])[:, -1].detach().cpu().numpy()
        for key, err in zip(batch_keys(batch), errors):
            out[key] = float(err)
    return out


@torch.no_grad()
def predict_cliper(data_dir, batch_size, device):
    model = fit_cliper_ridge(data_dir)
    ds = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = {}
    for batch in loader:
        batch = move_to_device(batch, device)
        bsz, pred_len = batch["future_latlon"].shape[:2]
        x_num = batch["x_num"].detach().cpu().numpy().reshape(bsz, -1)
        hist = batch["history_latlon"].detach().cpu()
        origin = batch["origin"].detach().cpu()
        hist_disp = latlon_to_disp(hist, origin.unsqueeze(1)).numpy().reshape(bsz, -1)
        x = np.concatenate([x_num, hist_disp / 500.0], axis=1)
        pred_raw = torch.tensor(model.predict(x).reshape(bsz, pred_len, 2), device=device, dtype=torch.float32)
        pred_latlon = disp_to_latlon(pred_raw, batch["origin"])
        errors = haversine_km(pred_latlon, batch["future_latlon"])[:, -1].detach().cpu().numpy()
        for key, err in zip(batch_keys(batch), errors):
            out[key] = float(err)
    return out


@torch.no_grad()
def predict_pp(checkpoint_file, data_dir, batch_size, device):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    train_args = Namespace(**checkpoint["args"])
    metadata = load_metadata(data_dir)
    model = make_pp_model(train_args, metadata, device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    ds = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = {}
    for batch in loader:
        batch = move_to_device(batch, device)
        features = model(batch["x_num"], batch["x_text"], batch["analog_pos"], batch["analog_neg"])
        pred_raw = deterministic_raw_prediction(features, batch, metadata, train_args, device)
        pred_latlon = disp_to_latlon(pred_raw, batch["origin"])
        errors = haversine_km(pred_latlon, batch["future_latlon"])[:, -1].detach().cpu().numpy()
        for key, err in zip(batch_keys(batch), errors):
            out[key] = float(err)
    return out


@torch.no_grad()
def predict_sequence(checkpoint_file, data_dir, batch_size, device):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    train_args = Namespace(**checkpoint["args"])
    metadata = load_metadata(data_dir)
    target_mean = torch.tensor(metadata["stats"]["target_mean"], device=device, dtype=torch.float32)
    target_std = torch.tensor(metadata["stats"]["target_std"], device=device, dtype=torch.float32)
    model = make_sequence_model(train_args, metadata, device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    ds = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = {}
    for batch in loader:
        batch = move_to_device(batch, device)
        pred_norm = model(batch["x_num"])
        pred_raw = pred_norm * target_std.view(1, 1, 2) + target_mean.view(1, 1, 2)
        pred_latlon = disp_to_latlon(pred_raw, batch["origin"])
        errors = haversine_km(pred_latlon, batch["future_latlon"])[:, -1].detach().cpu().numpy()
        for key, err in zip(batch_keys(batch), errors):
            out[key] = float(err)
    return out


def average_error_maps(error_maps):
    keys = sorted(set.intersection(*(set(m.keys()) for m in error_maps)))
    return {key: float(np.mean([m[key] for m in error_maps])) for key in keys}


def pp_checkpoint(model_key, lead, seed):
    if model_key == "b2_numeric_transformer":
        return Path("checkpoints_official_12to1_leadspecific") / f"b2_numeric_official_12to1_lead{lead}_s{seed}" / "best_model.pt"
    if model_key == "b3_typhoformer":
        return Path("checkpoints_official_12to1_leadspecific") / f"b3_typhoformer_official_12to1_lead{lead}_s{seed}" / "best_model.pt"
    if model_key == "b4_typhoformerpp":
        return Path("checkpoints_official_12to1_leadspecific_safeneg") / f"b4_plus_dual_safeneg_lead{lead}_s{seed}" / "best_model.pt"
    if model_key == "numeric_cv_residual":
        return Path("checkpoints_official_12to1_numeric_dual_safeneg") / f"numeric_dual_lead{lead}_s{seed}" / "best_model.pt"
    raise KeyError(model_key)


def sequence_checkpoint(model_name, lead, seed):
    return Path("checkpoints_official_12to1_sequence_baselines_safeneg") / f"{model_name}_lead{lead}_s{seed}" / "best_model.pt"


def collect_error_maps(batch_size, device):
    by_lead = {}
    for lead in LEADS:
        data_dir = f"data_official_pp_12to1_lead{lead}_safeneg"
        by_lead[lead] = {
            "info": dataset_info(data_dir),
            "persistence": predict_persistence(data_dir, batch_size, device),
            "constant_velocity": predict_cv(data_dir, batch_size, device),
            "cliper_ridge": predict_cliper(data_dir, batch_size, device),
        }
        for model_key in ["b2_numeric_transformer", "b3_typhoformer", "b4_typhoformerpp"]:
            by_lead[lead][model_key] = average_error_maps(
                [predict_pp(pp_checkpoint(model_key, lead, seed), data_dir, batch_size, device) for seed in SEEDS]
            )
        for model_name in SEQUENCE_BASELINES:
            by_lead[lead][model_name] = average_error_maps(
                [predict_sequence(sequence_checkpoint(model_name, lead, seed), data_dir, batch_size, device) for seed in SEEDS]
            )
        numeric_dual_maps = []
        for seed in SEEDS:
            path = pp_checkpoint("numeric_cv_residual", lead, seed)
            if path.exists():
                numeric_dual_maps.append(predict_pp(path, data_dir, batch_size, device))
        if numeric_dual_maps:
            by_lead[lead]["numeric_cv_residual"] = average_error_maps(numeric_dual_maps)
            by_lead[lead]["numeric_cv_residual_seed_count"] = len(numeric_dual_maps)
    return by_lead


def mean_for_keys(error_map, keys):
    vals = [error_map[key] for key in keys if key in error_map]
    return float(np.mean(vals)) if vals else float("nan")


def common_subset_summary(by_lead):
    common_keys = set(by_lead[4]["info"].keys())
    rows = []
    candidate_names = [
        "persistence",
        "constant_velocity",
        "cliper_ridge",
        "b2_numeric_transformer",
        "b3_typhoformer",
        "gru",
        "lstm",
        "informer",
        "autoformer",
        "tsmixer",
    ]
    for lead in LEADS:
        keys = sorted(common_keys.intersection(by_lead[lead]["info"].keys()))
        candidate_means = {name: mean_for_keys(by_lead[lead][name], keys) for name in candidate_names}
        best_name = min(candidate_means, key=candidate_means.get)
        b4 = mean_for_keys(by_lead[lead]["b4_typhoformerpp"], keys)
        b3 = mean_for_keys(by_lead[lead]["b3_typhoformer"], keys)
        row = {
            "lead": lead,
            "lead_hours": lead * 6,
            "windows": len(keys),
            "storms": len({by_lead[lead]["info"][key]["storm_id"] for key in keys}),
            "b3": b3,
            "b4": b4,
            "best_baseline": best_name,
            "best_baseline_mean": candidate_means[best_name],
            "b4_gain_vs_best_percent": 100.0 * (candidate_means[best_name] - b4) / candidate_means[best_name],
        }
        if "numeric_cv_residual" in by_lead[lead]:
            row["numeric_cv_residual"] = mean_for_keys(by_lead[lead]["numeric_cv_residual"], keys)
            row["numeric_cv_residual_seed_count"] = by_lead[lead]["numeric_cv_residual_seed_count"]
        rows.append(row)
    return {
        "definition": "The common subset is the 24h-clean test window set, intersected by storm ID and window index with every shorter-lead test set.",
        "rows": rows,
    }


def turning_summary(by_lead):
    lead = 4
    info = by_lead[lead]["info"]
    keys = sorted(info.keys())
    turns = np.array([info[key]["turn_deg"] for key in keys], dtype=np.float32)
    q1, q2 = np.quantile(turns, [1.0 / 3.0, 2.0 / 3.0])
    bins = [
        ("low", -1e-6, q1),
        ("medium", q1, q2),
        ("high", q2, 181.0),
    ]
    rows = []
    for label, lo, hi in bins:
        if label == "low":
            bin_keys = [key for key in keys if info[key]["turn_deg"] <= hi]
        elif label == "high":
            bin_keys = [key for key in keys if info[key]["turn_deg"] > lo]
        else:
            bin_keys = [key for key in keys if lo < info[key]["turn_deg"] <= hi]
        cv = mean_for_keys(by_lead[lead]["constant_velocity"], bin_keys)
        b3 = mean_for_keys(by_lead[lead]["b3_typhoformer"], bin_keys)
        b4 = mean_for_keys(by_lead[lead]["b4_typhoformerpp"], bin_keys)
        rows.append(
            {
                "turn_group": label,
                "windows": len(bin_keys),
                "storms": len({info[key]["storm_id"] for key in bin_keys}),
                "mean_turn_deg": float(np.mean([info[key]["turn_deg"] for key in bin_keys])),
                "cv": cv,
                "b3": b3,
                "b4": b4,
                "b4_minus_cv": b4 - cv,
                "b4_minus_b3": b4 - b3,
            }
        )
    return {
        "definition": "24h test windows grouped by tertiles of the absolute heading change between the last two observed 6h motion vectors.",
        "tertile_cutpoints_deg": [float(q1), float(q2)],
        "rows": rows,
    }


def per_storm_distribution(by_lead, fig_dir):
    summary = {"rows": []}
    best_deltas = []
    b3_deltas = []
    labels = []
    for lead in LEADS:
        info = by_lead[lead]["info"]
        best_name = BEST_MAIN_BASELINE[lead]
        storm_values = defaultdict(lambda: {"b4": [], "best": [], "b3": []})
        for key, meta in info.items():
            if key not in by_lead[lead]["b4_typhoformerpp"]:
                continue
            storm = meta["storm_id"]
            storm_values[storm]["b4"].append(by_lead[lead]["b4_typhoformerpp"][key])
            storm_values[storm]["best"].append(by_lead[lead][best_name][key])
            storm_values[storm]["b3"].append(by_lead[lead]["b3_typhoformer"][key])
        lead_best = []
        lead_b3 = []
        for storm, vals in storm_values.items():
            b4 = float(np.mean(vals["b4"]))
            best = float(np.mean(vals["best"]))
            b3 = float(np.mean(vals["b3"]))
            lead_best.append(b4 - best)
            lead_b3.append(b4 - b3)
            summary["rows"].append(
                {
                    "lead": lead,
                    "lead_hours": lead * 6,
                    "storm_id": storm,
                    "best_baseline": best_name,
                    "b4_minus_best": b4 - best,
                    "b4_minus_b3": b4 - b3,
                }
            )
        best_deltas.append(lead_best)
        b3_deltas.append(lead_b3)
        labels.append(f"{lead * 6}h")

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    for ax, data, title in [
        (axes[0], best_deltas, "B4 - strongest non-B4 baseline"),
        (axes[1], b3_deltas, "B4 - B3"),
    ]:
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Lead")
        ax.set_ylabel("Per-storm mean DeltaR difference (km)")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    pdf_path = fig_dir / "per_storm_b4_deltas.pdf"
    png_path = fig_dir / "per_storm_b4_deltas.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    summary["figure_pdf"] = str(pdf_path)
    summary["figure_png"] = str(png_path)
    return summary


def parse_training_runtime():
    groups = {
        "GRU": [Path("checkpoints_official_12to1_sequence_baselines_safeneg") / f"gru_lead{lead}_s{seed}" for lead in LEADS for seed in SEEDS],
        "LSTM": [Path("checkpoints_official_12to1_sequence_baselines_safeneg") / f"lstm_lead{lead}_s{seed}" for lead in LEADS for seed in SEEDS],
        "Informer-style": [Path("checkpoints_official_12to1_sequence_baselines_safeneg") / f"informer_lead{lead}_s{seed}" for lead in LEADS for seed in SEEDS],
        "Autoformer-style": [Path("checkpoints_official_12to1_sequence_baselines_safeneg") / f"autoformer_lead{lead}_s{seed}" for lead in LEADS for seed in SEEDS],
        "TSMixer": [Path("checkpoints_official_12to1_sequence_baselines_safeneg") / f"tsmixer_lead{lead}_s{seed}" for lead in LEADS for seed in SEEDS],
        "B2 Numeric Transformer": [Path("checkpoints_official_12to1_leadspecific") / f"b2_numeric_official_12to1_lead{lead}_s{seed}" for lead in LEADS for seed in SEEDS],
        "B3 TyphoFormer-style": [Path("checkpoints_official_12to1_leadspecific") / f"b3_typhoformer_official_12to1_lead{lead}_s{seed}" for lead in LEADS for seed in SEEDS],
        "B4 TyphoFormer++": [Path("checkpoints_official_12to1_leadspecific_safeneg") / f"b4_plus_dual_safeneg_lead{lead}_s{seed}" for lead in LEADS for seed in SEEDS],
        "Numeric Transformer + CV-residual": [Path("checkpoints_official_12to1_numeric_dual_safeneg") / f"numeric_dual_lead{lead}_s{seed}" for lead in LEADS for seed in SEEDS],
    }
    rows = []
    for label, run_dirs in groups.items():
        minutes = []
        epochs = []
        available = 0
        for run_dir in run_dirs:
            log_path = run_dir / "train_log.jsonl"
            if not log_path.exists():
                continue
            available += 1
            rows_json = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if rows_json:
                epochs.append(len(rows_json))
                minutes.append(sum(float(row.get("seconds", 0.0)) for row in rows_json) / 60.0)
        if minutes:
            rows.append(
                {
                    "model": label,
                    "runs_available": available,
                    "runs_expected": len(run_dirs),
                    "median_epochs": float(statistics.median(epochs)),
                    "median_train_minutes": float(statistics.median(minutes)),
                }
            )
    return {"definition": "Training time is parsed from per-epoch train_log.jsonl seconds on the local workstation.", "rows": rows}


def time_b4_inference(batch_size, device):
    timings = []
    for lead in LEADS:
        data_dir = f"data_official_pp_12to1_lead{lead}_safeneg"
        path = pp_checkpoint("b4_typhoformerpp", lead, 42)
        start = time.perf_counter()
        errors = predict_pp(path, data_dir, batch_size, device)
        elapsed = time.perf_counter() - start
        timings.append(
            {
                "lead": lead,
                "lead_hours": lead * 6,
                "windows": len(errors),
                "seconds": elapsed,
                "windows_per_second": len(errors) / elapsed if elapsed > 0 else float("nan"),
            }
        )
    return timings


def write_markdown(summary, output_md):
    lines = ["# Additional Official 12->1 Diagnostics", ""]
    lines.append("## Common 24h-Clean Test Subset")
    lines.append("")
    lines.append("| Lead | W/S | Best non-B4 | Best mean | B3 | B4 | B4 gain vs best |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|")
    for row in summary["common_subset"]["rows"]:
        lines.append(
            f"| {row['lead_hours']}h | {row['windows']}/{row['storms']} | {row['best_baseline']} | "
            f"{row['best_baseline_mean']:.3f} | {row['b3']:.3f} | {row['b4']:.3f} | {row['b4_gain_vs_best_percent']:.1f}% |"
        )
    lines.append("")
    lines.append("## 24h Turning Subgroups")
    lines.append("")
    lines.append("| Turn group | W/S | Mean turn deg | CV | B3 | B4 | B4-CV | B4-B3 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["turning_subgroups"]["rows"]:
        lines.append(
            f"| {row['turn_group']} | {row['windows']}/{row['storms']} | {row['mean_turn_deg']:.1f} | "
            f"{row['cv']:.3f} | {row['b3']:.3f} | {row['b4']:.3f} | {row['b4_minus_cv']:.3f} | {row['b4_minus_b3']:.3f} |"
        )
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append("| Model | Runs | Median epochs | Median train min/run |")
    lines.append("|---|---:|---:|---:|")
    for row in summary["runtime"]["rows"]:
        lines.append(
            f"| {row['model']} | {row['runs_available']}/{row['runs_expected']} | "
            f"{row['median_epochs']:.0f} | {row['median_train_minutes']:.1f} |"
        )
    lines.append("")
    lines.append("## B4 Inference Timing")
    lines.append("")
    lines.append("| Lead | Windows | Seconds | Windows/s |")
    lines.append("|---:|---:|---:|---:|")
    for row in summary["b4_inference_timing"]:
        lines.append(
            f"| {row['lead_hours']}h | {row['windows']} | {row['seconds']:.3f} | {row['windows_per_second']:.1f} |"
        )
    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default="official_12to1_additional_diagnostics.json")
    parser.add_argument("--output-md", default="official_12to1_additional_diagnostics.md")
    parser.add_argument("--fig-dir", default="diagnostic_figs")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_lead = collect_error_maps(args.batch_size, device)
    summary = {
        "protocol": "Official HURDAT2 strict-6h lead-specific clean 12->1 diagnostics on 2022-2024 test storms.",
        "common_subset": common_subset_summary(by_lead),
        "turning_subgroups": turning_summary(by_lead),
        "per_storm_distribution": per_storm_distribution(by_lead, args.fig_dir),
        "runtime": parse_training_runtime(),
        "b4_inference_timing": time_b4_inference(args.batch_size, device),
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, args.output_md)
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
