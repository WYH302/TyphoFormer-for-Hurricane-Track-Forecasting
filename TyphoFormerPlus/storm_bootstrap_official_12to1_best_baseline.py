import argparse
import json
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_official_cliper_baselines import fit_cliper_ridge
from eval_typhoformerpp import deterministic_raw_prediction, make_model as make_pp_model
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
BEST_BASELINE = {
    1: "constant_velocity",
    2: "cliper_ridge",
    3: "informer",
    4: "informer",
}


def bootstrap_ci(values, rng, draws):
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    boot = values[indices].mean(axis=1)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def finalize_storm_sums(sums, counts):
    return {
        storm_id: {"mean": sums[storm_id] / counts[storm_id], "count": counts[storm_id]}
        for storm_id in sorted(sums)
    }


def add_storm_errors(sums, counts, storm_ids, errors):
    for storm_id, err in zip(storm_ids, errors):
        sums[str(storm_id)] += float(err)
        counts[str(storm_id)] += 1


@torch.no_grad()
def per_storm_pp(checkpoint_file, data_dir, batch_size, device):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    train_args = Namespace(**checkpoint["args"])
    metadata = load_metadata(data_dir)
    dataset = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = make_pp_model(train_args, metadata, device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    sums = defaultdict(float)
    counts = defaultdict(int)
    for batch in loader:
        storm_ids = list(batch["storm_id"])
        batch = move_to_device(batch, device)
        features = model(batch["x_num"], batch["x_text"], batch["analog_pos"], batch["analog_neg"])
        pred_raw = deterministic_raw_prediction(features, batch, metadata, train_args, device)
        pred_latlon = disp_to_latlon(pred_raw, batch["origin"])
        errors = haversine_km(pred_latlon, batch["future_latlon"])[:, -1].detach().cpu().numpy()
        add_storm_errors(sums, counts, storm_ids, errors)
    return finalize_storm_sums(sums, counts)


@torch.no_grad()
def per_storm_constant_velocity(data_dir, batch_size, device):
    metadata = load_metadata(data_dir)
    lead_steps = metadata_lead_steps(metadata, device=device, dtype=torch.float32)
    dataset = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    sums = defaultdict(float)
    counts = defaultdict(int)
    for batch in loader:
        storm_ids = list(batch["storm_id"])
        batch = move_to_device(batch, device)
        origin = batch["origin"]
        prev = batch["history_latlon"][:, -2]
        step_disp = latlon_to_disp(origin, prev)
        pred_latlon = disp_to_latlon(step_disp.unsqueeze(1) * lead_steps.view(1, -1, 1), origin)
        errors = haversine_km(pred_latlon, batch["future_latlon"])[:, -1].detach().cpu().numpy()
        add_storm_errors(sums, counts, storm_ids, errors)
    return finalize_storm_sums(sums, counts)


@torch.no_grad()
def per_storm_cliper_ridge(data_dir, batch_size, device):
    model = fit_cliper_ridge(data_dir)
    dataset = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    sums = defaultdict(float)
    counts = defaultdict(int)
    for batch in loader:
        storm_ids = list(batch["storm_id"])
        batch = move_to_device(batch, device)
        bsz = batch["future_latlon"].shape[0]
        x_num = batch["x_num"].detach().cpu().numpy().reshape(bsz, -1)
        hist = batch["history_latlon"].detach().cpu()
        origin_cpu = batch["origin"].detach().cpu()
        hist_disp = latlon_to_disp(hist, origin_cpu.unsqueeze(1)).numpy().reshape(bsz, -1)
        features = np.concatenate([x_num, hist_disp / 500.0], axis=1)
        pred_raw = torch.tensor(model.predict(features).reshape(bsz, -1, 2), device=device, dtype=torch.float32)
        pred_latlon = disp_to_latlon(pred_raw, batch["origin"])
        errors = haversine_km(pred_latlon, batch["future_latlon"])[:, -1].detach().cpu().numpy()
        add_storm_errors(sums, counts, storm_ids, errors)
    return finalize_storm_sums(sums, counts)


@torch.no_grad()
def per_storm_sequence(checkpoint_file, data_dir, batch_size, device):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    train_args = Namespace(**checkpoint["args"])
    metadata = load_metadata(data_dir)
    target_mean, target_std = stats_tensors(metadata, device)
    dataset = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = make_sequence_model(train_args, metadata, device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    sums = defaultdict(float)
    counts = defaultdict(int)
    for batch in loader:
        storm_ids = list(batch["storm_id"])
        batch = move_to_device(batch, device)
        pred_norm = model(batch["x_num"])
        pred_raw = denormalize_disp(pred_norm, target_mean, target_std)
        pred_latlon = disp_to_latlon(pred_raw, batch["origin"])
        errors = haversine_km(pred_latlon, batch["future_latlon"])[:, -1].detach().cpu().numpy()
        add_storm_errors(sums, counts, storm_ids, errors)
    return finalize_storm_sums(sums, counts)


def average_by_storm(per_seed):
    by_storm = defaultdict(list)
    for storm_errors in per_seed:
        for storm_id, item in storm_errors.items():
            by_storm[storm_id].append(item["mean"])
    return {
        storm_id: {"mean": float(np.mean(values)), "count": len(values)}
        for storm_id, values in sorted(by_storm.items())
    }


def pp_checkpoint(lead, seed):
    return Path("checkpoints_official_12to1_leadspecific_safeneg") / f"b4_plus_dual_safeneg_lead{lead}_s{seed}" / "best_model.pt"


def sequence_checkpoint(model_name, lead, seed):
    return Path("checkpoints_official_12to1_sequence_baselines_safeneg") / f"{model_name}_lead{lead}_s{seed}" / "best_model.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--output-json", default="official_12to1_storm_bootstrap_best_baseline_summary.json")
    parser.add_argument("--output-md", default="official_12to1_storm_bootstrap_best_baseline_summary.md")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_lead = {}
    for lead in LEADS:
        data_dir = f"data_official_pp_12to1_lead{lead}_safeneg"
        b4_avg = average_by_storm([per_storm_pp(pp_checkpoint(lead, seed), data_dir, args.batch_size, device) for seed in SEEDS])

        baseline_name = BEST_BASELINE[lead]
        if baseline_name == "constant_velocity":
            baseline_avg = per_storm_constant_velocity(data_dir, args.batch_size, device)
        elif baseline_name == "cliper_ridge":
            baseline_avg = per_storm_cliper_ridge(data_dir, args.batch_size, device)
        else:
            baseline_avg = average_by_storm(
                [per_storm_sequence(sequence_checkpoint(baseline_name, lead, seed), data_dir, args.batch_size, device) for seed in SEEDS]
            )

        storm_ids = sorted(set(b4_avg) & set(baseline_avg))
        b4_values = np.array([b4_avg[s]["mean"] for s in storm_ids], dtype=np.float64)
        base_values = np.array([baseline_avg[s]["mean"] for s in storm_ids], dtype=np.float64)
        diffs = b4_values - base_values
        per_lead[str(lead)] = {
            "lead_hours": lead * 6,
            "storm_count": len(storm_ids),
            "baseline": baseline_name,
            "baseline_storm_mean": float(base_values.mean()),
            "b4_storm_mean": float(b4_values.mean()),
            "paired_delta_b4_minus_baseline": float(diffs.mean()),
            "paired_delta_ci95": bootstrap_ci(diffs, rng, args.bootstrap_draws),
            "storm_win_rate_b4": float(np.mean(diffs < 0.0)),
        }

    summary = {
        "protocol": "Storm-level paired bootstrap for B4 safe TyphoFormer++ raw vs strongest non-proposed Table 2 baseline per lead.",
        "best_baseline_by_lead": BEST_BASELINE,
        "seeds": SEEDS,
        "bootstrap_draws": args.bootstrap_draws,
        "per_lead": per_lead,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Official 12->1 Storm-Level Bootstrap vs Best Non-Proposed Baseline",
        "",
        "B4 - best baseline paired DeltaR in km. Negative means B4 is better. B4 and neural baselines average seeds first, then bootstrap test storms.",
        "",
        "| Lead | Storms | Best baseline | Baseline storm mean | B4 storm mean | B4-baseline delta | 95% CI | B4 better storms |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for lead in LEADS:
        row = per_lead[str(lead)]
        ci = row["paired_delta_ci95"]
        lines.append(
            f"| {row['lead_hours']}h | {row['storm_count']} | {row['baseline']} | "
            f"{row['baseline_storm_mean']:.3f} | {row['b4_storm_mean']:.3f} | "
            f"{row['paired_delta_b4_minus_baseline']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{100.0 * row['storm_win_rate_b4']:.1f}% |"
        )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
