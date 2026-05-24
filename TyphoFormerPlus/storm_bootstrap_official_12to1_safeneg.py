import argparse
import json
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_typhoformerpp import deterministic_raw_prediction, make_model
from typhoformerpp_common import TyphoPlusDataset, disp_to_latlon, haversine_km, load_metadata, move_to_device, seed_everything


LEADS = [1, 2, 3, 4]
SEEDS = [42, 123, 2024]


def checkpoint_path(model_name, lead, seed):
    if model_name == "b3":
        return Path("checkpoints_official_12to1_leadspecific") / f"b3_typhoformer_official_12to1_lead{lead}_s{seed}" / "best_model.pt"
    if model_name == "b4":
        return Path("checkpoints_official_12to1_leadspecific_safeneg") / f"b4_plus_dual_safeneg_lead{lead}_s{seed}" / "best_model.pt"
    raise ValueError(f"Unknown model {model_name}")


@torch.no_grad()
def per_storm_errors(checkpoint_file, data_dir, batch_size, device):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    train_args = Namespace(**checkpoint["args"])
    metadata = load_metadata(data_dir)
    dataset = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = make_model(train_args, metadata, device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    sums = defaultdict(float)
    counts = defaultdict(int)
    all_errors = []
    for batch in loader:
        storm_ids = list(batch["storm_id"])
        batch = move_to_device(batch, device)
        features = model(batch["x_num"], batch["x_text"], batch["analog_pos"], batch["analog_neg"])
        pred_raw = deterministic_raw_prediction(features, batch, metadata, train_args, device)
        pred_latlon = disp_to_latlon(pred_raw, batch["origin"])
        errors = haversine_km(pred_latlon, batch["future_latlon"])[:, -1].detach().cpu().numpy()
        all_errors.extend(float(x) for x in errors)
        for storm_id, err in zip(storm_ids, errors):
            sums[str(storm_id)] += float(err)
            counts[str(storm_id)] += 1
    return {
        storm_id: {"mean": sums[storm_id] / counts[storm_id], "count": counts[storm_id]}
        for storm_id in sorted(sums)
    }, float(np.mean(all_errors))


def bootstrap_ci(values, rng, draws):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [None, None]
    indices = rng.integers(0, values.size, size=(draws, values.size))
    boot = values[indices].mean(axis=1)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--output-json", default="official_12to1_storm_bootstrap_safeneg_summary.json")
    parser.add_argument("--output-md", default="official_12to1_storm_bootstrap_safeneg_summary.md")
    args = parser.parse_args()

    seed_everything(42)
    rng = np.random.default_rng(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_lead = {}
    for lead in LEADS:
        data_dir = f"data_official_pp_12to1_lead{lead}_safeneg"
        per_seed = {}
        by_storm = defaultdict(lambda: {"b3": [], "b4": [], "count": 0})
        for seed in SEEDS:
            b3_storm, b3_window = per_storm_errors(checkpoint_path("b3", lead, seed), data_dir, args.batch_size, device)
            b4_storm, b4_window = per_storm_errors(checkpoint_path("b4", lead, seed), data_dir, args.batch_size, device)
            storm_ids = sorted(set(b3_storm) & set(b4_storm))
            diffs = np.array([b4_storm[s]["mean"] - b3_storm[s]["mean"] for s in storm_ids], dtype=np.float64)
            per_seed[str(seed)] = {
                "storm_count": len(storm_ids),
                "b3_window_mean": b3_window,
                "b4_window_mean": b4_window,
                "b3_storm_mean": float(np.mean([b3_storm[s]["mean"] for s in storm_ids])),
                "b4_storm_mean": float(np.mean([b4_storm[s]["mean"] for s in storm_ids])),
                "paired_delta_b4_minus_b3": float(np.mean(diffs)),
                "paired_delta_ci95": bootstrap_ci(diffs, rng, args.bootstrap_draws),
                "storm_win_rate_b4": float(np.mean(diffs < 0.0)),
            }
            for storm_id in storm_ids:
                by_storm[storm_id]["b3"].append(b3_storm[storm_id]["mean"])
                by_storm[storm_id]["b4"].append(b4_storm[storm_id]["mean"])
                by_storm[storm_id]["count"] = b3_storm[storm_id]["count"]

        storm_ids = sorted(by_storm)
        b3_avg = np.array([np.mean(by_storm[s]["b3"]) for s in storm_ids], dtype=np.float64)
        b4_avg = np.array([np.mean(by_storm[s]["b4"]) for s in storm_ids], dtype=np.float64)
        diffs = b4_avg - b3_avg
        per_lead[str(lead)] = {
            "lead_hours": lead * 6,
            "storm_count": len(storm_ids),
            "b3_storm_mean": float(b3_avg.mean()),
            "b4_storm_mean": float(b4_avg.mean()),
            "paired_delta_b4_minus_b3": float(diffs.mean()),
            "paired_delta_ci95": bootstrap_ci(diffs, rng, args.bootstrap_draws),
            "storm_win_rate_b4": float(np.mean(diffs < 0.0)),
            "per_seed": per_seed,
        }

    summary = {
        "protocol": "Storm-level paired bootstrap for B4 safe TyphoFormer++ raw vs B3 leak-free TyphoFormer on official strict-6h lead-specific 12->1 test storms.",
        "seeds": SEEDS,
        "bootstrap_draws": args.bootstrap_draws,
        "per_lead": per_lead,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Official 12->1 Storm-Level Bootstrap Safe-Negative Summary",
        "",
        "B4 - B3 paired DeltaR in km. Negative means B4 is better. Storm means average seeds first, then bootstrap test storms.",
        "",
        "| Lead | Storms | B3 storm mean | B4 storm mean | B4-B3 delta | 95% CI | B4 better storms |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lead in LEADS:
        row = per_lead[str(lead)]
        ci = row["paired_delta_ci95"]
        lines.append(
            f"| {row['lead_hours']}h | {row['storm_count']} | {row['b3_storm_mean']:.3f} | "
            f"{row['b4_storm_mean']:.3f} | {row['paired_delta_b4_minus_b3']:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {100.0 * row['storm_win_rate_b4']:.1f}% |"
        )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
