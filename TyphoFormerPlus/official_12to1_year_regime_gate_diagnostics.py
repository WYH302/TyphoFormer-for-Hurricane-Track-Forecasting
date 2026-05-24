import argparse
import json
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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
    metadata_lead_hours,
    metadata_lead_steps,
    move_to_device,
    stats_tensors,
)


LEADS = [1, 2, 3, 4]
SEEDS = [42, 123, 2024]
YEARS = [2022, 2023, 2024]
BEST_BASELINE_BY_LEAD = {
    1: "constant_velocity",
    2: "cliper_ridge",
    3: "informer",
    4: "informer",
}


def safe_float(value):
    return float(value) if np.isfinite(value) else None


def final_error(pred_latlon, target_latlon):
    return haversine_km(pred_latlon, target_latlon)[:, -1].detach().cpu().numpy()


def feature_index(metadata, name):
    return metadata["feature_names"].index(name)


def denormalize_last_feature(batch, metadata, name):
    idx = feature_index(metadata, name)
    mean = float(metadata["stats"]["x_mean"][idx])
    std = float(metadata["stats"]["x_std"][idx])
    return batch["x_num"][:, -1, idx].detach().cpu().numpy() * std + mean


def turn_degrees(history_latlon):
    hist = history_latlon.detach().cpu().numpy()
    out = []
    for row in hist:
        if row.shape[0] < 3:
            out.append(0.0)
            continue
        v0 = row[-2] - row[-3]
        v1 = row[-1] - row[-2]
        n0 = float(np.linalg.norm(v0))
        n1 = float(np.linalg.norm(v1))
        if n0 < 1e-6 or n1 < 1e-6:
            out.append(0.0)
            continue
        cosang = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
        out.append(float(np.degrees(np.arccos(cosang))))
    return np.asarray(out, dtype=np.float32)


def dataset_records(data_dir, metadata, batch_size):
    ds = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    records = []
    turn_values = []
    speed_values = []
    for batch in loader:
        max_wind = denormalize_last_feature(batch, metadata, "max_wind")
        speed6 = denormalize_last_feature(batch, metadata, "speed6")
        lat = batch["history_latlon"][:, -1, 0].detach().cpu().numpy()
        turns = turn_degrees(batch["history_latlon"])
        years = batch["future_year"][:, 0].detach().cpu().numpy().astype(int)
        windows = batch["window"].detach().cpu().numpy().astype(int)
        storm_ids = list(batch["storm_id"])
        for idx, storm_id in enumerate(storm_ids):
            records.append(
                {
                    "storm_id": str(storm_id),
                    "window": int(windows[idx]),
                    "year": int(years[idx]),
                    "max_wind": float(max_wind[idx]),
                    "speed6": float(speed6[idx]),
                    "lat": float(lat[idx]),
                    "turn_deg": float(turns[idx]),
                }
            )
        turn_values.extend(turns.tolist())
        speed_values.extend(speed6.tolist())
    turn_cuts = np.percentile(np.asarray(turn_values), [33.333, 66.667])
    speed_cuts = np.percentile(np.asarray(speed_values), [33.333, 66.667])
    for record in records:
        wind = record["max_wind"]
        if wind < 34:
            record["intensity_group"] = "weak_lt34kt"
        elif wind < 64:
            record["intensity_group"] = "tropical_storm_34_63kt"
        else:
            record["intensity_group"] = "hurricane_ge64kt"
        alat = abs(record["lat"])
        if alat < 20:
            record["latitude_group"] = "low_lat_lt20"
        elif alat < 30:
            record["latitude_group"] = "mid_lat_20_30"
        else:
            record["latitude_group"] = "high_lat_ge30"
        if record["speed6"] <= turn_cuts[0] * 0 + speed_cuts[0]:
            record["speed_group"] = "slow"
        elif record["speed6"] <= speed_cuts[1]:
            record["speed_group"] = "medium"
        else:
            record["speed_group"] = "fast"
        if record["turn_deg"] <= turn_cuts[0]:
            record["turn_group"] = "low"
        elif record["turn_deg"] <= turn_cuts[1]:
            record["turn_group"] = "medium"
        else:
            record["turn_group"] = "high"
        record["high_lat_recurvature"] = "yes" if alat >= 30 and record["turn_group"] == "high" else "no"
    return records, {"turn_tertiles_deg": [float(x) for x in turn_cuts], "speed_tertiles_kmh": [float(x) for x in speed_cuts]}


@torch.no_grad()
def constant_velocity_errors(data_dir, metadata, batch_size, device):
    ds = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    lead_steps = metadata_lead_steps(metadata, device=device, dtype=torch.float32)
    values = []
    for batch in loader:
        batch = move_to_device(batch, device)
        prev = batch["history_latlon"][:, -2]
        origin = batch["origin"]
        step_disp = latlon_to_disp(origin, prev)
        pred = disp_to_latlon(step_disp.unsqueeze(1) * lead_steps.view(1, -1, 1), origin)
        values.extend(final_error(pred, batch["future_latlon"]).tolist())
    return np.asarray(values, dtype=np.float32)


@torch.no_grad()
def cliper_errors(data_dir, metadata, batch_size, device):
    model = fit_cliper_ridge(str(data_dir))
    ds = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    values = []
    for batch in loader:
        batch = move_to_device(batch, device)
        bsz = batch["future_latlon"].shape[0]
        x_num = batch["x_num"].detach().cpu().numpy().reshape(bsz, -1)
        hist = batch["history_latlon"].detach().cpu()
        org = batch["origin"].detach().cpu()
        hist_disp = latlon_to_disp(hist, org.unsqueeze(1)).numpy().reshape(bsz, -1)
        x = np.concatenate([x_num, hist_disp / 500.0], axis=1)
        pred_raw = torch.tensor(model.predict(x).reshape(bsz, metadata["pred_len"], 2), device=device, dtype=torch.float32)
        pred = disp_to_latlon(pred_raw, batch["origin"])
        values.extend(final_error(pred, batch["future_latlon"]).tolist())
    return np.asarray(values, dtype=np.float32)


@torch.no_grad()
def plus_errors(checkpoint_path, data_dir, batch_size, device, collect_gates=False):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = Namespace(**checkpoint["args"])
    metadata = load_metadata(str(data_dir))
    model = make_plus_model(train_args, metadata, device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    target_mean, target_std = stats_tensors(metadata, device)
    ds = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    errors = []
    dual_gates = []
    residual_gates = []
    for batch in loader:
        batch = move_to_device(batch, device)
        features = model(batch["x_num"], batch["x_text"], batch["analog_pos"], batch["analog_neg"])
        pred_norm = deterministic_prediction(features, batch, metadata, train_args, device)
        pred_raw = denormalize_disp(pred_norm, target_mean, target_std)
        pred = disp_to_latlon(pred_raw, batch["origin"])
        errors.extend(final_error(pred, batch["future_latlon"]).tolist())
        if collect_gates:
            dual_gates.extend(features["dual_gate"][:, -1, 0].detach().cpu().numpy().tolist())
            residual_gates.extend(features["residual_gate"][:, -1, 0].detach().cpu().numpy().tolist())
    result = {"errors": np.asarray(errors, dtype=np.float32)}
    if collect_gates:
        result["dual_gate"] = np.asarray(dual_gates, dtype=np.float32)
        result["residual_gate"] = np.asarray(residual_gates, dtype=np.float32)
    return result


@torch.no_grad()
def sequence_errors(checkpoint_path, data_dir, batch_size, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = Namespace(**checkpoint["args"])
    metadata = load_metadata(str(data_dir))
    model = make_sequence_model(train_args, metadata, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    target_mean, target_std = stats_tensors(metadata, device)
    ds = TyphoPlusDataset(str(Path(data_dir) / "test"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    values = []
    for batch in loader:
        batch = move_to_device(batch, device)
        pred_norm = model(batch["x_num"])
        pred_raw = denormalize_disp(pred_norm, target_mean, target_std)
        pred = disp_to_latlon(pred_raw, batch["origin"])
        values.extend(final_error(pred, batch["future_latlon"]).tolist())
    return np.asarray(values, dtype=np.float32)


def seed_average(arrays):
    return np.stack(arrays, axis=0).mean(axis=0)


def group_stats(records, errors, key):
    grouped = defaultdict(list)
    storms = defaultdict(set)
    for record, err in zip(records, errors):
        group = record[key]
        grouped[group].append(float(err))
        storms[group].add(record["storm_id"])
    out = {}
    for group, vals in grouped.items():
        out[group] = {
            "windows": len(vals),
            "storms": len(storms[group]),
            "mean": safe_float(np.mean(vals)),
        }
    return dict(sorted(out.items()))


def per_storm_delta(records, model_errors, baseline_errors):
    grouped_model = defaultdict(list)
    grouped_base = defaultdict(list)
    for record, m_err, b_err in zip(records, model_errors, baseline_errors):
        grouped_model[record["storm_id"]].append(float(m_err))
        grouped_base[record["storm_id"]].append(float(b_err))
    rows = []
    for storm_id in grouped_model:
        model_mean = float(np.mean(grouped_model[storm_id]))
        base_mean = float(np.mean(grouped_base[storm_id]))
        rows.append(
            {
                "storm_id": storm_id,
                "windows": len(grouped_model[storm_id]),
                "b4": model_mean,
                "best_baseline": base_mean,
                "b4_minus_best": model_mean - base_mean,
            }
        )
    rows.sort(key=lambda row: row["b4_minus_best"], reverse=True)
    return {"b4_worse_top5": rows[:5], "b4_better_top5": list(reversed(rows[-5:]))}


def gate_stats(records, gates, group_key):
    grouped = defaultdict(list)
    for record, value in zip(records, gates):
        grouped[record[group_key]].append(float(value))
    out = {}
    for group, vals in grouped.items():
        arr = np.asarray(vals, dtype=np.float32)
        out[group] = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }
    return dict(sorted(out.items()))


def load_hurdat_max_year(path):
    if not Path(path).exists():
        return None
    try:
        df = pd.read_csv(path)
        date_col = [col for col in df.columns if col.strip() == "date"][0]
        return int(pd.to_numeric(df[date_col], errors="coerce").dropna().astype(int).max() // 10000)
    except Exception:
        return None


def write_markdown(path, summary):
    lines = [
        "# Official 12->1 Year, Regime, and Gate Diagnostics",
        "",
        f"Local HURDAT material max year: {summary['temporal_extension']['local_hurdat_max_year']}. 2025 locked evaluation available: {summary['temporal_extension']['has_2025_records']}.",
        "",
        "## Yearly B4 vs Best Baseline",
        "",
        "| Lead | Year | Windows | Storms | Best baseline | Best mean | B4 mean | B4-best |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in summary["yearly_rows"]:
        lines.append(
            f"| {row['lead_hours']}h | {row['year']} | {row['windows']} | {row['storms']} | {row['best_baseline']} | "
            f"{row['best_baseline_mean']:.3f} | {row['b4_mean']:.3f} | {row['b4_minus_best']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 24h Regime Snapshot",
            "",
            "| Grouping | Group | Windows | Storms | Best | B4 | No-text | Numeric CV-residual |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lead4 = summary["regime_rows"].get("4", {})
    for grouping, group_rows in lead4.items():
        for group, stats in group_rows.items():
            lines.append(
                f"| {grouping} | {group} | {stats['windows']} | {stats['storms']} | {stats['best_baseline']:.3f} | "
                f"{stats['b4']:.3f} | {stats['b4_no_text']:.3f} | {stats['numeric_cv_residual']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## B4 Dual-Gate Means by Turn Group",
            "",
            "| Lead | Turn group | Count | Mean | P10 | P90 |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for lead, rows in summary["gate_rows"].items():
        for group, stats in rows["dual_gate_by_turn"].items():
            lines.append(
                f"| {int(lead) * 6}h | {group} | {stats['count']} | {stats['mean']:.3f} | {stats['p10']:.3f} | {stats['p90']:.3f} |"
            )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="../local_artifacts/TyphoFormerPlus")
    parser.add_argument("--hurdat-csv", default="HURDAT_2new_3000.csv")
    parser.add_argument("--output-json", default="official_12to1_year_regime_gate_diagnostics.json")
    parser.add_argument("--output-md", default="official_12to1_year_regime_gate_diagnostics.md")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.artifact_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = {
        "protocol": "Official HURDAT2 strict-6h lead-specific safe-negative 12->1 test diagnostics.",
        "device": str(device),
        "temporal_extension": {
            "local_hurdat_max_year": load_hurdat_max_year(args.hurdat_csv),
            "has_2025_records": load_hurdat_max_year(args.hurdat_csv) is not None and load_hurdat_max_year(args.hurdat_csv) >= 2025,
        },
        "yearly_rows": [],
        "regime_rows": {},
        "gate_rows": {},
        "failure_cases_24h": {},
    }

    for lead in LEADS:
        data_dir = root / f"data_official_pp_12to1_lead{lead}_safeneg"
        metadata = load_metadata(str(data_dir))
        records, cutpoints = dataset_records(data_dir, metadata, args.batch_size)
        print(f"Lead {lead * 6}h records={len(records)} cutpoints={cutpoints}")

        cv = constant_velocity_errors(data_dir, metadata, args.batch_size, device)
        cliper = cliper_errors(data_dir, metadata, args.batch_size, device)

        b4_arrays = []
        b4_dual_gates = []
        b4_residual_gates = []
        no_text_arrays = []
        numeric_arrays = []
        informer_arrays = []
        for seed in SEEDS:
            b4_path = root / "checkpoints_official_12to1_leadspecific_safeneg" / f"b4_plus_dual_safeneg_lead{lead}_s{seed}" / "best_model.pt"
            b4_result = plus_errors(b4_path, data_dir, args.batch_size, device, collect_gates=True)
            b4_arrays.append(b4_result["errors"])
            b4_dual_gates.append(b4_result["dual_gate"])
            b4_residual_gates.append(b4_result["residual_gate"])

            no_text_path = root / "checkpoints_official_12to1_text_ablation_safeneg" / f"b4_plus_dual_no_text_lead{lead}_s{seed}" / "best_model.pt"
            if no_text_path.exists():
                no_text_arrays.append(plus_errors(no_text_path, data_dir, args.batch_size, device)["errors"])

            numeric_path = root / "checkpoints_official_12to1_numeric_dual_safeneg" / f"numeric_dual_lead{lead}_s{seed}" / "best_model.pt"
            if numeric_path.exists():
                numeric_arrays.append(plus_errors(numeric_path, data_dir, args.batch_size, device)["errors"])

            informer_path = root / "checkpoints_official_12to1_sequence_baselines_safeneg" / f"informer_lead{lead}_s{seed}" / "best_model.pt"
            if informer_path.exists():
                informer_arrays.append(sequence_errors(informer_path, data_dir, args.batch_size, device))

        b4 = seed_average(b4_arrays)
        b4_no_text = seed_average(no_text_arrays) if no_text_arrays else np.full_like(b4, np.nan)
        numeric = seed_average(numeric_arrays) if numeric_arrays else np.full_like(b4, np.nan)
        informer = seed_average(informer_arrays) if informer_arrays else np.full_like(b4, np.nan)
        baselines = {
            "constant_velocity": cv,
            "cliper_ridge": cliper,
            "informer": informer,
        }
        best_name = BEST_BASELINE_BY_LEAD[lead]
        best = baselines[best_name]

        for year in YEARS:
            mask = np.asarray([record["year"] == year for record in records], dtype=bool)
            if not mask.any():
                continue
            storms = {record["storm_id"] for record, keep in zip(records, mask) if keep}
            summary["yearly_rows"].append(
                {
                    "lead": lead,
                    "lead_hours": lead * 6,
                    "year": year,
                    "windows": int(mask.sum()),
                    "storms": len(storms),
                    "best_baseline": best_name,
                    "best_baseline_mean": float(np.mean(best[mask])),
                    "b4_mean": float(np.mean(b4[mask])),
                    "b4_minus_best": float(np.mean(b4[mask]) - np.mean(best[mask])),
                }
            )

        regime = {}
        for key in ["intensity_group", "latitude_group", "speed_group", "turn_group", "high_lat_recurvature"]:
            best_stats = group_stats(records, best, key)
            b4_stats = group_stats(records, b4, key)
            no_text_stats = group_stats(records, b4_no_text, key)
            numeric_stats = group_stats(records, numeric, key)
            regime[key] = {}
            for group in b4_stats:
                regime[key][group] = {
                    "windows": b4_stats[group]["windows"],
                    "storms": b4_stats[group]["storms"],
                    "best_baseline": best_stats[group]["mean"],
                    "b4": b4_stats[group]["mean"],
                    "b4_minus_best": b4_stats[group]["mean"] - best_stats[group]["mean"],
                    "b4_no_text": no_text_stats[group]["mean"],
                    "full_minus_no_text": b4_stats[group]["mean"] - no_text_stats[group]["mean"],
                    "numeric_cv_residual": numeric_stats[group]["mean"],
                    "full_minus_numeric_cv_residual": b4_stats[group]["mean"] - numeric_stats[group]["mean"],
                }
        summary["regime_rows"][str(lead)] = regime

        dual_gate = np.concatenate(b4_dual_gates)
        residual_gate = np.concatenate(b4_residual_gates)
        repeated_records = records * len(SEEDS)
        summary["gate_rows"][str(lead)] = {
            "dual_gate_by_turn": gate_stats(repeated_records, dual_gate, "turn_group"),
            "residual_gate_by_turn": gate_stats(repeated_records, residual_gate, "turn_group"),
            "dual_gate_overall": {
                "count": int(dual_gate.size),
                "mean": float(dual_gate.mean()),
                "p10": float(np.percentile(dual_gate, 10)),
                "p90": float(np.percentile(dual_gate, 90)),
            },
            "residual_gate_overall": {
                "count": int(residual_gate.size),
                "mean": float(residual_gate.mean()),
                "p10": float(np.percentile(residual_gate, 10)),
                "p90": float(np.percentile(residual_gate, 90)),
            },
            "cutpoints": cutpoints,
        }

        if lead == 4:
            summary["failure_cases_24h"] = per_storm_delta(records, b4, best)

    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(args.output_md, summary)
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
