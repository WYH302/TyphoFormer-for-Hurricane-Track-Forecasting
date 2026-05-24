import argparse
import json
import statistics
from pathlib import Path


SEEDS = [42, 123, 2024]
LEADS = [1, 2, 3, 4]


def mean_std(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def eval_metrics(path: Path, lead: int, calibrated=False):
    data = json.loads(path.read_text(encoding="utf-8"))
    if "model_raw" in data:
        metrics = data["model_cv_calibrated" if calibrated else "model_raw"]
    else:
        metrics = data
    hour = lead * 6
    return {
        "err": metrics[f"err{hour}"],
        "mae": metrics.get(f"mae{hour}", metrics.get("mae")),
        "ade": metrics.get("ade"),
        "fde": metrics.get("fde"),
    }


def aggregate_per_lead(per_lead):
    out = {}
    for lead in LEADS:
        lead_key = str(lead)
        seed_rows = per_lead[lead_key]
        out[lead_key] = {"lead_hours": lead * 6}
        for metric in ["err", "mae", "ade", "fde"]:
            vals = [seed_rows[str(seed)][metric] for seed in SEEDS]
            out[lead_key][metric] = mean_std(vals)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-summary", default="official_leadspecific_12to1_summary.json")
    parser.add_argument("--safeneg-dir", default="checkpoints_official_12to1_leadspecific_safeneg")
    parser.add_argument("--output-json", default="official_leadspecific_12to1_safeneg_summary.json")
    parser.add_argument("--output-md", default="official_leadspecific_12to1_safeneg_summary.md")
    args = parser.parse_args()

    old = json.loads(Path(args.old_summary).read_text(encoding="utf-8"))
    safe_dir = Path(args.safeneg_dir)

    models = {
        "b2_numeric_transformer": old["models"]["b2_numeric_transformer"],
        "b3_typhoformer_leakfree": old["models"]["b3_typhoformer_leakfree"],
    }

    for name, calibrated in [
        ("b4_typhoformerpp_safeneg_raw", False),
        ("b4_typhoformerpp_safeneg_val_cv_calibrated", True),
    ]:
        per_lead = {str(lead): {} for lead in LEADS}
        for lead in LEADS:
            for seed in SEEDS:
                eval_path = (
                    safe_dir
                    / f"b4_plus_dual_safeneg_lead{lead}_s{seed}"
                    / "eval_test_calibrated.json"
                )
                per_lead[str(lead)][str(seed)] = eval_metrics(eval_path, lead, calibrated=calibrated)
        models[name] = {"per_lead": per_lead, "aggregate": aggregate_per_lead(per_lead)}

    summary = {
        "protocol": "Official NHC Atlantic HURDAT2, lead-specific clean 12->1, strict 6h. B4 uses safe validation/test negatives and does not fuse negative analogs as inference context.",
        "seeds": SEEDS,
        "data_audit": old["data_audit"],
        "baselines": old["baselines"],
        "baselines_2024": old["baselines_2024"],
        "models": models,
        "official_table1_reference": old.get("official_table1_reference", {}),
        "notes": {
            "b2_b3": "B2 and B3 do not consume analog negatives; their old checkpoints are unchanged because the safe-negative regeneration preserves inputs and targets.",
            "b4": "B4 was rerun/evaluated on safe-negative data with target-free validation/test negative analogs and no negative analog fusion.",
        },
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    labels = [
        ("persistence", "Persistence", "baselines"),
        ("constant_velocity", "Constant Velocity", "baselines"),
        ("cliper_ridge", "CLIPER-style ridge", "baselines"),
        ("b2_numeric_transformer", "B2 Numeric Transformer", "models"),
        ("b3_typhoformer_leakfree", "B3 leak-free TyphoFormer", "models"),
        ("b4_typhoformerpp_safeneg_raw", "B4 TyphoFormer++ safe raw", "models"),
        (
            "b4_typhoformerpp_safeneg_val_cv_calibrated",
            "B4 TyphoFormer++ safe + val CV calibration",
            "models",
        ),
    ]
    lines = [
        "# Official Lead-Specific 12->1 Safe-Negative Summary",
        "",
        "| Model | 6h | 12h | 18h | 24h |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label, group in labels:
        cells = []
        if group == "baselines":
            source = summary[group][key]
            for lead in LEADS:
                cells.append(f"{source[str(lead)]['err']:.3f}")
        else:
            source = summary[group][key]["aggregate"]
            for lead in LEADS:
                stat = source[str(lead)]["err"]
                cells.append(f"{stat['mean']:.3f} +/- {stat['std']:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
