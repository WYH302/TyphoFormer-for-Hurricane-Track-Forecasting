import argparse
import json
import statistics
from pathlib import Path


LEADS = [1, 2, 3, 4]
SEEDS = [42, 123, 2024]


def mean_std(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def load_metric(path, lead):
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("model_raw", data.get("model", data))
    return metrics[f"err{lead * 6}"]


def load_existing_row(label, values):
    row = {"label": label, "aggregate": {}}
    for lead in LEADS:
        value = values[str(lead)]
        if isinstance(value, dict) and "mean" in value:
            row["aggregate"][str(lead)] = {"lead_hours": lead * 6, "err": value}
        elif isinstance(value, (int, float)):
            row["aggregate"][str(lead)] = {"lead_hours": lead * 6, "err": {"mean": value, "std": 0.0}}
        else:
            row["aggregate"][str(lead)] = {"lead_hours": lead * 6, "err": value}
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints_official_12to1_text_ablation_safeneg")
    parser.add_argument("--lead-summary-json", default="official_leadspecific_12to1_safeneg_summary.json")
    parser.add_argument("--output-json", default="official_12to1_text_ablation_safeneg_summary.json")
    parser.add_argument("--output-md", default="official_12to1_text_ablation_safeneg_summary.md")
    args = parser.parse_args()

    lead_summary = json.loads(Path(args.lead_summary_json).read_text(encoding="utf-8"))
    rows = {}
    for key, label in [
        ("b2_numeric_transformer", "B2/B3 without text-template"),
        ("b3_typhoformer_leakfree", "B3 with hash text-template"),
        ("b4_typhoformerpp_safeneg_raw", "B4 full with hash text-template"),
        ("b4_typhoformerpp_safeneg_val_cv_calibrated", "B4 full + val CV calibration"),
    ]:
        values = {}
        for lead in LEADS:
            values[str(lead)] = lead_summary["models"][key]["aggregate"][str(lead)]["err"]
        rows[key] = load_existing_row(label, values)

    base = Path(args.checkpoints_dir)
    per_lead = {}
    aggregate = {}
    for lead in LEADS:
        values = []
        seed_rows = {}
        for seed in SEEDS:
            eval_path = base / f"b4_plus_dual_no_text_lead{lead}_s{seed}" / "eval_test.json"
            if eval_path.exists():
                err = load_metric(eval_path, lead)
                values.append(err)
                seed_rows[str(seed)] = {"err": err, "eval_path": str(eval_path)}
        per_lead[str(lead)] = seed_rows
        aggregate[str(lead)] = {
            "lead_hours": lead * 6,
            "err": mean_std(values) if values else None,
        }
    rows["b4_no_text"] = {
        "label": "B4 without text-template",
        "per_lead": per_lead,
        "aggregate": aggregate,
    }

    summary = {
        "protocol": "Official HURDAT2 strict-6h lead-specific clean 12->1 text-template ablation on safe-negative data.",
        "seeds": SEEDS,
        "rows": rows,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Official 12->1 Safe-Negative Text-Template Ablation",
        "",
        "Values are DeltaR km. Neural rows are mean +/- std over seeds 42/123/2024.",
        "",
        "| Row | 6h | 12h | 18h | 24h |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in [
        "b2_numeric_transformer",
        "b3_typhoformer_leakfree",
        "b4_no_text",
        "b4_typhoformerpp_safeneg_raw",
        "b4_typhoformerpp_safeneg_val_cv_calibrated",
    ]:
        row = rows[key]
        cells = []
        for lead in LEADS:
            stat = row["aggregate"][str(lead)]["err"]
            cells.append("-" if stat is None else f"{stat['mean']:.3f} +/- {stat['std']:.3f}")
        lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
