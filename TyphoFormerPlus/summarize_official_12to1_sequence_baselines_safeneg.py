import argparse
import json
import re
import statistics
from pathlib import Path


MODELS = ["gru", "lstm", "informer", "autoformer", "tsmixer"]
LEADS = [1, 2, 3, 4]
SEEDS = [42, 123, 2024]


def mean_std(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints_official_12to1_sequence_baselines_safeneg")
    parser.add_argument("--output-json", default="official_12to1_sequence_baselines_safeneg_summary.json")
    parser.add_argument("--output-md", default="official_12to1_sequence_baselines_safeneg_summary.md")
    args = parser.parse_args()

    base = Path(args.checkpoints_dir)
    rows = []
    for eval_path in sorted(base.glob("*/eval_test.json")):
        match = re.match(r"([a-z]+)_lead(\d+)_s(\d+)$", eval_path.parent.name)
        if not match:
            continue
        model, lead_text, seed_text = match.groups()
        lead = int(lead_text)
        seed = int(seed_text)
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        metrics = data.get("model", data)
        hour = lead * 6
        rows.append(
            {
                "model": model,
                "lead": lead,
                "lead_hours": hour,
                "seed": seed,
                "err": metrics[f"err{hour}"],
                "mae": metrics.get(f"mae{hour}", metrics.get("mae")),
                "ade": metrics["ade"],
                "fde": metrics["fde"],
                "run_dir": str(eval_path.parent),
            }
        )

    summary = {"protocol": "Official HURDAT2 strict-6h lead-specific clean 12->1 sequence baselines on safe-negative data.", "rows": rows, "models": {}}
    for model in MODELS:
        per_lead = {}
        for lead in LEADS:
            lead_rows = [r for r in rows if r["model"] == model and r["lead"] == lead]
            per_seed = {str(r["seed"]): r for r in lead_rows}
            per_lead[str(lead)] = {
                "lead_hours": lead * 6,
                "per_seed": per_seed,
                "err": mean_std([r["err"] for r in lead_rows]) if lead_rows else None,
                "mae": mean_std([r["mae"] for r in lead_rows]) if lead_rows else None,
            }
        summary["models"][model] = per_lead

    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Official 12->1 Sequence Baselines Safe-Negative Summary",
        "",
        "| Model | 6h | 12h | 18h | 24h |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "gru": "GRU",
        "lstm": "LSTM",
        "informer": "Informer-style",
        "autoformer": "Autoformer-style",
        "tsmixer": "TSMixer",
    }
    for model in MODELS:
        cells = []
        for lead in LEADS:
            stat = summary["models"][model][str(lead)]["err"]
            cells.append("-" if stat is None else f"{stat['mean']:.3f} +/- {stat['std']:.3f}")
        lines.append(f"| {labels[model]} | " + " | ".join(cells) + " |")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
