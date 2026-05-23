import argparse
import json
import statistics
from pathlib import Path


SEEDS = [42, 123, 2024]
LEADS = [1, 2, 3, 4]


def mean_std(values):
    values = list(values)
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def load_metric(eval_path, lead):
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    metrics = data["model"] if "model" in data else data.get("model_raw", data)
    return {
        "err": metrics[f"err{lead * 6}"],
        "mae": metrics.get(f"mae{lead * 6}", metrics.get("mae")),
        "ade": metrics.get("ade"),
        "fde": metrics.get("fde"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints_official_12to1_numeric_dual_safeneg")
    parser.add_argument("--output-json", default="official_12to1_numeric_dual_safeneg_summary.json")
    parser.add_argument("--output-md", default="official_12to1_numeric_dual_safeneg_summary.md")
    args = parser.parse_args()

    base = Path(args.checkpoints_dir)
    rows = []
    aggregate = {}
    for lead in LEADS:
        lead_rows = []
        for seed in SEEDS:
            run_dir = base / f"numeric_dual_lead{lead}_s{seed}"
            eval_path = run_dir / "eval_test.json"
            if not eval_path.exists():
                continue
            metric = load_metric(eval_path, lead)
            row = {
                "lead": lead,
                "lead_hours": lead * 6,
                "seed": seed,
                "run_dir": str(run_dir),
                **metric,
            }
            rows.append(row)
            lead_rows.append(row)
        if lead_rows:
            aggregate[str(lead)] = {
                "lead_hours": lead * 6,
                "seed_count": len(lead_rows),
                "err": mean_std(row["err"] for row in lead_rows),
                "mae": mean_std(row["mae"] for row in lead_rows),
            }

    summary = {
        "protocol": "Official HURDAT2 strict-6h lead-specific clean 12->1 numeric-only Transformer with direct/CV-residual dual head.",
        "seeds": SEEDS,
        "rows": rows,
        "aggregate": aggregate,
        "notes": "Same TyphoFormer++ backbone and dual head as the core model, but variant=numeric disables text and analog inputs.",
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Official 12->1 Numeric Transformer + CV-Residual Diagnostic",
        "",
        "Values are DeltaR km. Three-seed means are reported only when all seeds are available.",
        "",
        "| Variant | 6h | 12h | 18h | 24h |",
        "|---|---:|---:|---:|---:|",
    ]
    cells = []
    for lead in LEADS:
        stat = aggregate.get(str(lead))
        if not stat:
            cells.append("--")
        elif stat["seed_count"] == len(SEEDS):
            cells.append(f"{stat['err']['mean']:.3f} +/- {stat['err']['std']:.3f}")
        else:
            cells.append(f"{stat['err']['mean']:.3f} ({stat['seed_count']} seed)")
    lines.append("| Numeric Transformer + CV-residual head | " + " | ".join(cells) + " |")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
