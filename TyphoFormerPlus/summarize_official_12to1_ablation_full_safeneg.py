import argparse
import json
import re
import statistics
from pathlib import Path


VARIANTS = [
    ("b3_leakfree_eval", "B3 leak-free TyphoFormer"),
    ("b3_dual_head", "B3 + dual CV-residual head"),
    ("b3_positive_analog", "B3 + positive analog only"),
    ("b3_alignment_rank", "B3 + alignment/rank only"),
    ("b4_safe_existing", "B4 safe TyphoFormer++"),
]
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
    hour = lead * 6
    return {
        "err": metrics[f"err{hour}"],
        "mae": metrics.get(f"mae{hour}", metrics.get("mae")),
        "ade": metrics.get("ade"),
        "fde": metrics.get("fde"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints_official_12to1_ablation_full_safeneg")
    parser.add_argument("--output-json", default="official_12to1_ablation_full_safeneg_summary.json")
    parser.add_argument("--output-md", default="official_12to1_ablation_full_safeneg_summary.md")
    args = parser.parse_args()

    base = Path(args.checkpoints_dir)
    rows = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        match = re.search(r"_lead(\d+)_s(\d+)$", run_dir.name)
        if not match:
            continue
        lead = int(match.group(1))
        seed = int(match.group(2))
        variant = None
        label = None
        for key, name in VARIANTS:
            if run_dir.name.startswith(key):
                variant = key
                label = name
                break
        if variant is None:
            continue
        eval_path = run_dir / "eval_test.json"
        if not eval_path.exists():
            continue
        metric = load_metric(eval_path, lead)
        rows.append({"variant": variant, "label": label, "lead": lead, "lead_hours": lead * 6, "seed": seed, **metric, "run_dir": str(run_dir)})

    variants = {}
    for key, label in VARIANTS:
        aggregate = {}
        per_lead = {}
        for lead in LEADS:
            lead_rows = [r for r in rows if r["variant"] == key and r["lead"] == lead]
            per_lead[str(lead)] = {str(r["seed"]): r for r in lead_rows}
            aggregate[str(lead)] = {
                "lead_hours": lead * 6,
                "err": mean_std([r["err"] for r in lead_rows]) if lead_rows else None,
                "mae": mean_std([r["mae"] for r in lead_rows]) if lead_rows else None,
            }
        variants[key] = {"label": label, "per_lead": per_lead, "aggregate": aggregate}

    summary = {
        "protocol": "Official HURDAT2 strict-6h lead-specific clean 12->1 full ablation on safe-negative data.",
        "seeds": SEEDS,
        "leads": [lead * 6 for lead in LEADS],
        "variants": variants,
        "rows": rows,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Official 12->1 Full Safe-Negative Ablation",
        "",
        "Values are DeltaR km, mean +/- std over seeds 42/123/2024.",
        "",
        "| Variant | 6h | 12h | 18h | 24h |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in VARIANTS:
        cells = []
        for lead in LEADS:
            stat = variants[key]["aggregate"][str(lead)]["err"]
            cells.append("-" if stat is None else f"{stat['mean']:.3f} +/- {stat['std']:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
