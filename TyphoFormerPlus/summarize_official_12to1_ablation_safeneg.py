import argparse
import json
import re
from pathlib import Path


VARIANTS = [
    ("b3_leakfree_eval", "B3 leak-free TyphoFormer"),
    ("b3_dual_head_only", "B3 + dual CV-residual head"),
    ("b3_positive_analog_only", "B3 + positive analog only"),
    ("b3_alignment_rank_only", "B3 + alignment/rank only"),
    ("b3_analog_rank_align", "B3 + analog + alignment/rank"),
    ("b4_full", "B4 full TyphoFormer++"),
]


def load_metric(path: Path, lead: int):
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("model_raw", data.get("model", data))
    key = f"err{lead * 6}"
    return {
        "delta_r_km": metrics[key],
        "mae": metrics.get(f"mae{lead * 6}", metrics.get("mae")),
        "ade": metrics.get("ade"),
        "fde": metrics.get("fde"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints_official_12to1_ablation_safeneg")
    parser.add_argument("--output-json", default="official_12to1_ablation_safeneg_summary.json")
    parser.add_argument("--output-md", default="official_12to1_ablation_safeneg_summary.md")
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
        variant_key = None
        variant_label = None
        for key, label in VARIANTS:
            if run_dir.name.startswith(key):
                variant_key = key
                variant_label = label
                break
        if variant_key is None:
            continue
        eval_path = run_dir / "eval_test.json"
        if not eval_path.exists():
            continue
        metric = load_metric(eval_path, lead)
        rows.append(
            {
                "variant": variant_key,
                "label": variant_label,
                "lead": lead,
                "lead_hours": lead * 6,
                "seed": seed,
                **metric,
                "run_dir": str(run_dir),
            }
        )

    by_variant = {}
    for key, label in VARIANTS:
        entry = {"label": label, "lead12": None, "lead24": None, "runs": []}
        for row in rows:
            if row["variant"] != key:
                continue
            entry["runs"].append(row)
            if row["lead"] == 2:
                entry["lead12"] = row["delta_r_km"]
            elif row["lead"] == 4:
                entry["lead24"] = row["delta_r_km"]
        by_variant[key] = entry

    summary = {
        "protocol": "Official HURDAT2 strict-6h lead-specific clean 12->1 attribution ablation, safe target-free validation/test negatives, seed 42.",
        "leads": [12, 24],
        "seed": 42,
        "variants": by_variant,
        "rows": rows,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Official 12->1 Safe-Negative Ablation",
        "",
        "Protocol: official HURDAT2 strict-6h lead-specific clean 12->1, seed 42.",
        "",
        "| Variant | 12h DeltaR km | 24h DeltaR km |",
        "|---|---:|---:|",
    ]
    for key, label in VARIANTS:
        entry = by_variant[key]
        lead12 = entry["lead12"]
        lead24 = entry["lead24"]
        lead12_text = f"{lead12:.3f}" if lead12 is not None else "-"
        lead24_text = f"{lead24:.3f}" if lead24 is not None else "-"
        lines.append(f"| {label} | {lead12_text} | {lead24_text} |")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
