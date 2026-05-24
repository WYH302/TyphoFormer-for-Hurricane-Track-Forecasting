import argparse
import json
import os
from statistics import mean, stdev


SEEDS = [42, 123, 2024]
LEAD_KEYS = ["err6", "err12", "err18", "err24"]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values):
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": mean(values), "std": stdev(values)}


def collect_runs(base_dir, run_prefix, result_key="model"):
    per_seed = {}
    for seed in SEEDS:
        path = os.path.join(base_dir, f"{run_prefix}{seed}", "eval_test.json")
        data = read_json(path)
        per_seed[str(seed)] = data[result_key]
    aggregate = {
        key: mean_std([per_seed[str(seed)][key] for seed in SEEDS])
        for key in LEAD_KEYS
    }
    aggregate.update(
        {
            key: mean_std([per_seed[str(seed)][key] for seed in SEEDS])
            for key in ["ade", "fde"]
            if key in per_seed[str(SEEDS[0])]
        }
    )
    return {"per_seed": per_seed, "aggregate": aggregate}


def collect_calibrated(base_dir, run_prefix):
    raw = {}
    calibrated = {}
    calibration = {}
    for seed in SEEDS:
        path = os.path.join(base_dir, f"{run_prefix}{seed}", "eval_test_calibrated.json")
        data = read_json(path)
        raw[str(seed)] = data["model_raw"]
        calibrated[str(seed)] = data["model_cv_calibrated"]
        calibration[str(seed)] = data["calibration"]
    return {
        "raw_per_seed": raw,
        "calibrated_per_seed": calibrated,
        "raw_aggregate": {
            key: mean_std([raw[str(seed)][key] for seed in SEEDS])
            for key in LEAD_KEYS + ["ade", "fde"]
        },
        "calibrated_aggregate": {
            key: mean_std([calibrated[str(seed)][key] for seed in SEEDS])
            for key in LEAD_KEYS + ["ade", "fde"]
        },
        "calibration": calibration,
    }


def fmt(ms):
    return f"{ms['mean']:.2f} +/- {ms['std']:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints_pp_6h_12to4")
    parser.add_argument("--output-json", default="strict6h_12to4_summary.json")
    args = parser.parse_args()

    b2 = collect_runs(args.checkpoints_dir, "b2_numeric_direct_6h_12to4_256_e120_s")
    b3 = collect_runs(args.checkpoints_dir, "b3_typhoformer_leakfree_direct_6h_12to4_256_e120_s")
    b4 = collect_calibrated(args.checkpoints_dir, "b4_plus_dual_aux_fde_6h_12to4_256_e160_s")

    baseline_path = os.path.join(
        args.checkpoints_dir,
        "b2_numeric_direct_6h_12to4_256_e120_s42",
        "eval_test.json",
    )
    baseline_data = read_json(baseline_path)
    summary = {
        "protocol": "strict 6h, input 12 steps (72h), output 4 steps (24h), leak-free splits",
        "seeds": SEEDS,
        "baselines": {
            "persistence": baseline_data["persistence"],
            "constant_velocity": baseline_data["constant_velocity"],
        },
        "models": {
            "b2_numeric_transformer": b2,
            "b3_typhoformer_leakfree": b3,
            "b4_typhoformerpp_raw": {
                "per_seed": b4["raw_per_seed"],
                "aggregate": b4["raw_aggregate"],
            },
            "b4_typhoformerpp_val_cv_calibrated": {
                "per_seed": b4["calibrated_per_seed"],
                "aggregate": b4["calibrated_aggregate"],
                "calibration": b4["calibration"],
            },
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Model | 6h | 12h | 18h | 24h")
    print("---|---:|---:|---:|---:")
    print(
        "Persistence | "
        + " | ".join(f"{summary['baselines']['persistence'][key]:.2f}" for key in LEAD_KEYS)
    )
    print(
        "Constant Velocity | "
        + " | ".join(f"{summary['baselines']['constant_velocity'][key]:.2f}" for key in LEAD_KEYS)
    )
    rows = [
        ("B2 Numeric Transformer", b2["aggregate"]),
        ("B3 TyphoFormer leak-free", b3["aggregate"]),
        ("B4 TyphoFormer++ raw", b4["raw_aggregate"]),
        ("B4 TyphoFormer++ + val CV calibration", b4["calibrated_aggregate"]),
    ]
    for name, aggregate in rows:
        print(name + " | " + " | ".join(fmt(aggregate[key]) for key in LEAD_KEYS))
    print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
