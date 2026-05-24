import argparse
import json
import os
from statistics import mean, stdev


SEEDS = [42, 123, 2024]
LEADS = ["6", "12", "18", "24"]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values):
    return {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0}


def aggregate(per_seed, prefix):
    out = {}
    for lead in LEADS:
        key = f"{prefix}{lead}"
        out[key] = mean_std([per_seed[str(seed)][key] for seed in SEEDS])
    return out


def collect_model(base, prefix, calibrated=False, eval_name="eval_test.json"):
    raw = {}
    cal = {}
    calibration = {}
    for seed in SEEDS:
        path = os.path.join(base, f"{prefix}{seed}", eval_name)
        data = read_json(path)
        if calibrated:
            raw[str(seed)] = data["model_raw"]
            cal[str(seed)] = data["model_cv_calibrated"]
            calibration[str(seed)] = data.get("calibration")
        else:
            raw[str(seed)] = data["model"]
    result = {"per_seed": raw, "mae": aggregate(raw, "mae"), "dr": aggregate(raw, "err")}
    if calibrated:
        result["calibrated_per_seed"] = cal
        result["calibrated_mae"] = aggregate(cal, "mae")
        result["calibrated_dr"] = aggregate(cal, "err")
        result["calibration"] = calibration
    return result


def fmt(ms):
    return f"{ms['mean']:.3f} +/- {ms['std']:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints_official_12to4")
    parser.add_argument("--baselines-json", default="data_official_pp_12to4/official_baselines_test.json")
    parser.add_argument("--baselines-2024-json", default="data_official_pp_12to4/official_baselines_test_2024.json")
    parser.add_argument("--output-json", default="official_12to4_summary.json")
    args = parser.parse_args()

    baselines = read_json(args.baselines_json)
    baselines_2024 = read_json(args.baselines_2024_json) if os.path.exists(args.baselines_2024_json) else None
    b2 = collect_model(args.checkpoints_dir, "b2_numeric_official_12to4_s")
    b3 = collect_model(args.checkpoints_dir, "b3_typhoformer_official_12to4_s")
    b4 = collect_model(
        args.checkpoints_dir,
        "b4_plus_dual_official_12to4_s",
        calibrated=True,
        eval_name="eval_test_calibrated.json",
    )
    b2_2024 = collect_model(
        args.checkpoints_dir,
        "b2_numeric_official_12to4_s",
        eval_name="eval_test_2024.json",
    )
    b3_2024 = collect_model(
        args.checkpoints_dir,
        "b3_typhoformer_official_12to4_s",
        eval_name="eval_test_2024.json",
    )
    b4_2024 = collect_model(
        args.checkpoints_dir,
        "b4_plus_dual_official_12to4_s",
        calibrated=True,
        eval_name="eval_test_2024_calibrated.json",
    )
    summary = {
        "protocol": "Official NHC Atlantic HURDAT2, strict 6h, storm-year split, 12 observed records -> 4 future records",
        "seeds": SEEDS,
        "baselines": baselines,
        "baselines_2024": baselines_2024,
        "models": {
            "b2_numeric_transformer": b2,
            "b3_typhoformer_leakfree": b3,
            "b4_typhoformerpp_raw": {"per_seed": b4["per_seed"], "mae": b4["mae"], "dr": b4["dr"]},
            "b4_typhoformerpp_val_cv_calibrated": {
                "per_seed": b4["calibrated_per_seed"],
                "mae": b4["calibrated_mae"],
                "dr": b4["calibrated_dr"],
                "calibration": b4["calibration"],
            },
        },
        "models_2024_subset": {
            "b2_numeric_transformer": b2_2024,
            "b3_typhoformer_leakfree": b3_2024,
            "b4_typhoformerpp_raw": {
                "per_seed": b4_2024["per_seed"],
                "mae": b4_2024["mae"],
                "dr": b4_2024["dr"],
            },
            "b4_typhoformerpp_val_cv_calibrated": {
                "per_seed": b4_2024["calibrated_per_seed"],
                "mae": b4_2024["calibrated_mae"],
                "dr": b4_2024["calibrated_dr"],
                "calibration": b4_2024["calibration"],
            },
        },
        "official_table1_reference": {
            "typhoformer_delta_r_all": {"6h": 31.539, "12h": 38.084, "18h": 42.435, "24h": 49.562},
            "typhoformer_delta_r_2024": {"6h": 31.274, "12h": 37.934, "18h": 42.973, "24h": 50.881},
            "interpretation": "TyphoFormer++ beats our leak-free B3 reproduction and local baselines, but it does not beat the official Table 1 TyphoFormer values beyond 6h.",
        },
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("MAE table")
    print("Model | 6h | 12h | 18h | 24h")
    print("---|---:|---:|---:|---:")
    for name in ["persistence", "constant_velocity", "cliper_ridge"]:
        print(name + " | " + " | ".join(f"{baselines[name][f'mae{lead}']:.3f}" for lead in LEADS))
    for name, data in [
        ("B2 Numeric Transformer", b2),
        ("B3 TyphoFormer leak-free", b3),
        ("B4 TyphoFormer++ raw", summary["models"]["b4_typhoformerpp_raw"]),
        ("B4 TyphoFormer++ + val CV calibration", summary["models"]["b4_typhoformerpp_val_cv_calibrated"]),
    ]:
        print(name + " | " + " | ".join(fmt(data["mae"][f"mae{lead}"]) for lead in LEADS))

    print("\nDelta R table")
    print("Model | 6h | 12h | 18h | 24h")
    print("---|---:|---:|---:|---:")
    for name in ["persistence", "constant_velocity", "cliper_ridge"]:
        print(name + " | " + " | ".join(f"{baselines[name][f'err{lead}']:.2f}" for lead in LEADS))
    for name, data in [
        ("B2 Numeric Transformer", b2),
        ("B3 TyphoFormer leak-free", b3),
        ("B4 TyphoFormer++ raw", summary["models"]["b4_typhoformerpp_raw"]),
        ("B4 TyphoFormer++ + val CV calibration", summary["models"]["b4_typhoformerpp_val_cv_calibrated"]),
    ]:
        print(name + " | " + " | ".join(fmt(data["dr"][f"err{lead}"]) for lead in LEADS))
    print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
