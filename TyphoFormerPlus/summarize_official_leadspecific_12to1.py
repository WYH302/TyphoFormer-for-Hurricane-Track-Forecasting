import argparse
import json
import os
from statistics import mean, stdev


SEEDS = [42, 123, 2024]
LEADS = [1, 2, 3, 4]
LEAD_HOURS = {lead: lead * 6 for lead in LEADS}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values):
    return {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0}


def metric_key(prefix, lead):
    return f"{prefix}{LEAD_HOURS[lead]}"


def collect_single_model(checkpoints_dir, run_prefix, result_file, result_key):
    per_lead = {}
    aggregate = {}
    for lead in LEADS:
        per_seed = {}
        for seed in SEEDS:
            path = os.path.join(checkpoints_dir, f"{run_prefix}{lead}_s{seed}", result_file)
            data = read_json(path)[result_key]
            per_seed[str(seed)] = data
        err_key = metric_key("err", lead)
        mae_key = metric_key("mae", lead)
        per_lead[str(lead)] = per_seed
        aggregate[str(lead)] = {
            "lead_hours": LEAD_HOURS[lead],
            "err": mean_std([per_seed[str(seed)][err_key] for seed in SEEDS]),
            "mae": mean_std([per_seed[str(seed)][mae_key] for seed in SEEDS]),
            "ade": mean_std([per_seed[str(seed)]["ade"] for seed in SEEDS]),
            "fde": mean_std([per_seed[str(seed)]["fde"] for seed in SEEDS]),
        }
    return {"per_lead": per_lead, "aggregate": aggregate}


def collect_b4(checkpoints_dir, subset_2024=False):
    suffix = "eval_test_2024_calibrated.json" if subset_2024 else "eval_test_calibrated.json"
    raw = collect_single_model(checkpoints_dir, "b4_plus_dual_official_12to1_lead", suffix, "model_raw")
    calibrated = collect_single_model(checkpoints_dir, "b4_plus_dual_official_12to1_lead", suffix, "model_cv_calibrated")
    calibration = {}
    for lead in LEADS:
        calibration[str(lead)] = {}
        for seed in SEEDS:
            path = os.path.join(checkpoints_dir, f"b4_plus_dual_official_12to1_lead{lead}_s{seed}", suffix)
            calibration[str(lead)][str(seed)] = read_json(path)["calibration"]
    calibrated["calibration"] = calibration
    return raw, calibrated


def collect_baselines(subset_2024=False):
    filename = "official_baselines_test_2024.json" if subset_2024 else "official_baselines_test.json"
    out = {"persistence": {}, "constant_velocity": {}, "cliper_ridge": {}}
    for lead in LEADS:
        data = read_json(os.path.join(f"data_official_pp_12to1_lead{lead}", filename))
        for name in out:
            err_key = metric_key("err", lead)
            mae_key = metric_key("mae", lead)
            out[name][str(lead)] = {
                "lead_hours": LEAD_HOURS[lead],
                "err": data[name][err_key],
                "mae": data[name][mae_key],
                "ade": data[name]["ade"],
                "fde": data[name]["fde"],
            }
    return out


def collect_data_audit():
    audit = {}
    for lead in LEADS:
        meta = read_json(os.path.join(f"data_official_pp_12to1_lead{lead}", "metadata.json"))
        audit[str(lead)] = {
            "lead_hours": meta["lead_hours"],
            "target_lead_step": meta["target_lead_step"],
            "splits": meta["splits"],
            "data_audit": meta["data_audit"],
            "source_audit": meta["source_audit"],
            "stats_fit_split": meta["stats_fit_split"],
            "analog_candidate_split": meta["analog_candidate_split"],
        }
    return audit


def fmt(ms):
    return f"{ms['mean']:.3f} +/- {ms['std']:.3f}"


def print_delta_table(title, baselines, models):
    print(title)
    print("Model | 6h | 12h | 18h | 24h")
    print("---|---:|---:|---:|---:")
    for name, label in [
        ("persistence", "Persistence"),
        ("constant_velocity", "Constant Velocity"),
        ("cliper_ridge", "CLIPER-style ridge"),
    ]:
        row = [f"{baselines[name][str(lead)]['err']:.2f}" for lead in LEADS]
        print(label + " | " + " | ".join(row))
    for label, data in models:
        row = [fmt(data["aggregate"][str(lead)]["err"]) for lead in LEADS]
        print(label + " | " + " | ".join(row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints_official_12to1_leadspecific")
    parser.add_argument("--output-json", default="official_leadspecific_12to1_summary.json")
    args = parser.parse_args()

    b2 = collect_single_model(args.checkpoints_dir, "b2_numeric_official_12to1_lead", "eval_test.json", "model")
    b3 = collect_single_model(args.checkpoints_dir, "b3_typhoformer_official_12to1_lead", "eval_test.json", "model")
    b4_raw, b4_cal = collect_b4(args.checkpoints_dir, subset_2024=False)
    b2_2024 = collect_single_model(args.checkpoints_dir, "b2_numeric_official_12to1_lead", "eval_test_2024.json", "model")
    b3_2024 = collect_single_model(args.checkpoints_dir, "b3_typhoformer_official_12to1_lead", "eval_test_2024.json", "model")
    b4_raw_2024, b4_cal_2024 = collect_b4(args.checkpoints_dir, subset_2024=True)

    summary = {
        "protocol": "Official NHC Atlantic HURDAT2, lead-specific clean 12->1 tasks, strict 6h, train-period storms 2004-2021, test storms 2022-2024",
        "seeds": SEEDS,
        "data_audit": collect_data_audit(),
        "baselines": collect_baselines(subset_2024=False),
        "baselines_2024": collect_baselines(subset_2024=True),
        "models": {
            "b2_numeric_transformer": b2,
            "b3_typhoformer_leakfree": b3,
            "b4_typhoformerpp_raw": b4_raw,
            "b4_typhoformerpp_val_cv_calibrated": b4_cal,
        },
        "models_2024_subset": {
            "b2_numeric_transformer": b2_2024,
            "b3_typhoformer_leakfree": b3_2024,
            "b4_typhoformerpp_raw": b4_raw_2024,
            "b4_typhoformerpp_val_cv_calibrated": b4_cal_2024,
        },
        "official_table1_reference": {
            "typhoformer_delta_r_all": {"6h": 31.539, "12h": 38.084, "18h": 42.435, "24h": 49.562},
            "typhoformer_delta_r_2024": {"6h": 31.274, "12h": 37.934, "18h": 42.973, "24h": 50.881},
            "note": "This lead-specific clean 12->1 protocol is closer to the public repo's next-step setting than joint 12->4, but it still remains a local reconstruction rather than an official reproduction.",
        },
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = [
        ("B2 Numeric Transformer", b2),
        ("B3 TyphoFormer leak-free", b3),
        ("B4 TyphoFormer++ raw", b4_raw),
        ("B4 TyphoFormer++ + val CV calibration", b4_cal),
    ]
    print_delta_table("Delta R table, 2022-2024 test", summary["baselines"], rows)
    rows_2024 = [
        ("B2 Numeric Transformer", b2_2024),
        ("B3 TyphoFormer leak-free", b3_2024),
        ("B4 TyphoFormer++ raw", b4_raw_2024),
        ("B4 TyphoFormer++ + val CV calibration", b4_cal_2024),
    ]
    print()
    print_delta_table("Delta R table, 2024 subset", summary["baselines_2024"], rows_2024)
    print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
