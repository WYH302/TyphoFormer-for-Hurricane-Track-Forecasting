"""Paired storm-cluster bootstrap; average seed errors, never treat seeds as storms."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from run_controlled import json_write


def load_group(run_paths):
    errors, reference, seeds, costs, protocol = [], None, [], [], None
    for run in map(Path, run_paths):
        config = json.loads((run / "frozen_config.json").read_text(encoding="utf-8"))
        current_protocol = {key: value for key, value in config.items() if key not in {"seed", "device", "device_name", "threads"}}
        if protocol is None:
            protocol = current_protocol
        elif current_protocol != protocol:
            raise ValueError("Runs within one group differ in more than seed/device")
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        if status["status"] != "completed" or not status["test_evaluated"]:
            raise ValueError(f"Incomplete run {run}")
        with np.load(run / "test_predictions.npz", allow_pickle=False) as saved:
            keys = list(zip(saved["storm_id"].astype(str), saved["window"].tolist()))
            order = sorted(range(len(keys)), key=lambda i: keys[i])
            ordered = [keys[i] for i in order]
            leads = saved["lead_hours"].tolist()
            truth = saved["future_latlon"][order]
            err = saved["pred_error_km"][order]
            if reference is None:
                reference = (ordered, leads, truth)
            elif reference[:2] != (ordered, leads) or not np.array_equal(reference[2], truth):
                raise ValueError("Seed runs do not predict exactly the same windows/leads/labels")
            if not np.isfinite(err).all():
                raise ValueError("Nonfinite errors")
            errors.append(err)
        seeds.append(config["seed"])
        costs.append({"seed": config["seed"], "parameters": config["parameters"],
                      "seconds": status["total_seconds_train_plus_validation"],
                      "selected_epoch": status["selected_epoch"], "head": config["head"], "rmw": config["rmw"]})
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seed in group")
    return {"keys": reference[0], "leads": reference[1], "truth": reference[2],
            "errors": np.mean(errors, axis=0), "seeds": sorted(seeds), "costs": costs, "protocol": protocol}


def paired_bootstrap(a, b, storms, replicates=10000, seed=20260913):
    """Positive difference means A has higher error than B (B improves)."""
    ids = sorted(set(storms))
    indices = [np.flatnonzero(storms == sid) for sid in ids]
    difference = a - b
    sums = np.stack([difference[ix].sum(axis=0) for ix in indices])
    counts = np.asarray([len(ix) for ix in indices])
    means = sums / counts[:, None]
    rng = np.random.default_rng(seed)
    storm_draws, window_draws = [], []
    for _ in range(replicates):
        chosen = rng.integers(0, len(ids), size=len(ids))
        storm_draws.append(means[chosen].mean(axis=0))
        window_draws.append(sums[chosen].sum(axis=0) / counts[chosen].sum())
    return {"n_storms": len(ids), "n_windows": len(storms),
            "difference_A_minus_B_window_mean_km": difference.mean(axis=0).tolist(),
            "difference_A_minus_B_storm_equal_km": means.mean(axis=0).tolist(),
            "window_weighted_cluster_95ci_km": np.percentile(window_draws, [2.5, 97.5], axis=0).T.tolist(),
            "storm_equal_cluster_95ci_km": np.percentile(storm_draws, [2.5, 97.5], axis=0).T.tolist(),
            "fraction_storms_B_improves": (means > 0).mean(axis=0).tolist(),
            "replicates": replicates, "bootstrap_seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", required=True, help="NAME=run1,run2,run3 (repeat for each model)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.replicates < 100:
        raise ValueError("Use at least 100 replicates")
    groups = {}
    for definition in args.group:
        name, paths = definition.split("=", 1)
        if name in groups:
            raise ValueError("Duplicate group name")
        groups[name] = load_group(paths.split(","))
    if len(groups) < 2:
        raise ValueError("At least two groups are required")
    report = {"estimand": "Mean error over fixed training seeds, then paired storm-cluster resampling. Not an ensemble prediction. CIs are conditional on these seeds and observed storms; no multiple-comparison correction.",
              "groups": {}, "comparisons": []}
    for name, group in groups.items():
        storms = np.asarray([k[0] for k in group["keys"]])
        report["groups"][name] = {"seeds": group["seeds"], "costs": group["costs"], "lead_hours": group["leads"],
                                  "window_mean_km": group["errors"].mean(axis=0).tolist(),
                                  "storm_equal_mean_km": np.mean([group["errors"][storms == sid].mean(axis=0) for sid in sorted(set(storms))], axis=0).tolist()}
    for (name_a, a), (name_b, b) in itertools.combinations(groups.items(), 2):
        if a["keys"] != b["keys"] or a["leads"] != b["leads"] or a["seeds"] != b["seeds"] or not np.array_equal(a["truth"], b["truth"]):
            raise ValueError("Pairing requires identical windows, labels, leads and seed sets")
        stats = paired_bootstrap(a["errors"], b["errors"], np.asarray([k[0] for k in a["keys"]]), args.replicates)
        changed = sorted(key for key in set(a["protocol"]) | set(b["protocol"]) if a["protocol"].get(key) != b["protocol"].get(key))
        report["comparisons"].append({"A": name_a, "B": name_b, "lead_hours": a["leads"],
                                      "differing_protocol_fields": changed,
                                      "output_head_only_comparison": set(changed) <= {"head", "parameters"}, **stats})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_write(args.output, report)


if __name__ == "__main__":
    main()
