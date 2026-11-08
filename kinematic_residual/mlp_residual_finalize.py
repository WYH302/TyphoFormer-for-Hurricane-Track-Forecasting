"""Finish evaluation of the already frozen 24 MLP fits; never train or select.

The first runner parsed year from storm_id[-4:], but IDs include storm names.
Use the explicit numeric storm_year field instead. Preserve the original
training source, source hashes, checkpoints, failed log and selection lock.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_locked import verify_development_samples
from mlp_residual_baseline import NumericMLPResidual, subset, write_predictions
from run_controlled import digest, json_write, load_split


def verify_years(paths, expected):
    years = set()
    for path in paths:
        row = np.load(path, allow_pickle=True).item()
        year = row.get("storm_year")
        if not isinstance(year, (int, np.integer)):
            raise ValueError(f"Missing explicit numeric storm_year in {path}")
        years.add(int(year))
    if not years or not years <= set(expected):
        raise ValueError(f"Unexpected explicit storm years: {years}")
    return sorted(years)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    revision = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-root", type=Path, default=revision / "data")
    parser.add_argument("--output", type=Path, default=revision / "results/mlp_residual")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    root, out = args.data_root.resolve(), args.output.resolve()
    protocol = json.loads((out / "protocol_lock.json").read_text(encoding="utf-8"))
    selection_lock = json.loads((out / "all_selections_frozen.json").read_text(encoding="utf-8"))
    if not selection_lock["all_24_models_selected_before_2025_loading"] or len(selection_lock["selections"]) != 24:
        raise ValueError("Incomplete pre-evaluation selection lock")
    for filename, expected in protocol["source_sha256"].items():
        if digest(Path(__file__).parent / filename) != expected:
            raise ValueError(f"Original frozen training dependency changed: {filename}")
    amendment = {"finalizer_sha256": digest(Path(__file__)),
                 "original_protocol_sha256": digest(out / "protocol_lock.json"),
                 "all_selections_sha256": digest(out / "all_selections_frozen.json"),
                 "execution_repair": "Replace the invalid post-training storm_id suffix year check with validation of the explicit numeric storm_year field. Training sources, checkpoints, hyperparameters and all 24 validation selections are unchanged. No training is performed by this finalizer."}
    amendment_path = out / "finalization_lock.json"
    if amendment_path.exists() and json.loads(amendment_path.read_text(encoding="utf-8")) != amendment:
        raise ValueError("Finalization lock differs")
    if not amendment_path.exists():
        json_write(amendment_path, amendment)
    models, metadata_by_lead = {}, {}
    for entry in selection_lock["selections"]:
        name = entry["name"]
        run = out / "runs" / name
        if digest(run / "best.pt") != entry["checkpoint_sha256"]:
            raise ValueError(f"Selected checkpoint changed: {name}")
        config = json.loads((run / "frozen_config.json").read_text(encoding="utf-8"))
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        if status["completed_epochs"] != 40 or status["selected_epoch"] != entry["selected_epoch"]:
            raise ValueError(f"Incomplete or reselected training: {name}")
        saved = torch.load(run / "best.pt", map_location="cpu", weights_only=False)["model"]
        model = NumericMLPResidual(saved["input_mean"], saved["input_std"], saved["target_mean"], saved["target_std"], saved["lead_steps"])
        model.load_state_dict(saved)
        if sum(p.numel() for p in model.parameters()) != config["parameters"]:
            raise ValueError("MLP parameter count differs from frozen configuration")
        lead = config["lead_steps"][0]
        metadata = json.loads((Path(config["data_dir"]) / "metadata.json").read_text(encoding="utf-8"))
        if digest(Path(config["data_dir"]) / "metadata.json") != protocol["development"][str(lead)]["metadata_sha256"]:
            raise ValueError("Development metadata changed")
        metadata_by_lead[lead] = metadata
        models[name] = (model, config, status, lead)
    # Verify all four development identities before loading even the first 2025 array.
    for lead, metadata in metadata_by_lead.items():
        extension = root / f"extension2025_lead{lead}"
        ext_meta = json.loads((extension / "metadata.json").read_text(encoding="utf-8"))
        for field in ("stats", "feature_names", "input_len", "lead_steps"):
            if metadata[field] != ext_meta[field]:
                raise ValueError(f"Extension changed {field}")
        config = next(values[1] for values in models.values() if values[-1] == lead)
        identity = verify_development_samples(config, extension)
        json_write(out / f"finalization_development_identity_lead{lead}.json", identity)
    summary = []
    for lead in (1, 2, 3, 4):
        metadata = metadata_by_lead[lead]
        development_storms = set(metadata["split_storm_ids"]["train"] + metadata["split_storm_ids"]["val"])
        for population, folder, years in (("test2022_2024", root / f"controlled_12to1_lead{lead}", {2022, 2023, 2024}),
                                         ("extension2025", root / f"extension2025_lead{lead}", {2025})):
            paths = sorted((folder / "test").glob("*.npy"))
            test_lock = {path.name: digest(path) for path in paths}
            lock = {"finalization_lock_sha256": digest(amendment_path), "test_sha256": test_lock,
                    "metadata_sha256": digest(folder / "metadata.json")}
            path_lock = out / f"finalized_evaluation_lock_{population}_lead{lead}.json"
            if path_lock.exists() and json.loads(path_lock.read_text(encoding="utf-8")) != lock:
                raise ValueError("Evaluation data lock changed")
            if not path_lock.exists():
                json_write(path_lock, lock)
            actual_years = verify_years(paths, years)
            data = load_split(folder, "test", metadata, list(range(metadata["input_dim"])))
            if data["files"] != test_lock or development_storms & set(key[0] for key in data["keys"]):
                raise ValueError("Test changed or overlaps development")
            for name, (model, config, status, fit_lead) in models.items():
                if fit_lead != lead:
                    continue
                keep = [metadata["feature_names"].index(feature) for feature in config["features"]]
                destination = out / ("runs" if population == "test2022_2024" else "holdout2025") / name
                metrics = write_predictions(destination, model, subset(data, keep), metadata, config, status, population)
                summary.append({"name": name, "population": population, "lead_hours": lead * 6,
                    "rmw": config["rmw"], "seed": config["seed"], "windows": metrics["n_windows"],
                    "storms": metrics["n_storms"], "window_mean_km": metrics["window_mean_km"][0],
                    "storm_equal_mean_km": metrics["storm_equal_mean_km"][0],
                    "fit_seconds": status["total_seconds_train_plus_validation"], "selected_epoch": status["selected_epoch"]})
            print(json.dumps({"evaluated": population, "lead_hours": lead * 6, "explicit_years": actual_years}), flush=True)
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    result = {"status": "completed", "completed_fits": 24, "prediction_files": 48,
              "evaluated_populations": ["test2022_2024", "extension2025"],
              "finalization_seconds": time.perf_counter() - started,
              "total_seconds_train_plus_validation": sum(entry["fit_seconds"] for entry in selection_lock["selections"]),
              "no_retraining_or_reselection_in_finalization": True, "results": summary}
    json_write(out / "run_summary.json", result)
    json_write(out / "status.json", {key: value for key, value in result.items() if key != "results"})
    print(json.dumps({key: value for key, value in result.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    main()
