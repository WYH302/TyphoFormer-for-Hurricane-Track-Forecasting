"""Newly reconstructed MLP residual comparator; independent fixed CPU protocol.

This is not a claim to reproduce the historical MLP's architecture or numbers.
All 24 fits are declared before fitting and selected before 2025 data loading.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from controlled_heads import cv_displacement
from evaluate_locked import verify_development_samples
from provenance import EVALUATION_FILES, source_hashes
from run_controlled import batches, digest, evaluate, json_write, load_split, save_checkpoint
from vendor.common import normalize_disp

SEEDS = (42, 123, 2024)
RMW_MODES = ("on", "off")
EPOCHS = 40
BATCH_SIZE = 32


class NumericMLPResidual(nn.Module):
    def __init__(self, input_mean, input_std, target_mean, target_std, lead_steps):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(len(input_mean), 128), nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2))
        for name, values in (("input_mean", input_mean), ("input_std", input_std),
                             ("target_mean", target_mean), ("target_std", target_std),
                             ("lead_steps", lead_steps)):
            self.register_buffer(name, torch.as_tensor(values, dtype=torch.float32).clone())
        if len(lead_steps) != 1 or (self.input_std <= 0).any() or (self.target_std <= 0).any():
            raise ValueError("Require one explicit lead and positive scales")

    def forward(self, x, history):
        normalized_x = (x.flatten(1) - self.input_mean) / self.input_std
        residual = self.network(normalized_x).reshape(-1, 1, 2)
        cv_raw = cv_displacement(history, self.lead_steps)
        cv_norm = normalize_disp(cv_raw, self.target_mean, self.target_std)
        prediction = cv_norm + residual
        return {"pred": prediction, "cv_residual": prediction, "cv": cv_norm}


def subset(data, keep):
    return {**data, "tensors": {**data["tensors"], "x": data["tensors"]["x"][..., keep]}}


def normalization(train_x):
    flat = train_x.flatten(1)
    mean = flat.mean(0)
    std = flat.std(0, unbiased=False)
    return mean, torch.where(std < 1e-6, torch.ones_like(std), std)


def fit_one(run, config, metadata, train, val):
    run.mkdir(parents=True, exist_ok=True)
    config_path = run / "frozen_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Frozen MLP configuration changed")
    else:
        json_write(config_path, config)
    mean, std = normalization(train["tensors"]["x"])
    torch.manual_seed(config["seed"])
    model = NumericMLPResidual(mean, std, metadata["stats"]["target_mean"],
                               metadata["stats"]["target_std"], metadata["lead_steps"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    start, best, selected_epoch, history = 0, float("inf"), 0, []
    if (run / "latest.pt").exists():
        state = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best, selected_epoch, history = state["epoch"], state["best"], state["selected_epoch"], state["history"]
    for epoch in range(start, EPOCHS):
        started = time.perf_counter()
        model.train()
        total, seen = 0.0, 0
        for index, (_, batch) in enumerate(batches(train, BATCH_SIZE, "cpu", True, config["seed"] + epoch)):
            torch.manual_seed(config["seed"] * 1000000 + epoch * 10000 + index)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["x"], batch["history"])["pred"]
            loss = F.mse_loss(prediction, batch["y"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite MLP training loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch["y"])
            seen += len(batch["y"])
        validation, _ = evaluate(model, val, BATCH_SIZE, "cpu")
        score = float(np.mean(validation["window_mean_km"]))
        improved = score < best
        if improved:
            best, selected_epoch = score, epoch + 1
        history.append({"epoch": epoch + 1, "train_normalized_mse": total / seen,
                        "validation_window_mean_km": score,
                        "seconds_train_plus_validation": time.perf_counter() - started})
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                 "best": best, "selected_epoch": selected_epoch, "history": history}
        if improved:
            save_checkpoint(run / "best.pt", state)
        save_checkpoint(run / "latest.pt", state)
        json_write(run / "training_history.json", history)
    best_state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model"])
    validation, _ = evaluate(model, val, BATCH_SIZE, "cpu")
    json_write(run / "validation.json", validation)
    status = {"status": "completed", "test_evaluated": False, "selected_epoch": selected_epoch,
              "total_seconds_train_plus_validation": sum(item["seconds_train_plus_validation"] for item in history),
              "completed_epochs": EPOCHS, "device": "cpu", "threads": 1,
              "parameters": sum(parameter.numel() for parameter in model.parameters())}
    json_write(run / "status.json", status)
    return model, status, validation


def write_predictions(destination, model, data, metadata, config, training_status, population):
    destination.mkdir(parents=True, exist_ok=True)
    metrics, arrays = evaluate(model, data, BATCH_SIZE, "cpu")
    np.savez_compressed(destination / "test_predictions.npz", **arrays,
        storm_id=np.asarray([key[0] for key in data["keys"]]), window=np.asarray([key[1] for key in data["keys"]]),
        lead_hours=np.asarray(metadata["lead_steps"]) * 6, future_latlon=data["tensors"]["future"].numpy(),
        origin=data["tensors"]["origin"].numpy(), history_latlon=data["tensors"]["history"].numpy(),
        target_raw_km=data["tensors"]["raw"].numpy())
    json_write(destination / "frozen_config.json", config)
    json_write(destination / "test.json", {**metrics, "population": population, "test_file_sha256": data["files"]})
    json_write(destination / "status.json", {**training_status, "test_evaluated": True, "population": population})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-root", type=Path, default=default_root / "data")
    parser.add_argument("--output", type=Path, default=default_root / "results/mlp_residual")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    root, out = args.data_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sources = source_hashes(EVALUATION_FILES + ("mlp_residual_baseline.py",))
    datasets, development = {}, {}
    # Load only train/validation before declaring the complete protocol.
    for lead in (1, 2, 3, 4):
        folder = root / f"controlled_12to1_lead{lead}"
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if metadata["lead_steps"] != [lead] or metadata["input_len"] != 12:
            raise ValueError("Unexpected lead or history length")
        train, val = [load_split(folder, split, metadata, list(range(metadata["input_dim"]))) for split in ("train", "val")]
        if set(key[0] for key in train["keys"]) & set(key[0] for key in val["keys"]):
            raise ValueError("Development storms overlap")
        datasets[lead] = (metadata, train, val)
        development[str(lead)] = {"metadata_sha256": digest(folder / "metadata.json"),
                                  "train": train["files"], "val": val["files"]}
    jobs = [{"lead": lead, "rmw": rmw, "seed": seed} for lead in (1, 2, 3, 4) for rmw in RMW_MODES for seed in SEEDS]
    protocol = {"architecture": "flattened 12 x retained numeric features -> 128 ReLU Dropout(0.1) -> 128 ReLU Dropout(0.1) -> 2 std-scaled residual coordinates",
        "historical_status": "newly reconstructed comparator, not a matched reproduction of historical MLP architecture/results",
        "jobs": jobs, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "AdamW", "lr": 1e-4,
        "weight_decay": 1e-4, "scheduler": "none", "early_stopping": False, "gradient_clipping": "none",
        "loss": "MSE of absolute standardized displacement after adding CV; residual offsets are divided by target std without subtracting target mean again",
        "normalization": "Prepared historical train-only per-feature scaling, followed by train-only population mean/std per flattened column; columns with std < 1e-6 use 1",
        "cv_convention": "legacy_previous_origin", "RMW_off": "remove rmax/rmw columns at every time step before fitting; no text/analog/derived modalities consumed",
        "selection": "Each fixed seed/run selects lowest validation window-mean haversine km; first epoch wins exact ties. No architecture/hyperparameter search.",
        "extension_rule": "Seal all 24 validation selections before loading any 2025 test arrays; verify complete non-ID development field identity and train statistics before extension evaluation",
        "device": "cpu", "threads": 1, "torch": torch.__version__, "numpy": np.__version__,
        "source_sha256": sources, "development": development}
    lock_path = out / "protocol_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing MLP protocol/data/source changed")
    else:
        json_write(lock_path, protocol)
    fitted, selections = {}, []
    for index, job in enumerate(jobs):
        lead, rmw, seed = job["lead"], job["rmw"], job["seed"]
        metadata, train_all, val_all = datasets[lead]
        keep = [i for i, name in enumerate(metadata["feature_names"]) if rmw == "on" or not any(token in name.lower() for token in ("rmax", "rmw"))]
        if rmw == "off" and len(keep) == metadata["input_dim"]:
            raise ValueError("RMW-off did not remove a feature")
        train, val = subset(train_all, keep), subset(val_all, keep)
        name = f"lead{lead}_mlp_residual_rmw_{rmw}_seed{seed}"
        run = out / "runs" / name
        config = {"head": "mlp_residual", "rmw": rmw, "seed": seed, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": 1e-4, "weight_decay": 1e-4, "dropout": 0.1,
            "architecture": protocol["architecture"], "historical_status": protocol["historical_status"],
            "data_dir": str(root / f"controlled_12to1_lead{lead}"), "device": "cpu", "threads": 1,
            "features": [metadata["feature_names"][i] for i in keep], "lead_steps": [lead],
            "data_sha256": {"train": train_all["files"], "val": val_all["files"]},
            "code_sha256": sources, "protocol_sha256": digest(lock_path),
            "parameters": (12 * len(keep) + 1) * 128 + 129 * 128 + 129 * 2}
        json_write(out / "status.json", {"status": "training", "completed_fits": index, "total_fits": 24, "run": name})
        model, status, validation = fit_one(run, config, metadata, train, val)
        selections.append({"name": name, "lead_hours": lead * 6, "rmw": rmw, "seed": seed,
                           "selected_epoch": status["selected_epoch"], "validation": validation,
                           "checkpoint_sha256": digest(run / "best.pt"), "parameters": config["parameters"],
                           "fit_seconds": status["total_seconds_train_plus_validation"]})
        fitted[name] = (model, config, status, keep, lead)
        print(json.dumps({"fit": index + 1, "total": 24, "name": name,
                          "selected_epoch": status["selected_epoch"], "seconds": status["total_seconds_train_plus_validation"]}), flush=True)
    selections_lock = {"protocol_sha256": digest(lock_path), "all_24_models_selected_before_2025_loading": True,
                       "selections": selections}
    selection_path = out / "all_selections_frozen.json"
    if selection_path.exists() and json.loads(selection_path.read_text(encoding="utf-8")) != selections_lock:
        raise ValueError("Frozen MLP model selection differs")
    if not selection_path.exists():
        json_write(selection_path, selections_lock)
    # Complete consumed-development identity verification before any 2025 test load.
    for lead in (1, 2, 3, 4):
        metadata = datasets[lead][0]
        extension = root / f"extension2025_lead{lead}"
        ext_meta = json.loads((extension / "metadata.json").read_text(encoding="utf-8"))
        for field in ("stats", "feature_names", "input_len", "lead_steps"):
            if metadata[field] != ext_meta[field]:
                raise ValueError(f"2025 changed frozen {field}")
        config = next(values[1] for values in fitted.values() if values[-1] == lead)
        identity = verify_development_samples(config, extension)
        json_write(out / f"development_identity_lead{lead}.json", identity)
    summary = []
    for lead in (1, 2, 3, 4):
        metadata = datasets[lead][0]
        development_storms = set(metadata["split_storm_ids"]["train"] + metadata["split_storm_ids"]["val"])
        for population, folder in (("test2022_2024", root / f"controlled_12to1_lead{lead}"),
                                   ("extension2025", root / f"extension2025_lead{lead}")):
            test_lock = {path.name: digest(path) for path in sorted((folder / "test").glob("*.npy"))}
            json_write(out / f"evaluation_lock_{population}_lead{lead}.json", {
                "all_selections_sha256": digest(selection_path), "test_sha256": test_lock,
                "metadata_sha256": digest(folder / "metadata.json")})
            data = load_split(folder, "test", metadata, list(range(metadata["input_dim"])))
            if data["files"] != test_lock or development_storms & set(key[0] for key in data["keys"]):
                raise ValueError("Test changed or overlaps development")
            expected_years = {2025} if population == "extension2025" else {2022, 2023, 2024}
            if not {int(key[0][-4:]) for key in data["keys"]} <= expected_years:
                raise ValueError("Unexpected test storm year")
            for name, (model, config, status, keep, fit_lead) in fitted.items():
                if fit_lead != lead:
                    continue
                destination = out / ("runs" if population == "test2022_2024" else "holdout2025") / name
                metrics = write_predictions(destination, model, subset(data, keep), metadata, config, status, population)
                summary.append({"name": name, "population": population, "lead_hours": lead * 6,
                    "rmw": config["rmw"], "seed": config["seed"], "windows": metrics["n_windows"],
                    "storms": metrics["n_storms"], "window_mean_km": metrics["window_mean_km"][0],
                    "storm_equal_mean_km": metrics["storm_equal_mean_km"][0],
                    "fit_seconds": status["total_seconds_train_plus_validation"], "selected_epoch": status["selected_epoch"]})
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    final = {"status": "completed", "completed_fits": 24, "evaluated_populations": ["test2022_2024", "extension2025"],
             "elapsed_seconds_this_invocation": time.perf_counter() - started,
             "total_seconds_train_plus_validation": sum(item["fit_seconds"] for item in selections),
             "no_2025_model_selection": True, "results": summary}
    json_write(out / "run_summary.json", final)
    json_write(out / "status.json", {key: value for key, value in final.items() if key != "results"})
    print(json.dumps({key: value for key, value in final.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    main()
