"""Train one frozen, numeric-only output-head control with epoch-level resume.

Uses trusted project .npy dictionaries (pickle); never load untrusted datasets.
Default fits on train, selects on validation, and does not read test samples.
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from controlled_heads import ControlledNumericModel
from provenance import TRAINING_FILES, source_hashes
from vendor.common import denormalize_disp, disp_to_latlon, haversine_km, latlon_to_disp, seed_everything


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def load_split(root, split, metadata, indices):
    paths = sorted((root / split).glob("*.npy"))
    if not paths:
        raise FileNotFoundError(root / split)
    rows = [np.load(p, allow_pickle=True).item() for p in paths]
    tensors = {}
    mapping = {"x": "input_num", "y": "target_disp", "raw": "target_disp_raw_km",
               "history": "history_latlon", "origin": "origin", "future": "future_latlon"}
    for key, field in mapping.items():
        tensors[key] = torch.from_numpy(np.stack([r[field] for r in rows]).astype(np.float32))
        if not torch.isfinite(tensors[key]).all():
            raise ValueError(f"Nonfinite {split}/{field}")
    tensors["x"] = tensors["x"][..., indices]
    mean, std = [torch.tensor(metadata["stats"][k]) for k in ("target_mean", "target_std")]
    torch.testing.assert_close(tensors["y"] * std + mean, tensors["raw"], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(tensors["origin"], tensors["history"][:, -1])
    torch.testing.assert_close(latlon_to_disp(tensors["future"], tensors["origin"][:, None]),
                               tensors["raw"], rtol=1e-4, atol=0.003)
    if tensors["y"].shape[1] != len(metadata["lead_steps"]):
        raise ValueError("Target length differs from explicit lead_steps")
    if tensors["x"].shape[1] != metadata["input_len"]:
        raise ValueError("Input length differs from metadata")
    keys = [(str(r["storm_id"]), int(r["window"])) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate storm/window keys")
    for row in rows:
        if "history_timestamp_hours" in row and "future_timestamp_hours" in row:
            history = np.asarray(row["history_timestamp_hours"])
            future = np.asarray(row["future_timestamp_hours"])
            if not np.all(np.diff(history) == 6):
                raise ValueError("History is not strict six-hourly")
            if not np.array_equal(future - history[-1], np.asarray(metadata["lead_steps"]) * 6):
                raise ValueError("Target lead differs from timestamp evidence")
    return {"tensors": tensors, "keys": keys,
            "files": {p.name: digest(p) for p in paths}}


def batches(data, size, device, shuffle=False, shuffle_seed=0):
    generator = torch.Generator().manual_seed(shuffle_seed)
    order = torch.randperm(len(data["keys"]), generator=generator) if shuffle else torch.arange(len(data["keys"]))
    for ids in order.split(size):
        yield ids, {k: v[ids].to(device) for k, v in data["tensors"].items()}


@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    outputs, errors = {}, []
    for _, batch in batches(data, batch_size, device):
        result = model(batch["x"], batch["history"])
        for key in ("pred", "direct", "cv_residual", "cv"):
            if key in result:
                raw = denormalize_disp(result[key], model.target_mean, model.target_std)
                latlon = disp_to_latlon(raw, batch["origin"])
                outputs.setdefault(key + "_latlon", []).append(latlon.cpu().numpy())
                outputs.setdefault(key + "_error_km", []).append(haversine_km(latlon, batch["future"]).cpu().numpy())
        if "gate" in result:
            outputs.setdefault("gate", []).append(result["gate"].cpu().numpy())
    outputs = {key: np.concatenate(values) for key, values in outputs.items()}
    errors = outputs["pred_error_km"]
    ids = np.asarray([k[0] for k in data["keys"]])
    storm_errors = np.stack([errors[ids == sid].mean(axis=0) for sid in sorted(set(ids))])
    summary = {"window_mean_km": errors.mean(axis=0).tolist(),
               "storm_equal_mean_km": storm_errors.mean(axis=0).tolist(),
               "n_windows": len(ids), "n_storms": len(set(ids))}
    return summary, outputs


def snapshot(model, optimizer, epoch, best, stale, history):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            "best": best, "stale": stale, "history": history,
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--head", choices=["direct", "residual", "fixed", "adaptive"], required=True)
    p.add_argument("--rmw", choices=["on", "off"], default="on")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cv-convention", choices=["legacy_previous_origin", "forecast_origin"], default="legacy_previous_origin")
    p.add_argument("--aux-direct", type=float, default=0.0)
    p.add_argument("--aux-residual", type=float, default=0.0)
    p.add_argument("--aux-gate", type=float, default=0.0, help="Separate experiment: shrink gate toward 0.5")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-epochs", type=int, default=0, help="Intentional resumable wall-time checkpoint")
    p.add_argument("--evaluate-test", action="store_true", help="Read test only after fitting/validation selection")
    return p.parse_args()


def main():
    args = arguments()
    torch.set_num_threads(args.threads)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Positive epochs and batch size required")
    if min(args.aux_direct, args.aux_residual, args.aux_gate) < 0:
        raise ValueError("Auxiliary coefficients must be nonnegative")
    if (args.aux_direct or args.aux_residual or args.aux_gate) and args.head != "adaptive":
        raise ValueError("Auxiliary-loss experiment is separate and restricted to adaptive head")
    root, out = args.data_dir.resolve(), args.output.resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if "lead_steps" not in metadata or metadata.get("interval_hours", 6) != 6:
        raise ValueError("Require explicit six-hour lead_steps; old ambiguous metadata is unsupported")
    names = metadata["feature_names"]
    excluded = [name for name in names if "rmw" in name.lower() or "rmax" in name.lower()]
    if args.rmw == "off" and not excluded:
        raise ValueError("No identifiable RMW feature: specify provenance before ablating")
    indices = [i for i, name in enumerate(names) if args.rmw == "on" or name not in excluded]
    train, val = [load_split(root, split, metadata, indices) for split in ("train", "val")]
    train_ids, val_ids = [set(k[0] for k in data["keys"]) for data in (train, val)]
    if train_ids & val_ids:
        raise ValueError("Train/validation storms overlap")
    seed_everything(args.seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    model = ControlledNumericModel(len(indices), metadata["input_len"], metadata["lead_steps"],
                                  metadata["stats"]["target_mean"], metadata["stats"]["target_std"],
                                  args.head, args.d_model, args.num_heads, args.num_layers,
                                  args.dropout, args.cv_convention).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    code_root = Path(__file__).resolve().parent
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in {"resume", "evaluate_test", "stop_after_epochs", "output"}}
    config.update({"data_dir": str(root), "features": [names[i] for i in indices],
                   "excluded_features": excluded if args.rmw == "off" else [],
                   "text_analog_ranking_alignment": "disabled; no derived RMW modalities consumed",
                   "loss": "normalized displacement MSE + explicitly specified separate auxiliaries",
                   "selection": "lowest validation window mean haversine km averaged across output leads",
                   "scheduler": "none", "grad_clip_norm": 1.0,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "torch": torch.__version__, "numpy": np.__version__,
                   "device_name": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
                   "metadata_sha256": digest(root / "metadata.json"),
                   "code_sha256": source_hashes(TRAINING_FILES, code_root),
                   "data_sha256": {"train": train["files"], "val": val["files"]}})
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "frozen_config.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Run exists: pass --resume; never silently overwrite")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Resume configuration/data/code changed")
    else:
        if args.resume:
            raise FileNotFoundError("Cannot resume without frozen_config.json")
        json_write(manifest_path, config)
    start, best, stale, history = 0, float("inf"), 0, []
    # Reset after construction: batch order and initial training RNG match all heads.
    seed_everything(args.seed + 10000)
    if args.resume:
        state = torch.load(out / "latest.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best, stale, history = state["epoch"], state["best"], state["stale"], state["history"]
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    stop = min(args.epochs, start + args.stop_after_epochs) if args.stop_after_epochs else args.epochs
    for epoch in range(start, stop):
        if args.patience and stale >= args.patience:
            break
        begin = time.perf_counter()
        model.train()
        train_loss, seen = 0.0, 0
        for batch_index, (_, batch) in enumerate(batches(train, args.batch_size, args.device, shuffle=True, shuffle_seed=args.seed + epoch)):
            # Head-specific dropout consumption must not perturb the next batch's
            # backbone dropout or the next epoch's shuffle schedule.
            torch.manual_seed(args.seed * 1000000 + epoch * 10000 + batch_index)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["x"], batch["history"])
            loss = F.mse_loss(prediction["pred"], batch["y"])
            for coefficient, key in ((args.aux_direct, "direct"), (args.aux_residual, "cv_residual")):
                if coefficient:
                    loss = loss + coefficient * F.mse_loss(prediction[key], batch["y"])
            if args.aux_gate:
                loss = loss + args.aux_gate * (prediction["gate"] - 0.5).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; latest completed epoch remains resumable")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch["y"])
            seen += len(batch["y"])
        val_summary, _ = evaluate(model, val, args.batch_size, args.device)
        score = float(np.mean(val_summary["window_mean_km"]))
        improved = score < best
        best, stale = (score, 0) if improved else (best, stale + 1)
        elapsed = time.perf_counter() - begin
        history.append({"epoch": epoch + 1, "train_loss": train_loss / seen,
                        "validation_mean_km": score, "seconds_train_plus_validation": elapsed})
        state = snapshot(model, optimizer, epoch + 1, best, stale, history)
        save_checkpoint(out / "latest.pt", state)
        if improved:
            save_checkpoint(out / "best.pt", state)
        json_write(out / "training_history.json", history)
        print(json.dumps(history[-1]), flush=True)
    completed = len(history) >= args.epochs or bool(args.patience and stale >= args.patience)
    if not completed:
        json_write(out / "status.json", {"status": "paused_after_completed_epoch", "completed_epochs": len(history)})
        return
    selected = torch.load(out / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(selected["model"])
    summary, predictions = evaluate(model, val, args.batch_size, args.device)
    json_write(out / "validation.json", summary)
    if args.evaluate_test:
        test = load_split(root, "test", metadata, indices)
        if (train_ids | val_ids) & set(k[0] for k in test["keys"]):
            raise ValueError("Test storms overlap development splits")
        summary, predictions = evaluate(model, test, args.batch_size, args.device)
        np.savez_compressed(out / "test_predictions.npz", **predictions,
                            storm_id=np.asarray([k[0] for k in test["keys"]]),
                            window=np.asarray([k[1] for k in test["keys"]]),
                            lead_hours=np.asarray(metadata["lead_steps"]) * 6,
                            origin=test["tensors"]["origin"].numpy(),
                            history_latlon=test["tensors"]["history"].numpy(),
                            future_latlon=test["tensors"]["future"].numpy())
        json_write(out / "test.json", {**summary, "test_file_sha256": test["files"]})
    json_write(out / "status.json", {"status": "completed", "selected_epoch": selected["epoch"],
                                     "total_seconds_train_plus_validation": sum(h["seconds_train_plus_validation"] for h in history),
                                     "test_evaluated": args.evaluate_test,
                                     "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None})


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
