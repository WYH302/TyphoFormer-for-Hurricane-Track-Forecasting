"""Evaluate an already selected checkpoint on an independently prepared holdout.

This never trains or changes model selection. Feature definitions, target stats,
leads and training/validation fields must be unchanged. The two unconsumed
global analog-ID arrays may differ after reindexing; all changes are recorded.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from controlled_heads import ControlledNumericModel
from provenance import EVALUATION_FILES, source_hashes
from run_controlled import digest, evaluate, json_write, load_split


def verify_development_samples(config, new_root):
    allowed_unused = {"analog_pos_ids", "analog_neg_ids"}
    differences = []
    for split in ("train", "val"):
        paths = {path.name: path for path in sorted((new_root / split).glob("*.npy"))}
        expected = config["data_sha256"][split]
        if set(paths) != set(expected):
            raise ValueError(f"Holdout preparation changed {split} sample filenames")
        for name, path in paths.items():
            old_path = Path(config["data_dir"]) / split / name
            if digest(old_path) != expected[name]:
                raise ValueError(f"Archived development sample changed: {split}/{name}")
            new_sha = digest(path)
            if new_sha == expected[name]:
                continue
            old, new = [np.load(p, allow_pickle=True).item() for p in (old_path, path)]
            if old.keys() != new.keys():
                raise ValueError(f"Changed record fields: {split}/{name}")
            changed = []
            for key in old:
                a, b = np.asarray(old[key]), np.asarray(new[key])
                equal = np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" and b.dtype.kind in "fc" else np.array_equal(a, b)
                if not equal:
                    changed.append(key)
            if set(changed) - allowed_unused:
                raise ValueError(f"Holdout changed frozen values {split}/{name}: {changed}")
            differences.append({"split": split, "file": name, "old_sha256": expected[name],
                                "new_sha256": new_sha, "changed_fields": changed})
    return {"allowed_differences": sorted(allowed_unused),
            "justification": "Numeric-only model does not consume analog IDs; insertion of a different test-year set can reindex global sample IDs. Every other field is compared exactly; original files remain unchanged.",
            "n_binary_different_files": len(differences), "files": differences}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", action="store_true", help="Retry a partial evaluation only with the identical checkpoint/data lock")
    args = p.parse_args()
    config = json.loads((args.run / "frozen_config.json").read_text(encoding="utf-8"))
    status = json.loads((args.run / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise ValueError("Development run is not completed")
    original = json.loads((Path(config["data_dir"]) / "metadata.json").read_text(encoding="utf-8"))
    new = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    for key in ("input_len", "lead_steps", "feature_names", "stats"):
        if original[key] != new[key]:
            raise ValueError(f"Holdout changed frozen {key}")
    development_audit = verify_development_samples(config, args.data_dir)
    code_root = Path(__file__).resolve().parent
    for path, expected in config["code_sha256"].items():
        if digest(code_root / path) != expected:
            raise ValueError(f"Frozen implementation changed: {path}")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Use --resume only to retry the identical frozen evaluation")
    indices = [new["feature_names"].index(name) for name in config["features"]]
    model = ControlledNumericModel(len(indices), new["input_len"], new["lead_steps"],
                                  new["stats"]["target_mean"], new["stats"]["target_std"],
                                  config["head"], config["d_model"], config["num_heads"],
                                  config["num_layers"], config["dropout"], config["cv_convention"]).to(args.device)
    checkpoint = torch.load(args.run / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    lock = {"development_config_sha256": digest(args.run / "frozen_config.json"),
               "evaluation_code_sha256": source_hashes(EVALUATION_FILES),
               "checkpoint_sha256": digest(args.run / "best.pt"), "selected_epoch": checkpoint["epoch"],
               "holdout_metadata_sha256": digest(args.data_dir / "metadata.json"),
               "holdout_file_sha256": {p.name: digest(p) for p in sorted((args.data_dir / "test").glob("*.npy"))},
               "selection_rule": "unchanged development validation selection"}
    lock_path = args.output / "evaluation_lock.json"
    if args.output.exists():
        if not lock_path.exists() or json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise ValueError("Partial evaluation lock differs or is missing")
        attempts_path = args.output / "evaluation_attempts.json"
        attempts = json.loads(attempts_path.read_text(encoding="utf-8")) if attempts_path.exists() else []
    else:
        args.output.mkdir(parents=True)
        # Seal chosen checkpoint before reading holdout arrays/results.
        json_write(lock_path, lock)
        attempts = []
    attempts.append({"attempt": len(attempts) + 1, "same_lock_retry": args.resume})
    json_write(args.output / "evaluation_attempts.json", attempts)
    json_write(args.output / "development_identity_audit.json", development_audit)
    test = load_split(args.data_dir, "test", new, indices)
    if test["files"] != lock["holdout_file_sha256"]:
        raise ValueError("Holdout files changed during evaluation")
    development_storms = set(original["split_storm_ids"]["train"] + original["split_storm_ids"]["val"])
    if development_storms & set(k[0] for k in test["keys"]):
        raise ValueError("Holdout storms overlap development storms")
    summary, predictions = evaluate(model, test, config["batch_size"], args.device)
    np.savez_compressed(args.output / "test_predictions.npz", **predictions,
                        storm_id=np.asarray([k[0] for k in test["keys"]]),
                        window=np.asarray([k[1] for k in test["keys"]]),
                        lead_hours=np.asarray(new["lead_steps"]) * 6,
                        origin=test["tensors"]["origin"].numpy(),
                        history_latlon=test["tensors"]["history"].numpy(),
                        future_latlon=test["tensors"]["future"].numpy())
    json_write(args.output / "test.json", {**summary, "test_file_sha256": test["files"]})
    json_write(args.output / "frozen_config.json", config)
    json_write(args.output / "status.json", {**status, "test_evaluated": True, "external_holdout": True})


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    main()
