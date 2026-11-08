"""Quantify N1 on an explicitly selected split without changing historical models."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from controlled_heads import cv_displacement
from run_controlled import digest, json_write, load_split
from vendor.common import constant_velocity_future_disp, disp_to_latlon, haversine_km


def describe(array):
    array = np.asarray(array)
    return {"mean": array.mean(axis=0).tolist(), "median": np.median(array, axis=0).tolist(),
            "p95": np.percentile(array, 95, axis=0).tolist(), "maximum": array.max(axis=0).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    data = load_split(args.data_dir, args.split, metadata, list(range(metadata["input_dim"])))
    t = data["tensors"]
    legacy = cv_displacement(t["history"], metadata["lead_steps"])
    corrected = cv_displacement(t["history"], metadata["lead_steps"], "forecast_origin")
    independent = constant_velocity_future_disp(t["history"], len(metadata["lead_steps"]), metadata["lead_steps"])
    torch.testing.assert_close(legacy, independent, rtol=0, atol=0)
    old_ll, new_ll = [disp_to_latlon(value, t["origin"]) for value in (legacy, corrected)]
    old_error, new_error = [haversine_km(value, t["future"]).numpy() for value in (old_ll, new_ll)]
    json_write(args.output, {"interpretation": "Historical common CV and independent evaluator share the previous-origin convention; forecast-origin is a coordinate sensitivity alternative, not a retroactive correction or a trained-model result.",
        "split": args.split, "metadata_sha256": digest(args.data_dir / "metadata.json"),
        "lead_hours": (np.asarray(metadata["lead_steps"]) * 6).tolist(), "n_windows": len(data["keys"]),
        "independent_legacy_equivalence": "exact float32 equality passed",
        "displacement_difference_norm_km": describe(torch.linalg.vector_norm(legacy - corrected, dim=-1).numpy()),
        "forecast_position_difference_km": describe(haversine_km(old_ll, new_ll).numpy()),
        "legacy_cv_error_km": describe(old_error), "forecast_origin_cv_error_km": describe(new_error),
        "legacy_minus_forecast_origin_error_km": describe(old_error - new_error)})


if __name__ == "__main__":
    main()
