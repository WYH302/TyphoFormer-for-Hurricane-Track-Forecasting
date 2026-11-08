"""Post-training, warmed FP32 inference cost on common historical validation inputs.

No fitting, tuning, or forecast-score comparison. CPU/GPU are separate panels.
All timings exclude loading and host/device transfers; report medians and IQR.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from controlled_heads import ControlledNumericModel, cv_displacement
from mlp_residual_baseline import NumericMLPResidual
from run_controlled import digest

REV = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class RidgeInference(nn.Module):
    def __init__(self, path, lead_steps):
        super().__init__()
        with np.load(path) as arrays:
            for name in ("x_mean", "x_std", "coef", "intercept"):
                self.register_buffer(name, torch.tensor(arrays[name], dtype=torch.float32))
        self.register_buffer("lead_steps", torch.tensor(lead_steps, dtype=torch.float32))

    def forward(self, x, history):
        z = (x.flatten(1) - self.x_mean) / self.x_std
        return cv_displacement(history, self.lead_steps) + (z @ self.coef.T + self.intercept).reshape(-1, 1, 2)


def models(metadata):
    matrix = REV / "results/controlled_matrix/runs"
    for head in ("direct", "residual", "fixed", "adaptive"):
        run = matrix / f"lead4_{head}_rmw_on_seed42"
        config = read(run / "frozen_config.json")
        model = ControlledNumericModel(len(config["features"]), metadata["input_len"], metadata["lead_steps"],
                                      metadata["stats"]["target_mean"], metadata["stats"]["target_std"],
                                      head, config["d_model"], config["num_heads"], config["num_layers"],
                                      config["dropout"], config["cv_convention"])
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        yield head, model, run / "best.pt", True
    run = REV / "results/mlp_residual/runs/lead4_mlp_residual_rmw_on_seed42"
    if not run.exists():
        run = REV / "results/mlp_residual/runs/lead4_rmw_on_seed42"
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    model = NumericMLPResidual(state["input_mean"], state["input_std"], state["target_mean"], state["target_std"], state["lead_steps"])
    model.load_state_dict(state)
    yield "mlp_residual", model, run / "best.pt", True
    path = REV / "results/ridge/lead4_rmw_on_model.npz"
    yield "ridge_residual", RidgeInference(path, metadata["lead_steps"]), path, False


@torch.inference_mode()
def measure(model, x, history, normalized, device, warmup, repeats):
    model = model.to(device).eval()
    x, history = x.to(device), history.to(device)

    def invoke():
        result = model(x, history)
        return result["pred"] * model.target_std + model.target_mean if normalized else result

    sync = torch.cuda.synchronize if device == "cuda" else lambda: None
    for _ in range(warmup):
        value = invoke()
    sync()
    assert value.shape == (len(x), 1, 2) and torch.isfinite(value).all()
    elapsed = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter_ns()
        invoke()
        sync()
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    model.cpu()
    if device == "cuda":
        torch.cuda.empty_cache()
    q = np.percentile(elapsed, [25, 50, 75])
    return dict(batch_size=len(x), device=device, median_ms=float(q[1]), q25_ms=float(q[0]), q75_ms=float(q[2]),
                median_ms_per_window=float(q[1] / len(x)), warmup=warmup, repeats=repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    queue = read(REV / "results/controlled_matrix/queue_status.json")
    if queue["status"] != "completed":
        raise RuntimeError("Wait for the formal GPU queue to finish before timing")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data = REV / "data/controlled_12to1_lead4"
    metadata = read(data / "metadata.json")
    files = sorted((data / "val").glob("*.npy"))[:32]
    arrays = [np.load(p, allow_pickle=True).item() for p in files]
    x = torch.tensor(np.stack([a["input_num"] for a in arrays]), dtype=torch.float32)
    history = torch.tensor(np.stack([a["history_latlon"] for a in arrays]), dtype=torch.float32)
    rows = []
    for name, model, checkpoint, normalized in models(metadata):
        parameters = sum(p.numel() for p in model.parameters()) if normalized else model.coef.numel() + model.intercept.numel()
        stored_bytes = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))
        for device in ("cpu", "cuda"):
            for batch in (1, 32):
                row = measure(model, x[:batch], history[:batch], normalized, device, args.warmup, args.repeats)
                rows.append(dict(model=name, learned_coefficients_or_weights=parameters, stored_tensor_bytes=stored_bytes,
                                 checkpoint_sha256=digest(checkpoint), **row))
    result = dict(protocol="All models RMW-on, 24h, seed42 where applicable; FP32; warmed resident tensors; output is displacement in km; excludes loading, transfers and geographic decoding. Ridge uses equivalent Torch matrix arithmetic for this benchmark, not sklearn training time.",
                  cpu_identifier=os.environ.get("PROCESSOR_IDENTIFIER"), gpu=torch.cuda.get_device_name(0),
                  python=platform.python_version(), torch=torch.__version__, cpu_threads=1,
                  validation_input_sha256={p.name: digest(p) for p in files}, script_sha256=digest(Path(__file__)), rows=rows)
    output = REV / "results/diagnostics/inference_cost.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(models=6, timings=len(rows), output=str(output)), ensure_ascii=True))


if __name__ == "__main__":
    main()
