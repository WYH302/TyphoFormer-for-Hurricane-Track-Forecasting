import argparse
import json
import os
from argparse import Namespace
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.TyphoFormerPlus import TyphoFormerPlus
from typhoformerpp_common import (
    TyphoPlusDataset,
    constant_velocity_future_disp,
    denormalize_disp,
    disp_to_latlon,
    format_metrics,
    haversine_km,
    latlon_to_disp,
    load_metadata,
    metadata_lead_hours,
    metadata_lead_steps,
    move_to_device,
    normalize_disp,
    seed_everything,
    stats_tensors,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--ode-steps", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--calibrate-cv-blend", action="store_true")
    parser.add_argument("--calibration-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--calibration-step", type=float, default=0.05)
    parser.add_argument("--year-filter", type=int, default=0)
    return parser.parse_args()


def make_model(train_args, metadata, device):
    variant = getattr(train_args, "variant", "plus")
    use_text = variant in {"typhoformer", "plus"} and not getattr(train_args, "no_text", False)
    use_positive_analog = variant == "plus" and not getattr(train_args, "no_positive_analog", False)
    use_negative_analog = variant == "plus" and not getattr(train_args, "no_negative_analog", False)
    use_negative_context = use_negative_analog and getattr(train_args, "fuse_negative_analog", False)
    use_analog = use_positive_analog or use_negative_context
    model = TyphoFormerPlus(
        input_dim=metadata["input_dim"],
        text_dim=metadata["text_dim"],
        pred_len=metadata["pred_len"],
        d_model=train_args.d_model,
        num_heads=train_args.num_heads,
        num_layers=train_args.num_layers,
        dropout=train_args.dropout,
        modality_dropout=train_args.modality_dropout,
        decoder_type=train_args.decoder,
        use_text=use_text,
        use_analog=use_analog,
        use_positive_analog=use_positive_analog,
        use_negative_context=use_negative_context,
        max_input_len=metadata["input_len"],
    ).to(device)
    model.lead_steps = metadata.get("lead_steps", list(range(1, int(metadata["pred_len"]) + 1)))
    return model


def add_errors(sums, pred_latlon, target_latlon, lead_hours=None):
    errors = haversine_km(pred_latlon, target_latlon)
    mae = torch.mean(torch.abs(pred_latlon - target_latlon), dim=-1)
    bsz = errors.shape[0]
    sums["count"] += bsz
    sums["ade"] += errors.mean(dim=1).sum().item()
    sums["fde"] += errors[:, -1].sum().item()
    sums["mae"] += mae.mean(dim=1).sum().item()
    if lead_hours is None:
        lead_hours = [6 * (idx + 1) for idx in range(errors.shape[1])]
    for idx, hour in enumerate(lead_hours):
        if idx < errors.shape[1]:
            sums[f"err{int(hour)}"] += errors[:, idx].sum().item()
            sums[f"mae{int(hour)}"] += mae[:, idx].sum().item()


def finalize(sums):
    count = sums.pop("count")
    return {key: value / count for key, value in sums.items()}


def cv_base_norm(batch, metadata, device):
    target_mean, target_std = stats_tensors(metadata, device)
    cv_raw = constant_velocity_future_disp(
        batch["history_latlon"],
        metadata["pred_len"],
        metadata_lead_steps(metadata, device=batch["history_latlon"].device, dtype=batch["history_latlon"].dtype),
    )
    return normalize_disp(cv_raw, target_mean, target_std)


def deterministic_prediction(features, batch, metadata, train_args, device):
    direct = features["direct_pred"]
    base = cv_base_norm(batch, metadata, device)
    cv_res = base + features["residual_gate"] * features["residual_pred"]
    if getattr(train_args, "residual", "none") == "dual":
        return features["dual_gate"] * cv_res + (1.0 - features["dual_gate"]) * direct
    if getattr(train_args, "residual", "none") == "cv":
        return cv_res
    return direct


def deterministic_raw_prediction(features, batch, metadata, train_args, device):
    target_mean, target_std = stats_tensors(metadata, device)
    pred_norm = deterministic_prediction(features, batch, metadata, train_args, device)
    return denormalize_disp(pred_norm, target_mean, target_std)


def flow_base_norm(features, batch, metadata, train_args, device):
    flow_base = getattr(train_args, "flow_base", "zero")
    if flow_base == "cv":
        return cv_base_norm(batch, metadata, device)
    if flow_base == "det":
        return deterministic_prediction(features, batch, metadata, train_args, device)
    return batch["target"].new_zeros(batch["target"].shape)


@torch.no_grad()
def calibrate_cv_blend(model, loader, metadata, train_args, args, device):
    if train_args.decoder != "deterministic":
        raise ValueError("CV blend calibration is only supported for deterministic checkpoints.")
    model.eval()
    model_raw, cv_raw, target_latlon, origin = [], [], [], []
    for batch in tqdm(loader, desc="Calibration", leave=False, disable=args.disable_progress):
        batch = move_to_device(batch, device)
        features = model(batch["x_num"], batch["x_text"], batch["analog_pos"], batch["analog_neg"])
        model_raw.append(deterministic_raw_prediction(features, batch, metadata, train_args, device).cpu())
        cv_raw.append(
            constant_velocity_future_disp(
                batch["history_latlon"],
                metadata["pred_len"],
                metadata_lead_steps(metadata, device=batch["history_latlon"].device, dtype=batch["history_latlon"].dtype),
            ).cpu()
        )
        target_latlon.append(batch["future_latlon"].cpu())
        origin.append(batch["origin"].cpu())

    model_raw = torch.cat(model_raw, dim=0).to(device)
    cv_raw = torch.cat(cv_raw, dim=0).to(device)
    target_latlon = torch.cat(target_latlon, dim=0).to(device)
    origin = torch.cat(origin, dim=0).to(device)

    step = max(args.calibration_step, 1e-3)
    grid = torch.arange(0.0, 1.0 + step * 0.5, step, device=device).clamp_max(1.0)
    alphas = []
    for lead in range(metadata["pred_len"]):
        best_alpha = grid[-1]
        best_err = float("inf")
        for alpha in grid:
            pred_raw = alpha * model_raw[:, lead : lead + 1] + (1.0 - alpha) * cv_raw[:, lead : lead + 1]
            pred_latlon = disp_to_latlon(pred_raw, origin)
            err = haversine_km(pred_latlon, target_latlon[:, lead : lead + 1]).mean().item()
            if err < best_err:
                best_err = err
                best_alpha = alpha
        alphas.append(float(best_alpha.item()))
    return torch.tensor(alphas, device=device, dtype=torch.float32)


@torch.no_grad()
def evaluate_baselines(loader, metadata, device, disable_progress=False):
    persistence = defaultdict(float)
    constant_velocity = defaultdict(float)
    lead_hours = metadata_lead_hours(metadata)
    lead_steps = metadata_lead_steps(metadata, device=device, dtype=torch.float32)
    for batch in tqdm(loader, desc="Baselines", leave=False, disable=disable_progress):
        batch = move_to_device(batch, device)
        if batch.get("future_year") is not None and getattr(evaluate_baselines, "year_filter", 0):
            mask = batch["future_year"][:, 0] == getattr(evaluate_baselines, "year_filter", 0)
            if not mask.any():
                continue
            batch = {key: value[mask] if torch.is_tensor(value) and value.shape[0] == mask.shape[0] else value for key, value in batch.items()}
        bsz, pred_len = batch["future_latlon"].shape[:2]
        origin = batch["origin"]
        pred_persist = origin.unsqueeze(1).expand(bsz, pred_len, 2)
        add_errors(persistence, pred_persist, batch["future_latlon"], lead_hours)

        prev = batch["history_latlon"][:, -2]
        step_disp = latlon_to_disp(origin, prev)
        pred_disp = step_disp.unsqueeze(1) * lead_steps.view(1, pred_len, 1)
        pred_cv = disp_to_latlon(pred_disp, origin)
        add_errors(constant_velocity, pred_cv, batch["future_latlon"], lead_hours)
    return {"persistence": finalize(persistence), "constant_velocity": finalize(constant_velocity)}


@torch.no_grad()
def evaluate_model(model, loader, metadata, train_args, args, device, cv_blend_alpha=None):
    model.eval()
    target_mean, target_std = stats_tensors(metadata, device)
    sums = defaultdict(float)
    lead_hours = metadata_lead_hours(metadata)
    for batch in tqdm(loader, desc="Model", leave=False, disable=args.disable_progress):
        batch = move_to_device(batch, device)
        if args.year_filter:
            mask = batch["future_year"][:, 0] == args.year_filter
            if not mask.any():
                continue
            batch = {key: value[mask] if torch.is_tensor(value) and value.shape[0] == mask.shape[0] else value for key, value in batch.items()}
        features = model(batch["x_num"], batch["x_text"], batch["analog_pos"], batch["analog_neg"])
        if train_args.decoder == "deterministic":
            pred_norm = deterministic_prediction(features, batch, metadata, train_args, device)
        else:
            base = flow_base_norm(features, batch, metadata, train_args, device)
            samples_norm = model.sample_flow(
                features["context"],
                num_samples=args.num_samples,
                ode_steps=args.ode_steps,
                noise_std=args.noise_std,
            )
            samples_norm = base.unsqueeze(1) + samples_norm
            pred_norm = samples_norm.mean(dim=1)

            bsz, num_samples, pred_len, _ = samples_norm.shape
            sample_raw = denormalize_disp(
                samples_norm.reshape(bsz * num_samples, pred_len, 2),
                target_mean,
                target_std,
            )
            sample_latlon = disp_to_latlon(
                sample_raw,
                batch["origin"].repeat_interleave(num_samples, dim=0),
            ).reshape(bsz, num_samples, pred_len, 2)
            sample_err = haversine_km(sample_latlon, batch["future_latlon"].unsqueeze(1))
            sums["minade"] += sample_err.mean(dim=-1).min(dim=1).values.sum().item()
            sums["minfde"] += sample_err[:, :, -1].min(dim=1).values.sum().item()
            scores = model.score(
                features["context"].repeat_interleave(num_samples, dim=0),
                samples_norm.reshape(bsz * num_samples, pred_len, 2),
            ).reshape(bsz, num_samples)
            top_idx = scores.argmax(dim=1)
            top_err = sample_err[torch.arange(bsz, device=device), top_idx]
            sums["topade"] += top_err.mean(dim=-1).sum().item()
            sums["topfde"] += top_err[:, -1].sum().item()

        pred_raw = denormalize_disp(pred_norm, target_mean, target_std)
        if cv_blend_alpha is not None:
            cv_raw = constant_velocity_future_disp(
                batch["history_latlon"],
                metadata["pred_len"],
                metadata_lead_steps(metadata, device=batch["history_latlon"].device, dtype=batch["history_latlon"].dtype),
            )
            alpha = cv_blend_alpha.to(device=device, dtype=pred_raw.dtype).view(1, -1, 1)
            pred_raw = alpha * pred_raw + (1.0 - alpha) * cv_raw
        pred_latlon = disp_to_latlon(pred_raw, batch["origin"])
        add_errors(sums, pred_latlon, batch["future_latlon"], lead_hours)
    return finalize(sums)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = Namespace(**checkpoint["args"])
    data_dir = args.data_dir or train_args.data_dir
    metadata = load_metadata(data_dir)

    dataset = TyphoPlusDataset(os.path.join(data_dir, args.split))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = make_model(train_args, metadata, device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        print(f"Checkpoint loaded with missing={list(missing)} unexpected={list(unexpected)}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Split: {args.split} ({len(dataset)} samples)")

    evaluate_baselines.year_filter = args.year_filter
    results = evaluate_baselines(loader, metadata, device, disable_progress=args.disable_progress)
    if args.calibrate_cv_blend:
        calibration_ds = TyphoPlusDataset(os.path.join(data_dir, args.calibration_split))
        calibration_loader = DataLoader(
            calibration_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        alpha = calibrate_cv_blend(model, calibration_loader, metadata, train_args, args, device)
        results["model_raw"] = evaluate_model(model, loader, metadata, train_args, args, device)
        results["model_cv_calibrated"] = evaluate_model(model, loader, metadata, train_args, args, device, cv_blend_alpha=alpha)
        results["calibration"] = {
            "type": "lead_time_cv_model_blend",
            "split": args.calibration_split,
            "step": args.calibration_step,
            "alpha_model_weight": [float(x) for x in alpha.detach().cpu().tolist()],
        }
    else:
        results["model"] = evaluate_model(model, loader, metadata, train_args, args, device)

    print("\nEvaluation summary")
    for name, metrics in results.items():
        if isinstance(metrics, dict) and "ade" in metrics:
            print(f"{name}: {format_metrics(metrics)}")
        else:
            print(f"{name}: {metrics}")

    output_json = args.output_json
    if not output_json:
        output_json = os.path.join(os.path.dirname(args.checkpoint), f"eval_{args.split}.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {output_json}")


if __name__ == "__main__":
    main()
