import argparse
import json
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.TyphoFormerPlus import TyphoFormerPlus
from typhoformerpp_common import (
    TyphoPlusDataset,
    batch_metrics,
    constant_velocity_future_disp,
    denormalize_disp,
    disp_to_latlon,
    format_metrics,
    haversine_km,
    load_metadata,
    metadata_lead_hours,
    metadata_lead_steps,
    move_to_device,
    normalize_disp,
    seed_everything,
    smoothness_loss,
    stats_tensors,
    weighted_haversine_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_pp")
    parser.add_argument("--save-dir", default="checkpoints_pp")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--decoder", choices=["deterministic", "flow"], default="deterministic")
    parser.add_argument("--variant", choices=["numeric", "typhoformer", "plus"], default="plus")
    parser.add_argument("--residual", choices=["none", "cv", "dual"], default="none")
    parser.add_argument("--flow-base", choices=["zero", "cv", "det"], default="zero")
    parser.add_argument("--loss", choices=["mse", "weighted_haversine"], default="mse")
    parser.add_argument(
        "--select-key",
        choices=[
            "auto",
            "ade",
            "fde",
            "err6",
            "err12",
            "err18",
            "err24",
            "err30",
            "err36",
            "err42",
            "err48",
            "err54",
            "err60",
            "err66",
            "err72",
            "minade",
            "minfde",
        ],
        default="auto",
    )
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--no-positive-analog", action="store_true")
    parser.add_argument("--no-negative-analog", action="store_true")
    parser.add_argument(
        "--fuse-negative-analog",
        action="store_true",
        help="Fuse negative analog embeddings as an input modality. Disabled by default to keep inference target-free.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--modality-dropout", type=float, default=0.2)
    parser.add_argument("--lambda-align", type=float, default=0.05)
    parser.add_argument("--lambda-rank", type=float, default=0.10)
    parser.add_argument("--lambda-smooth", type=float, default=0.02)
    parser.add_argument("--lambda-direct-aux", type=float, default=0.0)
    parser.add_argument("--lambda-cv-aux", type=float, default=0.0)
    parser.add_argument("--lambda-gate-prior", type=float, default=0.0)
    parser.add_argument("--lambda-short-residual-anchor", type=float, default=0.0)
    parser.add_argument("--short-anchor-steps", type=int, default=4)
    parser.add_argument("--horizon-weight-24", type=float, default=0.5)
    parser.add_argument("--horizon-weight-48", type=float, default=1.0)
    parser.add_argument("--horizon-weight-72", type=float, default=1.5)
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument(
        "--disable-gate-prior",
        action="store_true",
        help="Use zero prior bias for the dual direct/CV-residual gate.",
    )
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--ode-steps", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def make_model(args, metadata, device):
    use_text = args.variant in {"typhoformer", "plus"} and not args.no_text
    use_positive_analog = args.variant == "plus" and not args.no_positive_analog
    use_negative_analog = args.variant == "plus" and not args.no_negative_analog
    use_negative_context = use_negative_analog and args.fuse_negative_analog
    use_analog = use_positive_analog or use_negative_context
    model = TyphoFormerPlus(
        input_dim=metadata["input_dim"],
        text_dim=metadata["text_dim"],
        pred_len=metadata["pred_len"],
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        modality_dropout=args.modality_dropout,
        decoder_type=args.decoder,
        use_text=use_text,
        use_analog=use_analog,
        use_positive_analog=use_positive_analog,
        use_negative_context=use_negative_context,
        max_input_len=metadata["input_len"],
    ).to(device)
    if getattr(args, "disable_gate_prior", False):
        model.dual_gate_prior.zero_()
    model.lead_steps = metadata.get("lead_steps", list(range(1, int(metadata["pred_len"]) + 1)))
    return model


def rank_loss(model, context, target, analog_neg, margin):
    score_pos = model.score(context, target)
    score_neg = model.score(context, analog_neg[:, 0])
    return F.relu(margin - score_pos + score_neg).mean()


def effective_lambdas(args):
    lambda_align = args.lambda_align if args.variant == "plus" else 0.0
    lambda_rank = args.lambda_rank if args.variant == "plus" and not args.no_negative_analog else 0.0
    return lambda_align, lambda_rank


def cv_base_norm(batch, metadata, device):
    target_mean, target_std = stats_tensors(metadata, device)
    cv_raw = constant_velocity_future_disp(
        batch["history_latlon"],
        metadata["pred_len"],
        metadata_lead_steps(metadata, device=batch["history_latlon"].device, dtype=batch["history_latlon"].dtype),
    )
    return normalize_disp(cv_raw, target_mean, target_std)


def deterministic_prediction(features, batch, metadata, args, device):
    direct = features["direct_pred"]
    base = cv_base_norm(batch, metadata, device)
    cv_res = base + features["residual_gate"] * features["residual_pred"]
    if args.residual == "dual":
        return features["dual_gate"] * cv_res + (1.0 - features["dual_gate"]) * direct
    if args.residual == "cv":
        return cv_res
    return direct


def flow_base_norm(model, features, batch, metadata, args, device):
    if args.flow_base == "cv":
        return cv_base_norm(batch, metadata, device)
    if args.flow_base == "det":
        base_args = argparse.Namespace(**vars(args))
        base_args.residual = "cv" if args.residual == "cv" else "none"
        return deterministic_prediction(features, batch, metadata, base_args, device).detach()
    return batch["target"].new_zeros(batch["target"].shape)


def trajectory_task_loss(pred_norm, target_norm, batch, metadata, args, device):
    if args.loss == "mse":
        return F.mse_loss(pred_norm, target_norm)
    target_mean, target_std = stats_tensors(metadata, device)
    pred_raw = denormalize_disp(pred_norm, target_mean, target_std)
    pred_latlon = disp_to_latlon(pred_raw, batch["origin"])
    return weighted_haversine_loss(
        pred_latlon,
        batch["future_latlon"],
        lead_weights=[
            (4, args.horizon_weight_24),
            (8, args.horizon_weight_48),
            (12, args.horizon_weight_72),
        ],
    )


def prefix_batch(batch, steps):
    out = dict(batch)
    for key in ["target", "future_latlon"]:
        out[key] = batch[key][:, :steps]
    return out


def dual_auxiliary_losses(model, features, batch, metadata, args, device):
    if args.residual != "dual":
        zero = batch["target"].new_tensor(0.0)
        return zero, zero, zero, zero
    base = cv_base_norm(batch, metadata, device)
    residual_delta = features["residual_gate"] * features["residual_pred"]
    cv_res = base + residual_delta
    direct_aux = trajectory_task_loss(features["direct_pred"], batch["target"], batch, metadata, args, device)
    short_steps = min(8, batch["target"].shape[1])
    cv_aux = trajectory_task_loss(
        cv_res[:, :short_steps],
        batch["target"][:, :short_steps],
        prefix_batch(batch, short_steps),
        metadata,
        args,
        device,
    )
    prior = model.dual_gate_prior.to(device=device, dtype=features["dual_gate"].dtype)
    if prior.shape[1] < features["dual_gate"].shape[1]:
        prior = torch.cat([prior, prior[:, -1:].expand(-1, features["dual_gate"].shape[1] - prior.shape[1])], dim=1)
    prior = torch.sigmoid(prior[:, : features["dual_gate"].shape[1]]).unsqueeze(-1)
    gate_prior = F.mse_loss(features["dual_gate"], prior.expand_as(features["dual_gate"]))
    anchor_steps = min(max(args.short_anchor_steps, 0), residual_delta.shape[1])
    if anchor_steps > 0:
        residual_anchor = torch.mean(residual_delta[:, :anchor_steps] * residual_delta[:, :anchor_steps])
    else:
        residual_anchor = batch["target"].new_tensor(0.0)
    return direct_aux, cv_aux, gate_prior, residual_anchor


def train_one_epoch(model, loader, optimizer, scaler, args, metadata, device):
    model.train()
    totals = defaultdict(float)
    seen = 0
    autocast_enabled = args.amp and device.type == "cuda"
    progress = tqdm(loader, desc="Training", leave=False, disable=args.disable_progress)
    for batch in progress:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            features = model(batch["x_num"], batch["x_text"], batch["analog_pos"], batch["analog_neg"])
            align = TyphoFormerPlus.alignment_loss(features)
            lambda_align, lambda_rank = effective_lambdas(args)
            if lambda_rank > 0:
                rank = rank_loss(model, features["context"], batch["target"], batch["analog_neg"], args.rank_margin)
            else:
                rank = batch["target"].new_tensor(0.0)

            if args.decoder == "deterministic":
                pred = deterministic_prediction(features, batch, metadata, args, device)
                task = trajectory_task_loss(pred, batch["target"], batch, metadata, args, device)
                smooth = smoothness_loss(pred)
                direct_aux, cv_aux, gate_prior, residual_anchor = dual_auxiliary_losses(model, features, batch, metadata, args, device)
                loss = (
                    task
                    + args.lambda_smooth * smooth
                    + lambda_rank * rank
                    + lambda_align * align
                    + args.lambda_direct_aux * direct_aux
                    + args.lambda_cv_aux * cv_aux
                    + args.lambda_gate_prior * gate_prior
                    + args.lambda_short_residual_anchor * residual_anchor
                )
            else:
                base = flow_base_norm(model, features, batch, metadata, args, device)
                target_residual = batch["target"] - base
                noise = torch.randn_like(batch["target"]) * args.noise_std
                t = torch.rand(batch["target"].shape[0], device=device)
                y_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * target_residual
                v_target = target_residual - noise
                v_pred = model.flow_velocity(y_t, t, features["context"])
                task = F.mse_loss(v_pred, v_target)
                smooth = task.new_tensor(0.0)
                loss = task + lambda_rank * rank + lambda_align * align

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bsz = batch["target"].shape[0]
        seen += bsz
        totals["loss"] += loss.item() * bsz
        totals["task"] += task.item() * bsz
        totals["rank"] += rank.item() * bsz
        totals["align"] += align.item() * bsz
        totals["smooth"] += smooth.item() * bsz
        if args.decoder == "deterministic":
            totals["direct_aux"] += direct_aux.item() * bsz
            totals["cv_aux"] += cv_aux.item() * bsz
            totals["gate_prior"] += gate_prior.item() * bsz
            totals["residual_anchor"] += residual_anchor.item() * bsz
        progress.set_postfix(loss=f"{totals['loss'] / seen:.4f}")

    return {key: value / seen for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader, metadata, args, device, split_name="Val"):
    model.eval()
    target_mean, target_std = stats_tensors(metadata, device)
    sums = defaultdict(float)
    seen = 0
    losses = defaultdict(float)

    for batch in tqdm(loader, desc=split_name, leave=False, disable=args.disable_progress):
        batch = move_to_device(batch, device)
        features = model(batch["x_num"], batch["x_text"], batch["analog_pos"], batch["analog_neg"])
        lambda_align, lambda_rank = effective_lambdas(args)
        if lambda_rank > 0:
            rank = rank_loss(model, features["context"], batch["target"], batch["analog_neg"], args.rank_margin)
        else:
            rank = batch["target"].new_tensor(0.0)
        align = TyphoFormerPlus.alignment_loss(features)

        if args.decoder == "deterministic":
            pred_norm = deterministic_prediction(features, batch, metadata, args, device)
            task = trajectory_task_loss(pred_norm, batch["target"], batch, metadata, args, device)
            direct_aux, cv_aux, gate_prior, residual_anchor = dual_auxiliary_losses(model, features, batch, metadata, args, device)
        else:
            base = flow_base_norm(model, features, batch, metadata, args, device)
            samples_norm = model.sample_flow(
                features["context"],
                num_samples=args.val_samples,
                ode_steps=args.ode_steps,
                noise_std=args.noise_std,
            )
            samples_norm = base.unsqueeze(1) + samples_norm
            pred_norm = samples_norm.mean(dim=1)
            task = trajectory_task_loss(pred_norm, batch["target"], batch, metadata, args, device)
            direct_aux = task.new_tensor(0.0)
            cv_aux = task.new_tensor(0.0)
            gate_prior = task.new_tensor(0.0)
            residual_anchor = task.new_tensor(0.0)

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

        loss = (
            task
            + lambda_rank * rank
            + lambda_align * align
            + args.lambda_direct_aux * direct_aux
            + args.lambda_cv_aux * cv_aux
            + args.lambda_gate_prior * gate_prior
            + args.lambda_short_residual_anchor * residual_anchor
        )
        pred_raw = denormalize_disp(pred_norm, target_mean, target_std)
        pred_latlon = disp_to_latlon(pred_raw, batch["origin"])
        errors = haversine_km(pred_latlon, batch["future_latlon"])

        bsz = batch["target"].shape[0]
        seen += bsz
        losses["loss"] += loss.item() * bsz
        losses["task"] += task.item() * bsz
        losses["rank"] += rank.item() * bsz
        losses["align"] += align.item() * bsz
        sums["ade"] += errors.mean(dim=1).sum().item()
        sums["fde"] += errors[:, -1].sum().item()
        for idx, hour in enumerate(metadata_lead_hours(metadata)):
            if idx < errors.shape[1]:
                sums[f"err{hour}"] += errors[:, idx].sum().item()

    metrics = {key: value / seen for key, value in sums.items()}
    metrics.update({key: value / seen for key, value in losses.items()})
    return metrics


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = load_metadata(args.data_dir)
    run_name = args.run_name or f"typhoformerpp_{args.decoder}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Run directory: {run_dir}")
    print(
        f"Data: input_len={metadata['input_len']} pred_len={metadata['pred_len']} train={metadata['splits']['train']} "
        f"| variant={args.variant} decoder={args.decoder} residual={args.residual} flow_base={args.flow_base} loss={args.loss}"
    )

    train_ds = TyphoPlusDataset(os.path.join(args.data_dir, "train"))
    val_ds = TyphoPlusDataset(os.path.join(args.data_dir, "val"))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = make_model(args, metadata, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    config = {"args": vars(args), "metadata": metadata}
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    select_key = args.select_key
    if select_key == "auto":
        select_key = "minade" if args.decoder == "flow" else "ade"
    best_score = float("inf")
    epochs_without_improvement = 0
    best_path = os.path.join(run_dir, "best_model.pt")
    log_path = os.path.join(run_dir, "train_log.jsonl")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, args, metadata, device)
        val_metrics = evaluate(model, val_loader, metadata, args, device, split_name="Validation")
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_stats,
            "val": val_metrics,
            "seconds": time.time() - epoch_start,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| train loss={train_stats['loss']:.4f} task={train_stats['task']:.4f} "
            f"| val loss={val_metrics['loss']:.4f} "
            f"| {format_metrics(val_metrics)} "
            f"| {row['seconds']:.1f}s"
        )

        score = val_metrics.get(select_key, val_metrics["ade"])
        if score < best_score:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "metadata": metadata,
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path} ({select_key}={score:.3f})")
        else:
            epochs_without_improvement += 1
            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping after {args.early_stop_patience} epochs without {select_key} improvement.")
                break

    torch.save(
        {"model_state": model.state_dict(), "args": vars(args), "metadata": metadata, "epoch": epoch},
        os.path.join(run_dir, "last_model.pt"),
    )
    print(f"Training finished in {(time.time() - start) / 60.0:.2f} min. Best val {select_key.upper()}={best_score:.3f} km")


if __name__ == "__main__":
    main()
