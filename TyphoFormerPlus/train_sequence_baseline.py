import argparse
import json
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from typhoformerpp_common import (
    TyphoPlusDataset,
    denormalize_disp,
    disp_to_latlon,
    format_metrics,
    haversine_km,
    load_metadata,
    metadata_lead_hours,
    move_to_device,
    seed_everything,
    stats_tensors,
    weighted_haversine_loss,
)


class RecurrentForecaster(nn.Module):
    def __init__(self, input_dim, pred_len, hidden_dim=192, num_layers=2, dropout=0.1, cell="gru"):
        super().__init__()
        cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = cls(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * 2),
        )
        self.pred_len = pred_len

    def forward(self, x):
        out, _ = self.rnn(x)
        h = out[:, -1]
        return self.head(h).reshape(x.shape[0], self.pred_len, 2)


class MixerBlock(nn.Module):
    def __init__(self, seq_len, hidden_dim, dropout):
        super().__init__()
        self.time_norm = nn.LayerNorm(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len * 2, seq_len),
        )
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x):
        y = self.time_norm(x).transpose(1, 2)
        x = x + self.time_mlp(y).transpose(1, 2)
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class TSMixerForecaster(nn.Module):
    def __init__(self, input_dim, input_len, pred_len, hidden_dim=192, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([MixerBlock(input_len, hidden_dim, dropout) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * 2),
        )
        self.pred_len = pred_len

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = h.mean(dim=1)
        return self.head(h).reshape(x.shape[0], self.pred_len, 2)


class InformerStyleForecaster(nn.Module):
    def __init__(self, input_dim, input_len, pred_len, hidden_dim=192, num_layers=2, dropout=0.1, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.distill = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * 2),
        )
        self.pred_len = pred_len

    def forward(self, x):
        h = self.input_proj(x) + self.pos_embed[:, : x.shape[1]]
        h = self.encoder(h)
        h = h + self.distill(h.transpose(1, 2)).transpose(1, 2)
        h = h[:, -1]
        return self.head(h).reshape(x.shape[0], self.pred_len, 2)


class MovingAverage(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        pad = (self.kernel_size - 1) // 2
        y = x.transpose(1, 2)
        y = F.pad(y, (pad, pad), mode="replicate")
        y = F.avg_pool1d(y, kernel_size=self.kernel_size, stride=1)
        return y.transpose(1, 2)


class AutoformerStyleForecaster(nn.Module):
    def __init__(self, input_dim, input_len, pred_len, hidden_dim=192, num_layers=2, dropout=0.1, num_heads=4):
        super().__init__()
        self.decomp = MovingAverage(kernel_size=3)
        self.seasonal_proj = nn.Linear(input_dim, hidden_dim)
        self.trend_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * 2),
        )
        self.pred_len = pred_len

    def forward(self, x):
        trend = self.decomp(x)
        seasonal = x - trend
        seasonal_h = self.seasonal_proj(seasonal) + self.pos_embed[:, : x.shape[1]]
        seasonal_h = self.encoder(seasonal_h).mean(dim=1)
        trend_h = self.trend_proj(trend).mean(dim=1)
        h = torch.cat([seasonal_h, trend_h], dim=-1)
        return self.head(h).reshape(x.shape[0], self.pred_len, 2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--save-dir", default="checkpoints_official_12to1_sequence_baselines")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--model", choices=["gru", "lstm", "informer", "autoformer", "tsmixer"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def make_model(args, metadata, device):
    if args.model in {"gru", "lstm"}:
        return RecurrentForecaster(
            metadata["input_dim"],
            metadata["pred_len"],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            cell=args.model,
        ).to(device)
    if args.model == "informer":
        return InformerStyleForecaster(
            metadata["input_dim"],
            metadata["input_len"],
            metadata["pred_len"],
            hidden_dim=args.hidden_dim,
            num_layers=max(args.num_layers, 1),
            dropout=args.dropout,
        ).to(device)
    if args.model == "autoformer":
        return AutoformerStyleForecaster(
            metadata["input_dim"],
            metadata["input_len"],
            metadata["pred_len"],
            hidden_dim=args.hidden_dim,
            num_layers=max(args.num_layers, 1),
            dropout=args.dropout,
        ).to(device)
    return TSMixerForecaster(
        metadata["input_dim"],
        metadata["input_len"],
        metadata["pred_len"],
        hidden_dim=args.hidden_dim,
        num_layers=max(args.num_layers, 1),
        dropout=args.dropout,
    ).to(device)


def task_loss(pred_norm, batch, metadata, device):
    target_mean, target_std = stats_tensors(metadata, device)
    pred_raw = denormalize_disp(pred_norm, target_mean, target_std)
    pred_latlon = disp_to_latlon(pred_raw, batch["origin"])
    return weighted_haversine_loss(pred_latlon, batch["future_latlon"], lead_weights=[])


def train_one_epoch(model, loader, optimizer, args, metadata, device):
    model.train()
    totals = defaultdict(float)
    seen = 0
    for batch in tqdm(loader, desc="Training", leave=False, disable=args.disable_progress):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch["x_num"])
        loss = task_loss(pred, batch, metadata, device)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        bsz = batch["target"].shape[0]
        totals["loss"] += loss.item() * bsz
        seen += bsz
    return {key: value / seen for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader, metadata, args, device):
    model.eval()
    target_mean, target_std = stats_tensors(metadata, device)
    sums = defaultdict(float)
    count = 0
    lead_hours = metadata_lead_hours(metadata)
    for batch in tqdm(loader, desc="Eval", leave=False, disable=args.disable_progress):
        batch = move_to_device(batch, device)
        pred_norm = model(batch["x_num"])
        pred_raw = denormalize_disp(pred_norm, target_mean, target_std)
        pred_latlon = disp_to_latlon(pred_raw, batch["origin"])
        errors = haversine_km(pred_latlon, batch["future_latlon"])
        mae = torch.mean(torch.abs(pred_latlon - batch["future_latlon"]), dim=-1)
        bsz = errors.shape[0]
        count += bsz
        sums["ade"] += errors.mean(dim=1).sum().item()
        sums["fde"] += errors[:, -1].sum().item()
        sums["mae"] += mae.mean(dim=1).sum().item()
        for idx, hour in enumerate(lead_hours):
            sums[f"err{hour}"] += errors[:, idx].sum().item()
            sums[f"mae{hour}"] += mae[:, idx].sum().item()
    return {key: value / count for key, value in sums.items()}


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = load_metadata(args.data_dir)
    run_name = args.run_name or f"{args.model}_s{args.seed}"
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_ds = TyphoPlusDataset(os.path.join(args.data_dir, "train"))
    val_ds = TyphoPlusDataset(os.path.join(args.data_dir, "val"))
    test_ds = TyphoPlusDataset(os.path.join(args.data_dir, "test"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_model(args, metadata, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    best_score = float("inf")
    stale = 0
    best_path = os.path.join(run_dir, "best_model.pt")
    log_path = os.path.join(run_dir, "train_log.jsonl")
    config = {"args": vars(args), "metadata": metadata}
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Run directory: {run_dir}")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, args, metadata, device)
        val_metrics = evaluate(model, val_loader, metadata, args, device)
        scheduler.step()
        row = {"epoch": epoch, "train": train_stats, "val": val_metrics, "seconds": time.time() - t0}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train loss={train_stats['loss']:.4f} "
            f"| val {format_metrics(val_metrics)} | {row['seconds']:.1f}s"
        )
        score = val_metrics["fde"]
        if score < best_score:
            best_score = score
            stale = 0
            torch.save({"model_state": model.state_dict(), "args": vars(args), "metadata": metadata, "epoch": epoch, "val_metrics": val_metrics}, best_path)
            print(f"Saved best checkpoint: {best_path} (fde={score:.3f})")
        else:
            stale += 1
            if args.early_stop_patience > 0 and stale >= args.early_stop_patience:
                print(f"Early stopping after {args.early_stop_patience} epochs without fde improvement.")
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, metadata, args, device)
    with open(os.path.join(run_dir, "eval_test.json"), "w", encoding="utf-8") as f:
        json.dump({"model": test_metrics}, f, indent=2)
    print(f"Test: {format_metrics(test_metrics)}")
    print(f"Finished in {(time.time() - start) / 60.0:.2f} min.")


if __name__ == "__main__":
    main()
