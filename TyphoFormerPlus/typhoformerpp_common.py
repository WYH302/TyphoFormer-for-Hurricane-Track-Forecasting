import glob
import json
import os
import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


EARTH_RADIUS_KM = 6371.0


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_metadata(data_dir: str) -> Dict:
    with open(os.path.join(data_dir, "metadata.json"), "r", encoding="utf-8") as f:
        return json.load(f)


class TyphoPlusDataset(Dataset):
    def __init__(self, data_dir: str):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not self.files:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        return {
            "x_num": torch.tensor(data["input_num"], dtype=torch.float32),
            "x_text": torch.tensor(data["input_text"], dtype=torch.float32),
            "target": torch.tensor(data["target_disp"], dtype=torch.float32),
            "target_raw": torch.tensor(data["target_disp_raw_km"], dtype=torch.float32),
            "future_latlon": torch.tensor(data["future_latlon"], dtype=torch.float32),
            "origin": torch.tensor(data["origin"], dtype=torch.float32),
            "history_latlon": torch.tensor(data["history_latlon"], dtype=torch.float32),
            "analog_pos": torch.tensor(data["analog_pos"], dtype=torch.float32),
            "analog_neg": torch.tensor(data["analog_neg"], dtype=torch.float32),
            "storm_id": data["storm_id"],
            "storm_year": int(data["storm_year"]) if "storm_year" in data else -1,
            "future_year": torch.tensor(data["future_year"], dtype=torch.long)
            if "future_year" in data
            else torch.full((data["future_latlon"].shape[0],), -1, dtype=torch.long),
            "window": int(data["window"]),
        }


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def stats_tensors(metadata: Dict, device: torch.device):
    mean = torch.tensor(metadata["stats"]["target_mean"], device=device, dtype=torch.float32)
    std = torch.tensor(metadata["stats"]["target_std"], device=device, dtype=torch.float32)
    return mean, std


def metadata_lead_steps(metadata: Dict, device=None, dtype=torch.float32) -> torch.Tensor:
    values = metadata.get("lead_steps")
    if values is None:
        values = list(range(1, int(metadata["pred_len"]) + 1))
    return torch.tensor(values, device=device, dtype=dtype)


def metadata_lead_hours(metadata: Dict):
    values = metadata.get("lead_hours")
    if values is None:
        interval = int(metadata.get("interval_hours", 6))
        values = [interval * (i + 1) for i in range(int(metadata["pred_len"]))]
    return [int(v) for v in values]


def denormalize_disp(y_norm: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    return y_norm * target_std.view(1, 1, 2) + target_mean.view(1, 1, 2)


def disp_to_latlon(disp_km: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    lat0_deg = origin[:, 0].unsqueeze(1)
    lon0_deg = origin[:, 1].unsqueeze(1)
    lat0_rad = torch.deg2rad(lat0_deg)
    lat = lat0_deg + torch.rad2deg(disp_km[..., 1] / EARTH_RADIUS_KM)
    lon = lon0_deg + torch.rad2deg(disp_km[..., 0] / (EARTH_RADIUS_KM * torch.cos(lat0_rad).clamp_min(1e-6)))
    lon = (lon + 180.0) % 360.0 - 180.0
    return torch.stack([lat, lon], dim=-1)


def latlon_to_disp(latlon: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    lat = torch.deg2rad(latlon[..., 0])
    lon = latlon[..., 1]
    lat0 = torch.deg2rad(origin[..., 0])
    lon0 = origin[..., 1]
    dlat = lat - lat0
    dlon_deg = (lon - lon0 + 180.0) % 360.0 - 180.0
    dlon = torch.deg2rad(dlon_deg)
    dx = EARTH_RADIUS_KM * torch.cos(lat0) * dlon
    dy = EARTH_RADIUS_KM * dlat
    return torch.stack([dx, dy], dim=-1)


def constant_velocity_future_disp(history_latlon: torch.Tensor, pred_len: int, lead_steps=None) -> torch.Tensor:
    """Future local displacement from the last observed point by 6h constant velocity."""
    if history_latlon.shape[1] < 2:
        return history_latlon.new_zeros(history_latlon.shape[0], pred_len, 2)
    prev = history_latlon[:, -2]
    origin = history_latlon[:, -1]
    step_disp = latlon_to_disp(origin, prev)
    if lead_steps is None:
        multipliers = torch.arange(1, pred_len + 1, device=history_latlon.device, dtype=history_latlon.dtype)
    else:
        multipliers = torch.as_tensor(lead_steps, device=history_latlon.device, dtype=history_latlon.dtype)
    return step_disp.unsqueeze(1) * multipliers.view(1, pred_len, 1)


def normalize_disp(y_raw: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    return (y_raw - target_mean.view(1, 1, 2)) / target_std.view(1, 1, 2)


def haversine_km(pred_latlon: torch.Tensor, target_latlon: torch.Tensor) -> torch.Tensor:
    lat1 = torch.deg2rad(pred_latlon[..., 0])
    lon1 = torch.deg2rad(pred_latlon[..., 1])
    lat2 = torch.deg2rad(target_latlon[..., 0])
    lon2 = torch.deg2rad(target_latlon[..., 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2
    a = torch.clamp(a, 0.0, 1.0)
    return EARTH_RADIUS_KM * 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(1.0 - a))


def smoothness_loss(y_norm: torch.Tensor) -> torch.Tensor:
    if y_norm.shape[1] < 3:
        return y_norm.sum() * 0.0
    accel = y_norm[:, 2:] - 2.0 * y_norm[:, 1:-1] + y_norm[:, :-2]
    return torch.mean(accel * accel)


def weighted_haversine_loss(pred_latlon: torch.Tensor, target_latlon: torch.Tensor, lead_weights=None) -> torch.Tensor:
    """ADE plus selected lead-time losses, scaled to thousands of km."""
    errors = haversine_km(pred_latlon, target_latlon)
    loss = errors.mean()
    if lead_weights is None:
        lead_weights = [(4, 0.5), (8, 1.0), (12, 1.5)]
    for step, weight in lead_weights:
        if errors.shape[1] >= step:
            loss = loss + weight * errors[:, step - 1].mean()
    return loss / 1000.0


def batch_metrics(pred_latlon: torch.Tensor, target_latlon: torch.Tensor) -> Dict[str, float]:
    errors = haversine_km(pred_latlon, target_latlon)
    metrics = {
        "ade": errors.mean().item(),
        "fde": errors[:, -1].mean().item(),
    }
    for step in range(1, errors.shape[1] + 1):
        metrics[f"err{step * 6}"] = errors[:, step - 1].mean().item()
    return metrics


def add_lead_error_sums(sums: Dict[str, float], errors: torch.Tensor, lead_hours):
    for idx, hour in enumerate(lead_hours):
        if idx < errors.shape[1]:
            sums[f"err{int(hour)}"] += errors[:, idx].sum().item()


def format_metrics(metrics: Dict[str, float]) -> str:
    lead_keys = sorted(
        [key for key in metrics if key.startswith("err") and key[3:].isdigit()],
        key=lambda key: int(key[3:]),
    )
    mae_keys = sorted(
        [key for key in metrics if key.startswith("mae") and key[3:].isdigit()],
        key=lambda key: int(key[3:]),
    )
    keys = ["mae", "ade", "fde"] + mae_keys + lead_keys + ["topade", "topfde", "minade", "minfde"]
    parts = []
    for key in keys:
        if key in metrics:
            unit = "" if key.startswith("mae") else "km"
            parts.append(f"{key.upper()}={metrics[key]:.3f}{unit}")
    return " | ".join(parts)
