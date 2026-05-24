import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from typhoformerpp_common import (
    TyphoPlusDataset,
    disp_to_latlon,
    haversine_km,
    latlon_to_disp,
    load_metadata,
    metadata_lead_hours,
    metadata_lead_steps,
    move_to_device,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_official_pp_12to4")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--year-filter", type=int, default=0)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def add_errors(sums, pred_latlon, target_latlon, lead_hours):
    errors = haversine_km(pred_latlon, target_latlon)
    mae = torch.mean(torch.abs(pred_latlon - target_latlon), dim=-1)
    bsz = errors.shape[0]
    sums["count"] += bsz
    sums["mae"] += mae.mean(dim=1).sum().item()
    sums["ade"] += errors.mean(dim=1).sum().item()
    sums["fde"] += errors[:, -1].sum().item()
    for idx, hour in enumerate(lead_hours):
        if idx < errors.shape[1]:
            sums[f"mae{int(hour)}"] += mae[:, idx].sum().item()
            sums[f"err{int(hour)}"] += errors[:, idx].sum().item()


def finalize(sums):
    count = sums.pop("count")
    return {key: value / count for key, value in sums.items()}


def dataset_arrays(data_dir):
    ds = TyphoPlusDataset(data_dir)
    xs, ys = [], []
    for idx in range(len(ds)):
        item = ds[idx]
        x_num = item["x_num"].numpy().reshape(-1)
        hist = item["history_latlon"].numpy()
        origin = item["origin"].numpy()
        hist_disp = latlon_to_disp(
            torch.tensor(hist).unsqueeze(0),
            torch.tensor(origin).view(1, 1, 2),
        ).numpy().reshape(-1)
        features = np.concatenate([x_num, hist_disp / 500.0], axis=0)
        xs.append(features.astype(np.float32))
        ys.append(item["target_raw"].numpy().reshape(-1).astype(np.float32))
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def fit_cliper_ridge(data_dir):
    x_train, y_train = dataset_arrays(os.path.join(data_dir, "train"))
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.array([0.01, 0.1, 1.0, 10.0, 100.0], dtype=np.float32)),
    )
    model.fit(x_train, y_train)
    return model


@torch.no_grad()
def evaluate(data_dir, split, model, batch_size, year_filter=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = load_metadata(data_dir)
    lead_hours = metadata_lead_hours(metadata)
    lead_steps = metadata_lead_steps(metadata, device=device, dtype=torch.float32)
    ds = TyphoPlusDataset(os.path.join(data_dir, split))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    sums = {"persistence": defaultdict(float), "constant_velocity": defaultdict(float), "cliper_ridge": defaultdict(float)}
    for batch in loader:
        batch = move_to_device(batch, device)
        if year_filter:
            mask = batch["future_year"][:, 0] == year_filter
            if not mask.any():
                continue
            batch = {key: value[mask] if torch.is_tensor(value) and value.shape[0] == mask.shape[0] else value for key, value in batch.items()}
        bsz, pred_len = batch["future_latlon"].shape[:2]
        origin = batch["origin"]
        persist = origin.unsqueeze(1).expand(bsz, pred_len, 2)
        add_errors(sums["persistence"], persist, batch["future_latlon"], lead_hours)

        prev = batch["history_latlon"][:, -2]
        step_disp = latlon_to_disp(origin, prev)
        cv = disp_to_latlon(step_disp.unsqueeze(1) * lead_steps.view(1, pred_len, 1), origin)
        add_errors(sums["constant_velocity"], cv, batch["future_latlon"], lead_hours)

        x_num = batch["x_num"].detach().cpu().numpy().reshape(bsz, -1)
        hist = batch["history_latlon"].detach().cpu()
        org = batch["origin"].detach().cpu()
        hist_disp = latlon_to_disp(hist, org.unsqueeze(1)).numpy().reshape(bsz, -1)
        x = np.concatenate([x_num, hist_disp / 500.0], axis=1)
        pred_raw = torch.tensor(model.predict(x).reshape(bsz, pred_len, 2), device=device, dtype=torch.float32)
        pred = disp_to_latlon(pred_raw, origin)
        add_errors(sums["cliper_ridge"], pred, batch["future_latlon"], lead_hours)
    return {key: finalize(value) for key, value in sums.items()}


def main():
    args = parse_args()
    model = fit_cliper_ridge(args.data_dir)
    results = evaluate(args.data_dir, args.split, model, args.batch_size, args.year_filter)
    output = args.output_json or os.path.join(args.data_dir, f"official_baselines_{args.split}.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
