"""
Prepare TyphoFormer++ data from the files bundled with this repository.

This script fixes the original demo-data leakage issue by putting historical
lat/lon and derived motion features in the input window. Targets are future
local displacements, normalized with train-only statistics. Analog retrieval is
also train-only for val/test and excludes the same storm during train retrieval.
"""

import argparse
import glob
import json
import os
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0


RADIUS_COLS = [
    "wind_radii_34_NE",
    "wind_radii_34_SE",
    "wind_radii_34_SW",
    "wind_radii_34_NW",
    "wind_radii_50_NE",
    "wind_radii_50_SE",
    "wind_radii_50_SW",
    "wind_radii_50_NW",
    "wind_radii_64_NE",
    "wind_radii_64_SE",
    "wind_radii_64_SW",
    "wind_radii_64_NW",
]

FEATURE_NAMES = [
    "lat",
    "lon",
    "dx6",
    "dy6",
    "speed6",
    "heading_sin",
    "heading_cos",
    "accel6",
    "turn_sin",
    "turn_cos",
    "month_sin",
    "month_cos",
    "max_wind",
    "min_pressure",
] + RADIUS_COLS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="HURDAT_2new_3000.csv")
    parser.add_argument("--embedding-dir", default="embedding_chunks")
    parser.add_argument("--output-dir", default="data_pp")
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--k-pos", type=int, default=5)
    parser.add_argument("--k-neg", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-6h", action="store_true")
    parser.add_argument("--standard-times-only", action="store_true")
    parser.add_argument("--interval-hours", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def convert_latlon(value):
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("N") or value.endswith("E"):
            return float(value[:-1])
        if value.endswith("S") or value.endswith("W"):
            return -float(value[:-1])
    return float(value)


def wrap_lon_delta_deg(lon: np.ndarray) -> np.ndarray:
    return (lon + 180.0) % 360.0 - 180.0


def hurdat_timestamps(df: pd.DataFrame) -> pd.Series:
    date_str = pd.to_numeric(df["date"], errors="raise").astype(int).astype(str).str.zfill(8)
    time_int = pd.to_numeric(df["time"], errors="raise").astype(int)
    hour = time_int // 100
    minute = time_int % 100
    return pd.to_datetime(date_str, format="%Y%m%d") + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_time_int"] = pd.to_numeric(out["time"], errors="raise").astype(int)
    out["_timestamp"] = hurdat_timestamps(out)
    return out


def latlon_to_disp_km(latlon: np.ndarray, origin: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(latlon[..., 0])
    lon = latlon[..., 1]
    lat0 = np.deg2rad(origin[..., 0])
    lon0 = origin[..., 1]
    dlat = lat - lat0
    dlon = np.deg2rad(wrap_lon_delta_deg(lon - lon0))
    dx = EARTH_RADIUS_KM * np.cos(lat0) * dlon
    dy = EARTH_RADIUS_KM * dlat
    return np.stack([dx, dy], axis=-1).astype(np.float32)


def load_embeddings(embedding_dir: str) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(embedding_dir, "emb_chunk_*.npy")))
    if not files:
        raise FileNotFoundError(f"No embedding chunks found in {embedding_dir}")
    return np.concatenate([np.load(path) for path in files], axis=0).astype(np.float32)


def split_storms(storm_ids: List[str], val_ratio: float, test_ratio: float, seed: int):
    ids = np.array(sorted(storm_ids), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_total = len(ids)
    n_test = max(1, int(round(n_total * test_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    test_ids = set(ids[:n_test].tolist())
    val_ids = set(ids[n_test : n_test + n_val].tolist())
    train_ids = set(ids[n_test + n_val :].tolist())
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def build_storm_arrays(storm_df: pd.DataFrame, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    storm_df = storm_df.sort_values(["date", "time"]).reset_index(drop=False)
    lat = storm_df["latitude"].map(convert_latlon).to_numpy(np.float32)
    lon = storm_df["longitude"].map(convert_latlon).to_numpy(np.float32)
    coords = np.stack([lat, lon], axis=1).astype(np.float32)
    count = len(storm_df)

    dx6 = np.zeros(count, dtype=np.float32)
    dy6 = np.zeros(count, dtype=np.float32)
    speed6 = np.zeros(count, dtype=np.float32)
    heading_sin = np.zeros(count, dtype=np.float32)
    heading_cos = np.ones(count, dtype=np.float32)
    accel6 = np.zeros(count, dtype=np.float32)
    turn_sin = np.zeros(count, dtype=np.float32)
    turn_cos = np.ones(count, dtype=np.float32)

    if count > 1:
        step_disp = latlon_to_disp_km(coords[1:], coords[:-1])
        dx6[1:] = step_disp[:, 0]
        dy6[1:] = step_disp[:, 1]
        dist = np.sqrt(dx6**2 + dy6**2)
        speed6 = dist / 6.0
        nonzero = dist > 1e-6
        heading_sin[nonzero] = dx6[nonzero] / dist[nonzero]
        heading_cos[nonzero] = dy6[nonzero] / dist[nonzero]
        accel6[1:] = speed6[1:] - speed6[:-1]
        for i in range(2, count):
            v0 = np.array([dx6[i - 1], dy6[i - 1]], dtype=np.float32)
            v1 = np.array([dx6[i], dy6[i]], dtype=np.float32)
            denom = np.linalg.norm(v0) * np.linalg.norm(v1)
            if denom > 1e-6:
                turn_sin[i] = (v0[0] * v1[1] - v0[1] * v1[0]) / denom
                turn_cos[i] = np.clip(np.dot(v0, v1) / denom, -1.0, 1.0)

    date_int = pd.to_numeric(storm_df["date"], errors="coerce").fillna(0).astype(int).to_numpy()
    time_int = pd.to_numeric(storm_df["time"], errors="coerce").fillna(0).astype(int).to_numpy()
    if "_timestamp" in storm_df.columns:
        timestamps = pd.to_datetime(storm_df["_timestamp"])
    else:
        timestamps = hurdat_timestamps(storm_df)
    timestamp_hours = timestamps.astype("int64").to_numpy() // (10**9 * 3600)
    month = ((date_int // 100) % 100).astype(np.float32)
    month_angle = 2.0 * np.pi * np.clip(month, 1, 12) / 12.0
    month_sin = np.sin(month_angle).astype(np.float32)
    month_cos = np.cos(month_angle).astype(np.float32)

    numeric_parts = [
        lat,
        lon,
        dx6,
        dy6,
        speed6,
        heading_sin,
        heading_cos,
        accel6,
        turn_sin,
        turn_cos,
        month_sin,
        month_cos,
    ]
    for col in ["max_wind", "min_pressure"] + RADIUS_COLS:
        values = pd.to_numeric(storm_df[col], errors="coerce").to_numpy(np.float32)
        values[values <= -999] = np.nan
        numeric_parts.append(values)

    x_num = np.stack(numeric_parts, axis=1).astype(np.float32)
    x_text = embeddings[storm_df["index"].to_numpy()]
    return {
        "coords": coords,
        "x_num": x_num,
        "x_text": x_text,
        "month": month,
        "date": date_int.astype(np.int64),
        "time": time_int.astype(np.int64),
        "timestamp_hours": timestamp_hours.astype(np.int64),
    }


def make_query(history_latlon: np.ndarray, x_num: np.ndarray) -> np.ndarray:
    origin = history_latlon[-1]
    hist_disp = latlon_to_disp_km(history_latlon, origin).reshape(-1) / 500.0
    speed = np.nan_to_num(x_num[:, FEATURE_NAMES.index("speed6")], nan=0.0) / 50.0
    heading = np.nan_to_num(
        x_num[:, [FEATURE_NAMES.index("heading_sin"), FEATURE_NAMES.index("heading_cos")]],
        nan=0.0,
    ).reshape(-1)
    month_last = x_num[-1, [FEATURE_NAMES.index("month_sin"), FEATURE_NAMES.index("month_cos")]]
    max_wind = np.nan_to_num(x_num[-1, FEATURE_NAMES.index("max_wind")], nan=0.0) / 100.0
    return np.concatenate([hist_disp, speed, heading, month_last, [max_wind]]).astype(np.float32)


def build_samples(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    splits,
    input_len: int,
    pred_len: int,
    strict_6h: bool = False,
    interval_hours: int = 6,
):
    samples = []
    audit = {
        "candidate_windows": {"train": 0, "val": 0, "test": 0},
        "skipped_non_strict_interval": {"train": 0, "val": 0, "test": 0},
        "generated_windows": {"train": 0, "val": 0, "test": 0},
    }
    grouped = df.groupby("typhoon_id")
    for storm_id, storm_df in grouped:
        split = next(name for name, ids in splits.items() if storm_id in ids)
        arrays = build_storm_arrays(storm_df, embeddings)
        coords = arrays["coords"]
        x_num = arrays["x_num"]
        x_text = arrays["x_text"]
        dates = arrays["date"]
        times = arrays["time"]
        timestamp_hours = arrays["timestamp_hours"]
        limit = len(coords) - input_len - pred_len + 1
        if limit <= 0:
            continue
        audit["candidate_windows"][split] += int(limit)
        for start in range(limit):
            end = start + input_len
            future_end = end + pred_len
            window_timestamp_hours = timestamp_hours[start:future_end]
            if strict_6h and not np.all(np.diff(window_timestamp_hours) == interval_hours):
                audit["skipped_non_strict_interval"][split] += 1
                continue
            history_latlon = coords[start:end]
            future_latlon = coords[end:future_end]
            origin = history_latlon[-1]
            future_disp = latlon_to_disp_km(future_latlon, origin)
            sample = {
                "storm_id": storm_id,
                "split": split,
                "window": start,
                "x_num_raw": x_num[start:end],
                "x_text": x_text[start:end],
                "history_latlon": history_latlon,
                "future_latlon": future_latlon,
                "origin": origin,
                "future_disp_raw": future_disp,
                "query_raw": make_query(history_latlon, x_num[start:end]),
                "history_date": dates[start:end],
                "history_time": times[start:end],
                "future_date": dates[end:future_end],
                "future_time": times[end:future_end],
                "history_timestamp_hours": timestamp_hours[start:end],
                "future_timestamp_hours": timestamp_hours[end:future_end],
            }
            samples.append(sample)
            audit["generated_windows"][split] += 1
    return samples, audit


def compute_stats(samples: List[Dict]) -> Dict[str, np.ndarray]:
    train = [s for s in samples if s["split"] == "train"]
    x_train = np.concatenate([s["x_num_raw"] for s in train], axis=0)
    x_mean = np.nanmean(x_train, axis=0)
    x_std = np.nanstd(x_train, axis=0)
    x_mean = np.where(np.isfinite(x_mean), x_mean, 0.0).astype(np.float32)
    x_std = np.where((x_std > 1e-6) & np.isfinite(x_std), x_std, 1.0).astype(np.float32)

    y_train = np.concatenate([s["future_disp_raw"] for s in train], axis=0)
    y_mean = y_train.mean(axis=0).astype(np.float32)
    y_std = y_train.std(axis=0).astype(np.float32)
    y_std = np.where(y_std > 1e-6, y_std, 1.0).astype(np.float32)

    q_train = np.stack([s["query_raw"] for s in train], axis=0)
    q_mean = q_train.mean(axis=0).astype(np.float32)
    q_std = q_train.std(axis=0).astype(np.float32)
    q_std = np.where(q_std > 1e-6, q_std, 1.0).astype(np.float32)
    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "target_mean": y_mean,
        "target_std": y_std,
        "query_mean": q_mean,
        "query_std": q_std,
    }


def normalize_samples(samples: List[Dict], stats: Dict[str, np.ndarray]):
    for sample in samples:
        x = sample["x_num_raw"].copy()
        x = np.where(np.isfinite(x), x, stats["x_mean"])
        sample["x_num"] = ((x - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
        sample["target_disp"] = (
            (sample["future_disp_raw"] - stats["target_mean"]) / stats["target_std"]
        ).astype(np.float32)
        sample["query"] = (
            (sample["query_raw"] - stats["query_mean"]) / stats["query_std"]
        ).astype(np.float32)


def pad_indices(indices: np.ndarray, k: int) -> np.ndarray:
    if len(indices) == 0:
        raise ValueError("No analog candidates available.")
    if len(indices) >= k:
        return indices[:k]
    pad = np.full(k - len(indices), indices[-1], dtype=indices.dtype)
    return np.concatenate([indices, pad], axis=0)


def attach_analogs(samples: List[Dict], stats: Dict[str, np.ndarray], k_pos: int, k_neg: int):
    train_indices = np.array([i for i, s in enumerate(samples) if s["split"] == "train"], dtype=np.int64)
    train_queries = np.stack([samples[i]["query"] for i in train_indices], axis=0)
    train_futures = np.stack([samples[i]["future_disp_raw"] for i in train_indices], axis=0)
    train_storms = np.array([samples[i]["storm_id"] for i in train_indices], dtype=object)

    for sample in samples:
        if sample["split"] == "train":
            valid_mask = train_storms != sample["storm_id"]
        else:
            valid_mask = np.ones(len(train_indices), dtype=bool)
        candidates = np.where(valid_mask)[0]
        if len(candidates) == 0:
            candidates = np.arange(len(train_indices))

        diff = train_queries[candidates] - sample["query"][None, :]
        dist = np.mean(diff * diff, axis=1)
        order_local = candidates[np.argsort(dist)]
        pos_local = pad_indices(order_local, k_pos)

        near_count = min(len(order_local), max(50, k_neg * 10))
        near = order_local[:near_count]
        future_diff = train_futures[near] - sample["future_disp_raw"][None, :, :]
        future_dist = np.mean(np.linalg.norm(future_diff, axis=-1), axis=1)
        neg_local = near[np.argsort(-future_dist)]
        neg_local = pad_indices(neg_local, k_neg)

        pos_raw = train_futures[pos_local]
        neg_raw = train_futures[neg_local]
        sample["analog_pos"] = ((pos_raw - stats["target_mean"]) / stats["target_std"]).astype(np.float32)
        sample["analog_neg"] = ((neg_raw - stats["target_mean"]) / stats["target_std"]).astype(np.float32)
        sample["analog_pos_ids"] = train_indices[pos_local].astype(np.int64)
        sample["analog_neg_ids"] = train_indices[neg_local].astype(np.int64)
        sample["analog_pos_storms"] = np.array(
            [samples[int(i)]["storm_id"] for i in sample["analog_pos_ids"]], dtype=object
        )
        sample["analog_neg_storms"] = np.array(
            [samples[int(i)]["storm_id"] for i in sample["analog_neg_ids"]], dtype=object
        )


def save_samples(samples: List[Dict], stats: Dict[str, np.ndarray], args):
    if os.path.exists(args.output_dir):
        if not args.force:
            raise FileExistsError(f"{args.output_dir} exists. Re-run with --force to replace it.")
        shutil.rmtree(args.output_dir)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    storm_sets = {"train": set(), "val": set(), "test": set()}
    for sample in samples:
        split = sample["split"]
        storm_sets[split].add(sample["storm_id"])
        name = f"{sample['storm_id']}_{sample['window']:03d}.npy"
        path = os.path.join(args.output_dir, split, name)
        payload = {
            "storm_id": sample["storm_id"],
            "window": np.int64(sample["window"]),
            "input_num": sample["x_num"].astype(np.float32),
            "input_text": sample["x_text"].astype(np.float32),
            "target_disp": sample["target_disp"].astype(np.float32),
            "target_disp_raw_km": sample["future_disp_raw"].astype(np.float32),
            "future_latlon": sample["future_latlon"].astype(np.float32),
            "origin": sample["origin"].astype(np.float32),
            "history_latlon": sample["history_latlon"].astype(np.float32),
            "history_date": sample["history_date"].astype(np.int64),
            "history_time": sample["history_time"].astype(np.int64),
            "future_date": sample["future_date"].astype(np.int64),
            "future_time": sample["future_time"].astype(np.int64),
            "history_timestamp_hours": sample["history_timestamp_hours"].astype(np.int64),
            "future_timestamp_hours": sample["future_timestamp_hours"].astype(np.int64),
            "analog_pos": sample["analog_pos"].astype(np.float32),
            "analog_neg": sample["analog_neg"].astype(np.float32),
            "analog_pos_ids": sample["analog_pos_ids"],
            "analog_neg_ids": sample["analog_neg_ids"],
            "analog_pos_storms": sample["analog_pos_storms"],
            "analog_neg_storms": sample["analog_neg_storms"],
        }
        np.save(path, payload)
        counts[split] += 1

    stats_json = {key: value.tolist() for key, value in stats.items()}
    metadata = {
        "input_csv": args.csv,
        "embedding_dir": args.embedding_dir,
        "input_len": args.input_len,
        "pred_len": args.pred_len,
        "k_pos": args.k_pos,
        "k_neg": args.k_neg,
        "feature_names": FEATURE_NAMES,
        "input_dim": len(FEATURE_NAMES),
        "text_dim": 384,
        "strict_6h": bool(getattr(args, "strict_6h", False)),
        "standard_times_only": bool(getattr(args, "standard_times_only", False)),
        "standard_time_values": [0, 600, 1200, 1800],
        "interval_hours": int(getattr(args, "interval_hours", 6)),
        "lead_hours": [int(getattr(args, "interval_hours", 6)) * (i + 1) for i in range(args.pred_len)],
        "history_hours": int(getattr(args, "interval_hours", 6)) * args.input_len,
        "forecast_hours": int(getattr(args, "interval_hours", 6)) * args.pred_len,
        "splits": {
            split: {"storms": len(storm_sets[split]), "samples": counts[split]}
            for split in ["train", "val", "test"]
        },
        "split_storm_ids": {
            split: sorted(storm_sets[split])
            for split in ["train", "val", "test"]
        },
        "data_audit": getattr(args, "data_audit", {}),
        "stats": stats_json,
        "notes": [
            "Targets are train-normalized local km displacements relative to the last observed point.",
            "Val/test analog retrieval uses train samples only.",
            "Train analog retrieval excludes all windows from the same storm.",
        ],
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata["splits"], indent=2))
    print(f"Saved TyphoFormer++ data to {args.output_dir}")


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    df.columns = [col.strip() for col in df.columns]
    raw_rows = len(df)
    df = add_time_columns(df)
    embeddings = load_embeddings(args.embedding_dir)
    if len(embeddings) != len(df):
        raise ValueError(f"Embedding rows {len(embeddings)} != CSV rows {len(df)}")
    required = {"typhoon_id", "date", "time", "latitude", "longitude", "max_wind", "min_pressure", *RADIUS_COLS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if args.standard_times_only:
        standard_mask = df["_time_int"].isin([0, 600, 1200, 1800])
        df = df.loc[standard_mask].copy()

    row_audit = {
        "raw_rows": int(raw_rows),
        "rows_after_standard_time_filter": int(len(df)),
        "dropped_non_standard_time_rows": int(raw_rows - len(df)),
    }

    splits = split_storms(df["typhoon_id"].unique().tolist(), args.val_ratio, args.test_ratio, args.seed)
    samples, window_audit = build_samples(
        df,
        embeddings,
        splits,
        args.input_len,
        args.pred_len,
        strict_6h=args.strict_6h,
        interval_hours=args.interval_hours,
    )
    if not samples:
        raise RuntimeError("No samples were generated. Check input_len/pred_len.")
    args.data_audit = {**row_audit, **window_audit}
    stats = compute_stats(samples)
    normalize_samples(samples, stats)
    attach_analogs(samples, stats, args.k_pos, args.k_neg)
    save_samples(samples, stats, args)


if __name__ == "__main__":
    main()
