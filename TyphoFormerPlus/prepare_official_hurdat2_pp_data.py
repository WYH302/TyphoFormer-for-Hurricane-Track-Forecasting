"""
Prepare official NHC Atlantic HURDAT2 data for TyphoFormer++ experiments.

Protocol:
- official Atlantic HURDAT2 text file
- storm-year split: train-period storms, validation subset from train-period storms,
  test-period storms
- strict synoptic 6-hour records only
- targets are future local km displacements relative to the last observed point
- normalization and analog retrieval are fitted from the train split only
"""

import argparse
import json
import math
import os
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer


EARTH_RADIUS_KM = 6371.0
STANDARD_TIMES = {0, 600, 1200, 1800}

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

STATUS_CODES = ["DB", "LO", "TD", "TS", "HU", "EX", "SD", "SS", "WV"]

FEATURE_NAMES = [
    "lat",
    "lon",
    "month_sin",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "hour_sin",
    "hour_cos",
    "max_wind",
    "min_pressure",
    "rmax",
    *RADIUS_COLS,
    "dx6",
    "dy6",
    "speed6",
    "heading_sin",
    "heading_cos",
    "accel6",
    "turn_sin",
    "turn_cos",
] + [f"status_{code}" for code in STATUS_CODES]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hurdat2", default="data_official/hurdat2_atl_1851_2024.txt")
    parser.add_argument("--output-dir", default="data_official_pp_12to4")
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--pred-len", type=int, default=4)
    parser.add_argument(
        "--target-lead-step",
        type=int,
        default=0,
        help="For clean lead-specific 12->1 tasks, set to 1/2/3/4. A value of 0 keeps contiguous joint pred_len targets.",
    )
    parser.add_argument("--interval-hours", type=int, default=6)
    parser.add_argument("--train-years", default="2004-2021")
    parser.add_argument("--test-years", default="2022-2024")
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--k-pos", type=int, default=5)
    parser.add_argument("--k-neg", type=int, default=5)
    parser.add_argument("--text-dim", type=int, default=384)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_year_range(text: str):
    if "-" in text:
        start, end = text.split("-", 1)
        return int(start), int(end)
    year = int(text)
    return year, year


def parse_latlon(value: str) -> float:
    value = value.strip()
    if value.endswith(("N", "E")):
        return float(value[:-1])
    if value.endswith(("S", "W")):
        return -float(value[:-1])
    return float(value)


def wrap_lon_delta_deg(lon: np.ndarray) -> np.ndarray:
    return (lon + 180.0) % 360.0 - 180.0


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


def timestamp_hours(date_int: np.ndarray, time_int: np.ndarray) -> np.ndarray:
    date_str = pd.Series(date_int).astype(str).str.zfill(8)
    hour = time_int // 100
    minute = time_int % 100
    ts = pd.to_datetime(date_str, format="%Y%m%d") + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
    return ts.astype("int64").to_numpy() // (10**9 * 3600)


def parse_hurdat2(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    i = 0
    while i < len(lines):
        parts = [part.strip() for part in lines[i].split(",")]
        if len(parts) < 3 or not parts[0].startswith("AL"):
            raise ValueError(f"Unexpected HURDAT2 header at line {i + 1}: {lines[i]}")
        storm_id = parts[0]
        storm_name = parts[1]
        record_count = int(parts[2])
        storm_year = int(storm_id[4:8])
        for j in range(record_count):
            fields = [part.strip() for part in lines[i + 1 + j].split(",")]
            if len(fields) < 20:
                raise ValueError(f"Unexpected HURDAT2 data line at {i + 2 + j}: {lines[i + 1 + j]}")
            numeric = []
            for value in fields[6:]:
                try:
                    numeric.append(int(value))
                except ValueError:
                    numeric.append(-999)
            while len(numeric) < 15:
                numeric.append(-999)
            rows.append(
                {
                    "typhoon_id": f"{storm_id}_{storm_name}",
                    "storm_id": storm_id,
                    "storm_name": storm_name,
                    "storm_year": storm_year,
                    "date": int(fields[0]),
                    "time": int(fields[1]),
                    "record_identifier": fields[2],
                    "system_status": fields[3],
                    "latitude": parse_latlon(fields[4]),
                    "longitude": parse_latlon(fields[5]),
                    "max_wind": numeric[0],
                    "min_pressure": numeric[1],
                    "wind_radii_34_NE": numeric[2],
                    "wind_radii_34_SE": numeric[3],
                    "wind_radii_34_SW": numeric[4],
                    "wind_radii_34_NW": numeric[5],
                    "wind_radii_50_NE": numeric[6],
                    "wind_radii_50_SE": numeric[7],
                    "wind_radii_50_SW": numeric[8],
                    "wind_radii_50_NW": numeric[9],
                    "wind_radii_64_NE": numeric[10],
                    "wind_radii_64_SE": numeric[11],
                    "wind_radii_64_SW": numeric[12],
                    "wind_radii_64_NW": numeric[13],
                    "rmax": numeric[14],
                }
            )
        i += record_count + 1
    df = pd.DataFrame(rows)
    df["_timestamp_hours"] = timestamp_hours(df["date"].to_numpy(np.int64), df["time"].to_numpy(np.int64))
    return df


def storm_splits(df: pd.DataFrame, args) -> Dict[str, set]:
    train_start, train_end = parse_year_range(args.train_years)
    test_start, test_end = parse_year_range(args.test_years)
    train_period = sorted(df.loc[df["storm_year"].between(train_start, train_end), "typhoon_id"].unique().tolist())
    test_ids = set(df.loc[df["storm_year"].between(test_start, test_end), "typhoon_id"].unique().tolist())
    rng = np.random.default_rng(args.split_seed)
    ids = np.array(train_period, dtype=object)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * args.val_ratio))) if len(ids) > 1 else 0
    val_ids = set(ids[:n_val].tolist())
    train_ids = set(ids[n_val:].tolist())
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def build_text(row: pd.Series) -> str:
    return (
        f"Atlantic storm {row['storm_name']} ({row['storm_id']}) at {int(row['date'])} {int(row['time']):04d} UTC. "
        f"Status {row['system_status']}, position {row['latitude']:.1f} latitude {row['longitude']:.1f} longitude, "
        f"maximum sustained wind {int(row['max_wind'])} kt, minimum central pressure {int(row['min_pressure'])} hPa, "
        f"34 kt wind radii NE SE SW NW {int(row['wind_radii_34_NE'])} {int(row['wind_radii_34_SE'])} "
        f"{int(row['wind_radii_34_SW'])} {int(row['wind_radii_34_NW'])} nmi, "
        f"50 kt radii {int(row['wind_radii_50_NE'])} {int(row['wind_radii_50_SE'])} {int(row['wind_radii_50_SW'])} {int(row['wind_radii_50_NW'])}, "
        f"64 kt radii {int(row['wind_radii_64_NE'])} {int(row['wind_radii_64_SE'])} {int(row['wind_radii_64_SW'])} {int(row['wind_radii_64_NW'])}, "
        f"radius of maximum wind {int(row['rmax'])} nmi."
    )


def text_embeddings(texts: List[str], dim: int) -> np.ndarray:
    vectorizer = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2", lowercase=True)
    return vectorizer.transform(texts).astype(np.float32).toarray()


def build_storm_arrays(storm_df: pd.DataFrame, text_emb: np.ndarray) -> Dict[str, np.ndarray]:
    storm_df = storm_df.sort_values(["date", "time"]).reset_index(drop=False)
    coords = storm_df[["latitude", "longitude"]].to_numpy(np.float32)
    count = len(storm_df)
    date_int = storm_df["date"].to_numpy(np.int64)
    time_int = storm_df["time"].to_numpy(np.int64)
    ts_hours = storm_df["_timestamp_hours"].to_numpy(np.int64)
    timestamp = pd.to_datetime(pd.Series(date_int).astype(str), format="%Y%m%d")
    day = timestamp.dt.dayofyear.to_numpy(np.float32)
    month = timestamp.dt.month.to_numpy(np.float32)
    hour = (time_int // 100).astype(np.float32)

    dx6 = np.zeros(count, dtype=np.float32)
    dy6 = np.zeros(count, dtype=np.float32)
    speed6 = np.zeros(count, dtype=np.float32)
    heading_sin = np.zeros(count, dtype=np.float32)
    heading_cos = np.ones(count, dtype=np.float32)
    accel6 = np.zeros(count, dtype=np.float32)
    turn_sin = np.zeros(count, dtype=np.float32)
    turn_cos = np.ones(count, dtype=np.float32)
    if count > 1:
        step = latlon_to_disp_km(coords[1:], coords[:-1])
        dx6[1:] = step[:, 0]
        dy6[1:] = step[:, 1]
        dist = np.sqrt(dx6 * dx6 + dy6 * dy6)
        speed6 = dist / 6.0
        nz = dist > 1e-6
        heading_sin[nz] = dx6[nz] / dist[nz]
        heading_cos[nz] = dy6[nz] / dist[nz]
        accel6[1:] = speed6[1:] - speed6[:-1]
        for i in range(2, count):
            v0 = np.array([dx6[i - 1], dy6[i - 1]], dtype=np.float32)
            v1 = np.array([dx6[i], dy6[i]], dtype=np.float32)
            denom = np.linalg.norm(v0) * np.linalg.norm(v1)
            if denom > 1e-6:
                turn_sin[i] = (v0[0] * v1[1] - v0[1] * v1[0]) / denom
                turn_cos[i] = np.clip(np.dot(v0, v1) / denom, -1.0, 1.0)

    numeric_parts = [
        coords[:, 0],
        coords[:, 1],
        np.sin(2 * np.pi * month / 12.0).astype(np.float32),
        np.cos(2 * np.pi * month / 12.0).astype(np.float32),
        np.sin(2 * np.pi * day / 366.0).astype(np.float32),
        np.cos(2 * np.pi * day / 366.0).astype(np.float32),
        np.sin(2 * np.pi * hour / 24.0).astype(np.float32),
        np.cos(2 * np.pi * hour / 24.0).astype(np.float32),
    ]
    for col in ["max_wind", "min_pressure", "rmax"] + RADIUS_COLS:
        values = pd.to_numeric(storm_df[col], errors="coerce").to_numpy(np.float32)
        values[values <= -999] = np.nan
        numeric_parts.append(values)
    numeric_parts.extend([dx6, dy6, speed6, heading_sin, heading_cos, accel6, turn_sin, turn_cos])
    status = np.zeros((count, len(STATUS_CODES)), dtype=np.float32)
    for i, code in enumerate(storm_df["system_status"].astype(str).tolist()):
        if code in STATUS_CODES:
            status[i, STATUS_CODES.index(code)] = 1.0
    x_num = np.concatenate([np.stack(numeric_parts, axis=1).astype(np.float32), status], axis=1)
    return {
        "coords": coords,
        "x_num": x_num,
        "x_text": text_emb[storm_df["index"].to_numpy()],
        "date": date_int,
        "time": time_int,
        "timestamp_hours": ts_hours,
        "storm_year": storm_df["storm_year"].to_numpy(np.int64),
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
    wind = np.nan_to_num(x_num[-1, FEATURE_NAMES.index("max_wind")], nan=0.0) / 100.0
    return np.concatenate([hist_disp, speed, heading, month_last, [wind]]).astype(np.float32)


def split_name(storm_id: str, splits: Dict[str, set]):
    for name, ids in splits.items():
        if storm_id in ids:
            return name
    return None


def build_samples(df: pd.DataFrame, text_emb: np.ndarray, splits: Dict[str, set], args):
    if args.target_lead_step > 0 and args.pred_len != 1:
        raise ValueError("--target-lead-step is only supported with --pred-len 1.")
    max_future_offset = args.target_lead_step if args.target_lead_step > 0 else args.pred_len
    samples = []
    audit = {
        "candidate_windows": {"train": 0, "val": 0, "test": 0},
        "skipped_non_split_year": 0,
        "skipped_non_standard_time": {"train": 0, "val": 0, "test": 0},
        "skipped_non_strict_interval": {"train": 0, "val": 0, "test": 0},
        "generated_windows": {"train": 0, "val": 0, "test": 0},
    }
    for storm_id, storm_df in df.groupby("typhoon_id"):
        split = split_name(storm_id, splits)
        if split is None:
            audit["skipped_non_split_year"] += 1
            continue
        arrays = build_storm_arrays(storm_df, text_emb)
        coords = arrays["coords"]
        limit = len(coords) - args.input_len - max_future_offset + 1
        if limit <= 0:
            continue
        audit["candidate_windows"][split] += int(limit)
        for start in range(limit):
            end = start + args.input_len
            check_end = end + max_future_offset
            times = arrays["time"][start:check_end]
            if any(int(t) not in STANDARD_TIMES for t in times):
                audit["skipped_non_standard_time"][split] += 1
                continue
            window_ts = arrays["timestamp_hours"][start:check_end]
            if not np.all(np.diff(window_ts) == args.interval_hours):
                audit["skipped_non_strict_interval"][split] += 1
                continue
            history_latlon = coords[start:end]
            if args.target_lead_step > 0:
                future_indices = np.array([end + args.target_lead_step - 1], dtype=np.int64)
            else:
                future_indices = np.arange(end, end + args.pred_len, dtype=np.int64)
            future_latlon = coords[future_indices]
            origin = history_latlon[-1]
            future_disp = latlon_to_disp_km(future_latlon, origin)
            samples.append(
                {
                    "storm_id": storm_id,
                    "storm_year": int(arrays["storm_year"][0]),
                    "split": split,
                    "window": start,
                    "x_num_raw": arrays["x_num"][start:end],
                    "x_text": arrays["x_text"][start:end],
                    "history_latlon": history_latlon,
                    "future_latlon": future_latlon,
                    "origin": origin,
                    "future_disp_raw": future_disp,
                    "query_raw": make_query(history_latlon, arrays["x_num"][start:end]),
                    "history_date": arrays["date"][start:end],
                    "history_time": arrays["time"][start:end],
                    "future_date": arrays["date"][future_indices],
                    "future_time": arrays["time"][future_indices],
                    "history_timestamp_hours": arrays["timestamp_hours"][start:end],
                    "future_timestamp_hours": arrays["timestamp_hours"][future_indices],
                    "future_year": (arrays["date"][future_indices] // 10000).astype(np.int64),
                }
            )
            audit["generated_windows"][split] += 1
    return samples, audit


def compute_stats(samples: List[Dict]) -> Dict[str, np.ndarray]:
    train = [s for s in samples if s["split"] == "train"]
    if not train:
        raise RuntimeError("No train samples available.")
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
    return {"x_mean": x_mean, "x_std": x_std, "target_mean": y_mean, "target_std": y_std, "query_mean": q_mean, "query_std": q_std}


def normalize_samples(samples: List[Dict], stats: Dict[str, np.ndarray]):
    for sample in samples:
        x = sample["x_num_raw"].copy()
        x = np.where(np.isfinite(x), x, stats["x_mean"])
        sample["x_num"] = ((x - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
        sample["target_disp"] = ((sample["future_disp_raw"] - stats["target_mean"]) / stats["target_std"]).astype(np.float32)
        sample["query"] = ((sample["query_raw"] - stats["query_mean"]) / stats["query_std"]).astype(np.float32)


def pad_indices(indices: np.ndarray, k: int) -> np.ndarray:
    if len(indices) == 0:
        raise ValueError("No analog candidates available.")
    if len(indices) >= k:
        return indices[:k]
    return np.concatenate([indices, np.full(k - len(indices), indices[-1], dtype=indices.dtype)])


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
        if sample["split"] == "train":
            # Training may use the supervised future to construct hard negatives
            # for ranking supervision. Validation/test negatives must not depend
            # on the current sample's future trajectory.
            future_diff = train_futures[near] - sample["future_disp_raw"][None, :, :]
            future_dist = np.mean(np.linalg.norm(future_diff, axis=-1), axis=1)
            neg_local = pad_indices(near[np.argsort(-future_dist)], k_neg)
            sample["analog_neg_policy"] = "train_target_hard_negative"
        else:
            # Operational target-free negatives: choose query-near alternatives
            # after the positive analogs using observed-history features only.
            query_only = order_local[k_pos : k_pos + max(k_neg, 1)]
            if len(query_only) == 0:
                query_only = order_local[: max(k_neg, 1)]
            neg_local = pad_indices(query_only, k_neg)
            sample["analog_neg_policy"] = "target_free_query_neighbor"
        sample["analog_pos"] = ((train_futures[pos_local] - stats["target_mean"]) / stats["target_std"]).astype(np.float32)
        sample["analog_neg"] = ((train_futures[neg_local] - stats["target_mean"]) / stats["target_std"]).astype(np.float32)
        sample["analog_pos_ids"] = train_indices[pos_local].astype(np.int64)
        sample["analog_neg_ids"] = train_indices[neg_local].astype(np.int64)
        sample["analog_pos_storms"] = np.array([samples[int(i)]["storm_id"] for i in sample["analog_pos_ids"]], dtype=object)
        sample["analog_neg_storms"] = np.array([samples[int(i)]["storm_id"] for i in sample["analog_neg_ids"]], dtype=object)


def save_samples(samples: List[Dict], stats: Dict[str, np.ndarray], splits: Dict[str, set], audit: Dict, args, source_audit: Dict):
    if os.path.exists(args.output_dir):
        if not args.force:
            raise FileExistsError(f"{args.output_dir} exists. Re-run with --force.")
        shutil.rmtree(args.output_dir)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    storm_sets = {"train": set(), "val": set(), "test": set()}
    for sample in samples:
        split = sample["split"]
        storm_sets[split].add(sample["storm_id"])
        path = os.path.join(args.output_dir, split, f"{sample['storm_id']}_{sample['window']:03d}.npy")
        payload = {
            "storm_id": sample["storm_id"],
            "storm_year": np.int64(sample["storm_year"]),
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
            "future_year": sample["future_year"].astype(np.int64),
            "history_timestamp_hours": sample["history_timestamp_hours"].astype(np.int64),
            "future_timestamp_hours": sample["future_timestamp_hours"].astype(np.int64),
            "analog_pos": sample["analog_pos"].astype(np.float32),
            "analog_neg": sample["analog_neg"].astype(np.float32),
            "analog_pos_ids": sample["analog_pos_ids"],
            "analog_neg_ids": sample["analog_neg_ids"],
            "analog_pos_storms": sample["analog_pos_storms"],
            "analog_neg_storms": sample["analog_neg_storms"],
            "analog_neg_policy": sample["analog_neg_policy"],
        }
        np.save(path, payload)
        counts[split] += 1
    metadata = {
        "source": "NHC Atlantic HURDAT2 official text",
        "hurdat2": args.hurdat2,
        "input_len": args.input_len,
        "pred_len": args.pred_len,
        "target_lead_step": args.target_lead_step,
        "lead_steps": [args.target_lead_step] if args.target_lead_step > 0 else [i + 1 for i in range(args.pred_len)],
        "interval_hours": args.interval_hours,
        "strict_6h": True,
        "standard_times_only": True,
        "standard_time_values": sorted(STANDARD_TIMES),
        "train_years": args.train_years,
        "test_years": args.test_years,
        "validation": f"{args.val_ratio:.3f} of train-period storms, split_seed={args.split_seed}",
        "lead_hours": [args.interval_hours * args.target_lead_step]
        if args.target_lead_step > 0
        else [args.interval_hours * (i + 1) for i in range(args.pred_len)],
        "history_hours": args.interval_hours * args.input_len,
        "forecast_hours": args.interval_hours * (args.target_lead_step if args.target_lead_step > 0 else args.pred_len),
        "feature_names": FEATURE_NAMES,
        "input_dim": len(FEATURE_NAMES),
        "text_dim": args.text_dim,
        "text_embedding": "rule_based_description_hashing_vectorizer_384d_no_fit",
        "k_pos": args.k_pos,
        "k_neg": args.k_neg,
        "splits": {split: {"storms": len(storm_sets[split]), "samples": counts[split]} for split in ["train", "val", "test"]},
        "split_storm_ids": {split: sorted(storm_sets[split]) for split in ["train", "val", "test"]},
        "stats_fit_split": "train",
        "analog_candidate_split": "train",
        "analog_negative_policy": {
            "train": "target-dependent hard negatives are used only as supervised ranking negatives",
            "val_test": "target-free query-neighbor negatives use only observed-history query features",
        },
        "source_audit": source_audit,
        "data_audit": audit,
        "stats": {key: value.tolist() for key, value in stats.items()},
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata["splits"], indent=2))
    print(f"Saved official HURDAT2 TyphoFormer++ data to {args.output_dir}")


def main():
    args = parse_args()
    df = parse_hurdat2(args.hurdat2)
    train_start, _ = parse_year_range(args.train_years)
    _, test_end = parse_year_range(args.test_years)
    source_audit = {
        "raw_rows": int(len(df)),
        "raw_storms": int(df["typhoon_id"].nunique()),
        "downloaded_year_min": int(df["storm_year"].min()),
        "downloaded_year_max": int(df["storm_year"].max()),
    }
    df = df.loc[df["storm_year"].between(train_start, test_end)].copy().reset_index(drop=True)
    source_audit.update(
        {
            "rows_after_protocol_year_filter": int(len(df)),
            "storms_after_protocol_year_filter": int(df["typhoon_id"].nunique()),
            "rows_non_standard_time_in_protocol_years": int((~df["time"].isin(sorted(STANDARD_TIMES))).sum()),
        }
    )
    descriptions = [build_text(row) for _, row in df.iterrows()]
    text_emb = text_embeddings(descriptions, args.text_dim)
    splits = storm_splits(df, args)
    samples, audit = build_samples(df, text_emb, splits, args)
    if not samples:
        raise RuntimeError("No samples generated.")
    stats = compute_stats(samples)
    normalize_samples(samples, stats)
    attach_analogs(samples, stats, args.k_pos, args.k_neg)
    save_samples(samples, stats, splits, audit, args, source_audit)


if __name__ == "__main__":
    main()
