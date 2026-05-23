#!/usr/bin/env python3
"""Audit strict-HURDAT2 lead-specific safe-negative datasets.

The script checks the saved artifact files rather than trusting summary tables.
It is intentionally read-only: it loads metadata and .npy records, verifies the
split/retrieval/window invariants, and prints a compact JSON summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


STANDARD_TIMES = {0, 600, 1200, 1800}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Folder containing data_official_pp_12to1_lead*_safeneg directories.",
    )
    parser.add_argument(
        "--max-files-per-split",
        type=int,
        default=0,
        help="Optional smoke-test cap. The default 0 audits every saved .npy file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for writing the JSON audit summary.",
    )
    return parser.parse_args()


def autodetect_root(script_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    local = script_dir
    if (local / "data_official_pp_12to1_lead1_safeneg").exists():
        return local
    sibling_artifact = script_dir.parent / "local_artifacts" / "TyphoFormerPlus"
    if (sibling_artifact / "data_official_pp_12to1_lead1_safeneg").exists():
        return sibling_artifact
    return local


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def as_strings(values: Any) -> list[str]:
    arr = np.asarray(values, dtype=object).reshape(-1)
    return [str(x) for x in arr.tolist()]


def fail(failures: list[str], lead: int, split: str, file_name: str, message: str) -> None:
    failures.append(f"lead{lead}/{split}/{file_name}: {message}")


def audit_sample(
    path: Path,
    lead: int,
    split: str,
    metadata: dict[str, Any],
    split_ids: dict[str, set[str]],
    failures: list[str],
) -> None:
    record = np.load(path, allow_pickle=True).item()
    storm_id = str(record["storm_id"])
    storm_year = int(np.asarray(record["storm_year"]).item())
    interval = int(metadata.get("interval_hours", 6))
    target_step = int(metadata["target_lead_step"])

    if storm_id not in split_ids[split]:
        fail(failures, lead, split, path.name, f"storm_id {storm_id} not listed in metadata split")

    if split == "test":
        if not (2022 <= storm_year <= 2024):
            fail(failures, lead, split, path.name, f"test storm year {storm_year} outside 2022-2024")
    elif not (2004 <= storm_year <= 2021):
        fail(failures, lead, split, path.name, f"{split} storm year {storm_year} outside 2004-2021")

    history_time = np.asarray(record["history_time"], dtype=int).reshape(-1)
    future_time = np.asarray(record["future_time"], dtype=int).reshape(-1)
    if any(int(t) not in STANDARD_TIMES for t in history_time.tolist() + future_time.tolist()):
        fail(failures, lead, split, path.name, "non-synoptic saved time")

    history_ts = np.asarray(record["history_timestamp_hours"], dtype=int).reshape(-1)
    future_ts = np.asarray(record["future_timestamp_hours"], dtype=int).reshape(-1)
    if history_ts.size > 1 and not np.all(np.diff(history_ts) == interval):
        fail(failures, lead, split, path.name, "history timestamps are not strict 6h")
    if future_ts.size != 1:
        fail(failures, lead, split, path.name, f"expected one lead-specific target, found {future_ts.size}")
    elif int(future_ts[0] - history_ts[-1]) != interval * target_step:
        fail(failures, lead, split, path.name, "target timestamp does not match requested lead")

    pos_storms = as_strings(record.get("analog_pos_storms", []))
    neg_storms = as_strings(record.get("analog_neg_storms", []))
    analog_storms = pos_storms + neg_storms
    if split in {"val", "test"}:
        outside = sorted({s for s in analog_storms if s not in split_ids["train"]})
        if outside:
            fail(failures, lead, split, path.name, f"analog storms outside train split: {outside[:5]}")
        policy = str(record.get("analog_neg_policy", ""))
        if policy != "target_free_query_neighbor":
            fail(failures, lead, split, path.name, f"unexpected validation/test negative policy {policy!r}")
    else:
        same_storm = sorted({s for s in analog_storms if s == storm_id})
        if same_storm:
            fail(failures, lead, split, path.name, "training analog retrieval included same-storm window")
        policy = str(record.get("analog_neg_policy", ""))
        if policy != "train_target_hard_negative":
            fail(failures, lead, split, path.name, f"unexpected training negative policy {policy!r}")


def audit_lead(root: Path, lead: int, max_files_per_split: int) -> dict[str, Any]:
    data_dir = root / f"data_official_pp_12to1_lead{lead}_safeneg"
    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    split_ids = {
        split: set(map(str, metadata["split_storm_ids"][split]))
        for split in ("train", "val", "test")
    }

    failures: list[str] = []
    split_overlap = {
        "train_val": len(split_ids["train"] & split_ids["val"]),
        "train_test": len(split_ids["train"] & split_ids["test"]),
        "val_test": len(split_ids["val"] & split_ids["test"]),
    }
    for name, count in split_overlap.items():
        if count:
            failures.append(f"lead{lead}: split overlap {name}={count}")

    split_counts: dict[str, int] = {}
    audited_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        files = sorted((data_dir / split).glob("*.npy"))
        split_counts[split] = len(files)
        expected = int(metadata["splits"][split]["samples"])
        if len(files) != expected:
            failures.append(f"lead{lead}/{split}: metadata samples {expected}, files {len(files)}")
        selected = files if max_files_per_split <= 0 else files[:max_files_per_split]
        audited_counts[split] = len(selected)
        for file_path in selected:
            audit_sample(file_path, lead, split, metadata, split_ids, failures)

    return {
        "lead": lead,
        "lead_hours": metadata["lead_hours"],
        "data_dir": data_dir.name,
        "metadata": {
            "stats_fit_split": metadata.get("stats_fit_split"),
            "analog_candidate_split": metadata.get("analog_candidate_split"),
            "analog_negative_policy": metadata.get("analog_negative_policy"),
            "source_audit": metadata.get("source_audit"),
            "data_audit": metadata.get("data_audit"),
        },
        "split_storms": {split: len(ids) for split, ids in split_ids.items()},
        "split_counts": split_counts,
        "audited_counts": audited_counts,
        "split_overlap": split_overlap,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = autodetect_root(script_dir, args.artifact_root)
    hurdat_path = root / "data_official" / "hurdat2_atl_1851_2024.txt"
    summary = {
        "artifact_root": root.name,
        "artifact_root_note": "Paths in this summary are relative to the detected or supplied artifact root.",
        "hurdat2_sha256": sha256_file(hurdat_path),
        "max_files_per_split": args.max_files_per_split,
        "leads": [audit_lead(root, lead, args.max_files_per_split) for lead in (1, 2, 3, 4)],
    }
    total_failures = sum(len(row["failures"]) for row in summary["leads"])
    summary["total_failures"] = total_failures
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
