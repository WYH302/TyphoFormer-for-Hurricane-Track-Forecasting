"""Tiny synthetic end-to-end checkpoint test; never a forecasting experiment."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from evaluate_locked import verify_development_samples
from run_controlled import digest
from run_matrix import jobs


class RunnerSmoke(unittest.TestCase):
    def test_locked_development_identity(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="synthetic_identity_", dir=here) as temporary:
            root = Path(temporary)
            old_root, new_root = root / "old", root / "new"
            config = {"data_dir": str(old_root), "data_sha256": {}}
            for split in ("train", "val"):
                for parent in (old_root, new_root):
                    (parent / split).mkdir(parents=True)
                old = {"input_num": np.ones((12, 3)), "analog_pos_ids": np.array([1, 2]), "analog_neg_ids": np.array([3, 4])}
                new = {**old, "analog_pos_ids": np.array([10, 20]), "analog_neg_ids": np.array([30, 40])}
                np.save(old_root / split / "sample.npy", old)
                np.save(new_root / split / "sample.npy", new)
                config["data_sha256"][split] = {"sample.npy": digest(old_root / split / "sample.npy")}
            report = verify_development_samples(config, new_root)
            self.assertEqual(report["n_binary_different_files"], 2)
            new["input_num"] = np.zeros((12, 3))
            np.save(new_root / "train" / "sample.npy", new)
            with self.assertRaisesRegex(ValueError, "frozen values"):
                verify_development_samples(config, new_root)

    def test_queue_matrix_separation(self):
        self.assertEqual(len(jobs(False)), 96)
        self.assertEqual(len(jobs(True)), 108)
        self.assertTrue(all(not job["aux"] for job in jobs(True)[:96]))
        self.assertTrue(all(job["aux"] and job["head"] == "adaptive" and job["rmw"] == "on" for job in jobs(True)[96:]))

    def test_epoch_resume_matches_uninterrupted(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="synthetic_sanity_", dir=here) as temporary:
            root = Path(temporary)
            data = root / "data"
            stats = {"target_mean": [3.0, -4.0], "target_std": [45.0, 80.0]}
            metadata = {"input_len": 12, "input_dim": 3, "lead_steps": [4], "interval_hours": 6,
                        "feature_names": ["lat", "lon", "rmax"], "stats": stats,
                        "split_storm_ids": {"train": ["TRAIN"], "val": ["VAL"], "test": ["TEST"]}}
            rng = np.random.default_rng(123)
            for split in ("train", "val", "test"):
                (data / split).mkdir(parents=True)
                for i in range(4):
                    history = np.stack([np.linspace(20, 21, 12), np.linspace(-60, -59, 12)], axis=-1).astype("float32")
                    raw = np.array([[30.0 + i, 60.0 + i]], dtype="float32")
                    origin = history[-1]
                    future = np.array([[origin[0] + np.rad2deg(raw[0, 1] / 6371),
                                        origin[1] + np.rad2deg(raw[0, 0] / (6371 * np.cos(np.deg2rad(origin[0]))))]], dtype="float32")
                    np.save(data / split / f"{i}.npy", {"input_num": rng.normal(size=(12, 3)).astype("float32"),
                            "target_disp": (raw - stats["target_mean"]) / stats["target_std"],
                            "target_disp_raw_km": raw, "history_latlon": history, "origin": origin,
                            "future_latlon": future, "storm_id": split.upper(), "window": i})
            (data / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            common = [sys.executable, str(here / "run_controlled.py"), "--data-dir", str(data),
                      "--head", "adaptive", "--rmw", "off", "--epochs", "2", "--batch-size", "2",
                      "--d-model", "16", "--num-layers", "1", "--dropout", "0.1", "--device", "cpu", "--threads", "1"]
            def run(extra):
                result = subprocess.run(common + extra, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run(["--output", str(root / "full"), "--evaluate-test"])
            run(["--output", str(root / "resumed"), "--stop-after-epochs", "1"])
            self.assertFalse((root / "resumed" / "test_predictions.npz").exists())
            run(["--output", str(root / "resumed"), "--resume", "--evaluate-test"])
            full = torch.load(root / "full" / "latest.pt", weights_only=False)
            resumed = torch.load(root / "resumed" / "latest.pt", weights_only=False)
            for name, value in full["model"].items():
                torch.testing.assert_close(value, resumed["model"][name], rtol=0, atol=0)
            with np.load(root / "full" / "test_predictions.npz") as a, np.load(root / "resumed" / "test_predictions.npz") as b:
                np.testing.assert_array_equal(a["pred_latlon"], b["pred_latlon"])
            config = json.loads((root / "full" / "frozen_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["features"], ["lat", "lon"])
            self.assertEqual(config["excluded_features"], ["rmax"])
            result = subprocess.run([sys.executable, str(here / "compare_storms.py"),
                                      "--group", "full=" + str(root / "full"), "--group", "resumed=" + str(root / "resumed"),
                                      "--replicates", "100", "--output", str(root / "comparison.json")],
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            comparison = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(comparison["comparisons"][0]["difference_A_minus_B_window_mean_km"], [0.0])


if __name__ == "__main__":
    unittest.main()
