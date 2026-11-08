"""Synthetic implementation checks only; not MLP forecasting evidence."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from controlled_heads import cv_displacement
from mlp_residual_baseline import NumericMLPResidual, fit_one, normalization, subset
from vendor.common import denormalize_disp, disp_to_latlon, normalize_disp


class MLPComparatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.x = torch.randn(4, 12, 3)
        self.history = torch.tensor([[[20.0, -60.0], [20.5, -59.0]]] * 4)

    def test_exact_architecture_zero_residual_and_future_independence(self):
        mean, std = normalization(self.x)
        model = NumericMLPResidual(mean, std, [13, -7], [45, 80], [4]).eval()
        linear = [module for module in model.network if isinstance(module, torch.nn.Linear)]
        self.assertEqual([(module.in_features, module.out_features) for module in linear], [(36, 128), (128, 128), (128, 2)])
        for parameter in model.network.parameters():
            torch.nn.init.zeros_(parameter)
        predicted = model(self.x, self.history)["pred"]
        torch.testing.assert_close(denormalize_disp(predicted, model.target_mean, model.target_std), cv_displacement(self.history, [4]))
        record = {"x": self.x, "history": self.history, "future": torch.full((4, 1, 2), 999.0)}
        before = model(record["x"], record["history"])["pred"]
        record["future"].fill_(-999.0)
        torch.testing.assert_close(before, model(record["x"], record["history"])["pred"], rtol=0, atol=0)

    def test_rmw_removed_before_train_only_normalization(self):
        data = {"keys": [("STORM", i) for i in range(4)], "tensors": {"x": self.x}}
        without = subset(data, [0, 1])
        self.assertEqual(without["tensors"]["x"].shape, (4, 12, 2))
        mean, std = normalization(without["tensors"]["x"])
        normalized = (without["tensors"]["x"].flatten(1) - mean) / std
        torch.testing.assert_close(normalized.mean(0), torch.zeros(24), atol=1e-6, rtol=0)

    def test_tiny_fit_and_resume_are_identical(self):
        root = Path(__file__).resolve().parent
        raw = cv_displacement(self.history, [4]) + torch.tensor([[[3.0, -2.0]]])
        origin = self.history[:, -1]
        mean, std = torch.tensor([13.0, -7.0]), torch.tensor([45.0, 80.0])
        data = {"keys": [("STORM", i) for i in range(4)], "tensors": {"x": self.x, "history": self.history,
                 "origin": origin, "y": normalize_disp(raw, mean, std), "raw": raw,
                 "future": disp_to_latlon(raw, origin)}}
        metadata = {"stats": {"target_mean": mean.tolist(), "target_std": std.tolist()}, "lead_steps": [4]}
        with tempfile.TemporaryDirectory(prefix="mlp_synthetic_", dir=root) as temporary:
            folder = Path(temporary)
            with patch("mlp_residual_baseline.EPOCHS", 2):
                full, _, _ = fit_one(folder / "full", {"seed": 42}, metadata, data, data)
            with patch("mlp_residual_baseline.EPOCHS", 1):
                fit_one(folder / "resumed", {"seed": 42}, metadata, data, data)
            with patch("mlp_residual_baseline.EPOCHS", 2):
                resumed, _, _ = fit_one(folder / "resumed", {"seed": 42}, metadata, data, data)
            for name, tensor in full.state_dict().items():
                torch.testing.assert_close(tensor, resumed.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
