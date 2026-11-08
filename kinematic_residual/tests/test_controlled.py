"""Implementation checks on synthetic tensors; these are NOT scientific results."""
import unittest

import numpy as np
import torch

from controlled_heads import ControlledNumericModel, cv_displacement
from compare_storms import paired_bootstrap
from vendor.common import constant_velocity_future_disp, denormalize_disp, disp_to_latlon, latlon_to_disp


class ControlledTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x = torch.randn(3, 12, 5)
        self.history = torch.tensor([[[20.0, -60.0], [20.5, -59.0]],
                                     [[45.0, -55.0], [46.0, -53.0]],
                                     [[10.0, 179.5], [10.2, -179.5]]])

    def model(self, head):
        torch.manual_seed(42)
        return ControlledNumericModel(5, 12, [4], [13.0, -7.0], [45.0, 80.0],
                                      head, d_model=16, num_heads=4, num_layers=1, dropout=0.0)

    def test_shared_backbone_initialization(self):
        models = [self.model(head) for head in ("direct", "residual", "fixed", "adaptive")]
        for model in models[1:]:
            for name, tensor in models[0].state_dict().items():
                if name.startswith(("traj_encoder.", "backbone.", "fusion.", "context_")):
                    torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)

    def test_future_label_invariance(self):
        # Full record labels/analog/text can change; inference only consumes x/history.
        sample = {"x": self.x, "history": self.history, "future": torch.randn(3, 1, 2),
                  "target": torch.randn(3, 1, 2), "analog_neg": torch.randn(3, 5, 1, 2)}
        for head in ("direct", "residual", "fixed", "adaptive"):
            model = self.model(head).eval()
            before = model(sample["x"], sample["history"])["pred"].detach()
            sample["future"].fill_(999.0)
            sample["target"].fill_(-999.0)
            sample["analog_neg"].fill_(float("nan"))
            after = model(sample["x"], sample["history"])["pred"].detach()
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_cv_zero_residual_and_lead(self):
        model = self.model("residual").eval()
        for parameter in model.residual_head.parameters():
            torch.nn.init.zeros_(parameter)
        norm = model(self.x, self.history)["pred"]
        raw = denormalize_disp(norm, model.target_mean, model.target_std)
        independent = constant_velocity_future_disp(self.history, 1, [4])
        torch.testing.assert_close(raw, independent)
        forced = self.model("adaptive").eval()(self.x, self.history, force_cv=True)["pred"]
        torch.testing.assert_close(norm, forced)
        adaptive = self.model("adaptive").eval()
        for parameter in adaptive.residual_head.parameters():
            torch.nn.init.zeros_(parameter)
        for parameter in adaptive.dual_gate_head.parameters():
            torch.nn.init.zeros_(parameter)
        adaptive.dual_gate_head[-1].bias.data.fill_(100.0)
        torch.testing.assert_close(adaptive(self.x, self.history)["pred"], norm)

    def test_fixed_and_adaptive_formulas(self):
        for head in ("fixed", "adaptive"):
            model = self.model(head).eval()
            result = model(self.x, self.history)
            expected = result["gate"] * result["cv_residual"] + (1 - result["gate"]) * result["direct"]
            torch.testing.assert_close(result["pred"], expected)
            if head == "fixed":
                torch.testing.assert_close(result["gate"], torch.full((3, 1, 1), 0.5))
            result["pred"].square().mean().backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_coordinate_roundtrip_and_convention_difference(self):
        displacement = cv_displacement(self.history, [1, 4])
        origin = self.history[:, -1]
        coordinates = disp_to_latlon(displacement, origin)
        recovered = latlon_to_disp(coordinates, origin[:, None])
        torch.testing.assert_close(recovered, displacement, rtol=1e-4, atol=0.003)
        legacy = cv_displacement(self.history, [4])
        origin_based = cv_displacement(self.history, [4], "forecast_origin")
        self.assertGreater((legacy - origin_based).abs().max().item(), 1.0)

    def test_storm_bootstrap_is_clustered(self):
        storms = np.array(["A", "A", "B"])
        a = np.array([[3.0], [3.0], [0.0]])
        b = np.zeros_like(a)
        result = paired_bootstrap(a, b, storms, 100, 42)
        self.assertEqual(result["n_storms"], 2)
        self.assertEqual(result["difference_A_minus_B_window_mean_km"], [2.0])
        self.assertEqual(result["difference_A_minus_B_storm_equal_km"], [1.5])
        self.assertEqual(result["fraction_storms_B_improves"], [0.5])


if __name__ == "__main__":
    unittest.main()
