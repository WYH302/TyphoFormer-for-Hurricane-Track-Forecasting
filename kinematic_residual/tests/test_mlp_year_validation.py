"""Regression for named HURDAT storm IDs in the evaluation-only finalizer."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlp_residual_finalize import verify_years


class YearValidationTests(unittest.TestCase):
    def test_explicit_year_is_used_instead_of_named_id_suffix(self):
        with tempfile.TemporaryDirectory(prefix="mlp_year_sanity_", dir=Path(__file__).parent) as temporary:
            path = Path(temporary) / "AL012022_ALEX.npy"
            np.save(path, {"storm_id": "AL012022_ALEX", "storm_year": np.int64(2022)})
            self.assertEqual(verify_years([path], {2022, 2023, 2024}), [2022])
            with self.assertRaisesRegex(ValueError, "Unexpected explicit"):
                verify_years([path], {2025})
            np.save(path, {"storm_id": "AL012022_ALEX"})
            with self.assertRaisesRegex(ValueError, "Missing explicit numeric"):
                verify_years([path], {2022})


if __name__ == "__main__":
    unittest.main()
