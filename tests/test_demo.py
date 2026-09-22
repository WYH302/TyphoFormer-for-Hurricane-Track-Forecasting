import importlib.util
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Load by path so an unrelated package named 'demo' cannot be selected.
spec = importlib.util.spec_from_file_location("local_results_demo", ROOT / "demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
from demo_ui import render

class DemoTests(unittest.TestCase):
    def test_rows_are_finite_and_provenanced(self):
        import math
        result = demo.load()
        self.assertTrue(result["rows"])
        self.assertEqual(len(result["sha256"]), 64)
        for row in result["rows"]:
            self.assertTrue(math.isfinite(row["value"]))
        self.assertTrue((ROOT / result["source"]).is_file())

    def test_viewer_is_offline_and_escapes_embedded_data(self):
        payload = demo.load()
        payload["rows"][0]["label"] = "</script><img src=x onerror=alert(1)>"
        page = render("Example <title>", "Only retained data.", payload)
        self.assertNotIn("</script><img", page)
        self.assertIn("Example &lt;title&gt;", page)
        self.assertNotIn("https://", page)
        self.assertIn('id="filters"', page)
        self.assertIn("Download selected rows", page)

if __name__ == "__main__":
    unittest.main()
