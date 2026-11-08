"""Focused process scheduling and provenance tests, with no model training."""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from process_scheduler import ProcessRegistry, schedule
from provenance import TRAINING_FILES, source_hashes
from run_matrix import execute


class SchedulerTests(unittest.TestCase):
    def test_bounded_processes_and_unique_status(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="scheduler_sanity_", dir=here) as temporary:
            root = Path(temporary)
            registry, updates = ProcessRegistry(), []
            parent_thread = threading.get_ident()
            def worker(name, delay):
                execute([sys.executable, "-c", f"import time; time.sleep({delay})"],
                        root / (name + ".log"), root / (name + ".json"), {"status": "training", "job": name}, registry)
            def progress(state):
                self.assertEqual(threading.get_ident(), parent_thread)
                updates.append(state)
            result = schedule([(f"job{i}", 0.1) for i in range(5)], worker, 2, progress, registry)
            self.assertEqual(len(result["completed"]), 5)
            self.assertFalse(result["failed"])
            self.assertEqual(max(len(state["active_jobs"]) for state in updates), 2)
            for i in range(5):
                status = json.loads((root / f"job{i}.json").read_text(encoding="utf-8"))
                self.assertEqual(status["job"], f"job{i}")
                self.assertEqual(status["status"], "completed")
            self.assertFalse(registry.processes)

    def test_failure_drains_running_and_does_not_dispatch_more(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="scheduler_failure_", dir=here) as temporary:
            root = Path(temporary)
            registry, started = ProcessRegistry(), []
            def worker(name, script):
                started.append(name)
                execute([sys.executable, "-c", script], root / (name + ".log"),
                        root / (name + ".json"), {"status": "training", "job": name}, registry)
            result = schedule([("failure", "raise SystemExit(3)"), ("running", "import time; time.sleep(0.3)"),
                               ("not_dispatched", "raise SystemExit(99)")], worker, 2, lambda state: None, registry)
            self.assertCountEqual(started, ["failure", "running"])
            self.assertEqual(result["completed"], ["running"])
            self.assertEqual(result["failed"][0]["job"], "failure")
            self.assertFalse(registry.processes)

    def test_ctrl_c_terminates_this_invocations_children(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="scheduler_interrupt_", dir=here) as temporary:
            root = Path(temporary)
            started = threading.Event()
            children = []
            class NotifyingRegistry(ProcessRegistry):
                def start(self, command, **kwargs):
                    process = super().start(command, **kwargs)
                    children.append(process)
                    started.set()
                    return process
            registry = NotifyingRegistry()
            def worker(name, payload):
                execute([sys.executable, "-c", "import time; time.sleep(30)"], root / "job.log",
                        root / "job.json", {"status": "training", "job": name}, registry)
            def interrupt(*args, **kwargs):
                self.assertTrue(started.wait(5))
                raise KeyboardInterrupt()
            with patch("process_scheduler.wait", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    schedule([("long", None)], worker, 1, lambda state: None, registry)
            self.assertTrue(children)
            self.assertTrue(all(child.poll() is not None for child in children))
            self.assertFalse(registry.processes)

    def test_unrelated_analysis_does_not_invalidate_training(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="provenance_sanity_", dir=here) as temporary:
            root = Path(temporary)
            for name in TRAINING_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("frozen", encoding="utf-8")
            before = source_hashes(TRAINING_FILES, root)
            (root / "analyze_revision.py").write_text("changed independently", encoding="utf-8")
            self.assertEqual(before, source_hashes(TRAINING_FILES, root))
            (root / "controlled_heads.py").write_text("core changed", encoding="utf-8")
            self.assertNotEqual(before, source_hashes(TRAINING_FILES, root))


if __name__ == "__main__":
    unittest.main()
