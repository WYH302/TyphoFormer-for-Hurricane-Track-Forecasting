"""Explicit executable dependencies; unrelated analysis edits cannot invalidate fits."""
import hashlib
from pathlib import Path


TRAINING_FILES = ("provenance.py", "run_controlled.py", "controlled_heads.py",
                  "vendor/common.py", "vendor/model.py")
EVALUATION_FILES = TRAINING_FILES + ("evaluate_locked.py",)
QUEUE_FILES = EVALUATION_FILES + ("run_matrix.py", "process_scheduler.py", "compare_storms.py")


def source_hashes(filenames, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in filenames}
