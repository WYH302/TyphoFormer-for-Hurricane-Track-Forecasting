import sys
from pathlib import Path

_PKG_DIR = str(Path(__file__).resolve().parents[1])
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
