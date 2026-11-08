"""Published alias of the unchanged A1 read-only coordination checks."""
import importlib.util
from pathlib import Path

_source = Path(__file__).resolve().parents[2] / "compute-dl/a1-continuation/coordination.py"
_spec = importlib.util.spec_from_file_location("_published_format_coordination", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def __getattr__(name):
    return getattr(_module, name)
