"""Published-layout adapter for the unchanged historical benchmark module."""
import importlib.util
from pathlib import Path

_source = Path(__file__).resolve().parent.parent / "bench.py"
_spec = importlib.util.spec_from_file_location("_published_stdbench", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def __getattr__(name):
    return getattr(_module, name)
