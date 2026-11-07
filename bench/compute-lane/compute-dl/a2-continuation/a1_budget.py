"""Published alias of the unchanged A1 budget, with its original limits."""
import importlib.util
from pathlib import Path

_source = Path(__file__).resolve().parent.parent / "a1-continuation/budget.py"
_spec = importlib.util.spec_from_file_location("_published_a1_budget", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def __getattr__(name):
    return getattr(_module, name)
