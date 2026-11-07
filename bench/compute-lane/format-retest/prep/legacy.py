"""Private legacy module graph: no sys.path or sys.modules modifications."""
import builtins
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PATHS = {
    "a2": ROOT / "compute-dl/a2-continuation/request_adapter.py",
    "client3": ROOT / "decl-control/client3.py",
    "reconciliation": ROOT / "decl-control/reconciliation.py",
    "client": ROOT / "phase2/client.py",
    "native_frames": ROOT / "phase2/native_frames.py",
    "disjointness": ROOT / "compute-dl/disjointness.py",
    "reference_a": ROOT / "compute-dl/reference_a.py",
    "reference_b": ROOT / "compute-dl/reference_b.py",
    "exclusion_sources": ROOT / "compute-dl/exclusion_sources.py",
}


class Legacy:
    def __init__(self):
        self.modules = {}
        self.private_sys = SimpleNamespace(**vars(sys))
        self.private_sys.path = list(sys.path)

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name == "sys":
            return self.private_sys
        if level == 0 and name in PATHS:
            return self.load(name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    def load(self, name):
        if name not in self.modules:
            path = PATHS[name]
            module = ModuleType("_format_retest_legacy_" + name)
            module.__file__ = str(path)
            module.__dict__["__builtins__"] = dict(vars(builtins), __import__=self._import)
            self.modules[name] = module
            try:
                exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
            except BaseException:
                del self.modules[name]
                raise
        return self.modules[name]
