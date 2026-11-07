"""Restore the omitted identical A1 module while keeping this caller's directory."""
from pathlib import Path

_source = Path(__file__).resolve().parent.parent / "a1-continuation/rank_ops.py"
exec(compile(_source.read_bytes(), str(_source), "exec"))
