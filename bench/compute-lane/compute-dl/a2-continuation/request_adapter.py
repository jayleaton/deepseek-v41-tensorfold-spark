"""Published alias of the unchanged A1 request adapter omitted as a duplicate."""
from pathlib import Path

# Execute the retained bytes in this namespace so the private Legacy importer
# keeps control of imports, and relative paths retain the original source base.
_source = Path(__file__).resolve().parent.parent / "a1-continuation/request_adapter.py"
__file__ = str(_source)
exec(compile(_source.read_bytes(), str(_source), "exec"))
