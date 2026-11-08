"""Read-only source pins; imports never execute model content."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent
SPEC_SHA256 = '0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453'
ORIGINAL_SPEC_SHA256 = 'c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'
STRICT_SHA256 = '1a2249a1b5e0b06f03a4300995d0c320e5a21630d261b3791501f7c83740fb9f'
SELECTOR_SHA256 = 'dcc8ef766247a1b02c993c9a50186b70b68c4fb5b6493e1bb42b3f7727186d36'
ANALYSIS_SHA256 = '71101f372473fc5be4a7f9896fb5b3e8ef375620c1dca87f48bb8559ddb292ae'


def pinned_bytes(path, expected):
    blob = Path(path).read_bytes()
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError('immutable source pin mismatch: ' + str(path))
    return blob


def load_local(name):
    path = ROOT / (name + '.py')
    spec = importlib.util.spec_from_file_location('_compute_a1_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_strict():
    path = ORIGINAL / 'grade.py'
    blob = pinned_bytes(path, STRICT_SHA256)
    module = types.ModuleType('_compute_a1_immutable_strict')
    module.__file__ = str(path)
    exec(compile(blob, str(path), 'exec'), module.__dict__)
    return module


def load_selector():
    # Import only the exact pinned pure selector AST. The full phase2 module
    # imports bench/Jail and a different analysis module; none are needed here.
    path = ORIGINAL.parent / 'phase2/grade_all.py'
    tree = ast.parse(pinned_bytes(path, SELECTOR_SHA256), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'gsm_final']
    if len(selected) != 1:
        raise ValueError('pinned final-span selector unavailable')
    namespace = {'re': __import__('re')}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['gsm_final']


def verify_protocols():
    pinned_bytes(ORIGINAL / 'T5-COMPUTE-DL-SPEC.md', ORIGINAL_SPEC_SHA256)
    pinned_bytes(ORIGINAL / 'T5-COMPUTE-DL-SPEC-A1.md', SPEC_SHA256)
    pinned_bytes(ORIGINAL / 'analysis.py', ANALYSIS_SHA256)
