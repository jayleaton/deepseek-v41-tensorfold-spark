"""This gate is a guard against accidental bypass in our own code. It is not a sandbox against deliberately obfuscated code."""
import ast
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest

from tree_sitter import Language, Parser
import tree_sitter_bash

ROOT = Path(__file__).resolve().parents[2]
# New host shell helpers are audited automatically. The publication scanner is
# a read-only checker; the central adapter has its own stricter test below.
SHELLS = tuple(sorted(p.relative_to(ROOT).as_posix()
                      for folder in ('scripts', 'docker')
                      for p in (ROOT / folder).rglob('*.sh')
                      if p.relative_to(ROOT).as_posix() not in
                      {'scripts/check-public.sh', 'scripts/safe_shell.sh'}))

BUILTINS = {'set', 'unset', 'export', 'readonly', 'local', 'declare',
            'return', 'exit', 'printf', 'echo', 'read', 'cd', 'pwd', 'shift', ':',
            'umask', 'compgen', 'wait', 'test', 'true', 'false', 'break', 'continue'}
ADAPTERS = {'safe_command', 'safe_capture', 'safe_filter', 'safe_shell', 'safe_main', 'safe_check_argv'}
# These are executable, trusted operator configuration, not arbitrary source
# targets. Library paths are separately pinned to the audited source tree.
SOURCES = {
    'scripts/serve.sh': {'"$safe_dir/safe_shell.sh"', '"$CONFIG"'},
    'scripts/run-zig.sh': {'"$safe_dir/safe_shell.sh"', '"$config"'},
    'scripts/guard-watch-zig.sh': {'"$safe_dir/safe_shell.sh"', '"$config"'},
    'scripts/build-native.sh': {'"$safe_dir/safe_shell.sh"'},
    'scripts/check-zig-image.sh': {'"$safe_dir/safe_shell.sh"'},
    'docker/zig/fetch-source.sh': {'"$safe_dir/../../scripts/safe_shell.sh"'},
}
# Checked argv forwarding is reviewed at these exact sites. It never permits a
# raw dynamic command node; that remains exclusive to safe_shell's function adapter.
FORWARDING = {
    ('scripts/guard-watch-zig.sh', 'watch_run'): 'safe_command "$@"',
    ('scripts/guard-watch-zig.sh', 'watch_capture'): 'safe_capture "$@"',
    ('scripts/guard-watch-zig.sh', '_watch_alert'): 'watch_run "$WATCH_ALERT" "$message"',
    ('scripts/run-zig.sh', None): 'safe_command "${args[@]}"',
}


def python_bypasses(source, reviewed_dispatch=False):
    """Conservative, scope-insensitive aliases reach a fixpoint; overwrites never erase a possible spawn."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [(0, 'unparsed Python payload')]
    aliases = {name: {name} for name in ('os', 'posix', 'subprocess', 'asyncio', 'pty', 'builtins',
                                        'multiprocessing', 'concurrent')}
    aliases.update({name: {name} for name in ('getattr', 'eval', 'exec', 'compile', '__import__')})
    slots = {}
    assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases.setdefault(alias.asname or alias.name.split('.')[0], set()).add(
                    alias.name if alias.asname else alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases.setdefault(alias.asname or alias.name, set()).add(f'{node.module}.{alias.name}')
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            aliases.setdefault(node.name, set()).add(f'<local:{node.name}>')
        elif isinstance(node, ast.Assign):
            assignments.extend((target, node.value) for target in node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            assignments.append((node.target, node.value))

    def resolve(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, set())
        if isinstance(node, ast.Attribute):
            return slots.get(ast.unparse(node), set()) | {f'{base}.{node.attr}' for base in resolve(node.value)}
        if isinstance(node, ast.Call) and resolve(node.func) & {'getattr', 'builtins.getattr'}:
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                bases = resolve(node.args[0])
                return {f'{base}.{node.args[1].value}' for base in bases} if bases else {'<dynamic callable>'}
            return {'<dynamic callable>'}
        if isinstance(node, ast.Call) and resolve(node.func) & {'asyncio.get_event_loop', 'asyncio.get_running_loop',
                                                               'asyncio.new_event_loop'}:
            return {'asyncio.loop'}
        if isinstance(node, ast.Subscript):
            return (slots.get(ast.unparse(node), set()) | resolve(node.value)) or {'<dynamic callable>'}
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return set().union(*(resolve(value) for value in node.elts))
        if isinstance(node, ast.Dict):
            return set().union(*(resolve(value) for value in node.values))
        if isinstance(node, ast.IfExp):
            return resolve(node.body) | resolve(node.orelse)
        if isinstance(node, ast.BoolOp):
            return set().union(*(resolve(value) for value in node.values))
        return set()

    def bind(target, value):
        if isinstance(target, ast.Name):
            before = len(aliases.setdefault(target.id, set()))
            aliases[target.id].update(resolve(value))
            return len(aliases[target.id]) != before
        if isinstance(target, (ast.Attribute, ast.Subscript)):
            key = ast.unparse(target)
            before = len(slots.setdefault(key, set()))
            slots[key].update(resolve(value))
            return len(slots[key]) != before
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
            changed = [bind(left, right) for left, right in zip(target.elts, value.elts)]
            return any(changed)
        return False

    # Union rather than overwrite handles branches, chained assignments and
    # aliases discovered later in the source, without pretending to model execution.
    for _ in range(len(assignments) + 1):
        changed = [bind(target, value) for target, value in assignments]
        if not any(changed):
            break
    else:
        return [(0, 'alias dataflow did not converge')]

    def forbidden(name):
        module, _, method = name.rpartition('.')
        if module in {'os', 'posix'}:
            return (method in {'system', 'popen', 'fork', 'forkpty', 'startfile'}
                    or method.startswith(('spawn', 'posix_spawn', 'exec')))
        if module == 'subprocess':
            return method in {'run', 'Popen', 'call', 'check_call', 'check_output', 'getoutput', 'getstatusoutput'}
        if module.startswith('asyncio'):
            return method in {'create_subprocess_exec', 'create_subprocess_shell', 'subprocess_exec', 'subprocess_shell'}
        if module == 'pty':
            return method in {'spawn', 'fork'}
        if module.startswith('multiprocessing'):
            return method in {'Process', 'Pool', 'get_context'}
        if module == 'concurrent.futures':
            return method == 'ProcessPoolExecutor'
        return name in {'<dynamic callable>', 'eval', 'exec', 'compile', '__import__',
                        'builtins.eval', 'builtins.exec', 'builtins.compile', 'builtins.__import__'}

    def literal(value):
        return value.value if isinstance(value, ast.Constant) else None

    reviewed_shape = ast.dump(ast.parse('getattr(ops, args.operation)()', mode='eval').body,
                              include_attributes=False)
    reviewed_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and ast.dump(node, include_attributes=False) == reviewed_shape]
    violations = [(node.lineno, 'wildcard process API import') for node in ast.walk(tree)
                  if isinstance(node, ast.ImportFrom) and node.module in
                  {'subprocess', 'os', 'posix', 'asyncio', 'pty', 'multiprocessing', 'concurrent.futures'}
                  and any(alias.name == '*' for alias in node.names)]
    for target, value in assignments:
        if (any(isinstance(part, (ast.Attribute, ast.Subscript)) for part in ast.walk(target))
                and any(name != '<dynamic callable>' and forbidden(name) for name in resolve(value))):
            violations.append((target.lineno, 'process callable escapes through stored attribute/item'))
    if reviewed_dispatch and len(reviewed_calls) != 1:
        violations.append((0, 'reviewed lifecycle dispatch changed'))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # native_ops.main validates the operation against argparse choices; the
        # exception is the exact call, never the file or every getattr invocation.
        reviewed = reviewed_dispatch and len(reviewed_calls) == 1 and node is reviewed_calls[0]
        targets = resolve(node.func)
        if targets & {'safe_exec.run', 'scripts.safe_exec.run'}:
            argv = node.args[0] if node.args else next(
                (keyword.value for keyword in node.keywords if keyword.arg == 'argv'), None)
            # Only explicit literal argv trampolines are inspected here. Runtime
            # argv validation belongs to safe_exec; this is not a general call graph.
            if isinstance(argv, (ast.List, ast.Tuple)) and argv.elts:
                executable = literal(argv.elts[0])
                executable = executable.rsplit('/', 1)[-1] if isinstance(executable, str) else None
                is_python = bool(executable and re.fullmatch(r'python(?:[23](?:\.\d+)*)?', executable))
                if is_python or executable in {'bash', 'sh', 'dash', 'zsh', 'ksh', 'ash'}:
                    values = [literal(value) for value in argv.elts]
                    flags = [index for index, value in enumerate(values) if value == '-c'
                             or (not is_python and isinstance(value, str) and value.startswith('-')
                                 and not value.startswith('--') and 'c' in value[1:])]
                    if flags:
                        index = flags[0] + 1
                        payload = literal(argv.elts[index]) if index < len(argv.elts) else None
                        if not isinstance(payload, str):
                            violations.append((node.lineno, 'dynamic interpreter adapter payload'))
                        else:
                            nested = python_bypasses(payload) if is_python else bypasses(payload.encode())
                            violations.extend((node.lineno, f'interpreter adapter payload: {reason}')
                                              for _, reason in nested)
        if any(forbidden(target) for target in targets) and not reviewed:
            violations.append((node.lineno, ast.unparse(node.func)))
        elif isinstance(node.func, (ast.Call, ast.Subscript)) and not targets and not reviewed:
            violations.append((node.lineno, 'unresolved callable dispatch'))
    return violations


def shell_literal(node):
    if node.type not in {'word', 'string', 'raw_string'}:
        return None
    if any(child.type in {'expansion', 'simple_expansion', 'command_substitution', 'arithmetic_expansion'}
           for child in nodes(node)):
        return None
    try:
        values = shlex.split(node.text.decode())
    except ValueError:
        return None
    return values[0] if len(values) == 1 else None


def nodes(root):
    yield root
    for child in root.children:
        yield from nodes(child)


def enclosing_function(node):
    while node.parent is not None:
        node = node.parent
        if node.type == 'function_definition':
            return node.child_by_field_name('name').text.decode()
    return None


def has_stdin_payload(node):
    while node.parent is not None and node.parent.type == 'redirected_statement':
        node = node.parent
    return any(part.type in {'heredoc_redirect', 'herestring_redirect'}
               or (part.type == 'file_redirect' and part.text.lstrip().startswith(b'<'))
               for part in nodes(node))


def bypasses(source, filename=None):
    parser = Parser(Language(tree_sitter_bash.language()))
    tree = parser.parse(source)
    assert not tree.root_node.has_error, 'unparsed launcher syntax invalidates boundary audit'
    functions = {n.child_by_field_name('name').text.decode() for n in nodes(tree.root_node)
                 if n.type == 'function_definition'}
    violations = []
    for node in nodes(tree.root_node):
        if node.type != 'command':
            continue
        command = node.child_by_field_name('name')
        if command is None:
            continue
        name = command.text.decode()
        arguments = node.children_by_field_name('argument')
        line = node.start_point.row + 1
        if name in {'source', '.'}:
            if len(arguments) != 1 or arguments[0].text.decode() not in SOURCES.get(filename, set()):
                violations.append((line, 'unreviewed executable source'))
        elif name == 'trap':
            payload = shell_literal(arguments[0]) if arguments else None
            if payload is None:
                violations.append((line, 'dynamic trap payload'))
            elif payload != '-':
                violations.extend((line, f'trap: {reason}') for _, reason in bypasses(payload.encode()))
        elif name not in BUILTINS | ADAPTERS | functions:
            violations.append((line, name))
        # Inspect the interpreter itself, not arbitrary later words in e.g.
        # Docker argv. Function bodies remain audited above, including wrappers.
        offset = {'safe_command': 0, 'safe_capture': 1, 'watch_run': 0, 'watch_capture': 1}.get(name)
        if offset is None:
            continue
        invocation = ' '.join([name, *(argument.text.decode() for argument in arguments)])
        if invocation == FORWARDING.get((filename, enclosing_function(node))):
            if has_stdin_payload(node):
                violations.append((line, 'stdin payload on dynamic argv forwarding'))
            continue
        if len(arguments) <= offset:
            violations.append((line, 'unreviewed argv forwarding'))
            continue
        argv = arguments[offset:]
        if shell_literal(argv[0]) == 'flock':
            # The supported lock syntax has optional -n, one lock path, then
            # argv. Inspect that program so a lock cannot hide bash stdin/-c.
            lock_at = 2 if len(argv) > 1 and shell_literal(argv[1]) == '-n' else 1
            argv = argv[lock_at + 1:]
            if not argv:
                violations.append((line, 'unreviewed flock dispatch'))
                continue
            if (filename == 'scripts/guard-watch-zig.sh' and enclosing_function(node) == '_watch_watch_tick'
                    and invocation == 'watch_run flock "$WATCH_STATE_DIR/lock" python3 "$safe_exec" --restart'):
                if has_stdin_payload(node):
                    violations.append((line, 'stdin payload on reviewed restart'))
                continue  # safe_exec admits only same-repo serve.sh watch --once.
        executable = shell_literal(argv[0])
        if executable is None:
            violations.append((line, 'unreviewed dynamic command dispatch'))
            continue
        executable = executable.rsplit('/', 1)[-1]
        is_python = bool(re.fullmatch(r'python(?:[23](?:\.\d+)*)?', executable))
        if not is_python and executable not in {'bash', 'sh', 'dash', 'zsh', 'ksh', 'ash'}:
            continue
        literals = [shell_literal(arg) for arg in argv[1:]]
        code_flags = [index for index, value in enumerate(literals)
                      if value == '-c' or (not is_python and value and value.startswith('-')
                                           and not value.startswith('--') and 'c' in value[1:])]
        if code_flags:
            index = code_flags[0] + 1
            payload = literals[index] if index < len(literals) else None
            if payload is None:
                violations.append((line, 'dynamic interpreter payload'))
            elif is_python:
                violations.extend((line, f'Python payload: {reason}') for _, reason in python_bypasses(payload))
            else:
                violations.extend((line, f'shell payload: {reason}') for _, reason in bypasses(payload.encode()))
        elif not literals or literals[0] is None or literals[0].startswith('-'):
            violations.append((line, 'unreviewed interpreter source'))
        else:
            # Only source files already covered by this gate are executable
            # script inputs. This excludes stdin paths and generated heredoc files.
            script = literals[0]
            audited = ((script.startswith('scripts/') and script.endswith('.py')
                        and (ROOT / script).is_file()) if is_python else script in SHELLS)
            if not audited:
                violations.append((line, 'unaudited interpreter script'))
    return violations


def test_shell_orchestrators_have_no_direct_subprocess_or_exec():
    violations = {name: bypasses((ROOT / name).read_bytes(), name) for name in SHELLS}
    assert not {name: values for name, values in violations.items() if values}


def test_audit_catches_new_commands_in_nested_and_quoted_contexts():
    # Prove the source guard catches future bypasses rather than grepping only
    # the known leaking command names or trusting a whole line containing safe_command.
    for snippet in (b'docker inspect x', b'exec docker run x', b'safe_command true; unexpected x',
                    b'echo "$(curl x)"', b'f() { ( "$@" ); }', b'true > >(ssh worker x)'):
        assert bypasses(snippet), 'subprocess guard missed an injected escape hatch'
    assert not bypasses(b'safe_command docker inspect x; printf "%s" "docker is data"')


def test_python_orchestrators_have_no_direct_subprocess_or_exec():
    exceptions = {'scripts/safe_exec.py', 'scripts/check_public_history.py', 'scripts/check_secrets.py'}
    forbidden = []
    for path in (ROOT / 'scripts').rglob('*.py'):
        if path.relative_to(ROOT).as_posix() in exceptions:
            continue
        name = path.relative_to(ROOT).as_posix()
        source = path.read_text()
        reviewed = name == 'scripts/native_ops.py'
        if reviewed:
            assert "choices=('preflight', 'start', 'stop', 'restart', 'status', 'watch')" in source
            assert 'ops = Lifecycle()' in source
        forbidden.extend((name, line, reason) for line, reason in python_bypasses(source, reviewed))
    assert not forbidden


@pytest.mark.parametrize('source', [
    'import os; os.system("true")',
    'import subprocess as sp; sp.Popen(["true"])',
    'import subprocess; spawn = subprocess.Popen; spawn(["true"])',
    'import subprocess; sp = subprocess; sp.Popen(["true"])',
    'import subprocess as original; module = original; spawn = module.Popen; launch = spawn; launch(["true"])',
    'import subprocess; launch = later; later = subprocess.Popen; launch(["true"])',
    'from subprocess import Popen as launch; launch(["true"])',
    'import os; getattr(os, "system")("true")',
    'import os; launch = getattr(os, "system"); launch("true")',
    'import os; module = os; get = getattr; launch = get(module, "system"); launch("true")',
    'import os; launch = getattr(os, setting); launch("true")',
    'getattr(receiver, operation)()',
    'launch = getattr(receiver, "method"); launch()',
    'launch = handlers[operation]; launch()',
    'import os; os.posix_spawn("/bin/true", ["true"], {})',
    'from os import posix_spawnp as launch; launch("true", ["true"], {})',
    'import os; os.spawnl(0, "/bin/true", "true")',
    'import os; os.spawnvp(0, "true", ["true"])',
    'import os; os.execl("/bin/true", "true")',
    'import os; os.execvpe("true", ["true"], {})',
    'import os; os.fork()',
    'import os; os.forkpty()',
    'import asyncio as a; a.create_subprocess_exec("true")',
    'from asyncio import create_subprocess_shell as launch; launch("true")',
    'import asyncio; loop = asyncio.get_running_loop(); loop.subprocess_exec(factory, "true")',
    'import pty as p; p.spawn(["true"])',
    'from pty import fork as launch; launch()',
    'import posix as p; p.system("true")',
    'import subprocess; launch = subprocess.Popen; launch = print; launch(["true"])',
    'import subprocess; first, launch = print, subprocess.Popen; launch(["true"])',
    'import subprocess; choices = [subprocess.Popen]; choices[0](["true"])',
    'import subprocess; self.spawn = subprocess.Popen; self.spawn(["true"])',
    'import subprocess as sp; launch = sp.Popen; self.spawn = launch',
    'import os; callbacks["launch"] = os.posix_spawn',
    'from subprocess import *; Popen(["true"])',
    'from os import *; system("true")',
    'import os; os.startfile("program.exe")',
    'import multiprocessing as mp; mp.Process(target=callback)',
    'from multiprocessing import Pool as workers; workers()',
    'import multiprocessing; multiprocessing.get_context("spawn")',
    'from concurrent.futures import ProcessPoolExecutor as workers; workers()',
    'import concurrent.futures as futures; futures.ProcessPoolExecutor()',
    'from safe_exec import run as launch; launch(["python3", "-c", "import os; os.system(\\\"true\\\")"])',
    'import safe_exec as adapter; adapter.run(["python3", "-c", payload])',
    'import scripts.safe_exec as adapter; adapter.run(["python3.12", "-c", "import os; os.fork()" ])',
    'from safe_exec import run as launch; launch(["bash", "-c", \'docker logs "$API_KEY"\'])',
    'import safe_exec as adapter; adapter.run(["dash", "-c", payload])',
    'import os; self.launch = getattr(os, operation); self.launch("true")',
    'import os; callbacks["launch"] = getattr(os, operation); callbacks["launch"]("true")',
])
def test_python_audit_rejects_spawn_aliases_and_dynamic_dispatch(source):
    assert python_bypasses(source), source


@pytest.mark.parametrize('source', [
    'import os; print(os.environ.get("MODE"))',
    'from safe_exec import run as launch; launch(["true"])',
    'import subprocess; print(subprocess.PIPE)',
    'def first(): pass\ndef second(): pass\n{"a": first, "b": second}[mode]()',
    'import os; module = os; read = module.getenv; print(read("MODE"))',
    'def callback(): pass\nself.callback = callback\nself.callback()',
    'self.setting = configuration["setting"]',
    'import concurrent.futures; concurrent.futures.ThreadPoolExecutor(max_workers=2)',
    'from concurrent.futures import ThreadPoolExecutor as workers; workers(max_workers=2)',
    'from safe_exec import run as launch; launch(["python3", "-c", "print(1)"])',
    'from safe_exec import run as launch; launch(command)',
    'from safe_exec import run as launch; launch(["bash", "-c", "printf hello"])',
    'self.label = getattr(settings, "label"); print(self.label)',
    'labels["label"] = getattr(settings, "label"); print(labels["label"])',
    'self.label = "hello"; print(self.label)',
    'def callback(): pass\ncallbacks["launch"] = callback\ncallbacks["launch"]()',
])
def test_python_audit_accepts_supported_nonspawning_code(source):
    assert not python_bypasses(source)


@pytest.mark.parametrize('source', [
    b'e=exec; "$e" docker inspect x',
    b'eval "$command" &',
    b'read -r -d "" command <<\'EOF\'\ndocker inspect x\nEOF\neval "$command"',
    b'trap \'eval "$command" &\' EXIT',
    b'source /dev/stdin <<\'EOF\'\ndocker inspect x\nEOF\n',
    b'read -r -d "" command <<\'EOF\'\ndocker inspect x\nEOF\ntrap "$command" EXIT',
    b'source "$unreviewed_path"',
    b'python3 -c "import os; os.system(\'true\')"',
    b'safe_command python3 -c "import os; os.system(\'true\')"',
    b'safe_capture text python3 -c "import subprocess; launch = subprocess.Popen; launch([\'true\'])"',
    b'safe_command python3 -c "$payload"',
    b'safe_command "$interpreter" -c "$payload"',
    b'safe_command "$command" "$argument"',
    b'f() { safe_command "$@"; }',
    b'safe_command bash -c "$payload"',
    b'safe_command bash -c \'eval "$command"\'',
    b'safe_command python3 <<\'EOF\'\nimport os; os.system("true")\nEOF\n',
    b'safe_command python3 /dev/stdin <<\'EOF\'\nimport os; os.system("true")\nEOF\n',
    b'safe_command bash <<\'EOF\'\ndocker inspect x\nEOF\n',
    b'safe_command bash /dev/stdin <<\'EOF\'\ndocker inspect x\nEOF\n',
    b'watch_run flock "$lock" bash <<\'EOF\'\ndocker inspect x\nEOF\n',
    b'watch_run flock "$lock" python3 -c "import os; os.system(\'true\')"',
    b'safe_command "$exe" <<\'EOF\'\ndocker inspect x\nEOF\n',
])
def test_shell_audit_rejects_executable_payload_escapes(source):
    assert bypasses(source), source


@pytest.mark.parametrize('interpreter', ['python3.12', '/usr/bin/python3.13', 'python2.7',
                                          'bash', 'dash', 'zsh', 'ksh', 'ash', 'sh'])
def test_versioned_python_and_shell_interpreter_sources(interpreter):
    payload = 'import os; os.system("true")' if 'python' in interpreter else 'docker inspect x'
    for flag in (['-c'] if 'python' in interpreter else ['-c', '-lc', '-ec']):
        assert bypasses(f'safe_command {interpreter} {flag} {shlex.quote(payload)}'.encode())
        assert bypasses(f'safe_command {interpreter} {flag} "$payload"'.encode())
    assert bypasses(f'safe_command {interpreter} <<\'EOF\'\n{payload}\nEOF\n'.encode())


@pytest.mark.parametrize('source', [
    b'safe_command docker inspect x; printf "%s" "docker is data"',
    b'safe_check_argv "${args[@]}"',
    b'safe_command python3 -c \'import os; print(os.environ.get("MODE"))\'',
    b'trap \'safe_command python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1])" "$scratch"\' EXIT',
    b'cat_data() { read -r value; printf "%s" "$value"; }; cat_data <<\'EOF\'\ndocker is data\nEOF\n',
    b'safe_command bash -c \'safe_command true\'',
])
def test_shell_audit_accepts_supported_checked_commands_and_data(source):
    assert not bypasses(source)


def test_source_exceptions_are_explicit_and_file_scoped():
    for filename, paths in SOURCES.items():
        for path in paths:
            source = f'source {path}'.encode()
            assert not bypasses(source, filename)
            assert bypasses(source)
    assert bypasses(b'source /dev/stdin', 'scripts/run-zig.sh')


def test_checked_forwarding_and_restart_exceptions_are_exact():
    watch_wrapper = 'watch_run() { safe_command "$@"; }\n'
    for (filename, function), command in FORWARDING.items():
        prelude = watch_wrapper if function == '_watch_alert' else ''
        source = prelude + (f'{function}() {{ {command}; }}' if function else command)
        assert not bypasses(source.encode(), filename)
        assert bypasses(source.encode())
        if function:
            with_input = f'{function}() {{ {command} <<\'EOF\'\ndocker inspect x\nEOF\n}}'
        else:
            with_input = command + ' <<\'EOF\'\ndocker inspect x\nEOF\n'
        assert bypasses((prelude + with_input).encode(), filename)
    command = 'watch_run flock "$WATCH_STATE_DIR/lock" python3 "$safe_exec" --restart'
    source = (watch_wrapper + f'_watch_watch_tick() {{ {command}; }}').encode()
    assert not bypasses(source, 'scripts/guard-watch-zig.sh')
    assert bypasses(source)
    assert bypasses(source.replace(b'--restart', b'--command bash'), 'scripts/guard-watch-zig.sh')


def test_rejected_assignment_alias_really_can_create_a_child(tmp_path):
    marker = tmp_path / 'child-marker'
    payload = 'from pathlib import Path; Path(' + repr(str(marker)) + ').write_text("child ran")'
    source = ('import subprocess\nspawn = subprocess.Popen\n'
              f'spawn([{sys.executable!r}, "-c", {payload!r}]).wait()\n')
    # The old gate only recognized module.attribute calls. Demonstrate that its
    # passing mutation actually spawns, then require the current gate to reject it.
    calls = [node.func for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    assert not any(isinstance(call, ast.Attribute) and isinstance(call.value, ast.Name)
                   and call.value.id == 'subprocess' for call in calls)
    assert python_bypasses(source)
    subprocess.run([sys.executable, '-c', source], check=True, capture_output=True, timeout=10)
    assert marker.read_text() == 'child ran'


def test_central_shell_adapter_has_only_supervisor_spawns():
    source = (ROOT / 'scripts/safe_shell.sh').read_bytes()
    tree = Parser(Language(tree_sitter_bash.language())).parse(source)
    assert not tree.root_node.has_error
    functions = {n.child_by_field_name('name').text.decode() for n in nodes(tree.root_node)
                 if n.type == 'function_definition'}
    for node in nodes(tree.root_node):
        if node.type != 'command':
            continue
        name = node.child_by_field_name('name').text.decode()
        if name in BUILTINS | functions | {'builtin'}:
            continue
        text = node.text.decode()
        if name == 'python3':
            assert text.startswith('python3 "$safe_exec" '), 'new direct Python subprocess bypass'
        elif name == 'exec':
            assert text.startswith('exec python3 "$safe_exec" --command bash '), 'new exec bypass'
        else:
            assert text == '"$function" "$@"', 'unreviewed central command bypass'
    assert 'declare -F -- "$function"' in source.decode(), 'only in-process functions may enter shell boundary'


def test_watchdog_new_operations_cannot_skip_shell_failure_boundary():
    source = (ROOT / 'scripts/guard-watch-zig.sh').read_bytes()
    tree = Parser(Language(tree_sitter_bash.language())).parse(source)
    # Internal implementations are invoked by guarded public operations. These
    # small protocol adapters never execute a body outside the shared supervisor.
    protocol = {'watch_redact', 'watch_run', 'watch_capture', 'watch_filter',
                'watch_predicate', 'wssh_control', 'wssh', 'watch_validate_numbers'}
    for node in nodes(tree.root_node):
        if node.type != 'function_definition':
            continue
        name = node.child_by_field_name('name').text.decode()
        if name.startswith('_watch_') or name in protocol:
            continue
        body = node.child_by_field_name('body').text.decode()
        assert body.lstrip('{ \n').startswith('safe_shell '), 'new watchdog operation lacks a shell failure boundary'
