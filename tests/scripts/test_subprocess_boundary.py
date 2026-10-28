"""Reject subprocess escape hatches in host launcher/watchdog orchestration."""
import ast
from pathlib import Path

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

BUILTINS = {'source', '.', 'set', 'unset', 'export', 'readonly', 'local', 'declare',
            'return', 'exit', 'printf', 'echo', 'read', 'cd', 'pwd', 'shift', ':',
            'umask', 'compgen', 'wait', 'trap', 'test', 'true', 'false', 'break', 'continue'}
ADAPTERS = {'safe_command', 'safe_capture', 'safe_filter', 'safe_shell', 'safe_main'}


def nodes(root):
    yield root
    for child in root.children:
        yield from nodes(child)


def bypasses(source):
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
        if command is not None and command.text.decode() not in BUILTINS | ADAPTERS | functions:
            violations.append((node.start_point.row + 1, command.text.decode()))
    return violations


def test_shell_orchestrators_have_no_direct_subprocess_or_exec():
    violations = {name: bypasses((ROOT / name).read_bytes()) for name in SHELLS}
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
    methods = {'run', 'Popen', 'call', 'check_call', 'check_output', 'system',
               'execv', 'execve', 'execvp', 'execvpe', 'fork', 'spawnv', 'spawnve',
               'create_subprocess_exec', 'create_subprocess_shell'}
    for path in (ROOT / 'scripts').rglob('*.py'):
        if path.relative_to(ROOT).as_posix() in exceptions:
            continue
        tree = ast.parse(path.read_text())
        aliases = {'subprocess', 'os', 'asyncio'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                aliases.update(alias.asname or alias.name for alias in node.names
                               if alias.name in aliases)
            if isinstance(node, ast.ImportFrom) and node.module in aliases:
                assert not any(alias.name in methods for alias in node.names), 'direct imported spawn bypass'
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id in aliases:
                    if node.func.attr in methods:
                        forbidden.append((str(path.relative_to(ROOT)), node.lineno))
    assert not forbidden


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
