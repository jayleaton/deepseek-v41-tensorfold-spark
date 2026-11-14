#!/usr/bin/env python3
"""Scan each reachable text blob once, preserving every historical path and pattern exclusion.

    check_public_history.py [--patch-private REGEX] PATTERN...

PATTERN is check-public.sh's name|regex|exclusions, matched case-insensitively. --patch-private adds the engine patch
series rule: REGEX, case-sensitive, over every blob that was ever a patches/NNNN-*.patch file (author lines exempt).
"""
from __future__ import annotations

import fnmatch
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from safe_exec import child_environment

# check-public.sh PATCH_AUTHOR_LINES: a patch's author / co-author line with a public GitHub noreply address.
PATCH_PATH = re.compile(r'patches/[0-9]{4}-[^/]*\.patch')
PATCH_AUTHOR = re.compile(rb'(From|Co-authored-by): [^<]*<[0-9]+\+[A-Za-z0-9-]+@users\.noreply\.github\.com>')


def git(*args):
    return subprocess.check_output(['git', *args], env=child_environment())


def grep(flags, regex, files):
    """grep FLAGS REGEX over FILES in argv chunks; yields (sha, line number, text) per hit."""
    for at in range(0, len(files), 200):
        result = subprocess.run(['grep', flags, '--', regex, *files[at:at+200]],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=child_environment())
        if result.returncode not in (0, 1):
            raise ValueError('history grep failed: ' + result.stderr.decode(errors='replace'))
        for line in result.stdout.splitlines():
            file, number, text = line.split(b':', 2)
            yield Path(file.decode()).name, number.decode(), text


def scan_patch_series(regex, objects, files):
    """The current-tree rule for patches/ (check-public.sh PATCH_PRIVATE) over every historical patch blob."""
    patch_files = [file for file in files if any(PATCH_PATH.fullmatch(path) for path in objects[Path(file).name])]
    hits = []
    for sha, number, text in grep('-HnIE', regex, patch_files):
        if PATCH_AUTHOR.fullmatch(text):
            continue
        for path, commit in objects[sha].items():
            if PATCH_PATH.fullmatch(path):
                hits.append(f'{commit}:{path}:{number}:' + text.decode(errors='replace'))
    if hits:
        print('== history: internal names in the engine patch series')
        for hit in sorted(set(hits)):
            print(hit[:220])
    return bool(hits)


def main() -> int:
    patterns = sys.argv[1:]
    patch_private = None
    if patterns[:1] == ['--patch-private']:
        if len(patterns) < 2:
            raise ValueError('--patch-private needs a regex')
        patch_private, patterns = patterns[1], patterns[2:]
    objects = {}
    for commit in git('rev-list', '--all').decode().splitlines():
        for row in git('ls-tree', '-rz', commit).split(b'\0'):
            if not row:
                continue
            metadata, path_bytes = row.split(b'\t', 1)
            _, kind, sha = metadata.split()
            path = path_bytes.decode('utf-8', 'surrogateescape')
            if kind != b'blob' or path == 'scripts/check-public.sh' or path.startswith('vendor/TensorFold/'):
                continue
            objects.setdefault(sha.decode(), {}).setdefault(path, commit)
    failed = False
    with tempfile.TemporaryDirectory(prefix='check-public-history-') as scratch:
        proc = subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=child_environment())
        files = []
        try:
            for sha in objects:
                proc.stdin.write((sha + '\n').encode())
                proc.stdin.flush()
                header = proc.stdout.readline().split()
                if len(header) != 3 or header[1] != b'blob':
                    raise ValueError('unexpected git cat-file response')
                data = proc.stdout.read(int(header[2]))
                if proc.stdout.read(1) != b'\n':
                    raise ValueError('truncated git cat-file response')
                if b'\0' not in data:
                    file = Path(scratch) / sha
                    file.write_bytes(data)
                    files.append(str(file))
        finally:
            proc.stdin.close()
            if proc.wait() != 0:
                raise ValueError('git cat-file failed')
        if patch_private is not None:
            failed |= scan_patch_series(patch_private, objects, files)
        for entry in patterns:
            name, _, rest = entry.partition('|')
            regex, _, exclusions = rest.rpartition('|')
            hits = []
            # Chunked grep keeps the exact ERE semantics used by the tree scan.
            for sha, number, text in grep('-HnIiE', regex, files):
                if name == 'home directories' and subprocess.run(
                        ['grep', '-qiE', '/home/(user|you|<you>|<user>|ubuntu)/'],
                        input=text, stdout=subprocess.DEVNULL, env=child_environment()).returncode == 0:
                    continue
                for path, commit in objects[sha].items():
                    if any(fnmatch.fnmatchcase(path, glob) for glob in exclusions.split()):
                        continue
                    if PATCH_PATH.fullmatch(path) and PATCH_AUTHOR.fullmatch(text):
                        continue
                    hits.append(f'{commit}:{path}:{number}:' + text.decode(errors='replace'))
            if hits:
                failed = True
                print('== history: ' + name)
                for hit in sorted(set(hits)):
                    print(hit[:220])
    return int(failed)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print('history scan failed: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
