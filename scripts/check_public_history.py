#!/usr/bin/env python3
"""Scan each reachable text blob once, preserving every historical path and pattern exclusion."""
from __future__ import annotations

import fnmatch
from pathlib import Path
import subprocess
import sys
import tempfile

from safe_exec import child_environment


def git(*args):
    return subprocess.check_output(['git', *args], env=child_environment())


def main() -> int:
    patterns = sys.argv[1:]
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
        for entry in patterns:
            name, _, rest = entry.partition('|')
            regex, _, exclusions = rest.rpartition('|')
            hits = []
            # Chunk argv for large histories; grep retains the exact ERE semantics used by the tree scan.
            for at in range(0, len(files), 200):
                result = subprocess.run(['grep', '-HnIiE', '--', regex, *files[at:at+200]],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=child_environment())
                if result.returncode not in (0, 1):
                    raise ValueError('history grep failed: ' + result.stderr.decode(errors='replace'))
                for line in result.stdout.splitlines():
                    file, number, text = line.split(b':', 2)
                    sha = Path(file.decode()).name
                    if name == 'home directories' and subprocess.run(
                            ['grep', '-qiE', '/home/(user|you|<you>|<user>|ubuntu)/'],
                            input=text, stdout=subprocess.DEVNULL, env=child_environment()).returncode == 0:
                        continue
                    for path, commit in objects[sha].items():
                        if any(fnmatch.fnmatchcase(path, glob) for glob in exclusions.split()):
                            continue
                        hits.append(f'{commit}:{path}:{number.decode()}:' + text.decode(errors='replace'))
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
