#!/usr/bin/env python3
"""Check a runner asset directory against config/runner-assets.sha256: exactly the listed files, each at its sha256."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))
from safe_output import cli, safe_print as print  # noqa: E402


def listed(manifest):
    pins = {}
    for line in manifest.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        digest, name = line.split('  ', 1)
        if len(digest) != 64 or name in pins or name.startswith('/') or '..' in Path(name).parts:
            raise ValueError(f'bad manifest line: {line}')
        pins[name] = digest
    return pins


def found(root):
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f'not a regular file: {path.relative_to(root)}')
        if path.is_file():
            with path.open('rb') as stream:
                files[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return files


def check(manifest, root):
    pins, files = listed(manifest), found(root)
    problems = [f'missing {n}' for n in sorted(pins.keys() - files.keys())]
    problems += [f'unlisted {n}' for n in sorted(files.keys() - pins.keys())]
    problems += [f'sha256 differs {n}' for n in sorted(pins.keys() & files.keys()) if pins[n] != files[n]]
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('assets', type=Path)
    args = parser.parse_args(argv)
    try:
        problems = check(args.manifest, args.assets)
    except (OSError, ValueError) as error:
        print(f'runner assets: {error}', file=sys.stderr)
        return 2
    for problem in problems:
        print(f'runner assets: {problem}', file=sys.stderr)
    if problems:
        return 1
    print(f'runner assets: {len(listed(args.manifest))} files match {args.manifest.name}')
    return 0


if __name__ == '__main__':
    sys.exit(cli(main))
