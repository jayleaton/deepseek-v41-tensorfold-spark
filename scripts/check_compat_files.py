#!/usr/bin/env python3
"""Checks for scripts/build-compat-image.sh.

  files DIR SHA256S LINKS   DIR holds exactly the regular files in SHA256S (sha256, ./path) and the links in LINKS
                            (./link -> target), nothing else; exit 1 otherwise
  entry FILE                FILE (tf-compat-entry's stderr with no GPU visible) reports the host CUDA driver
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path


def _entries(path: str) -> list[list[str]]:
    return [line.split() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")]


def compat_files(root: str, sums: str, links: str) -> bool:
    want_files = {p: h for h, p in _entries(sums)}
    want_links = {a: b for a, _, b in _entries(links)}
    got_files, got_links = {}, {}
    for directory, dirs, files in os.walk(root):
        for name in dirs + files:
            path = os.path.join(directory, name)
            rel = "./" + os.path.relpath(path, root)
            if os.path.islink(path):
                got_links[rel] = os.readlink(path)
            elif os.path.isfile(path):
                got_files[rel] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return got_files == want_files and got_links == want_links


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[0] == "files":
        if compat_files(*argv[1:]):
            return 0
        print("the compat driver files differ from docker/compat/compat.sha256 / compat.links", file=sys.stderr)
        return 1
    if len(argv) == 2 and argv[0] == "entry":
        if "tf-compat-entry: host CUDA driver" in Path(argv[1]).read_text(errors="replace"):
            return 0
        print("tf-compat-entry did not report the host CUDA driver with no GPU visible", file=sys.stderr)
        return 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
