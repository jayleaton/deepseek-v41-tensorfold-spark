#!/usr/bin/env python3
"""Download the in-model runner assets named in config/runner-assets.release, check the archive's sha256, unpack it into
an empty directory and check every file against config/runner-assets.sha256. Any failure names the file and the URL."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
import urllib.error
import urllib.request


sys.path.insert(0, str(Path(__file__).resolve().parent))
from safe_output import cli, safe_print as print  # noqa: E402
import check_runner_assets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'config/runner-assets.release'
MANIFEST = ROOT / 'config/runner-assets.sha256'


class AssetError(Exception):
    pass


def release(path=RELEASE):
    """The archive's `url` and `sha256` (one `key value` per line; # comments)."""
    fields = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            key, _, value = line.partition(' ')
            fields[key] = value.strip()
    url, digest = fields.get('url', ''), fields.get('sha256', '')
    if not url.startswith('https://') or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise AssetError(f'{path.name} must hold an https url and a 64-digit sha256')
    return url, digest


def download(url, dest, attempts=3, timeout=120):
    last = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, dest.open('wb') as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            return
        except (urllib.error.URLError, OSError) as error:
            last = error
            time.sleep(2 * (attempt + 1))
    raise AssetError(f'cannot download {PurePosixPath(url).name} from {url}: {last}')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def keep_dir_mode(member, path):
    """tarfile's 'data' filter, keeping the archive's directory modes: it leaves them to the umask, and the runner image
    layer (and with it the pair identity) would then differ between builders."""
    kept = tarfile.data_filter(member, path)
    return kept.replace(mode=member.mode & 0o755, deep=False) if member.isdir() else kept


def unpack(archive, dest, url):
    """Only plain files and directories under relative names; anything else stops before a byte is written there."""
    try:
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            for m in members:
                parts = PurePosixPath(m.name).parts
                if not (m.isfile() or m.isdir()) or m.name.startswith('/') or '..' in parts:
                    raise AssetError(f'{archive.name} from {url} holds an unsafe entry: {m.name}')
            tar.extractall(dest, members=members, filter=keep_dir_mode)
    except tarfile.TarError as error:
        raise AssetError(f'cannot unpack {archive.name} from {url}: {error}') from error


def fetch(dest, release_path=RELEASE, manifest=MANIFEST, archive=None):
    url, digest = release(release_path)
    url = os.environ.get('RUNNER_ASSETS_URL') or url  # a mirror must serve the same bytes (same sha256)
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise AssetError(f'{dest} must be empty')
    name = PurePosixPath(url).name
    local = Path(archive) if archive else dest.parent / f'.{dest.name}-{name}'
    try:
        if not archive:
            print(f'runner assets: downloading {name} from {url}')
            download(url, local)
        got = sha256(local)
        if got != digest:
            raise AssetError(f'sha256 mismatch for {name} from {url}: expected {digest}, got {got}')
        unpack(local, dest, url)
    finally:
        if not archive and local.exists():
            local.unlink()
    problems = check_runner_assets.check(manifest, dest)
    if problems:
        raise AssetError('; '.join(problems) + f' (in {name} from {url}; expected by {manifest.name})')
    print(f'runner assets: {name} sha256 {digest} ok, {len(check_runner_assets.listed(manifest))} files match {manifest.name}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dest', type=Path, help='an empty or new directory')
    parser.add_argument('--archive', help='an already downloaded archive (offline); still sha256-checked')
    args = parser.parse_args(argv)
    try:
        fetch(args.dest, archive=args.archive)
    except (AssetError, OSError, ValueError) as error:
        print(f'runner assets: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(cli(main))
