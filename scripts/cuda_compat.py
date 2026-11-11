#!/usr/bin/env python3
"""The CUDA forward-compatibility layer scripts/build-compat-image.sh builds.

  check DIR             compare a directory with config/cuda-compat.sha256 and config/cuda-compat.links (exactly those
                        files at those sha256s and those links)
  extract DIR           copy the driver out of the image config/cuda-compat.release pins into a new DIR and check it
  prepare BASE CONTEXT  write the build context: docker/zig/compat.Dockerfile with BASE's entrypoint behind
                        tf-compat-entry and its CMD kept, the entry and the extracted driver
  verify BASE TAG       with no GPU, check that TF_CUDA_COMPAT=0 and an image without a device both log the host driver,
                        that arguments reach the command unchanged and that TAG is BASE plus the layer; print a JSON
                        receipt

Any difference fails closed and names the entry and the image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from safe_exec import run as supervised_run  # noqa: E402
from safe_output import cli, safe_print as print  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'config/cuda-compat.release'
FILES = ROOT / 'config/cuda-compat.sha256'
LINKS = ROOT / 'config/cuda-compat.links'
DOCKERFILE = ROOT / 'docker/zig/compat.Dockerfile'
ENTRY = ROOT / 'docker/zig/tf-compat-entry'
ENTRY_PATH = '/opt/tensorfold/bin/tf-compat-entry'
DRIVER = '610.43.02'
PINNED = re.compile(r'^[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}$')
HEX = set('0123456789abcdef')


class CompatError(Exception):
    pass


def docker(*args, check=True, timeout=600):
    """One docker CLI call; the finished process, or CompatError naming the call when it fails and `check` is set."""
    done = supervised_run(['docker', *args], timeout=timeout)
    if check and done.returncode:
        raise CompatError(f'docker {" ".join(args)}: {done.stderr.strip() or done.returncode}')
    return done


def release(path=RELEASE):
    """The pinned `image`, its `mirror` (both name@sha256:digest) and the absolute `path` of the driver inside them."""
    fields = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            key, _, value = line.partition(' ')
            fields[key] = value.strip()
    image, mirror, where = fields.get('image', ''), fields.get('mirror', ''), fields.get('path', '')
    if not (PINNED.match(image) and PINNED.match(mirror) and where.startswith('/') and '..' not in where.split('/')):
        raise CompatError(f'{path.name} must pin image and mirror as name@sha256:<64 hex> and give an absolute path')
    return image, mirror, where


def flat(name, line):
    if not name or '/' in name or name in ('.', '..'):
        raise ValueError(f'bad manifest line: {line}')
    return name


def pins(files=FILES, links=LINKS):
    """name -> ('file', sha256) or ('link', target), from both manifests; names are plain names in one directory."""
    want = {}
    for line in files.read_text().splitlines():
        if line and not line.startswith('#'):
            digest, _, name = line.partition('  ')
            if len(digest) != 64 or not set(digest) <= HEX or flat(name, line) in want:
                raise ValueError(f'bad manifest line: {line}')
            want[name] = ('file', digest)
    for line in links.read_text().splitlines():
        if line and not line.startswith('#'):
            name, sep, target = line.partition(' -> ')
            if not sep or flat(name, line) in want or flat(target, line) == name:
                raise ValueError(f'bad links line: {line}')
            want[name] = ('link', target)
    return want


def found(root):
    """What the directory holds: plain files (by sha256) and links (by target); anything else is refused."""
    got = {}
    for path in sorted(root.iterdir()):
        if path.is_symlink():
            got[path.name] = ('link', os.readlink(path))
        elif path.is_file():
            with path.open('rb') as stream:
                got[path.name] = ('file', hashlib.file_digest(stream, 'sha256').hexdigest())
        else:
            raise ValueError(f'not a plain file or link: {path.name}')
    return got


def check(root, files=FILES, links=LINKS):
    want, got = pins(files, links), found(root)
    problems = [f'missing {n}' for n in sorted(want.keys() - got.keys())]
    problems += [f'unlisted {n}' for n in sorted(got.keys() - want.keys())]
    for name in sorted(want.keys() & got.keys()):
        (kind, value), (has, seen) = want[name], got[name]
        if kind != has:
            problems.append(f'{name} is a {has}, expected a {kind}')
        elif value != seen:
            problems.append(f'sha256 differs {name}' if kind == 'file'
                            else f'link differs {name}: {seen}, expected {value}')
    return problems


def present(ref):
    """True when a local image carries this exact digest (as its ID or one of its repository digests)."""
    try:
        out = docker('image', 'inspect', '-f', '{{.Id}} {{join .RepoDigests " "}}', ref).stdout
    except CompatError:
        return False
    digest = ref.rpartition('@')[2]
    return any(word == digest or word.endswith('@' + digest) for word in out.split())


def extract(dest, release_path=RELEASE, files=FILES, links=LINKS):
    """Copy the pinned image's driver directory to a new DEST and check it; the pinned image, else the mirror pulled."""
    image, mirror, where = release(release_path)
    if dest.exists() or dest.is_symlink():
        raise CompatError(f'{dest} must not exist yet')
    ref = image
    if not present(image):
        print(f'cuda compat: {image} is not here; pulling {mirror}')
        docker('pull', '--quiet', mirror, timeout=None)
        if not present(mirror):
            raise CompatError(f'{mirror} does not carry its pinned digest after the pull')
        ref = mirror
    container = docker('create', ref).stdout.strip()
    try:
        docker('cp', f'{container}:{where}', str(dest), timeout=None)
    finally:
        docker('rm', container)
    problems = check(dest, files, links)
    if problems:
        raise CompatError('; '.join(problems) + f' (in {where} of {ref}; expected by {files.name} and {links.name})')
    print(f'cuda compat: {where} of {ref}: {len(pins(files, links))} entries match {files.name} and {links.name}')
    return ref


def inspect(ref):
    return json.loads(docker('image', 'inspect', ref).stdout)[0]


def entry_lines(base):
    """The ENTRYPOINT and CMD that put tf-compat-entry ahead of BASE's own entrypoint and keep its CMD."""
    config = base['Config']
    entrypoint = config.get('Entrypoint') or []
    if any(arg.endswith('tf-compat-entry') for arg in entrypoint):
        raise CompatError('the base already runs tf-compat-entry; use the image scripts/build-runner-image.sh built')
    return ('ENTRYPOINT ' + json.dumps([ENTRY_PATH, *entrypoint]) + '\n'
            + 'CMD ' + json.dumps(config.get('Cmd') or []) + '\n')


def prepare(base_ref, context, release_path=RELEASE, files=FILES, links=LINKS):
    """The build context in a new directory: the Dockerfile, the entry and the checked driver."""
    lines = entry_lines(inspect(base_ref))
    context.mkdir()
    shutil.copyfile(ENTRY, context / 'tf-compat-entry')
    (context / 'tf-compat-entry').chmod(0o755)
    extract(context / 'compat', release_path, files, links)
    (context / 'Dockerfile').write_text(DOCKERFILE.read_text() + lines)


def receipt(base, image, tag, host, forced, passed):
    """The derived image's checks, or CompatError naming the first that fails; the receipt on success."""
    if not ('host CUDA driver' in host and 'host CUDA driver' in forced and 'TF_CUDA_COMPAT=0' in forced):
        raise CompatError(f"without a GPU the entry must log the host driver: '{host.strip()}' / '{forced.strip()}'")
    if not passed:
        raise CompatError('tf-compat-entry did not pass its arguments through')
    if (image['Config'].get('Cmd') or []) != (base['Config'].get('Cmd') or []):
        raise CompatError(f'{tag} does not keep the base CMD')
    if image['Config'].get('Entrypoint') != [ENTRY_PATH, *(base['Config'].get('Entrypoint') or [])]:
        raise CompatError(f'{tag} does not run tf-compat-entry ahead of the base entrypoint')
    layers = base['RootFS']['Layers']
    if image['RootFS']['Layers'][:len(layers)] != layers:
        raise CompatError(f'{tag} is not built over the base layers')
    return {'size_metric': 'docker image inspect .Size', 'tag': tag, 'image_id': image['Id'], 'base_id': base['Id'],
            'delta_bytes': image['Size'] - base['Size'], 'entrypoint': image['Config']['Entrypoint'],
            'cuda_compat': DRIVER}


def verify(base_ref, tag):
    def entry(*args, check=True):
        done = docker('run', '--rm', '--runtime=runc', '--network', 'none', '-e', 'NVIDIA_VISIBLE_DEVICES=void',
                      '-e', 'CUDA_VISIBLE_DEVICES=', '--entrypoint', ENTRY_PATH, *args, check=check)
        return done.stdout + done.stderr, done.returncode

    host, _ = entry(tag, '/bin/true')
    forced, _ = entry('-e', 'TF_CUDA_COMPAT=0', tag, '/bin/true')
    _, code = entry(tag, '/bin/sh', '-c', 'test "$1" = "hello world"', 'sh', 'hello world', check=False)
    print(json.dumps(receipt(inspect(base_ref), inspect(tag), tag, host, forced, code == 0), sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('check', 'extract', 'prepare', 'verify'))
    parser.add_argument('args', nargs='+')
    args = parser.parse_args(argv)
    if len(args.args) != (1 if args.action in ('check', 'extract') else 2):
        parser.error(f'{args.action}: wrong number of arguments')
    try:
        if args.action == 'extract':
            extract(Path(args.args[0]))
            return 0
        if args.action == 'prepare':
            prepare(args.args[0], Path(args.args[1]))
            return 0
        if args.action == 'verify':
            verify(*args.args)
            return 0
        problems = check(Path(args.args[0]))
    except (CompatError, OSError, ValueError) as error:
        print(f'cuda compat: {error}', file=sys.stderr)
        return 1 if args.action != 'check' else 2
    for problem in problems:
        print(f'cuda compat: {problem}', file=sys.stderr)
    if problems:
        return 1
    print(f'cuda compat: {len(pins())} entries match {FILES.name} and {LINKS.name}')
    return 0


if __name__ == '__main__':
    sys.exit(cli(main))
