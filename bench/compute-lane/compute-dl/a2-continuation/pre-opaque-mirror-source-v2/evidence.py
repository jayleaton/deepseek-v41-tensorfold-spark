"""Hash-identical historical rank0-host evidence mirrors; never redirects live code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
PREFIX='stdbench-compute-a2-20261106-v1'
MIRRORS=ROOT/PREFIX/'rank0-host-evidence'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def canonical(path):return Path(str(path).replace('<rank1-home>/','<rank0-home>/',1))
def primary(path):
    return Path(str(path).replace('<rank0-home>/','<rank1-home>/',1)) if str(ROOT).startswith('<rank1-home>/') else Path(path)
def mirror_path(path,base=None):
    source=canonical(path)
    if not source.is_absolute() or '..' in source.parts:raise ValueError('unsafe historical evidence identity')
    base=Path(base) if base is not None else MIRRORS
    result=base/source.relative_to('/')
    if any(p.is_symlink() for p in (result,*result.parents)):raise ValueError('evidence mirror cannot use symlinks')
    return result
def qualified_path(path,expected):
    """Only callers checking frozen historical evidence use this resolver."""
    direct=primary(path)
    if direct.is_file() and sha(direct)==expected:return direct
    if str(ROOT).startswith('<rank1-home>/'):
        mirror=mirror_path(path)
        if mirror.is_file() and sha(mirror)==expected:return mirror
    return direct  # The caller's mandatory hash check fails closed.
def scan(campaign,out):
    cfg=json.loads(Path(campaign).read_text());proof=json.loads(primary(cfg['cpu_receipt']).read_text())
    values=dict(proof['preparation_file_sha256'])
    audit=json.loads(primary(cfg['a1_audit']).read_text())
    for path,digest in audit['provenance']['input_file_sha256'].items():
        if path in values and values[path]!=digest:raise ValueError('conflicting historical evidence hash')
        values[path]=digest
    missing=[]
    for path,digest in values.items():
        actual=primary(path)
        if not actual.is_file() or sha(actual)!=digest:
            missing.append({'source':str(canonical(path)),'sha256':digest,'local_path':str(actual)})
    bad_code=[path for path,digest in cfg['code_sha256'].items()
              if not primary(path).is_file() or sha(primary(path))!=digest]
    receipt={'historical_missing_or_different':missing,'live_code_missing_or_different':bad_code,
             'count':len(missing),'source_sha256':sha(__file__),'GPU_requests':0}
    with Path(out).open('x') as target:json.dump(receipt,target,indent=2)
    print(json.dumps({'historical_missing_or_different':len(missing),'live_code_missing_or_different':len(bad_code),'GPU_requests':0}))
def copy_evidence(manifest):
    record=json.loads(Path(manifest).read_text())
    if record['live_code_missing_or_different']:raise ValueError('live code cannot be replaced by historical mirrors')
    if str(ROOT).startswith('<rank1-home>/'):raise ValueError('only authoritative rank0-host may produce these copies')
    for item in record['historical_missing_or_different']:
        source=Path(item['source']);expected=item['sha256'];target=mirror_path(source)
        if not source.is_file() or sha(source)!=expected:raise ValueError('authoritative evidence source drift: '+str(source))
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if not target.is_file() or sha(target)!=expected:raise ValueError('existing mirror differs')
        else:
            with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
            target.chmod(0o444)
        if sha(target)!=expected:raise ValueError('mirror copy hash mismatch')
    print(json.dumps({'passed':True,'mirrored':record['count'],'GPU_requests':0}))
if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    s=sub.add_parser('scan');s.add_argument('--campaign',required=True,type=Path);s.add_argument('--out',required=True,type=Path)
    c=sub.add_parser('copy');c.add_argument('--manifest',required=True,type=Path)
    args=parser.parse_args()
    if args.command=='scan':scan(args.campaign,args.out)
    else:copy_evidence(args.manifest)
