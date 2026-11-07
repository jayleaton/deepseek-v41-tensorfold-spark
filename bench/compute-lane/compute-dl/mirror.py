#!/usr/bin/env python3
"""Transfer only pinned stdbench bytes; preserve replaced owned prep artifacts."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

BASE=Path('<rank0-work>/stdbench')
PREFIX='stdbench-compute-dl-20261105-v1'
sha=lambda data:hashlib.sha256(data).hexdigest()

def pack(campaign,out):
    cfg=json.loads(campaign.read_text())
    entries={str(Path(p).relative_to(BASE)):h for p,h in cfg['code_sha256'].items()}
    entries[str(Path(cfg['cpu_receipt']).relative_to(BASE))]=cfg['cpu_receipt_sha256']
    entries[str(Path(cfg['go_file']).relative_to(BASE))]=sha(Path(cfg['go_file']).read_bytes())
    campaign_name='compute-dl/'+PREFIX+'/campaign-v1.json'
    entries[campaign_name]=sha(campaign.read_bytes())
    metadata={'files':entries,'spec_sha256':cfg['spec_sha256'],'campaign':campaign_name}
    with tarfile.open(out,'x') as archive:
        data=json.dumps(metadata,sort_keys=True).encode();info=tarfile.TarInfo('MIRROR.json');info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
        for name,expected in entries.items():
            path=campaign if name==campaign_name else BASE/name
            data=path.read_bytes()
            if sha(data)!=expected:raise ValueError('source drift while packaging: '+name)
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=path.stat().st_mode&0o777
            archive.addfile(info,io.BytesIO(data))
    print(json.dumps({'packed_files':len(entries),'archive_sha256':sha(out.read_bytes()),'spec_sha256':cfg['spec_sha256']}))

def install(archive_path):
    base=Path('<rank1-work>/stdbench')
    kept=changed=0
    with tarfile.open(archive_path,'r') as archive:
        members=archive.getmembers()
        if len({m.name for m in members})!=len(members):raise ValueError('duplicate archive entry')
        metadata=json.load(archive.extractfile('MIRROR.json'))
        if {m.name for m in members}!={'MIRROR.json',*metadata['files']}:raise ValueError('archive coverage differs')
        for member in members:
            if member.name=='MIRROR.json':continue
            relative=Path(member.name)
            if not member.isfile() or relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe archive member')
            target=base/relative
            if any(p.is_symlink() for p in (target,*target.parents)):raise ValueError('symlink destination')
            data=archive.extractfile(member).read()
            if sha(data)!=metadata['files'][member.name]:raise ValueError('archive data drift')
            if target.exists():
                old=target.read_bytes()
                if sha(old)==sha(data):kept+=1;continue
                if relative.parts[0]!='compute-dl':raise ValueError('refuse changing legacy artifact: '+str(relative))
                if any(p in relative.parts for p in ('cache','state','started','ownership')):raise ValueError('refuse runtime-state mutation')
                preserved=base/'compute-dl'/PREFIX/'mirror-preserved-v1'/relative
                preserved.parent.mkdir(parents=True,exist_ok=True)
                with preserved.open('xb') as output:output.write(old)
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.mirror-'+str(os.getpid()))
            with temporary.open('xb') as output:output.write(data)
            temporary.chmod(member.mode&0o777)
            os.replace(temporary,target);changed+=1
        for name,expected in metadata['files'].items():
            if sha((base/name).read_bytes())!=expected:raise ValueError('installed mirror readback drift')
    result={'passed':True,'unchanged_files':kept,'installed_owned_or_missing_files':changed,
            'archive_sha256':sha(archive_path.read_bytes()),'spec_sha256':metadata['spec_sha256']}
    with (base/'compute-dl'/PREFIX/'mirror-install.json').open('x') as output:json.dump(result,output,indent=2)
    print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('pack');p.add_argument('campaign',type=Path);p.add_argument('out',type=Path)
    p=sub.add_parser('install');p.add_argument('archive',type=Path)
    args=parser.parse_args()
    if args.action=='pack':pack(args.campaign,args.out)
    else:install(args.archive)
