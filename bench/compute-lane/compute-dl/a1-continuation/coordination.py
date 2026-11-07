#!/usr/bin/env python3
"""Read-only coordination-file checks. Holding G's flock is expected and allowed."""
import json
import os
from pathlib import Path
import stat
import sys

LOCK_PATH=Path('/tmp/rank1-host-gpu.lock')

def check(path=LOCK_PATH):
    path=Path(path)
    # O_NONBLOCK prevents a malformed FIFO from blocking this preflight.
    # No creation, write, truncate or flock operation is allowed here.
    fd=os.open(path,os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode): raise ValueError('coordination lock must be a regular readable file')
        os.read(fd,1)
        return {'passed':True,'path':str(path),'regular':True,'readable':True,
                'device':info.st_dev,'inode':info.st_ino,'flock_attempted':False,'open_mode':'O_RDONLY'}
    finally:
        os.close(fd)

if __name__=='__main__':
    if len(sys.argv)!=1: raise SystemExit('native coordination path only')
    print(json.dumps(check()))
