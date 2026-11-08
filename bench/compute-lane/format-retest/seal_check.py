#!/usr/bin/env python3
"""Independent local-rank pre-GO seal check; no queue, model or lock action."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'runtime'))
import protocol
def main():
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);a=p.parse_args()
    cfg=protocol.require_collection_seal(a.campaign)
    assert not protocol.local_path(cfg['go_file']).exists(),'GO must remain absent in final CPU qualification'
    print(json.dumps({'passed':True,'spec_sha256':cfg['spec_sha256'],'prefix':cfg['prefix'],
                      'source_files':len(cfg['code_sha256']),'scheduled':360,'GO_absent':True,'GPU_requests':0}))
if __name__=='__main__':main()
