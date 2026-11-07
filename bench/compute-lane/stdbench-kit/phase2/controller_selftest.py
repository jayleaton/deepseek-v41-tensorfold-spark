#!/usr/bin/env python3
"""Exercise the real execute/recovery control flow with isolated fake resource actions."""
import json
import tempfile
from pathlib import Path
import controller


class Fake(controller.Turn):
    def __init__(self, root, failure=None):
        self.failure = failure
        self.turn, self.out = 't01', root
        self.cfg, self.error = {}, None
        self.stopped_g = self.cleaning = False
        self.boots = [{'boot':'test', 'requests':[]}]
        self.events = []
        self.stop_count = 0
    def event(self, step, verdict, **detail):
        self.events.append((step,verdict))
    def preflight(self):
        pass
    def queue(self):
        pass
    def stop_g(self):
        self.stopped_g = True
    def acquire(self):
        if self.failure == 'acquire':
            raise RuntimeError('fake lease acquisition failed')
    def boot(self, boot):
        if self.failure == 'boot':
            raise RuntimeError('fake boot failed')
    def diagnostics(self, boot):
        if self.failure == 'diagnostics':
            raise RuntimeError('fake gate failed')
    def stop_boot(self, boot):
        self.stop_count += 1
        if self.failure == 'stop' and self.stop_count == 1:
            raise RuntimeError('fake first stop failed')
        self.events.append(('owned-rank','stopped'))
    def require_drained(self, boot):
        self.events.append(('owned-rank','drain_verified'))
    def release(self):
        self.events.append(('locks','released'))
        if self.failure == 'release':
            raise RuntimeError('fake fence/release failed')
        return []
    def restore(self):
        assert self.stopped_g
        self.events.append(('G','restored'))
        self.stopped_g = False


def run(out):
    cases = []
    with tempfile.TemporaryDirectory(prefix='stdbench-control-') as temporary:
        for case in (None, 'acquire', 'boot', 'diagnostics', 'stop', 'release'):
            fake = Fake(Path(temporary)/str(case), case)
            try:
                fake.execute()
            except RuntimeError:
                assert case is not None
            assert ('G','restored') in fake.events and not fake.stopped_g
            if ('locks','released') in fake.events:
                assert ('owned-rank','drain_verified') in fake.events or case == 'acquire'
            cases.append({'case':case or 'success','restored':True,'events':fake.events})
    Path(out).write_text(json.dumps({'passed':True,'cases':cases},indent=2)+'\n')


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
