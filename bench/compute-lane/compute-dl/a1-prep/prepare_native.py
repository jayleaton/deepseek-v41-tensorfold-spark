#!/usr/bin/env python3
"""CPU-only extraction and pin check; never imports the engine or contacts a server."""
import hashlib
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / 'decl-control/dispatch-high'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pins = json.loads((SOURCE/'unmodified-source-sha256.json').read_text())
    checked = {}
    for rel, expected in pins.items():
        if rel.startswith('zig/src/core/tokenizer/') or rel in (
            'zig/src/server/json.zig', 'zig/src/server/json_parse.zig',
            'zig/src/server/json_write.zig', 'zig/src/server/hf_text.zig',
            'zig/src/core/compute_runner.zig', 'zig/src/core/compute_lane.zig',
            'zig/src/server/model_text.zig', 'zig/src/server/compute_segments.zig'):
            p=SOURCE/rel
            if sha(p)!=expected: raise ValueError('pinned source drift: '+rel)
            checked[rel]=expected
    source=(SOURCE/'zig/src/server/compute_segments.zig').read_text()
    start=source.index('fn capTokens(')
    end=source.index('\n}\n',start)+2
    function=source[start:end]
    header='const std = @import("std");\nconst Allocator = @import("root").CapAllocator;\nconst Server = @import("root").CapServer;\n'
    (HERE/'cap_function.zig').write_text(header+'pub '+function+'\n')
    runner=(SOURCE/'zig/src/core/compute_runner.zig').read_text()
    body_start=runner.index('pub fn body(')
    body_end=runner.index('\n}\n',body_start)+2
    lane=(SOURCE/'zig/src/core/compute_lane.zig').read_text()
    outcome_start=lane.index('pub const Status =')
    outcome_end=lane.index('\n};',outcome_start)+3
    (HERE/'body_function.zig').write_text('const std = @import("std");\nconst Allocator = std.mem.Allocator;\n'+lane[outcome_start:outcome_end]+'\n'+runner[body_start:body_end]+'\n')
    json.dump({'source_pins':checked,'extracted_function_sha256':hashlib.sha256(function.encode()).hexdigest(),
               'cap_function_sha256':sha(HERE/'cap_function.zig')},(HERE/'native-source-receipt.json').open('w'),indent=2)
if __name__=='__main__': main()
