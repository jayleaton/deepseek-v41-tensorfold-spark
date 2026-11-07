#!/usr/bin/env bash
# PREPARATION ONLY. Parent runs this on Spark, then compiles/runs the mutant
# separately. Never touches the baseline snapshot or the source repository.
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ -e guard-mutant ]; then
    echo 'Refusing to overwrite guard-mutant' >&2
    exit 1
fi
mkdir guard-mutant
cp build.zig build.zig.zon dispatch_spy.c guard-mutant/
cp -a zig guard-mutant/zig
file=guard-mutant/zig/src/server/compute_turns.zig
before=$(sha256sum "$file")
# Retain a reference to srv so this is not an unused-parameter compile mutant.
sed -i 's/if (srv.compute == null or r.eos.len == 0/if ((srv.compute == null and false) or r.eos.len == 0/' "$file"
after=$(sha256sum "$file")
if [ "$before" = "$after" ]; then
    echo 'Mutation did not match pinned source; mutant VOID' >&2
    exit 1
fi
diff -u zig/src/server/compute_turns.zig "$file" > guard-mutant/source.patch || [ "$?" -eq 1 ]
