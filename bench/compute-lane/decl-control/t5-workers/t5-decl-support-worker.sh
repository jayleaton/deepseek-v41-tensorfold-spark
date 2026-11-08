#!/bin/bash
set -uo pipefail
out=$T5_OUT;mkdir -p "$out/audit"
ps -o sid= -p $$ | tr -d ' ' > "$out/sid"
python3 -B "$HOME/work/imr-t5-s1/transport/t5-decl-support.py" "$out/audit"
rc=$?;echo "$rc" > "$out/job-exit";exit "$rc"
