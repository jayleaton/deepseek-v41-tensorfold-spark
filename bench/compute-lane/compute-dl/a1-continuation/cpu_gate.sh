#!/usr/bin/env bash
# preflight-requires: <rank0-work>/stdbench/phase2/cpu-rank0-host.sh
# preflight-requires: <rank0-work>/stdbench/phase2/cpu_guard.py
set -euo pipefail
# The inherited shell producer's ignored SIGPIPE caused both original stalls.
# Reset the disposition BEFORE bash exec. Python defaults to SIG_IGN, so reset
# explicitly; bash and its producer then inherit SIG_DFL. pipefail waits for both.
exec python3 -B -c 'import os,signal,sys; signal.signal(signal.SIGPIPE,signal.SIG_DFL); os.execvp("bash",["bash","<rank0-work>/stdbench/phase2/cpu-rank0-host.sh",*sys.argv[1:]])' "$@"
