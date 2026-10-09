#!/usr/bin/env bash
# M4: each line of prompts.txt as a chat request at temperature 0 to the Zig server on this host (rank 0); the server
# records each finished reply's prompt and ids in its TF_DSV41_TRACE_TOKENS file (run_host.sh: OUTDIR/trace.jsonl).
#   ./requests.sh [PORT] [MAX_TOKENS]
set -u
PORT=${1:-8091}; MAX=${2:-256}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
n=0
while IFS= read -r p; do
    [[ -z "$p" ]] && continue
    n=$((n + 1))
    body=$(python3 -c 'import json, sys; print(json.dumps({"model": "dsv41", "messages": [{"role": "user", "content": sys.argv[1]}], "temperature": 0, "max_tokens": int(sys.argv[2])}))' "$p" "$MAX")
    t0=$(date +%s.%N)
    out=$(curl -s --max-time 600 "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' -d "$body")
    t1=$(date +%s.%N)
    python3 -c 'import json, sys; d = json.loads(sys.argv[1]); u = d.get("usage", {}); print(json.dumps({"n": int(sys.argv[2]), "seconds": round(float(sys.argv[4]) - float(sys.argv[3]), 2), "usage": u, "text": ((d.get("choices") or [{}])[0].get("message", {}).get("content") or "")[:80]}))' "$out" "$n" "$t0" "$t1" || echo "request $n failed: ${out:0:200}"
done < "$HERE/prompts.txt"
