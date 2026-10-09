#!/usr/bin/env bash
# The structured-output requests over HTTP (gate's grammar-http step): chat requests to tensorfold-dsv41
# with TF_DSV41_GRAMMAR=1, each reply checked against what its grammar promises (grammar_http.py: a JSON schema's instance, a
# JSON object, a call of an offered / the named tool with arguments of its schema). The server records each reply with
# its grammar (TF_DSV41_TRACE_TOKENS), which job.sh then replays through tf-dsv41-m1 generate (HTTP == CLI).
#   bash requests.sh PORT OUTDIR
set -u
PORT=$1 OUT=$2
mkdir -p "$OUT"
exec python3 -u -B "$(dirname "$0")/grammar_http.py" --port "$PORT" --out "$OUT"
