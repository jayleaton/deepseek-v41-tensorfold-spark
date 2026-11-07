#!/usr/bin/env bash
# CPU-only feasibility; no engine/repository edits, GPU work, endpoint, queue or lock mutation.
# preflight-requires: <rank0-work>/stdbench/decl-control/cpu_probe.py
# preflight-requires: <rank0-work>/stdbench/phase2/cpu-rootrank1_host.sh
set -euo pipefail
base=<rank0-work>/stdbench/decl-control
source=<rank0-work>/imr-te/build-3a429d6f122d/zig/src
mkdir "$base/native"
for file in json.zig json_parse.zig json_write.zig errors.zig tool_specs.zig compute_turns.zig; do cp "$source/server/$file" "$base/native/$file"; done
for file in template.zig tokenizer.zig encoder.zig; do cp "$source/families/deepseek_v41/serve/$file" "$base/native/$file"; done
for file in unicode.zig unicode_data.zig; do cp "$source/core/tokenizer/$file" "$base/native/$file"; done
cp "$base/api_stub.zig" "$base/native/api_stub.zig"
cp "$base/server_stub.zig" "$base/native/server.zig"
cp "$base/chat_stub.zig" "$base/native/chat.zig"
cp "$base/unicode_root.zig" "$base/native/unicode_root.zig"
ssh root@<rank1-link-ip> 'test ! -e <rank1-work>/stdbench/decl-control'
scp -r "$base" root@<rank1-link-ip>:<rank1-work>/stdbench/
bash <rank0-work>/stdbench/phase2/cpu-rootrank1_host.sh python3 <rank1-work>/stdbench/decl-control/cpu_probe.py
