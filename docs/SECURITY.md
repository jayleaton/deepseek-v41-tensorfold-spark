# Credential boundary

Rule: no credential reaches a log, stdout/stderr, an evidence file, Docker or SSH
argv, or an environment dump. The recipe meets it with three mechanisms, in this
order of importance. The output filter is a backstop, not the boundary.

## 1. Where credentials live

- **Serving keys** live in one operator-owned file, mode `0600`, outside the checkout.
  `NATIVE_AUTH_FILE` in the head config holds its absolute path, never its contents.
  The format is upstream's: one key per line, or `label[scopes]: key`, with `#` comments.
- **Client keys** (benchmarks, canary, preflight) come from the operator's `API_KEY`
  environment or an explicit flag, and stay inside that client process.
- **Shell config never holds a key.** A head config with `TENSORFOLD_API_KEY` and no
  `NATIVE_AUTH_FILE` fails with a migration error.

## 2. Which processes hold a key

| Process | Holds a key? | How |
|---|---|---|
| rank-0 `tensorfold-dsv41 serve` | yes | reads `/run/tensorfold/auth-keys`, a read-only mount, via `--api-key-file` |
| rank-0 container environment, argv, `docker inspect` | no | Docker receives only the mount path |
| rank 1, AOT and diagnostic containers | no | no mount, no key environment |
| host supervisors (`safe_exec.py`) | in memory only | reads the key file and configured values to mask and refuse them |
| every other child: Docker CLI, curl, ssh, git, guard/bench helpers | no | sanitized environment (below) |
| an operator-run client | its own key | sent as an HTTP header; never passed to a child |

## 3. Every child starts from a sanitized environment

All recipe code spawns processes only through `scripts/safe_exec.py` (Python) or
`scripts/safe_shell.sh` (shell). A static gate test fails on any other spawn or
shell-quoting site. Before each spawn the supervisor:

- **removes credential-named variables:** `*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`,
  `*AUTH*`, `*CREDENTIAL*`, `*COOKIE*`; names ending in `FILE`, `PATH` or `DIR` are pointers
  and stay;
- **removes aliases:** any variable whose value contains a known credential. When the
  variable is an operational setting (`TF_*`, `WATCH_*`, `PATH`, `IMAGE`, …), the
  command is refused instead, exit 78, so a default can never silently replace it;
- **removes shell and interpreter startup hooks:** `BASH_ENV`, `ENV`, `BASH_FUNC_*`,
  `PROMPT_COMMAND`, `SHELLOPTS`, `BASHOPTS`, `PS4`; it also starts Bash with
  `--noprofile --norc`. A child therefore cannot source a file that restores a key,
  or inherit `xtrace`;
- **refuses argv** containing a known credential, a rendering of one, or a credential
  flag, assignment or header, before anything starts.

A child that never holds a credential cannot print one, whatever encoding it uses:
`set`, `declare -p`, `printf %q`, `pprint`, JSON, tracebacks or `/proc/*/environ`.

## 4. Remote commands

Each remote path is one of: the watchdog evidence and inspection; native lifecycle
launch, stop and clock; the guard backend and sampler; the reference loader.
The head builds the remote command as checked argv and runs:

```
ssh -o RemoteCommand=none WORKER /usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin \
    /usr/bin/python3 $WORKER_REPO/scripts/safe_exec.py ... --command ARGV
```

(the reference loader runs `env -i PATH=… ARGV` directly). `env -i` discards the
worker's login and session environment, including server `SetEnv`/`AcceptEnv`
values and anything a profile exported. The worker's own supervisor then applies
section 3 on the worker. The local `ssh` client also receives a sanitized
environment, so `SendEnv` has nothing secret to forward.

## 5. The output backstop

Every stream and file the recipe writes passes `scripts/redact.py` through the
`safe_output.py` sinks.

**It promises:**

- It is line-based and stateless. Each line ending in `\n` or `\r` is redacted and
  released immediately; only an unterminated tail waits, for its terminator, 64 KiB,
  or process exit. No line is held for, or masked because of, another line.
- On each line it masks every known value: configured credential variables, explicit
  key flags and the keys in `NATIVE_AUTH_FILE`. It also masks the value's
  JSON-escaped, `repr`, bytes-`repr`, single-quoted shell and URL-quoted spellings,
  plus credential-named assignments, flags and headers, Bearer tokens, URL
  passwords, PEM private-key lines and common token shapes (`sk-`, `hf_`, `ghp_`, …).
- Complete documents (saved JSON files, HTTP bodies, event rows) are masked field
  by field, so numbers and JSON syntax survive. Plain-text documents also get
  whole-text known-value and PEM-block passes.
- Ordinary diagnostics are released verbatim, apostrophes and `NAME=value` metrics
  included.

**It does not promise to catch:**

- a value split across lines, wrapped, or in any encoding not listed above
  (base64, `$'…'`, hex, compression);
- a credential the head never knew: a worker-only key, or a file a child read on its own;
- a multi-line PEM body in a live stream.

Those cases are covered by sections 1–4: such a value must never be in a
child process to begin with.

## 6. Operator responsibilities

- Keep keys only in the `NATIVE_AUTH_FILE` file; do not export them in shell profiles
  on either node.
- Keep worker shell startup quiet. A login profile runs before `env -i` on the worker,
  and anything it prints reaches the head as output.
- The bind mount pins one inode: restart through the lifecycle after an atomic key
  replacement.
- Serving-process core dumps follow the host's `core_pattern`. With Docker's default
  root user they are root-owned. Evidence collection never gathers core files.
- Benchmark `--api-key KEY` is visible in `ps` on a shared host; prefer `API_KEY`.
