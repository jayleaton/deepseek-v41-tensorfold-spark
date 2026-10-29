# Credential boundary

Rule: no credential reaches a log, stdout/stderr, an evidence file, Docker or SSH
argv, or an environment dump. The recipe meets it with three mechanisms, in this
order of importance. The output filter is a backstop, not the boundary.

**What this is, and is not.** These mechanisms stop the recipe from *implicitly*
handing a key to its children: through their environment, their argv, or a
remote session. They are not OS isolation. Every child runs as the operator's
user, with that user's filesystem, `/proc` and Docker access. A child that goes
looking can read anything that user can read: the key file, if the operator owns
it; a parent's `/proc/PID/environ`; or the key mounted in rank 0, through
`docker exec`. Docker access is root-equivalent. The recipe's own diagnostics do
none of this, and the output backstop masks the ordinary ways a key could still
appear (section 5). If no non-serving process may be able to obtain a key, use
OS privileges: a root-owned key file (section 6) and an operator account without
Docker access.

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
| host supervisors (`safe_exec.py`, the `safe_shell.sh` boundary) | in memory | read the key file (when readable) and configured values, to mask and refuse them |
| recipe Python CLIs wrapped by `safe_output.cli()` | in memory | same as the supervisor: `preflight.py`, `canary.py`, `check_native_assets.py`, `native_ops.py`, `pack_engram.py`, `guard/{control,daemon,lifecycle,profile,sampler}.py`, `bench/{quality,soak,stress,structured}.py`, `bench/tooleval/chains.py` |
| every other child: Docker CLI, curl, ssh, git, diagnostics | not given one | sanitized environment and argv (below); same-user file and process access remains |
| an operator-run client | its own key | sent as an HTTP header; never passed to a child |

These are the trusted holders. A process "in memory" holds keys only to mask and
refuse them. That memory is in a core dump if the process crashes with core
dumps enabled (section 6).

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

So a child is never *given* a credential: its `set`, `declare -p`, `printf %q`,
`pprint`, JSON and tracebacks, and its own `/proc/self/environ`, contain none. This
covers what the child inherits, not what it can go and read (see the top of this
file).

`SSH_AUTH_SOCK` is a pointer to the operator's agent, not a key, so it is kept:
agent-only SSH to the worker works. Every recipe `ssh` passes
`-o ForwardAgent=no`, and the worker side starts from `env -i`, so the socket
never leaves the head. Docker receives only the variables the launcher names, so
it never reaches a container.

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
  JSON-escaped, `repr`, bytes-`repr`, single-quoted shell, double-quoted
  (`declare -p`) and URL-quoted spellings. Every line of 4 or more characters
  of a multi-line value counts as known on its own. It also masks
  credential-named assignments, flags and headers, Bearer tokens, URL passwords,
  PEM private-key lines and common token shapes (`sk-`, `hf_`, `ghp_`, …).
- Complete documents (saved JSON files, HTTP bodies, event rows) are masked field
  by field, so numbers and JSON syntax survive. Plain-text documents also get
  whole-text known-value and PEM-block passes.
- Ordinary diagnostics are released verbatim, apostrophes and `NAME=value` metrics
  included.

**It does not promise to catch:**

- a value a tool splits or wraps at arbitrary points (`pprint` word-wrapping),
  fragments shorter than 4 characters, or any encoding not listed above (base64,
  `$'…'` escapes, hex, compression);
- a credential the head never knew: a worker-only key, or a file a child read on its own;
- a multi-line PEM body in a live stream.

The backstop protects the recipe's own trusted diagnostics, which print keys
in none of these ways. It is not a defence against a process that deliberately
reads a key and re-encodes it. That process has crossed the trust boundary at
the top of this file.

## 6. Operator responsibilities

- Keep keys only in the `NATIVE_AUTH_FILE` file; do not export them in shell profiles
  on either node. The launcher refuses a file inside the checkout or inside any
  directory it mounts into a container (model, engram, assets, prepared, state,
  cache), because rank 1, AOT and diagnostic containers see those.
- For the strongest setting, make the file `root:root` mode `0600` and run the
  recipe as a user without it. Only the serving container (root) can then read
  it. Host diagnostics cannot, and the backstop then has no keys to mask, because
  none can be read. The launcher accepts a file owned by you or by root.
- Keep worker shell startup quiet. A login profile runs before `env -i` on the worker,
  and anything it prints reaches the head as output.
- The bind mount pins one inode: restart through the lifecycle after an atomic key
  replacement.
- Core dumps follow the host's `core_pattern` and `ulimit -c`. The serving
  process runs as Docker's default root user, so its cores are root-owned. A
  host supervisor or recipe CLI holds keys in memory to mask them, so its core
  (mode `0600`, owned by the operator) can contain a key. Keep `ulimit -c 0`,
  the usual default, for recipe shells, or point `core_pattern` at a root-only
  location. Evidence collection never gathers core files.
- Benchmark `--api-key KEY` is visible in `ps` on a shared host; prefer `API_KEY`.
