# CPU reference and HumanEval jail

Standalone artifact, no repository changes. Standards: ancestry search and
`../LEDGER.md` found no applicable AGENTS/assistant/CONTRIBUTING or build/lint
configuration. Checks are parent-run scoped <rank1-host> py_compile, command-only
selftest, and real Docker qualification. No Python, Docker, or tests were run
on the dev machine. Applying principle-prove-it-works means qualification
must inspect actual container results before claiming this jail is qualified.

Host API (only orchestration and JSON/file operations execute on host):

```python
from jail import Jail
jail = Jail("sha256:<full node-local ID>")
reference = jail.run_reference("print(17*19)")
grade = jail.grade_humaneval(normalized_task, delivered_final_content)
```

Module functions accept the same inputs plus mandatory keyword `image_id`.
Constructor also accepts `docker_prefix=("docker",)` (default), optionally
`("sudo", "-n", "docker")`, and `job_timeout=7200` (minimum7200). CLI has
`--sudo-docker`; it makes no assumption that passwordless sudo is available.
Parent intends to qualify on <rank1-host> as root under a root-user 2GiB/no-swap
systemd user scope with nice19/ionice3, outer timeout7200, no GPU use.

Image inspect must resolve `tensorfold-dsv41-imr:r-3a429d6f122d` to the supplied
full ID and label `org.tensorfold.revision` equal to
`3a429d6f122d9eb2900c43e622933d311e4be44f`. The run uses that immutable ID,
never pulls or mutates tags, and explicitly overrides entrypoint with
`/usr/bin/python3 -I -S /kit/jail_worker.py --supervise`. It never invokes the
engine entrypoint. Runtime is runc; no GPU/device flags; NVIDIA/CUDA visibility
is disabled. Only the phase2 directory is mounted, read-only at /kit. No
legacy mount is needed because tasks carry original prompt and tests.

Each container: network none, UID/GID65534, read-only root, all capabilities
dropped, no-new-privileges, pids2, memory768MiB and equal memory-swap (no swap),
one CPU, tmpfs /tmp rw/nosuid/noexec64MiB. Supervisor creates exactly one
Python child. Child has wall limit3s, CPU soft2/hard3s, address-space512MiB,
file-size0, no core, file-descriptor32 and RLIMIT_NPROC1. Thus candidate
threads/processes cannot add descendants; Docker pids2 also limits supervisor
plus child. Temporary candidate files are automatically removed.

Set `STDBENCH_CPU_JOB` in the parent environment. Every owned container gets
exactly the label `stdbench.cpu_job=<value>`, enabling the parent's CPU guard
to pause the exact container PID/descendants on QUIET and abort only its owned
containers. No other custom labels are added. Containers use unique names and
host-private cidfiles. Cleanup inspects exact CID/name/image/job label and
removes only that container; child timeout kills only the exact child PID.
No killpg, pattern kills, detached jobs, locks, queues, or engines are used.
Docker daemon processes are outside the user scope, so the parent's external
guard and these container resource limits remain necessary.

Reference semantics are independently implemented in CPython: fresh process
and namespace per call, random.seed(0), AST filename `<code>`, execute preceding
statements, evaluate a final bare expression and print repr unless None.
There is no persistent state. Results include stdout/stderr text and base64
raw bytes, exit_code/status, normalized_body plus base64/SHA256, elapsed time,
and truncation flags. Normalization appends stderr using the native separator,
non-ok status text, removes exactly one terminal LF, and uses `[no output]`
only for an empty result. For independent reference errors, traceback wrapper
frames are trimmed; native error validation is separate, never a successful
value comparison or invented result. Timeout/output-limit annotations describe
this reference jail's limits, not an assertion about native execution limits.

Combined stdout/stderr is capped at64KiB with bounded pipe draining. Native
result cap is512 tokens, a distinct limit: every result carries
`needs_native_token_cap_check=true`. Parent must use the frozen native tokenizer
and truncation/provenance evidence to check the delivered cap before comparing
successful native results byte-for-byte with normalized_body_base64. This jail
does not fake tokenization, certify returned-ID accounting, or call endpoints.
Native chronological ID/completion/inserted-token reconciliation belongs to
the parent's client/accounting layer.

HumanEval uses the last closed python/py fence; otherwise exact unfenced text.
Syntax parsing and execution happen only in the constrained child. A full
entry-point function retains exact original prompt import statements; no
dependencies are invented. A native body completion is concatenated directly
to original code_prefix (fallback prompt), without whitespace repair. Invalid
syntax/no complete entry point fails. Original test is executed followed by
check(entry_point). Only delivered final content is supplied; caller must not
substitute reasoning/tool text. A complete capped solution can still pass;
caller retains censor flags independently.

Only status ok plus a post-check pipe receipt gives grade correct. Premature
sys.exit/os._exit(0), timeout, missing receipt, malformed supervisor response
or container failure are incorrect. Receipts bind extracted program and
original test hashes. This protects accidental/premature exit; as with ordinary
in-process HumanEval harnesses, it is not a proof against intentionally hostile
Python that introspects/forges the harness. OS isolation protects the host;
receipts are operational completion evidence, not cryptographic attestation.

Parent CPU checks only, never <dev-host>:

```sh
python3 -m py_compile jail.py jail_worker.py
python3 jail.py --selftest --out /absolute/new-command-receipt.json
python3 jail.py --qualify --image-id sha256:FULL_ID --source-gz /absolute/HumanEval.jsonl.gz --selected /absolute/humaneval-selected.jsonl --out /absolute/new-qualification
```

`--selftest` constructs/inspects arguments only and explicitly returns
qualified=false. Actual qualification requires exactly24 unique selected
original HumanEval tasks; it checks exact prompt/test/entry-point identity
against original GZ, then grades original prompt+canonical_solution, deliberate
return None, infinite loop and os._exit(0) for each selected task inside Docker.
Reference controls include323,55, a final REPL expression, exact-one-LF handling
and fresh namespace. `qualification.json` preserves every control result,
source-GZ/selection hashes and jail source hashes. Exit0 requires all controls
to pass; failures are retained. Output directory/file must be new.

Qualification reads originals but executes no generated code on host. Parent
must pin/check the original dataset lock hash and selected manifest before
calling it and include these jail files plus actual qualification receipt in
the final freeze's additional code/hash mapping. No collection readiness or
GPU GO is inferred from the local code or a command-only selftest.
