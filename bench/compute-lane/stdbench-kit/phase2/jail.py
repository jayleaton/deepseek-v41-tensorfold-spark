#!/usr/bin/env python3
"""CPU Docker jail controller. Submitted code only executes inside pinned image."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import uuid

ROOT = Path(__file__).resolve().parent
IMAGE = "tensorfold-dsv41-imr:r-3a429d6f122d"
REVISION = "3a429d6f122d9eb2900c43e622933d311e4be44f"
JOB_LABEL = "stdbench.cpu_job"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


class Jail:
    def __init__(self, image_id, docker_prefix=("docker",), job_timeout=7200):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise ValueError("full requested image ID required")
        if not isinstance(docker_prefix, tuple) or not docker_prefix or docker_prefix[-1] != "docker":
            raise ValueError("docker_prefix must be a tuple ending with docker")
        if type(job_timeout) is not int or job_timeout < 7200:
            raise ValueError("whole Docker job timeout must be >=7200 seconds")
        self.image_id, self.docker_prefix, self.job_timeout = image_id, docker_prefix, job_timeout
        self.cpu_job = os.environ.get("STDBENCH_CPU_JOB")
        inspected = self._docker(["image", "inspect", IMAGE])
        record = json.loads(inspected.stdout)[0]
        if record["Id"] != image_id:
            raise ValueError("requested image ID differs from local pinned tag")
        if (record.get("Config", {}).get("Labels") or {}).get("org.tensorfold.revision") != REVISION:
            raise ValueError("image engine revision label differs from frozen revision")
        self.image_record = dict(tag=IMAGE, id=image_id, revision=REVISION)

    def _docker(self, args, check=True):
        return subprocess.run([*self.docker_prefix, *args], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=30, check=check)

    def command(self, name, cidfile):
        root = str(ROOT)
        if any(char in root for char in (",", "\n", "\r")):
            raise ValueError("unsupported mount path")
        args = [*self.docker_prefix, "run", "--rm", "--pull", "never", "-i", "--name", name,
                "--cidfile", str(cidfile), "--runtime", "runc", "--network", "none",
                "--user", "65534:65534", "--read-only", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--pids-limit", "2",
                "--memory", "768m", "--memory-swap", "768m", "--cpus", "1",
                "--tmpfs", "/tmp:rw,nosuid,noexec,size=64m,mode=1777",
                "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "CUDA_VISIBLE_DEVICES=",
                "--mount", f"type=bind,src={root},dst=/kit,readonly"]
        if self.cpu_job:
            args.extend(["--label", JOB_LABEL + "=" + self.cpu_job])
        return [*args, "--entrypoint", "/usr/bin/python3", self.image_id,
                "-I", "-S", "/kit/jail_worker.py", "--supervise"]

    def _remove_owned(self, name, cidfile):
        if not cidfile.exists():
            return
        cid = cidfile.read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{64}", cid):
            raise RuntimeError("invalid owned container ID receipt")
        result = self._docker(["container", "inspect", cid], check=False)
        if result.returncode:
            if "No such" in result.stderr:
                return  # --rm already removed our container.
            raise RuntimeError("owned container inspection failed: " + result.stderr)
        record = json.loads(result.stdout)[0]
        if record["Id"] != cid or record["Name"] != "/" + name or record["Image"] != self.image_id:
            raise RuntimeError("container ownership mismatch; refusing removal")
        if self.cpu_job and (record["Config"].get("Labels") or {}).get(JOB_LABEL) != self.cpu_job:
            raise RuntimeError("CPU job label mismatch; refusing removal")
        self._docker(["container", "rm", "--force", cid])

    def _run(self, payload):
        wire = canonical(payload).encode()
        if len(wire) > 2 * 1024 * 1024:
            raise ValueError("payload exceeds 2MiB")
        with tempfile.TemporaryDirectory(prefix="stdbench-jail-") as work:
            cidfile = Path(work) / "cid"
            name = "stdbench-jail-" + uuid.uuid4().hex
            process = subprocess.Popen(self.command(name, cidfile), stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            timed_out = False
            try:
                try:
                    stdout, stderr = process.communicate(wire, timeout=self.job_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process.kill()  # Exact Docker client PID, never a process group.
                    self._remove_owned(name, cidfile)
                    stdout, stderr = process.communicate(timeout=30)
            finally:
                if process.poll() is None:
                    process.kill()
                self._remove_owned(name, cidfile)
                process.wait(timeout=30)
            if timed_out:
                result = dict(status="harness_error", reason="docker_whole_job_timeout",
                              grade="incorrect", harness_receipt=False)
            elif process.returncode != 0:
                result = dict(status="harness_error", reason="docker_failed", grade="incorrect",
                              harness_receipt=False, docker_exit_code=process.returncode,
                              docker_stderr=stderr.decode("utf-8", "replace")[:65536])
            else:
                try:
                    result = json.loads(stdout)
                    if not isinstance(result, dict) or "status" not in result:
                        raise ValueError("missing supervisor status")
                except (ValueError, UnicodeError):
                    result = dict(status="harness_error", reason="malformed_supervisor_receipt",
                                  grade="incorrect", harness_receipt=False)
            result.update(image=self.image_record, payload_sha256=sha(wire),
                          worker_sha256=sha((ROOT / "jail_worker.py").read_bytes()), cpu_job=self.cpu_job)
            return result

    def run_reference(self, code):
        if not isinstance(code, str):
            raise ValueError("code must be text")
        return self._run(dict(mode="reference", code=code))

    def grade_humaneval(self, task, content):
        if not isinstance(task, dict) or not isinstance(content, str):
            raise ValueError("normalized task and delivered final text required")
        result = self._run(dict(mode="humaneval", task=task, content=content))
        result.update(task_sha256=sha(canonical(task).encode()), content_sha256=sha(content.encode()))
        return result


def run_reference(code, *, image_id, docker_prefix=("docker",)):
    return Jail(image_id, docker_prefix).run_reference(code)


def grade_humaneval(task, content, *, image_id, docker_prefix=("docker",)):
    return Jail(image_id, docker_prefix).grade_humaneval(task, content)


def qualify(jail, source_gz, selected, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    source_blob, selected_blob = Path(source_gz).read_bytes(), Path(selected).read_bytes()
    raw = [json.loads(line) for line in gzip.decompress(source_blob).splitlines() if line.strip()]
    originals = {row["task_id"]: row for row in raw}
    if len(originals) != len(raw):
        raise ValueError("duplicate original HumanEval task IDs")
    tasks = [json.loads(line) for line in selected_blob.splitlines() if line.strip()]
    if len(tasks) != 24 or len({t["id"] for t in tasks}) != 24:
        raise ValueError("exactly 24 unique selected normalized HumanEval tasks required")
    results = []
    for task in tasks:
        original = originals[task["id"]]
        if (task.get("suite") != "humaneval" or task.get("kind") != "humaneval" or
                task.get("test") != original["test"] or task.get("entry_point") != original["entry_point"] or
                task.get("code_prefix") != original["prompt"]):
            raise ValueError("normalized task differs from original HumanEval prompt/test/entry point")
        canonical_content = original["prompt"] + original["canonical_solution"]
        cases = [("canonical", canonical_content, "correct"),
                 ("wrong_return_none", original["prompt"] + "\n    return None\n", "incorrect"),
                 ("infinite_loop", "while True:\n    pass\n", "incorrect"),
                 ("early_exit", "import os\nos._exit(0)\n", "incorrect")]
        for case, content, expected in cases:
            # Timeout/exit fixtures are complete programs with the original
            # function definition retained; execution stops before base tests.
            if case in ("infinite_loop", "early_exit"):
                content = canonical_content + "\n" + content
            result = jail.grade_humaneval(task, content)
            valid = result.get("grade") == expected
            if case == "canonical":
                valid = valid and result.get("harness_receipt") is True
            elif case == "infinite_loop":
                valid = valid and result.get("status") == "timeout"
            elif case == "early_exit":
                valid = valid and result.get("exit_code") == 0 and not result.get("harness_receipt")
            else:
                valid = valid and result.get("status") == "exit_error" and not result.get("harness_receipt")
            results.append(dict(id=task["id"], case=case, qualified=bool(valid), result=result,
                                canonical_source_sha256=sha(canonical_content.encode()),
                                original_test_sha256=sha(original["test"].encode())))
    reference_cases = [("print323", "print(17*19)", "323"),
                       ("print55", "print(sum(range(11)))", "55"),
                       ("repl323", "17*19", "323"),
                       ("one_lf", "print('x' + chr(10))", "x\n"),
                       ("set_previous", "previous = 1", "[no output]"),
                       ("fresh_namespace", "'previous' in globals()", "False")]
    for case, code, expected in reference_cases:
        result = jail.run_reference(code)
        results.append(dict(case=case, qualified=result.get("status") == "ok" and
                            result.get("normalized_body") == expected, result=result))
    receipt = dict(qualified=all(row["qualified"] for row in results), image=jail.image_record,
                   original_source_gz_sha256=sha(source_blob), selected_sha256=sha(selected_blob),
                   sources_sha256={name: sha((ROOT / name).read_bytes()) for name in
                                   ("jail.py", "jail_worker.py", "jail.md")}, results=results)
    (out / "qualification.json").write_text(canonical(receipt) + "\n", encoding="utf-8")
    return receipt


def selftest(out):
    """Only constructs command/JSON; no Docker, process, or candidate execution."""
    jail = Jail.__new__(Jail)
    jail.image_id, jail.docker_prefix = "sha256:" + "a" * 64, ("docker",)
    jail.cpu_job = "fake-cpu-job"
    command = jail.command("stdbench-jail-fake", "/tmp/fake-cid")
    if command[command.index("--entrypoint") + 1] != "/usr/bin/python3" or "--gpus" in command:
        raise ValueError("unsafe command")
    for flag, value in (("--pids-limit", "2"), ("--memory", "768m"), ("--memory-swap", "768m"),
                        ("--network", "none"), ("--user", "65534:65534"), ("--runtime", "runc")):
        if command[command.index(flag) + 1] != value:
            raise ValueError("unexpected isolation option")
    with Path(out).open("x", encoding="utf-8") as stream:
        stream.write(canonical(dict(fake=True, qualified=False, command=command,
                                    note="command-only; real Docker qualification required")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--qualify", action="store_true")
    modes.add_argument("--selftest", action="store_true")
    modes.add_argument("--reference", help="UTF-8 code file; executed only in jail")
    modes.add_argument("--grade", help="normalized task JSON file")
    parser.add_argument("--content", help="delivered final answer file for --grade")
    parser.add_argument("--image-id")
    parser.add_argument("--sudo-docker", action="store_true")
    parser.add_argument("--source-gz")
    parser.add_argument("--selected")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.selftest:
        selftest(args.out)
        return
    jail = Jail(args.image_id or "", ("sudo", "-n", "docker") if args.sudo_docker else ("docker",))
    if args.qualify:
        if not args.source_gz or not args.selected:
            parser.error("--qualify requires --source-gz and --selected")
        receipt = qualify(jail, args.source_gz, args.selected, args.out)
        raise SystemExit(0 if receipt["qualified"] else 1)
    result = (jail.run_reference(Path(args.reference).read_text(encoding="utf-8")) if args.reference else
              jail.grade_humaneval(json.loads(Path(args.grade).read_text()),
                                   Path(args.content).read_text(encoding="utf-8")))
    with Path(args.out).open("x", encoding="utf-8") as stream:
        stream.write(canonical(result) + "\n")


if __name__ == "__main__":
    main()
