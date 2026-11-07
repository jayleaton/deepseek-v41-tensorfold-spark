"""Container-only code worker; parent supplies timeout and kills only its exact PID."""
import json
import os
import resource
import sys


def main():
    cpu, memory = int(sys.argv[1]), int(sys.argv[2])
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    payload = json.load(sys.stdin)
    namespace = {"__name__": "__main__"}
    exec(compile(payload["program"], "<completion-and-base-tests>", "exec"), namespace)
    # Accidental sys.exit/os._exit cannot bypass test completion. Not an anti-cheat sandbox.
    os.write(int(sys.argv[3]), b"stdbench-tests-complete")


if __name__ == "__main__":
    main()
