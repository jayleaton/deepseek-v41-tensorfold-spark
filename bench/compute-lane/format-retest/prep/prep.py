"""Parent-run offline entrypoint: checks, generate, stage. Never collects replies."""
import argparse
import json
from pathlib import Path

from common import PLAIN_FAMILIES, create, encoded, pinned, read_tasks, sha, spec_gate, strict_json
from generation import generate
from legacy import PATHS
from request_adapter import Adapter, layout

SETTINGS = {"chat_template_kwargs": {"enable_thinking": True}, "deadline_s": 270,
            "max_tokens": 8192, "model": "DeepSeek-V4.1-Flash-TF",
            "reasoning_effort": "high", "return_token_ids": True, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 1.0,
            "top_k": 20, "top_p": 0.95}
GRADER_SHA256 = "1a2249a1b5e0b06f03a4300995d0c320e5a21630d261b3791501f7c83740fb9f"


def generate_files(config, output):
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    # This log survives all later errors, including independent reference failure.
    with (output / "collisions.jsonl").open("xb") as log:
        def retain(collision):
            import os
            log.write(encoded(collision))
            log.flush()
            os.fsync(log.fileno())
        try:
            tasks, a, b, audit = generate(config, retain)
        except Exception as error:
            create(output / "preparation-failure.json", encoded({
                "status": "STOP_NO_REGENERATION", "error_type": type(error).__name__,
                "error": str(error), "spec_sha256": config.get("spec_sha256")}))
            raise
    payloads = {"tasks.jsonl": b"".join(encoded(t) for t in tasks),
                "reference-a.json": encoded(a), "reference-b.json": encoded(b)}
    for name, data in payloads.items():
        create(output / name, data)
    audit["files_sha256"] = {name: sha(data) for name, data in payloads.items()}
    create(output / "generation-audit.json", encoded(audit))
    return audit


def stage(config):
    spec_gate(config["spec_path"], config["spec_sha256"])
    raw = pinned(config["tasks_path"], config["tasks_sha256"])
    tasks = read_tasks(raw)
    audit = strict_json(pinned(config["generation_audit_path"], config["generation_audit_sha256"]))
    if audit["spec_sha256"] != config["spec_sha256"] or audit["files_sha256"]["tasks.jsonl"] != sha(raw):
        raise ValueError("generation audit task/spec mismatch")
    references = []
    for name in ("reference-a.json", "reference-b.json"):
        ref = strict_json(pinned(config["references"][name], audit["files_sha256"][name]))
        if type(ref) is not dict or set(ref) != {t["id"] for t in tasks}:
            raise ValueError("reference identity mismatch")
        references.append(ref)
    if encoded(references[0]) != encoded(references[1]):
        raise ValueError("reference mismatch")
    settings = strict_json(pinned(config["settings_path"], config["settings_sha256"]))
    tools = strict_json(pinned(config["tools_path"], config["tools_sha256"]))
    if encoded(settings) != encoded(SETTINGS) or type(tools) is not list or len(tools) != 1 or tools[0].get("function", {}).get("name") != "python":
        raise ValueError("pinned baseline settings/canonical declaration mismatch")
    if config.get("client_wait_s", 0) < 330:
        raise ValueError("client wait must be at least 330 seconds")
    # Configurable adapter policy, frozen production stage policy.
    if config.get("plain_families", list(PLAIN_FAMILIES)) != list(PLAIN_FAMILIES):
        raise ValueError("frozen spec plain policy is dp/graph/aggregation")
    for name in ("a2", "client3", "client", "reconciliation", "native_frames"):
        pinned(PATHS[name], config["legacy_source_pins"][name])
    pinned(PATHS["reference_a"].parent / "grade.py", GRADER_SHA256)
    adapter = Adapter(settings, tools)
    boots = layout(tasks, adapter)
    if len(boots["L"]) != 240 or len(boots["P"]) != 120:
        raise ValueError("frozen boot counts mismatch")
    return {"spec_sha256": config["spec_sha256"], "tasks_sha256": sha(raw),
            "settings_sha256": config["settings_sha256"], "tools_sha256": config["tools_sha256"],
            "legacy_source_pins": config["legacy_source_pins"], "grader_sha256": GRADER_SHA256,
            "client_wait_s": config["client_wait_s"], "boots": boots,
            "cpu_rendering_status": "PENDING_PARENT_CPU_RENDER_AND_COUNT_QUALIFICATION",
            "runtime_seal_status": "NOT_SEALED_NO_COLLECTION_AUTHORIZATION"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("checks")
    for name in ("generate", "stage"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--config-sha256", required=True)
        command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "checks":
        from checks import main as check
        check()
        return
    config = strict_json(pinned(args.config, args.config_sha256))
    if args.command == "generate":
        generate_files(config, args.output)
    else:
        # Preserve schema/body insertion order: A2 wire() intentionally does.
        create(args.output, (json.dumps(stage(config), ensure_ascii=False,
                                       separators=(",", ":"), allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
