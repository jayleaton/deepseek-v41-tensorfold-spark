#!/usr/bin/env python3
"""Offline, fail-closed phase-2 input/request freeze. Never launches a run."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import sys
import tempfile

SUITES = ("gsm8k", "math500", "humaneval")
SEEDS = (101, 102)
ARMS = ("plain", "lane")
COUNTS = dict(gsm8k=48, math500=20, humaneval=24)
DIAGNOSTICS = (
    ("python-17x19", "Use the Python tool to print 17*19, then report the result.", "323"),
    ("python-sum11", "Use the Python tool to print sum(range(11)), then report the result.", "55"),
)
DEFAULTS = dict(model="DeepSeek-V4.1-Flash-TF", reasoning_effort="high",
                temperature=1.0, top_p=0.95, top_k=20, max_tokens=8192,
                deadline_s=270, request_timeout=360, return_token_ids=True,
                stream=True, stream_options={"include_usage": True})
ROOT = Path(__file__).resolve().parent


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def object_sha(value):
    return sha(canonical(value).encode("utf-8"))


def checked(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("explicit lowercase SHA256 required: " + str(path))
    blob = Path(path).read_bytes()
    if sha(blob) != expected:
        raise ValueError("SHA256 mismatch: " + str(path))
    return blob


def load_json(blob):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(blob, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def emit_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def emit_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def freeze(config, out, config_blob=None):
    """All production and fabricated-fixture freezes use this same path."""
    allowed = {"spec", "spec_sha", "kit", "lock", "lock_sha", "inputs", "counts", "settings",
               "code_sha_files", "code_sha_file"}
    require(not set(config) - allowed, "unknown config fields")
    require(set(config["inputs"]) == set(SUITES), "exactly three suite inputs required")
    require(set(config["counts"]) == set(SUITES), "explicit counts for all suites required")
    require(config["counts"] == COUNTS, "T5 sizes fixed at GSM48/MATH20/HumanEval24; no shrink")
    out = Path(out).absolute()
    require(not out.exists() and not out.is_symlink(), "output already exists")
    settings = dict(DEFAULTS)
    require(not set(config.get("settings", {})) - set(settings), "unknown request setting")
    settings.update(config.get("settings", {}))
    require(isinstance(settings["model"], str) and settings["model"], "model required")
    require(settings["reasoning_effort"] in ("low", "medium", "high"), "invalid reasoning effort")
    for key in ("top_k", "max_tokens", "deadline_s", "request_timeout"):
        require(type(settings[key]) is int and settings[key] > 0, "positive integer required: " + key)
    for key in ("temperature", "top_p"):
        require(type(settings[key]) in (int, float) and math.isfinite(settings[key]), "finite number required: " + key)
    require(0 <= settings["temperature"] <= 2 and 0 < settings["top_p"] <= 1, "invalid sampling range")
    require(settings["request_timeout"] > settings["deadline_s"], "client timeout must exceed server deadline")
    require(settings["request_timeout"] >= 330, "T5 client wait must be >=330 seconds")
    require(all(canonical(settings[key]) == canonical(value) for key, value in DEFAULTS.items()
                if key != "request_timeout"), "T5 request settings are fixed except client wait >=330")
    spec_blob = checked(config["spec"], config["spec_sha"])
    require(not (config.get("code_sha_file") and config.get("code_sha_files")), "use only one code hash mapping form")
    code_map_blob = None
    code_map = config.get("code_sha_files", {})
    if config.get("code_sha_file"):
        descriptor = config["code_sha_file"]
        require(set(descriptor) == {"path", "sha256"}, "code_sha_file requires path and sha256")
        code_map_blob = checked(descriptor["path"], descriptor["sha256"])
        code_map = load_json(code_map_blob)
    require(isinstance(code_map, dict), "code hashes must map file paths to SHA256 strings")
    code_blobs = {path: checked(path, expected) for path, expected in sorted(code_map.items())}
    kit = Path(config.get("kit", ROOT.parent / "kit")).resolve()
    lock_path = Path(config.get("lock", kit / "datasets.lock.json"))
    lock_blob = checked(lock_path, config["lock_sha"])
    lock = load_json(lock_blob)
    require(lock.get("version") == 1, "unsupported dataset lock")
    entries = {entry["suite"]: entry for entry in lock["datasets"]}
    require(len(entries) == len(lock["datasets"]) and set(SUITES) <= set(entries), "invalid locked suites")
    # Snapshot before import; importing bench defines helpers, never runs its CLI.
    bench_path = kit / "bench.py"
    bench_blob = bench_path.read_bytes()
    module_spec = importlib.util.spec_from_file_location("stdbench_frozen_kit", bench_path)
    bench = importlib.util.module_from_spec(module_spec)
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module_spec.loader.exec_module(bench)
    finally:
        sys.dont_write_bytecode = old_bytecode
    require(bench_path.read_bytes() == bench_blob, "kit changed during import")
    selected, inputs = {}, {}
    for suite in SUITES:
        source = config["inputs"][suite]
        require(set(source) == {"path", "sha256"}, "input needs only path and sha256")
        blob = checked(source["path"], source["sha256"])
        rows = [load_json(line) for line in blob.decode("utf-8").splitlines() if line.strip()]
        bench.validate_tasks(rows)
        entry = entries[suite]
        require(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None and
                re.fullmatch(r"[0-9a-f]{40}", entry["revision"]) is not None, "invalid source lock")
        for row in rows:
            require(row["suite"] == suite, "mixed suite input")
            require(row.get("source_sha256") == entry["sha256"] and
                    row.get("source_revision") == entry["revision"], "normalized source provenance mismatch")
        count = config["counts"][suite]
        require(type(count) is int and 4 <= count <= len(rows), "count must be >=4 and <= available tasks")
        rows.sort(key=lambda task: (sha(("stdbench-v1:" + task["id"]).encode()), task["id"]))
        selected[suite] = rows[:count]
        inputs[suite] = blob

    # No output is created until all external inputs have been checked.
    out.mkdir(parents=True, exist_ok=False)
    (out / "sources").mkdir()
    (out / "sources/spec.bin").write_bytes(spec_blob)
    (out / "sources/bench.py").write_bytes(bench_blob)
    (out / "sources/datasets.lock.json").write_bytes(lock_blob)
    (out / "sources/freeze.py").write_bytes(Path(__file__).read_bytes())
    (out / "sources/README.md").write_bytes((ROOT / "README.md").read_bytes())
    (out / "sources/PLAN.md").write_bytes((ROOT.parent / "PLAN.md").read_bytes())
    if config_blob is not None:
        (out / "sources/operator-config.json").write_bytes(config_blob)
    if code_map_blob is not None:
        (out / "sources/code-sha-file.json").write_bytes(code_map_blob)
    code_sources = []
    for index, (path, blob) in enumerate(code_blobs.items()):
        rel = f"sources/code/{index:03d}-{Path(path).name}"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(blob)
        code_sources.append(dict(path=path, file=rel, sha256=sha(blob)))
    emit_json(out / "effective-config.json", config)
    emit_json(out / "settings.json", settings)
    for suite, blob in inputs.items():
        (out / ("sources/" + suite + ".normalized.jsonl")).write_bytes(blob)

    all_pairs, requests, runs, turns, summaries = [], [], [], [], {}
    assignment, block_orders, order_files = {}, {}, []
    for suite in SUITES:
        tasks = selected[suite]
        cut = (len(tasks) + 1) // 2
        emit_rows(out / (suite + "/selected.jsonl"), tasks)
        for block, block_tasks in enumerate((tasks[:cut], tasks[cut:]), 1):
            emit_rows(out / f"{suite}/block{block}.jsonl", block_tasks)
            pairs = sorted((task["id"], seed) for task in block_tasks for seed in SEEDS)
            random.Random(20261104).shuffle(pairs)
            block_orders[(suite, block)] = pairs
            grid = [dict(suite=suite, id=task_id, seed=seed) for task_id, seed in pairs]
            rel = f"{suite}/block{block}-ordered-pairs.jsonl"
            emit_rows(out / rel, grid)
            order_files.append(dict(suite=suite, block=block, file=rel,
                                    sha256=sha((out / rel).read_bytes()), pairs_sha256=object_sha(grid)))
            for task in block_tasks:
                assignment[(suite, task["id"])] = block
        summaries[suite] = dict(tasks=len(tasks), seeds=list(SEEDS),
                                pairs_per_arm=len(tasks) * 2, pilot_pairs_per_arm=4,
                                main_pairs_per_arm=len(tasks) * 2 - 4,
                                tasks_sha256=bench.digest(tasks), ids=[t["id"] for t in tasks])

    def payload(task, seed, arm):
        body = bench.request_payload(task, seed, arm, settings["model"],
                                     settings["max_tokens"], settings["deadline_s"])
        for key in ("temperature", "top_p", "top_k", "reasoning_effort", "stream",
                    "stream_options", "return_token_ids"):
            body[key] = settings[key]
        require(("tools" not in body if arm == "plain" else
                 body.get("tools") == [{"type": "code_interpreter"}]), "bad tool contract")
        return body

    def add_runs(stage, suite, block, arm_order, turn):
        tasks = selected[suite]
        for seed in ((101,) if stage == "pilot" else SEEDS):
            chosen = [task for index, task in enumerate(tasks)
                      if (index < 4 if stage == "pilot" else
                          assignment[(suite, task["id"])] == block and not (index < 4 and seed == 101))]
            if not chosen:
                continue
            rel = f"{suite}/{stage}-b{block}-s{seed}.jsonl"
            emit_rows(out / rel, chosen)
            for arm in arm_order:
                run_id = f"{stage}-{suite}-b{block}-s{seed}-{arm}"
                grid = [{"suite": suite, "id": task["id"], "seed": seed} for task in chosen]
                runs.append(dict(run_id=run_id, stage=stage, turn=turn, block=block,
                                 arm=arm, tasks=rel, seeds=[seed], expected_pairs=grid,
                                 expected_pairs_sha256=object_sha(grid), order_seed=20261104))
                for task, pair in zip(chosen, grid):
                    body = payload(task, seed, arm)
                    requests.append(dict(pair, arm=arm, run_id=run_id, body=body,
                                         body_sha256=object_sha(body)))
                    all_pairs.append(dict(pair, arm=arm, stage=stage, run_id=run_id))

    for suite in SUITES:
        add_runs("pilot", suite, 0, ARMS, None)
    layout = [("gsm8k", 1, ARMS), ("math500", 1, ARMS[::-1]),
              ("humaneval", 1, ARMS), ("gsm8k", 2, ARMS[::-1]),
              ("math500", 2, ARMS), ("humaneval", 2, ARMS[::-1])]
    for turn, (suite, block, order) in enumerate(layout, 1):
        add_runs("main", suite, block, order, turn)
        # Launchers must keep every run for one arm inside that arm's boot.
        turns.append(dict(turn=turn, suite=suite, block=block, arm_order=list(order),
                          arm_runs={arm: [r["run_id"] for r in runs if r["turn"] == turn and r["arm"] == arm]
                                    for arm in order}))
    expected = {(suite, task["id"], seed, arm) for suite in SUITES
                for task in selected[suite] for seed in SEEDS for arm in ARMS}
    actual = [(p["suite"], p["id"], p["seed"], p["arm"]) for p in all_pairs]
    require(len(actual) == len(set(actual)) and set(actual) == expected, "pilot/main coverage or duplicate failure")
    emit_rows(out / "expected-pairs.jsonl", all_pairs)
    emit_rows(out / "requests.jsonl", requests)
    # Stable controller API: one boot file per arm, exact wire bodies included.
    task_map = {(suite, task["id"]): task for suite in SUITES for task in selected[suite]}
    run_map = {run["run_id"]: run for run in runs}
    boots = []
    controller_turns = []
    for name, turn, order in [("pilot", None, ARMS)] + [
            (f"t{item['turn']:02d}", item["turn"], item["arm_order"]) for item in turns]:
        boot_names = []
        for arm in order:
            boot = name + "-" + arm
            boot_names.append(boot)
            rows = []
            for request in requests:
                run = run_map[request["run_id"]]
                if run["turn"] != turn or request["arm"] != arm:
                    continue
                task = task_map[(request["suite"], request["id"])]
                rows.append(dict(suite=request["suite"], id=request["id"], seed=request["seed"],
                                 task=task, task_sha256=bench.digest(task),
                                 body_template=request["body"], body_sha256=request["body_sha256"]))
            # Shuffle the complete block once, then remove already-counted pilot
            # units without reshuffling survivors or changing cohort membership.
            if turn is None:
                order_keys = [(suite, task_id, seed) for suite in SUITES for block in (1, 2)
                              for task_id, seed in block_orders[(suite, block)]
                              if seed == 101 and task_id in {t["id"] for t in selected[suite][:4]}]
            else:
                suite, block, _ = layout[turn - 1]
                pilot_ids = {t["id"] for t in selected[suite][:4]}
                order_keys = [(suite, task_id, seed) for task_id, seed in block_orders[(suite, block)]
                              if not (seed == 101 and task_id in pilot_ids)]
            by_key = {(r["suite"], r["id"], r["seed"]): r for r in rows}
            require(len(by_key) == len(rows) and set(by_key) == set(order_keys), "boot order grid mismatch")
            rows = [by_key[key] for key in order_keys]
            diagnostics = []
            for diagnostic_id, prompt, reference in DIAGNOSTICS:
                task = dict(suite="diagnostic", id=diagnostic_id, prompt=prompt,
                            kind="numeric", answer=reference)
                body = payload(task, 101, arm)
                diagnostics.append(dict(suite="diagnostic", id=diagnostic_id, seed=101,
                                        task=task, task_sha256=bench.digest(task),
                                        body_template=body, body_sha256=object_sha(body),
                                        reference=reference, included_in_cohort=False))
            rel = "boots/" + boot + ".json"
            emit_json(out / rel, dict(boot=boot, arm=arm, diagnostics=diagnostics, requests=rows))
            boots.append(dict(boot=boot, arm=arm, file=rel, sha256=sha((out / rel).read_bytes())))
        controller_turns.append(dict(name=name, boots=boot_names))
    # Full-suite matching manifests describe analysis inputs, not launchable main runs.
    for arm in ARMS:
        emit_json(out / f"analysis-{arm}.json", dict(arm=arm, seeds=list(SEEDS),
                  suites={suite: dict(tasks=selected[suite], tasks_sha256=bench.digest(selected[suite]))
                          for suite in SUITES}, include_stages=["pilot", "main"]))
    manifest = dict(version=1, spec_sha256=config["spec_sha"], settings=settings,
                    suites=summaries, seeds=list(SEEDS), selection="sha256(stdbench-v1: + id), id",
                    order_seed=20261104, pilot=dict(first_selected=4, seed=101, counted_in_final=True,
                                                   arm_order=list(ARMS)),
                    boots=boots, turns=controller_turns, main_run_groups=turns, runs=runs,
                    block_orders=order_files, code_sources=code_sources,
                    request_order="Random(20261104).shuffle(sorted((taskID, seed))); filter pilot after shuffle",
                    expected_pairs_sha256=object_sha(all_pairs),
                    request_templates_sha256=object_sha(requests),
                    source_provenance={suite: entries[suite] for suite in SUITES})
    lines = ["# Frozen suite counts", "", "Pilot pairs are included in each final denominator.", "",
             "| Suite | Tasks | Final pairs/arm | Pilot pairs/arm | Main pairs/arm |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for suite, summary in summaries.items():
        lines.append(f"| {suite} | {summary['tasks']} | {summary['pairs_per_arm']} | 4 | {summary['main_pairs_per_arm']} |")
    (out / "COUNTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest["files_sha256"] = {str(path.relative_to(out)): sha(path.read_bytes())
                                for path in sorted(out.rglob("*")) if path.is_file()}
    manifest["files_sha256_excludes"] = ["manifest.json", "file-hashes.json", "file-hashes.sha256"]
    emit_json(out / "manifest.json", manifest)
    inventory = {str(path.relative_to(out)): sha(path.read_bytes())
                 for path in sorted(out.rglob("*")) if path.is_file()}
    emit_json(out / "file-hashes.json", inventory)
    # Conventional root of trust: inventory is itself pinned separately, never self-hashed.
    (out / "file-hashes.sha256").write_text(sha((out / "file-hashes.json").read_bytes()) +
                                          "  file-hashes.json\n", encoding="utf-8")
    return manifest


def dry_run(out, kit, lock, lock_sha):
    """Fabricated data only; no transport, driver, replay, grading or subprocess."""
    receipt_path = Path(str(out) + ".dry-run-receipt.json")
    require(not receipt_path.exists() and not receipt_path.is_symlink(), "receipt already exists")
    with tempfile.TemporaryDirectory(prefix="stdbench-freeze-fixture-") as scratch:
        tmp = Path(scratch)
        spec = tmp / "fake-T5.txt"
        spec.write_text("FAKE offline fixture; not a T5 specification.\n", encoding="utf-8")
        entries = {e["suite"]: e for e in load_json(checked(lock, lock_sha))["datasets"]}
        config = dict(spec=str(spec), spec_sha=sha(spec.read_bytes()), kit=str(kit),
                      lock=str(lock), lock_sha=lock_sha, inputs={}, counts=dict(COUNTS),
                      code_sha_files={str(spec): sha(spec.read_bytes())})
        for suite in SUITES:
            rows = []
            for index in range(COUNTS[suite] + 2):
                row = dict(suite=suite, id=f"fake/{index}", prompt="Return 7.", answer="7",
                           kind="numeric", source_sha256=entries[suite]["sha256"],
                           source_revision=entries[suite]["revision"])
                if suite == "humaneval":
                    row.update(kind="humaneval", test="def check(f):\n    assert f() == 7\n",
                               entry_point="seven", prompt="def seven():\n    pass\n")
                rows.append(row)
            path = tmp / (suite + ".jsonl")
            emit_rows(path, rows)
            config["inputs"][suite] = dict(path=str(path), sha256=sha(path.read_bytes()))
        result = freeze(config, out)
        require(all(s["pairs_per_arm"] == COUNTS[suite] * 2 and
                    s["main_pairs_per_arm"] == COUNTS[suite] * 2 - 4
                    for suite, s in result["suites"].items()), "fixture counts failed")
        for bad in (dict(config, spec_sha="0" * 64),
                    dict(config, counts=dict(COUNTS, gsm8k=47)),
                    dict(config, settings={"request_timeout": 329}),
                    dict(config, code_sha_files={str(spec): "0" * 64})):
            try:
                freeze(bad, tmp / "must-not-exist")
            except ValueError:
                require(not (tmp / "must-not-exist").exists(), "invalid config created output")
            else:
                raise ValueError("invalid hash/count/timeout accepted")
        try:
            freeze(config, out)
        except ValueError:
            pass
        else:
            raise ValueError("existing output accepted")
        inventory = load_json((Path(out) / "file-hashes.json").read_bytes())
        require(all(sha((Path(out) / name).read_bytes()) == value for name, value in inventory.items()),
                "inventory verification failed")
        freeze(config, tmp / "repeat")
        require((Path(out) / "file-hashes.json").read_bytes() ==
                (tmp / "repeat/file-hashes.json").read_bytes(), "freeze is not deterministic")
        templates = [load_json(line) for line in (Path(out) / "requests.jsonl").read_text().splitlines()]
        require(len(templates) == 368 and all(object_sha(r["body"]) == r["body_sha256"]
                                           for r in templates), "fixture request hash failure")
        boot_pairs, boot_orders = [], {}
        for entry in result["boots"]:
            boot = load_json(checked(Path(out) / entry["file"], entry["sha256"]))
            require(len(boot["diagnostics"]) == 2, "two diagnostics required every boot")
            for diagnostic, (_, prompt, reference) in zip(boot["diagnostics"], DIAGNOSTICS):
                require(diagnostic["body_template"]["messages"] == [{"role": "user", "content": prompt}]
                        and diagnostic["reference"] == reference and diagnostic["seed"] == 101
                        and not diagnostic["included_in_cohort"], "diagnostic contract failure")
            for row in boot["requests"] + boot["diagnostics"]:
                body = row["body_template"]
                require(body["return_token_ids"] is True and body["stream"] is True and
                        body["stream_options"] == {"include_usage": True} and
                        object_sha(body) == row["body_sha256"], "stream/native-ID contract failure")
                require(("tools" not in body if boot["arm"] == "plain" else
                         body.get("tools") == [{"type": "code_interpreter"}]), "boot tool declaration failure")
            boot_orders[boot["boot"]] = [(r["suite"], r["id"], r["seed"]) for r in boot["requests"]]
            for row in boot["requests"]:
                require(object_sha(row["body_template"]) == row["body_sha256"] and
                        object_sha(row["task"]) == row["task_sha256"], "boot body/task hash failure")
                boot_pairs.append((row["suite"], row["id"], row["seed"], boot["arm"]))
        require(len(boot_pairs) == len(set(boot_pairs)) == 368, "boot duplicate/coverage failure")
        for turn in result["turns"]:
            first, second = turn["boots"]
            require(boot_orders[first] == boot_orders[second], "paired boot orders differ")
        for order in result["block_orders"]:
            grid = [load_json(line) for line in (Path(out) / order["file"]).read_text().splitlines()]
            pairs = sorted((row["id"], row["seed"]) for row in grid)
            random.Random(20261104).shuffle(pairs)
            require(pairs == [(row["id"], row["seed"]) for row in grid], "block shuffle mismatch")
        require([turn["name"] for turn in result["turns"]] == ["pilot"] +
                [f"t{index:02d}" for index in range(1, 7)], "controller turn names differ")
        receipt = dict(fake=True, normal_freeze_path=True, assertions=["spec hash rejection",
                       "existing directory rejection", "inventory hashes", "suite counts", "deterministic repeat",
                       "request body hashes", "controller boot task/body hashes and pair coverage",
                       "two fixed diagnostics per boot", "stream/native IDs", "registered block shuffle",
                       "paired boot order equality", "fixed size and client wait rejection", "code hash rejection",
                       "pilot/main exact coverage with no duplicate arm pairs"],
                       manifest_sha256=sha((Path(out) / "manifest.json").read_bytes()))
        # Receipt is external to the immutable freeze inventory.
        with receipt_path.open("x", encoding="utf-8") as stream:
            stream.write(canonical(receipt) + "\n")


def mappings(values, convert=str):
    result = {}
    for value in values or []:
        key, sep, item = value.partition("=")
        require(sep and key in SUITES and key not in result, "use unique suite=value entries")
        result[key] = convert(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    parser.add_argument("--config-sha")
    parser.add_argument("--spec")
    parser.add_argument("--spec-sha")
    parser.add_argument("--kit")
    parser.add_argument("--lock")
    parser.add_argument("--lock-sha")
    parser.add_argument("--input", action="append", help="suite=normalized.jsonl")
    parser.add_argument("--input-sha", action="append", help="suite=sha256")
    parser.add_argument("--count", action="append", help="suite=count, explicitly read from T5")
    parser.add_argument("--settings", help="JSON object overriding documented request defaults")
    parser.add_argument("--code-sha-file", help="JSON mapping code file paths to required SHA256 strings")
    parser.add_argument("--code-sha-file-sha", help="external SHA256 of the code mapping file")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.config:
            require(not args.dry_run and not any((args.spec, args.spec_sha, args.kit, args.lock,
                    args.lock_sha, args.input, args.input_sha, args.count, args.settings,
                    args.code_sha_file, args.code_sha_file_sha)), "config cannot mix with settings flags")
            blob = checked(args.config, args.config_sha)
            config = load_json(blob)
            freeze(config, args.out, blob)
        else:
            require(not args.config_sha, "config-sha requires config")
            require(bool(args.code_sha_file) == bool(args.code_sha_file_sha),
                    "code-sha-file and code-sha-file-sha must be supplied together")
            kit = Path(args.kit or ROOT.parent / "kit").resolve()
            lock = Path(args.lock or kit / "datasets.lock.json").resolve()
            if args.dry_run:
                require(not any((args.spec, args.spec_sha, args.input, args.input_sha, args.count, args.settings,
                                 args.code_sha_file, args.code_sha_file_sha)),
                        "dry-run supplies its own fake inputs/settings")
                dry_run(args.out, kit, lock, args.lock_sha)
            else:
                paths, hashes = mappings(args.input), mappings(args.input_sha)
                require(set(paths) == set(hashes) == set(SUITES), "all inputs and hashes required")
                require(args.spec and args.spec_sha, "exact T5 path and externally supplied spec-sha required")
                config = dict(spec=str(Path(args.spec).resolve()), spec_sha=args.spec_sha,
                              kit=str(kit), lock=str(lock), lock_sha=args.lock_sha,
                              inputs={s: dict(path=str(Path(paths[s]).resolve()), sha256=hashes[s]) for s in SUITES},
                              counts=mappings(args.count, int), settings=load_json(args.settings) if args.settings else {})
                if args.code_sha_file:
                    config["code_sha_file"] = dict(path=str(Path(args.code_sha_file).resolve()),
                                                   sha256=args.code_sha_file_sha)
                freeze(config, args.out)
        print(str(Path(args.out).absolute()))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, "freeze: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
