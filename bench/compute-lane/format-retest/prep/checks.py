"""Handwritten offline controls ONLY. Never calls generation or any transport."""
import copy
import sys

from common import OLD, encoded, read_tasks, strict_json
from prep import SETTINGS
from request_adapter import Adapter, require_accepted, validate_keys, validate_neutrality


def rejected(call):
    try:
        call()
    except (ValueError, KeyError, TypeError):
        return
    raise AssertionError("expected rejection")


def fixture():
    # Deliberately tiny HANDWRITTEN control; no cohort/candidate generation.
    return {"id": "fake-dp", "suite": "compute_dl", "family": "dp", "inputs": {"rows": 1, "cols": 1, "matrix": [[3]]},
            "prompt": "Handwritten grid [[3]]. " + OLD["dp"]}


class Decoder:
    pieces = {1: b"</think>", 2: b"3", 3: "<｜end▁of▁sentence｜>".encode(),
              4: "<｜User｜>".encode(), 5: "<｜Assistant｜>".encode()}

    def decode(self, ids):
        return b"".join(self.pieces[i] for i in ids)


def main():
    path_before, modules_before = list(sys.path), {k: sys.modules.get(k) for k in ("client", "client3", "reconciliation", "native_frames", "request_adapter")}
    tools = [{"type": "function", "function": {"name": "python", "description": "FAKE ONLY", "parameters": {"type": "object"}}}]
    adapter = Adapter(SETTINGS, tools)
    task = fixture()
    rows = validate_neutrality(adapter, task, 101)
    assert len({tuple(r["request_identity"]) for r in rows.values()}) == 4
    for wording in ("old", "new"):
        lane_body = copy.deepcopy(rows["L-" + wording]["body_template"])
        assert lane_body.pop("tools") == tools
        assert adapter.a2.wire(lane_body) == rows["P-" + wording]["wire"]
    for row in rows.values():
        assert row["body_template"]["seed"] == 101
        for key, value in SETTINGS.items():
            assert row["body_template"][key] == value
    template = rows["P-new"]
    cpu = {"key": template["key"], "request_identity": template["request_identity"], "wire_sha256": template["wire_sha256"],
           "plain_ids": [7, 8], "declared_lane_ids": [7, 8], "declared_off_ids": [7, 8]}
    ids = [1, 2, 3]
    digest = adapter.legacy.load("client").sha_ids(ids)
    raw = {"status": "ok", "transport_status": "ok", "actual_request_wire": template["wire"],
           "sse_done": True, "finish_reason": "stop", "sse_frames": [], "external_tool_calls": [],
           "usage": {"prompt_tokens": 2, "completion_tokens": 3}, "tensorfold": {"token_ids": ids, "token_sha256": digest}, "content": "3"}
    good = adapter.observe(raw, task, template, cpu, Decoder())
    require_accepted(good)
    assert good["diagnostics"]["input_accounting"]["processed_input_tokens"] == 2
    variants = []
    bad = copy.deepcopy(raw); bad["tensorfold"]["token_sha256"] = "bad"; variants.append(bad)
    bad = copy.deepcopy(raw); del bad["tensorfold"]["token_ids"]; variants.append(bad)
    bad = copy.deepcopy(raw); bad["usage"]["prompt_tokens"] = 99; variants.append(bad)
    bad = copy.deepcopy(raw); del bad["usage"]["prompt_tokens"]; variants.append(bad)
    bad = copy.deepcopy(raw); bad["sse_frames"] = [{"error": "fake native failure"}]; variants.append(bad)
    bad = copy.deepcopy(raw); bad["sse_done"] = False; variants.append(bad)
    bad = copy.deepcopy(raw); bad["actual_request_wire"] += " "; variants.append(bad)
    bad = copy.deepcopy(raw); bad["tensorfold"]["compute"] = {"calls": [{"name": "python"}], "prefix_tokens": 0, "inserted_tokens": 0}; variants.append(bad)
    for bad in variants:
        retained = adapter.observe(bad, task, template, cpu, Decoder())
        assert retained["raw_observation"] == bad and retained["errors"]
        rejected(lambda: require_accepted(retained))
    off = adapter.observe(variants[-1], task, template, cpu, Decoder())
    assert any("effective-OFF" in error for error in off["errors"])
    assert "native_accounting" in off["diagnostics"] and "input_accounting" in off["diagnostics"]
    assert off["diagnostics"]["input_accounting"]["processed_input_tokens"] is None
    unknown = adapter.observe(variants[3], task, template, cpu, Decoder())
    assert unknown["diagnostics"]["input_accounting"]["processed_input_tokens"] is None
    assert unknown["diagnostics"]["input_accounting"]["prompt_plus_returned_tokens"] is None
    wrong_cpu = copy.deepcopy(cpu); wrong_cpu["request_identity"][-1] = "P-old"
    retained = adapter.observe(raw, task, template, wrong_cpu, Decoder())
    assert retained["raw_observation"] == raw and not retained["accepted"]
    rejected(lambda: require_accepted(retained))
    drift = copy.deepcopy(template); drift["body_template"]["seed"] = 102
    rejected(lambda: adapter.validate_template(task, drift))
    for arm in rows:
        row = rows[arm]
        assert row["arm"] == ("lane" if arm.startswith("L-") else "plain")
    boot_rows = {"L": [], "P": []}
    for seed in (101, 102):
        for arm, row in validate_neutrality(adapter, task, seed).items():
            boot_rows[arm[0]].append(row)
    validate_keys(boot_rows, [task])
    bad = copy.deepcopy(boot_rows); bad["L"].append(bad["L"][0]); rejected(lambda: validate_keys(bad, [task]))
    bad = copy.deepcopy(boot_rows); bad["L"].pop(); rejected(lambda: validate_keys(bad, [task]))
    bad = copy.deepcopy(boot_rows); bad["L"][0]["request_identity"][2] = True; rejected(lambda: validate_keys(bad, [task]))
    bad = copy.deepcopy(boot_rows); del bad["P"][0]["request_identity"]; rejected(lambda: validate_keys(bad, [task]))
    rejected(lambda: strict_json(b'{"key":1,"key":2}'))
    rejected(lambda: strict_json(b'{"key":NaN}'))
    rejected(lambda: read_tasks(b""))
    rejected(lambda: read_tasks(encoded({})))
    assert sys.path == path_before
    assert all(sys.modules.get(k) is value for k, value in modules_before.items())
    print("PASS handwritten fake-response controls; no generation, network or runtime seal")


if __name__ == "__main__":
    main()
