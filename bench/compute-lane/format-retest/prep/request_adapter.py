"""Four logical arms over original A2 native evidence semantics. No transport."""
import copy
import random

from common import NEW, OLD, PLAIN_FAMILIES, encoded, sha
from legacy import Legacy

ARMS = ("L-old", "L-new", "P-old", "P-new")


class Adapter:
    def __init__(self, settings, tools, plain_families=PLAIN_FAMILIES):
        self.legacy = Legacy()
        self.a2 = self.legacy.load("a2")
        self.client = self.legacy.load("client3")
        self.settings, self.tools = copy.deepcopy(settings), copy.deepcopy(tools)
        self.plain_families = tuple(plain_families)
        if len(set(self.plain_families)) != len(self.plain_families):
            raise ValueError("duplicate plain policy family")

    def build_request(self, task, seed, arm):
        if arm not in ARMS or type(seed) is not int or seed not in (101, 102):
            raise ValueError("invalid logical arm/seed")
        if arm.startswith("P-") and task["family"] not in self.plain_families:
            raise ValueError("plain family not scheduled by policy")
        family, prompt = task["family"], task["prompt"]
        if not prompt.endswith(OLD[family]):
            raise ValueError("old suffix mismatch")
        variant = copy.deepcopy(task)
        if arm.endswith("-new"):
            variant["prompt"] = prompt[:-len(OLD[family])] + NEW[family]
        adapter_arm = "lane" if arm.startswith("L-") else "plain"
        template = self.a2.build_request(variant, seed, adapter_arm, self.settings, self.tools)
        template.update(logical_arm=arm, request_identity=[*template["key"], arm])
        return template

    def validate_template(self, task, template):
        expected = self.build_request(task, template["key"][2], template["logical_arm"])
        if (encoded(expected) != encoded(template)
                or self.a2.wire(template["body_template"]) != template["wire"]
                or sha(template["wire"].encode()) != template["wire_sha256"]):
            raise ValueError("template drift")

    def observe(self, raw, task, template, cpu_prompt, decoder):
        """Always return retained raw+diagnostics; caller persists, THEN require_accepted.

        No coercion of unknown input costs to zero. Original A2 classification is
        authoritative; its effective-OFF exception becomes a retained rejection.
        """
        result = {"raw_observation": copy.deepcopy(raw), "request_identity": copy.deepcopy(template.get("request_identity")),
                  "logical_arm": template.get("logical_arm"), "diagnostics": {}, "errors": []}
        errors, diagnostics = result["errors"], result["diagnostics"]
        try:
            self.validate_template(task, template)
            if cpu_prompt.get("request_identity") != template["request_identity"] or cpu_prompt.get("wire_sha256") != template["wire_sha256"]:
                raise ValueError("CPU prompt logical identity/wire mismatch")
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            errors.append("request/CPU binding: " + str(exc))
            result["accepted"] = False
            return result
        try:
            diagnostics.update(self.a2.classify_observation(raw, template, cpu_prompt, decoder))
        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
            errors.append("A2 classification: " + str(exc))
            # A2 deep-copies before raising effective-OFF. Recover diagnostics
            # with the ORIGINAL helpers, never a replacement accounting algorithm.
            row = copy.deepcopy(raw)
            row["arm"] = template["arm"]
            stages = (
                ("initial_prompt_reconciliation", lambda: self.a2.reconcile(row, dict(template, body=template["body_template"]), cpu_prompt)),
                ("attempts", lambda: self.client.assemble_attempts(row.get("sse_frames") or [])),
                ("native_accounting", lambda: self.client.accounting(row, template["body_template"]["max_tokens"])),
                ("native_binding", lambda: self.client.authenticate_calls(row, decoder)),
                ("input_accounting", lambda: self.client.input_accounting(row, row["native_binding"])),
                ("call_accounting", lambda: self.client.call_accounting(row, decoder)),
            )
            for name, operation in stages:
                try:
                    row[name] = operation()
                    diagnostics[name] = row[name]
                except (ValueError, TypeError, KeyError, AttributeError, IndexError) as error:
                    diagnostics[name] = {"error": str(error), "error_type": type(error).__name__}
        try:
            diagnostics["generation_errors"] = self.client.generation_errors(raw)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            diagnostics["generation_errors"] = [{"malformed": str(exc)}]
        for name, flag in (("initial_prompt_reconciliation", "qualified"), ("native_accounting", "complete"),
                           ("native_binding", "qualified"), ("call_accounting", "complete")):
            if diagnostics.get(name, {}).get(flag) is not True:
                errors.append(name + " incomplete/unqualified")
        if diagnostics["generation_errors"]:
            errors.append("native generation/framing errors")
        if raw.get("status") != "ok" or raw.get("transport_status") != "ok" or raw.get("external_tool_calls"):
            errors.append("ineligible status/transport/client-visible calls")
        if diagnostics.get("attempts", {}).get("malformed"):
            errors.append("malformed native tool attempts")
        result["accepted"] = not errors
        return result


def require_accepted(retained):
    """Call only AFTER durably recording the observe() return value."""
    if retained.get("accepted") is not True:
        raise ValueError("retained observation rejected: " + "; ".join(retained.get("errors", [])))
    return retained


def validate_neutrality(adapter, task, seed):
    applicable = ARMS if task["family"] in adapter.plain_families else ARMS[:2]
    rows = {arm: adapter.build_request(task, seed, arm) for arm in applicable}
    if len({tuple(r["request_identity"]) for r in rows.values()}) != len(rows):
        raise ValueError("logical identities collapsed")
    for prefix in ("L", "P"):
        if prefix + "-old" not in rows:
            continue
        old, new = [copy.deepcopy(rows[prefix + "-" + wording]["body_template"]) for wording in ("old", "new")]
        op, np = old.pop("messages"), new.pop("messages")
        if encoded(old) != encoded(new) or op[0]["content"][:-len(OLD[task["family"]])] != np[0]["content"][:-len(NEW[task["family"]])]:
            raise ValueError("wording changed non-suffix request fields")
    if "P-old" in rows:
        for wording in ("old", "new"):
            l = copy.deepcopy(rows["L-" + wording]["body_template"])
            p = rows["P-" + wording]["body_template"]
            if l.pop("tools") != adapter.tools or encoded(l) != encoded(p):
                raise ValueError("L/P differ beyond canonical declaration")
    return rows


def layout(tasks, adapter):
    """Exact frozen pair ordering, independent fresh RNGs for each cold boot."""
    boots = {}
    for boot in ("L", "P"):
        pairs = sorted((task["id"], seed) for task in tasks for seed in (101, 102)
                       if boot == "L" or task["family"] in adapter.plain_families)
        random.Random(20261107).shuffle(pairs)
        order = [False] * (len(pairs) // 2) + [True] * (len(pairs) // 2)
        random.Random(20261108).shuffle(order)
        lookup = {task["id"]: task for task in tasks}
        requests = []
        for (task_id, seed), new_first in zip(pairs, order):
            arms = ("new", "old") if new_first else ("old", "new")
            task = lookup[task_id]
            validate_neutrality(adapter, task, seed)
            requests.extend(adapter.build_request(task, seed, boot + "-" + arm) for arm in arms)
        boots[boot] = requests
    validate_keys(boots, tasks, adapter.plain_families)
    return boots


def validate_keys(boots, tasks, plain_families=PLAIN_FAMILIES):
    expected = {(t["suite"], t["id"], s, a) for t in tasks for s in (101, 102) for a in ARMS
                if a.startswith("L-") or t["family"] in plain_families}
    found = []
    if type(boots) is not dict or set(boots) != {"L", "P"}:
        raise ValueError("two boot arrays required")
    for boot, rows in boots.items():
        if type(rows) is not list:
            raise ValueError("boot must contain a request list")
        for row in rows:
            key = row.get("request_identity") if isinstance(row, dict) else None
            if type(key) is not list or len(key) != 4 or any(type(key[i]) is not str for i in (0, 1, 3)) or type(key[2]) is not int:
                raise ValueError("malformed request identity")
            if key[3] not in ARMS or not key[3].startswith(boot + "-") or row.get("logical_arm") != key[3] or row.get("key") != key[:3]:
                raise ValueError("logical/adapter key mismatch")
            found.append(tuple(key))
    if len(set(found)) != len(found) or set(found) != expected:
        raise ValueError("duplicate/missing/unexpected scheduled key")
    return True
