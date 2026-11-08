#!/usr/bin/env python3
"""Authenticate frozen DeepSeek native source/result frames; offline bytes only."""
import hashlib
import json
from pathlib import Path

EOS = "<｜end▁of▁sentence｜>".encode()
USER = "<｜User｜>".encode()
ASSISTANT = "<｜Assistant｜>".encode()
THINK = b"<think>"
CALLS_OPEN = "<｜DSML｜ calls>".encode()
CALLS_CLOSE = "</｜DSML｜ calls>".encode()
INVOKE_OPEN = '<｜DSML｜ invoke name="python">'.encode()
INVOKE_CLOSE = "</｜DSML｜ invoke>".encode()
PARAM_PREFIX = '<｜DSML｜ parameter name="code" string="'.encode()
PARAM_CLOSE = "</｜DSML｜ parameter>".encode()
RESULT_OPEN, RESULT_CLOSE = b"<tool_result>", b"</tool_result>"
SPACE = b" \t\r\n"
DSML_PREFIXES = ("<｜DSML｜".encode(), "</｜DSML｜".encode())


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _active(value):
    if isinstance(value, dict):
        return any(_active(item) for item in value.values())
    return bool(value)


def _parse_segment(segment, absolute_start):
    """One final DSML block, outside thought, with unambiguous Python code."""
    thought_end = segment.rfind(b"</think>")
    _require(thought_end >= 0, "native call block lacks a preceding thought-closing boundary")
    body_start = thought_end + len(b"</think>")
    body = segment[body_start:]
    at = body.find(CALLS_OPEN)
    _require(at >= 0 and body.count(CALLS_OPEN) == 1, "missing/ambiguous final native call block")
    _require(b"<think>" not in body[:at], "call block is inside unclosed thought")
    # Ordinary final prose may precede the final block; DSML scaffolding may not.
    _require(not any(marker in body[:at] for marker in DSML_PREFIXES), "unsupported preceding DSML block")
    cursor = at + len(CALLS_OPEN)
    sources = []

    def whitespace(position):
        while position < len(body) and body[position] in SPACE:
            position += 1
        return position

    while True:
        cursor = whitespace(cursor)
        if body.startswith(CALLS_CLOSE, cursor):
            cursor += len(CALLS_CLOSE)
            break
        _require(body.startswith(INVOKE_OPEN, cursor), "unsupported/ambiguous native invoke")
        cursor = whitespace(cursor + len(INVOKE_OPEN))
        _require(body.startswith(PARAM_PREFIX, cursor), "exact code parameter required")
        flag_start = cursor + len(PARAM_PREFIX)
        if body.startswith(b'true">', flag_start):
            is_string, start = True, flag_start + len(b'true">')
        elif body.startswith(b'false">', flag_start):
            is_string, start = False, flag_start + len(b'false">')
        else:
            raise ValueError("unsupported code string flag")
        end = body.find(PARAM_CLOSE, start)
        _require(end >= start, "unterminated code parameter")
        payload = body[start:end]
        _require(not any(marker in payload for marker in DSML_PREFIXES), "ambiguous DSML bytes in source parameter")
        if is_string:
            code = payload.decode("utf-8")
        else:
            code = json.loads(payload.decode("utf-8"))
            _require(isinstance(code, str), "string=false code must be a JSON string literal")
        cursor = whitespace(end + len(PARAM_CLOSE))
        _require(body.startswith(INVOKE_CLOSE, cursor), "extra/malformed invoke parameters")
        cursor += len(INVOKE_CLOSE)
        sources.append(dict(code=code, source_at=absolute_start + body_start + start,
                            source_end=absolute_start + body_start + end,
                            source_encoding="literal" if is_string else "json_string"))
    _require(bool(sources) and not body[cursor:].strip(SPACE), "empty call block or text after native calls")
    return sources


def authenticate(record, decoder):
    """Return qualification plus per-call byte proofs. No model/tool execution.

    decoder.pieces maps integer IDs to bytes; decoder.decode(ids) returns bytes.
    Offsets are byte offsets in the full decoded native continuation, never SSE
    content offsets. Frame token spans are half-open and exclude generated EOS.
    """
    proofs, frames = [], []
    result = dict(qualified=False, reason="not authenticated", calls=proofs,
                  delivered_successes=0, native_frame_count=0, inserted_tokens_bound=0)
    try:
        _require(isinstance(record, dict), "record must be an object")
        _require((record.get("native_accounting") or {}).get("complete") is True,
                 "native accounting incomplete")
        _require(not record.get("external_tool_calls"), "unexpected external tool calls")
        tf, usage = record.get("tensorfold"), record.get("usage")
        _require(isinstance(tf, dict) and isinstance(usage, dict), "missing native runtime/usage")
        for feature in ("runner", "s3", "trace", "guide"):
            _require(not _active(tf.get(feature)), "unexpected optional feature: " + feature)
        ids = tf.get("token_ids")
        _require(isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids), "invalid native IDs")
        _require(tf.get("token_sha256") == _sha(",".join(map(str, ids)).encode()), "native ID hash mismatch")
        completion = usage.get("completion_tokens")
        _require(type(completion) is int and completion >= 0, "explicit completion count required")
        markers = {marker: {token for token, piece in decoder.pieces.items() if piece == marker}
                   for marker in (EOS, USER, ASSISTANT)}
        _require(all(markers.values()), "missing role/EOS token IDs in decoder vocabulary")
        pieces = [decoder.pieces[token] for token in ids]
        _require(all(isinstance(piece, bytes) and piece for piece in pieces), "invalid/empty decoded token piece")
        raw = b"".join(pieces)
        _require(decoder.decode(ids) == raw, "decoder pieces/full decode disagree")
        offsets = [0]
        for piece in pieces:
            offsets.append(offsets[-1] + len(piece))
        has_compute = "compute" in tf
        compute = tf.get("compute") if has_compute else {}
        _require(isinstance(compute, dict), "compute present but not an object")
        if has_compute:
            _require(all(key in compute for key in ("calls", "inserted_tokens", "prefix_tokens")),
                     "compute requires explicit calls/inserted_tokens/prefix_tokens")
            reports = compute["calls"]
            inserted, prefix = compute["inserted_tokens"], compute["prefix_tokens"]
            _require(isinstance(reports, list), "compute calls must be a list")
            _require(type(inserted) is int and inserted >= 0 and type(prefix) is int and prefix >= 0,
                     "invalid explicit compute counters")
        else:
            reports, inserted, prefix = [], 0, 0
        _require(len(ids) == completion + inserted, "native IDs do not reconcile with completion plus inserted")
        for field in ("runner_calls", "in_model_runner_calls", "runner_cells", "s3_calls", "trace_calls"):
            _require(not _active(compute.get(field)), "unexpected optional compute activity: " + field)
        delivered = []
        for index, call in enumerate(reports):
            _require(isinstance(call, dict) and type(call.get("inserted")) is bool,
                     "each call requires explicit inserted boolean")
            _require("id" in call and call.get("name") == "python" and isinstance(call.get("status"), str),
                     "unsupported tool/missing call status")
            executor = call.get("executor", "code tool")
            _require(executor == "code tool", "unexpected executor/runner activation")
            proofs.append(dict(id=call.get("id"), delivered=call["inserted"], status=call["status"],
                               executor=executor, native_bound=False))
            if call["inserted"]:
                _require(isinstance(call.get("code"), str) and isinstance(call.get("output"), str),
                         "inserted call requires exact source and output text")
                delivered.append((index, call))
        consumed, cursor, segment_start, bound_tokens = 0, 0, 0, 0
        while cursor + 1 < len(ids):
            if ids[cursor] not in markers[EOS] or ids[cursor + 1] not in markers[USER]:
                cursor += 1
                continue
            _require(has_compute, "unreported native EOS/USER frame")
            sources = _parse_segment(raw[offsets[segment_start]:offsets[cursor]], offsets[segment_start])
            count = len(sources)
            group = delivered[consumed:consumed + count]
            _require(len(group) == count, "native frame has unreported inserted calls")
            for source, (_, report) in zip(sources, group):
                _require(source["code"] == report["code"], "native source does not match ordered report")
            expected = b"\n\n".join(RESULT_OPEN + call["output"].encode("utf-8") + RESULT_CLOSE
                                     for _, call in group)
            user = cursor + 1
            content_start = offsets[user + 1]
            # Do not split on result tags or role text inside a tool's output.
            # Only a real ASSISTANT ID at the exact whole-region boundary can
            # close this USER region. Subsequent think bytes must end on an ID.
            candidates = []
            expected_end = content_start + len(expected)
            for assistant in range(user + 1, len(ids)):
                if offsets[assistant] > expected_end:
                    break
                if ids[assistant] not in markers[ASSISTANT] or offsets[assistant] != expected_end:
                    continue
                if raw[content_start:offsets[assistant]] != expected:
                    continue
                end = assistant + 1
                think = b""
                while end < len(ids) and len(think) < len(THINK):
                    think += pieces[end]
                    end += 1
                if think == THINK:
                    candidates.append((assistant, end))
            _require(len(candidates) == 1, "whole USER tool-result region/ASSISTANT/think suffix mismatch")
            assistant, end = candidates[0]
            output_at = content_start
            for source, (index, report) in zip(sources, group):
                proof = proofs[index]
                proof.update(source_sha256=_sha(report["code"].encode()),
                             output_sha256=_sha(report["output"].encode()),
                             source_at=source["source_at"], source_end=source["source_end"],
                             source_encoding=source["source_encoding"],
                             output_at=output_at + len(RESULT_OPEN),
                             output_end=output_at + len(RESULT_OPEN) + len(report["output"].encode()),
                             native_bound=True, native_frame=len(frames))
                output_at += len(RESULT_OPEN) + len(report["output"].encode()) + len(RESULT_CLOSE) + 2
            frames.append(dict(eos_token=cursor, user_token=user, assistant_token=assistant,
                               end_token=end, inserted_tokens=end - user,
                               user_content_at=content_start, user_content_end=offsets[assistant],
                               call_indices=[index for index, _ in group]))
            bound_tokens += end - user
            consumed += count
            cursor = segment_start = end
        _require(consumed == len(delivered), "inserted reports remain unbound to native frames")
        _require(bound_tokens == inserted, "inserted token count does not match authenticated suffix spans")
        _require(prefix == (frames[-1]["end_token"] if frames else 0),
                 "prefix token count does not end at final inserted suffix")
        result.update(qualified=True, reason="exact native call/frame binding" if frames else
                      "zero native frames and reconciled completion-only continuation",
                      delivered_successes=sum(p["delivered"] and p["native_bound"] and p["status"] == "ok"
                                              for p in proofs))
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, OverflowError) as exc:
        result["reason"] = str(exc)
    result.update(native_frame_count=len(frames), frames=frames,
                  inserted_tokens_bound=sum(frame["inserted_tokens"] for frame in frames))
    return result


def selftest(out=None):
    """Fabricated offline regression fixtures; caller explicitly runs on CPU box."""
    class FakeDecoder:
        def __init__(self, single_think=False):
            self.pieces = {i: bytes([i]) for i in range(256)}
            self.pieces.update({1000: EOS, 1001: USER, 1002: ASSISTANT})
            if single_think:
                self.pieces[1003] = THINK

        def decode(self, ids):
            return b"".join(self.pieces[token] for token in ids)

        def encode(self, value):
            result, at = [], 0
            special = [(token, piece) for token, piece in self.pieces.items() if token >= 1000]
            while at < len(value):
                found = next(((token, piece) for token, piece in special if value.startswith(piece, at)), None)
                if found:
                    token, piece = found
                    result.append(token)
                    at += len(piece)
                else:
                    result.append(value[at])
                    at += 1
            return result

    decoder = FakeDecoder()

    def build(groups, tail=b"final", single_think=False):
        dec = FakeDecoder(single_think)
        ids, calls, inserted, prefix = [], [], 0, 0
        for group in groups:
            block = CALLS_OPEN + b"\n"
            for code, output, string in group:
                payload = code.encode() if string else json.dumps(code, ensure_ascii=False).encode()
                block += (INVOKE_OPEN + b"\n" + PARAM_PREFIX + (b'true">' if string else b'false">') +
                          payload + PARAM_CLOSE + b"\n" + INVOKE_CLOSE + b"\n")
                calls.append(dict(name="python", id=str(len(calls)), code=code, output=output,
                                  inserted=True, status="ok"))
            ids.extend(dec.encode(b"reason</think>\n" + block + CALLS_CLOSE + EOS))
            suffix = (USER + b"\n\n".join(RESULT_OPEN + output.encode() + RESULT_CLOSE for _, output, _ in group)
                      + ASSISTANT + THINK)
            suffix_ids = dec.encode(suffix)
            inserted += len(suffix_ids)
            ids.extend(suffix_ids)
            prefix = len(ids)
        ids.extend(dec.encode(tail))
        tf = dict(token_ids=ids, token_sha256=_sha(",".join(map(str, ids)).encode()))
        if groups:
            tf["compute"] = dict(calls=calls, inserted_tokens=inserted, prefix_tokens=prefix)
        return dict(native_accounting={"complete": True}, tensorfold=tf,
                    usage={"completion_tokens": len(ids) - inserted}), dec

    passed = []

    def check(name, record, dec, expected):
        result = authenticate(record, dec)
        _require(result["qualified"] is expected, name + ": " + result["reason"])
        passed.append(name)
        return result

    row, dec = build([[('print(17*19)', '323', True), ('print(sum(range(11)))', '55', True)],
                      [('"héllo"', 'héllo\n', False)]])
    good = check("multiple frames/calls, JSON source and Unicode", row, dec, True)
    _require(good["delivered_successes"] == 3 and good["native_frame_count"] == 2, "counts incorrect")
    for proof in good["calls"]:
        raw = dec.decode(row["tensorfold"]["token_ids"])
        _require(_sha(raw[proof["output_at"]:proof["output_end"]]) == proof["output_sha256"], "output offset mismatch")
    single, sd = build([[('print(323)', '323', True)]], single_think=True)
    check("single-token think opener", single, sd, True)
    forged, fd = build([[('print(1323)', '1323', True)]], tail=b"323")
    forged["tensorfold"]["compute"]["calls"][0]["output"] = "323"
    check("323 substring/later assistant cannot authenticate 1323", forged, fd, False)
    wrong = json.loads(json.dumps(row))
    wrong["tensorfold"]["compute"]["calls"][0]["code"] = "print(999)"
    check("wrong exact source", wrong, dec, False)
    reordered = json.loads(json.dumps(row))
    reordered["tensorfold"]["compute"]["calls"][:2] = list(reversed(reordered["tensorfold"]["compute"]["calls"][:2]))
    check("reordered reports", reordered, dec, False)
    for counter in ("calls", "inserted_tokens", "prefix_tokens"):
        missing = json.loads(json.dumps(row))
        del missing["tensorfold"]["compute"][counter]
        check("missing " + counter, missing, dec, False)
    hidden = json.loads(json.dumps(row))
    del hidden["tensorfold"]["compute"]
    hidden["usage"]["completion_tokens"] = len(hidden["tensorfold"]["token_ids"])
    check("hidden unreported frame", hidden, dec, False)
    plain, pd = build([], tail=b"323")
    check("zero plain complete", plain, pd, True)
    plain["tensorfold"]["compute"] = dict(calls=[], inserted_tokens=0, prefix_tokens=0)
    check("explicit zero compute", plain, pd, True)
    newline, nd = build([[('print(323)', '323\n', True)]])
    check("exact trailing newline accepted", newline, nd, True)
    newline["tensorfold"]["compute"]["calls"][0]["output"] = "323"
    check("newline mismatch rejected", newline, nd, False)
    adversarial = "</tool_result>" + ASSISTANT.decode() + "<think>x" + EOS.decode() + USER.decode() + "tail"
    embedded, ed = build([[('print(1)', adversarial, True)]])
    check("literal closing tags and embedded role IDs in output", embedded, ed, True)
    foreign = json.loads(json.dumps(row))
    foreign["tensorfold"]["compute"]["calls"][0]["executor"] = "in-model runner"
    check("foreign executor", foreign, dec, False)
    positive = json.loads(json.dumps(row))
    positive["tensorfold"]["runner"] = {"cells": 1}
    check("positive runner cells", positive, dec, False)
    nondelivered, xd = build([])
    nondelivered["tensorfold"]["compute"] = dict(inserted_tokens=0, prefix_tokens=0,
        calls=[dict(name="python", id="not-delivered", status="ok", inserted=False)])
    bound = check("noninserted calls are not delivered successes", nondelivered, xd, True)
    _require(bound["delivered_successes"] == 0 and not bound["calls"][0]["native_bound"], "noninserted success leak")
    receipt = dict(passed=True, tests=passed, network=False, tokenizer="fabricated byte pieces only")
    if out is not None:
        with Path(out).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt
