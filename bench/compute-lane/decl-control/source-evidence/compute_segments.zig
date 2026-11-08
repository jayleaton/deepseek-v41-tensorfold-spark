//! A lane reply's segments: at each held EOS its built-in calls run, their result turn is forced in, it goes on.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const chat = @import("chat.zig");
const reply_text = @import("reply_text.zig");
const tool_parse = @import("tool_parse.zig");
const tool_stream = @import("tool_stream.zig");
const prompt_mod = @import("prompt.zig");
const turns = @import("compute_turns.zig");
const log = @import("log.zig");
const Generation = @import("generation.zig").Generation;
const Server = @import("server.zig").Server;
const lane = api.compute_lane;
const Value = json.Value;
const Cx = @import("errors.zig").Cx;
const Allocator = std.mem.Allocator;

/// Tokens the reply keeps free after an inserted result, so the model can still answer.
const answer_room = 16;

/// What the earlier segments of a reply said and did, merged into its last segment's reply.
const Merge = struct {
    content: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    completion: usize = 0,
    reasoning_tokens: usize = 0,
    ids: std.ArrayList(u32) = .empty, // the engine's whole continuation: segments and inserted results
    inserted: usize = 0,
    owned: ?reply_text.Owned = null, // the segments' runner bytes in `content` (Generation.ownedOf)
    calls: std.ArrayList(Value) = .empty, // one report a call
    history: std.ArrayList(Value) = .empty, // the request's messages, then each segment's call turn and results
};

/// Continues a reply whose first segment `first` ended held at its EOS, until the engine ends it.
pub fn rest(first: *Generation, cx: *Cx, p: chat.Prepared, filter: *turns.Filter, gone: anytype) chat.Failure!chat.Reply {
    const a = cx.a;
    const srv = first.srv;
    const cfg = srv.info.compute.?;
    var m: Merge = .{};
    if (p.input.messages == .array) try m.history.appendSlice(a, p.input.messages.array);
    var g = first;
    var run_calls: u32 = 0;
    while (g.held) {
        const used = m.ids.items.len + g.collected.items.len;
        const answer = try segmentAnswer(g);
        const calls = builtInOnly(answer.calls);
        var plan: ?[]const u32 = null;
        var assistant: Value = .null;
        var reports: []Value = &.{};
        if (calls.len > 0 and used + answer_room < p.request.max_tokens) {
            assistant = try assistantTurn(a, answer.prose, answer.reasoning, calls);
            reports = try runAll(a, srv, cfg, calls, &run_calls, gone);
            plan = try suffix(srv, cx, p, &m.history, assistant, calls, reports);
            if (plan) |s| if (used + s.len + answer_room > p.request.max_tokens) {
                plan = null;
            };
        }
        const s = plan orelse {
            // no call to run, a client's call, or no room: the reply ends at this EOS as it would without the lane
            _ = srv.engine.resumeHeld(g.id, .{ .tokens = &.{} });
            g.held = false;
            try g.loop(gone);
            for (reports) |r| {
                try r.object.put(a, "inserted", .{ .bool = false });
                try m.calls.append(a, r);
            }
            return merge(a, &m, try finish(g, cx, p), g.collected.items);
        };
        _ = srv.engine.resumeHeld(g.id, .{ .tokens = s, .think_open = p.thinking });
        const seg = try finish(g, cx, p);
        try note(a, &m, g, seg, s);
        try m.history.append(a, assistant);
        for (calls, reports) |c, r| try m.history.append(a, try toolTurn(a, c, r.get("output").?.string));
        for (reports) |r| {
            try r.object.put(a, "inserted", .{ .bool = true });
            try m.calls.append(a, r);
            if (g.sink != null) filter.direct(try reportDelta(a, r)) catch return error.Cancelled;
        }
        g = try next(g, s, p.request.max_tokens - (used + s.len));
        g.base = m.ids.items.len; // the runner records' token indices count every segment and result before it
        filter.nextSegment();
        try g.loop(gone);
    }
    return merge(a, &m, try finish(g, cx, p), g.collected.items);
}

fn finish(g: *Generation, cx: *Cx, p: chat.Prepared) chat.Failure!chat.Reply {
    return g.finish(cx, p.prompt_len, p.received, p.thinking, p.effort, p.sampled, p.drafts);
}

/// The segment after an inserted result: the same reply, its first tokens the result's, its own text and calls.
fn next(g: *Generation, inserted: []const u32, left: usize) chat.Failure!*Generation {
    const n = try g.a.create(Generation);
    n.* = .{ .srv = g.srv, .a = g.a, .box = g.box, .id = g.id, .reply_id = g.reply_id, .seed = g.seed, .sink = g.sink, .thinking = g.thinking, .stops = g.stops, .ignore_eos = g.ignore_eos, .max_tokens = @intCast(left), .tools = g.tools, .token_ids = g.token_ids, .loop_stop = g.loop_stop, .hold = true, .skip = inserted, .consumed = g.consumed, .chunk_index = g.chunk_index, .first_ns = g.first_ns, .wrote_ns = g.wrote_ns, .cells_sent = g.cells_sent, .cells_sig = g.cells_sig, .deadline_ns = g.deadline_ns, .deadline_hit = g.deadline_hit };
    if (g.calls != null) n.calls = try tool_stream.Streamer.init(g.a, g.tools);
    return n;
}

/// The segment's calls when every one is the built-in tool's (a client's call ends the reply for its client).
fn builtInOnly(calls: ?[]Value) []Value {
    const all = calls orelse return &.{};
    for (all) |c| if (!turns.isBuiltInCall(c)) return &.{};
    return all;
}

const Answer = struct { prose: []const u8, reasoning: ?[]const u8, calls: ?[]Value };

/// The segment's answer as `Generation.finish` reads it: stops, a call closed at the EOS, the think split, the calls.
fn segmentAnswer(g: *Generation) chat.Failure!Answer {
    const a = g.a;
    const tokens = reply_text.stripTrailing(g.collected.items, g.srv.eos);
    const visible = g.stops.visible(try g.srv.text.decode(a, tokens), false);
    const close = try tool_parse.closeCall(a, visible, g.tools);
    const text = if (close.len > 0) try std.mem.concat(a, u8, &.{ visible, close }) else visible;
    var answer = text;
    var reasoning: ?[]const u8 = null;
    if (g.thinking) {
        const split = reply_text.splitThinkingTokens(a, reply_text.decodeText, g.srv.text, text, true, g.srv.markers, tokens, g.srv.close_id) catch return error.OutOfMemory;
        answer = split.answer;
        const r = reply_text.pyStrip(split.reasoning);
        if (r.len > 0) reasoning = r;
    }
    const parsed = try tool_parse.parse(a, answer, g.tools, null);
    return .{ .prose = parsed.content, .reasoning = reasoning, .calls = parsed.calls };
}

/// Runs each call in order (results stay in call order); past the reply's call cap a call gets that note instead.
fn runAll(a: Allocator, srv: *Server, cfg: lane.Config, calls: []const Value, ran: *u32, gone: anytype) chat.Failure![]Value {
    const Watch = struct {
        gone: @TypeOf(gone),
        fn left(ctx: *anyopaque) bool {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            return w.gone.check();
        }
    };
    var watch: Watch = .{ .gone = gone };
    const out = try a.alloc(Value, calls.len);
    for (calls, out) |c, *slot| {
        const code = try codeOf(a, c);
        var o: lane.Outcome = .{ .status = .unavailable, .stderr = "the call has no string argument \"code\"" };
        var body: []const u8 = undefined;
        var s3: ?s3_api.Outcome = null;
        var verify: ?bool = null;
        const t0 = s3_api.nowNs();
        if (ran.* >= cfg.max_calls) {
            body = "[call limit reached: answer without running more code]";
        } else {
            const hook: lane.Cancel = .{ .ctx = &watch, .gone = Watch.left };
            // S3: the circuit first; a decline, or verify's shadow run, goes to the sandbox as the lane always did
            if (code) |src| if (srv.compute_exec) |*e| {
                s3 = try e.run(a, src, hook);
            };
            if (s3) |r| if (r == .declined and r.declined == .cancelled) return error.Cancelled;
            const owned = ownedOf(s3);
            const shadow = owned != null and srv.compute_exec.?.cfg.verify;
            if (owned == null or shadow) {
                if (code) |src| o = try srv.compute.?.run(a, src, hook);
                if (o.status == .cancelled) return error.Cancelled;
                body = try lane.runner.body(a, o, cfg.max_ms, cfg.max_output_bytes);
            }
            if (owned) |w| if (shadow) {
                verify = std.mem.eql(u8, w.body, body);
                if (!verify.?) log.line("warning: compute lane verify: the circuit printed [{s}], the sandbox [{s}]", .{ w.body, body });
            } else {
                body = w.body;
                o = .{ .status = .ok, .exec_ns = w.exec_ns };
            };
            ran.* += 1;
        }
        slot.* = try report(a, c, code orelse "", o, try capTokens(a, srv, body, cfg.max_result_tokens));
        if (srv.compute_exec != null) try executor(a, slot.object, s3, verify, s3_api.nowNs() - t0);
    }
    return out;
}

const s3_api = api.compute_circuit;

/// The circuit's run when it ran the call to its halt; null when declined or S3 is off.
fn ownedOf(s3: ?s3_api.Outcome) ?s3_api.Owned {
    const r = s3 orelse return null;
    return switch (r) {
        .owned => |w| w,
        .declined => null,
    };
}

/// S3's report fields: who produced the inserted body (never both), what ran it or why the circuit declined.
fn executor(a: Allocator, r: *json.Object, s3: ?s3_api.Outcome, verify: ?bool, call_ns: u64) Allocator.Error!void {
    const w = ownedOf(s3);
    // the server's whole time on the call (translate, queue, run; both executors under verify), beside exec_ms's run
    try r.put(a, "call_ms", .{ .float = @as(f64, @floatFromInt(call_ns)) / 1e6 });
    const by_circuit = w != null and verify == null;
    try r.put(a, "executor", .{ .string = if (by_circuit) s3_api.in_model else s3_api.code_tool });
    if (by_circuit) {
        try r.put(a, "program", .{ .string = @tagName(w.?.kind) });
        try r.put(a, "steps", try json.intValue(a, w.?.steps));
    } else if (s3) |x| if (x == .declined) try r.put(a, "runner_abstain", .{ .string = @tagName(x.declined) });
    if (verify) |same| try r.put(a, "verify", .{ .string = if (same) "same" else "differ" });
}

/// The call's "code" argument, its arguments a JSON object or a JSON string; null when missing or not a string.
fn codeOf(a: Allocator, call: Value) Allocator.Error!?[]const u8 {
    const f = call.get("function") orelse return null;
    const args = f.get("arguments") orelse return null;
    const obj: Value = if (args == .string) switch (try json.parseText(a, args.string)) {
        .ok => |v| v,
        .err => return null,
    } else args;
    const code = obj.strField("code") orelse return null;
    return if (std.mem.trim(u8, code, " \t\r\n").len > 0) code else null;
}

/// The result text cut to `max` tokens (decoded back at a character boundary), with a note when cut.
fn capTokens(a: Allocator, srv: *Server, body: []const u8, max: u32) Allocator.Error![]const u8 {
    const ids = srv.text.encodePlain(a, body) catch return body;
    if (ids.len <= max) return body;
    var n: usize = max;
    while (n > 0) : (n -= 1) {
        const text = try srv.text.decode(a, ids[0..n]);
        if (std.unicode.utf8ValidateSlice(text)) return std.fmt.allocPrint(a, "{s}\n[output truncated at {d} tokens]", .{ text, max });
    }
    return std.fmt.allocPrint(a, "[output truncated at {d} tokens]", .{max});
}

fn report(a: Allocator, call: Value, code: []const u8, o: lane.Outcome, output: []const u8) Allocator.Error!Value {
    const r = try json.newObject(a);
    try r.put(a, "name", .{ .string = turns.name });
    try r.put(a, "id", call.get("id") orelse .null);
    try r.put(a, "code", .{ .string = code });
    try r.put(a, "status", .{ .string = @tagName(o.status) });
    try r.put(a, "exit_code", if (o.exit_code) |x| try json.intValue(a, x) else .null);
    try r.put(a, "exec_ms", .{ .float = @as(f64, @floatFromInt(o.exec_ns)) / 1e6 });
    try r.put(a, "output", .{ .string = output });
    return .{ .object = r };
}

fn reportDelta(a: Allocator, r: Value) Allocator.Error!Value {
    const d = try json.newObject(a);
    try d.put(a, "compute", r);
    return .{ .object = d };
}

/// The assistant turn the segment wrote, as a client would send it back: its prose, reasoning and calls.
fn assistantTurn(a: Allocator, prose: []const u8, reasoning: ?[]const u8, calls: []Value) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = "assistant" });
    try o.put(a, "content", .{ .string = prose });
    if (reasoning) |r| try o.put(a, "reasoning_content", .{ .string = r });
    try o.put(a, "tool_calls", .{ .array = calls });
    return .{ .object = o };
}

fn toolTurn(a: Allocator, call: Value, output: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = "tool" });
    try o.put(a, "tool_call_id", call.get("id") orelse .null);
    try o.put(a, "content", .{ .string = output });
    return .{ .object = o };
}

/// The template's tokens after the held EOS: render(history + call turn + results) past render(history + call turn).
fn suffix(srv: *Server, cx: *Cx, p: chat.Prepared, history: *const std.ArrayList(Value), assistant: Value, calls: []const Value, reports: []const Value) chat.Failure!?[]const u32 {
    const a = cx.a;
    var with: std.ArrayList(Value) = .empty;
    try with.appendSlice(a, history.items);
    try with.append(a, assistant);
    const before = prompt_mod.renderIds(srv, cx, .{ .array = try a.dupe(Value, with.items) }, p.input.tools, p.thinking, p.effort, false) catch |e| return lost(e);
    for (calls, reports) |c, r| try with.append(a, try toolTurn(a, c, r.get("output").?.string));
    const after = prompt_mod.renderIds(srv, cx, .{ .array = with.items }, p.input.tools, p.thinking, p.effort, true) catch |e| return lost(e);
    // the history must end at the model's own EOS and stand unchanged under the results: else no exact insert exists
    const ends = before.len > 0 and std.mem.indexOfScalar(u32, srv.eos, before[before.len - 1]) != null;
    if (!ends or after.len <= before.len or !std.mem.eql(u32, after[0..before.len], before)) {
        log.line("warning: compute lane: this template's tool-result turn does not extend the call turn; the reply ends at the call", .{});
        return null;
    }
    return after[before.len..];
}

fn lost(e: anyerror) chat.Failure!?[]const u32 {
    if (e == error.OutOfMemory) return error.OutOfMemory;
    log.line("warning: compute lane: the tool-result turn did not render ({t}); the reply ends at the call", .{e});
    return null;
}

/// A finished held segment into the merge: its words, its tokens and the result inserted after it.
fn note(a: Allocator, m: *Merge, g: *const Generation, seg: chat.Reply, inserted: []const u32) Allocator.Error!void {
    const parsed = try tool_parse.parseOwned(a, seg.content, g.tools, null, seg.owned);
    m.owned = reply_text.Owned.join(m.owned, reply_text.Owned.shift(parsed.owned, m.content.items.len));
    try m.content.appendSlice(a, parsed.content);
    if (seg.reasoning) |r| try m.reasoning.appendSlice(a, r);
    m.completion += seg.completion_tokens;
    m.reasoning_tokens += seg.reasoning_tokens;
    try m.ids.appendSlice(a, g.collected.items);
    try m.ids.appendSlice(a, inserted);
    m.inserted += inserted.len;
}

/// The last segment's reply with the earlier segments' words, counts, calls and tokens in front.
fn merge(a: Allocator, m: *Merge, last: chat.Reply, last_ids: []const u32) chat.Failure!chat.Reply {
    var r = last;
    if (m.content.items.len > 0) r.content = try std.mem.concat(a, u8, &.{ m.content.items, last.content });
    r.owned = reply_text.Owned.join(m.owned, reply_text.Owned.shift(last.owned, m.content.items.len));
    if (m.reasoning.items.len > 0) r.reasoning = try std.mem.concat(a, u8, &.{ m.reasoning.items, last.reasoning orelse "" });
    r.completion_tokens += m.completion;
    r.reasoning_tokens += m.reasoning_tokens;
    if (m.calls.items.len == 0 and m.inserted == 0) return r;
    const rt = r.runtime.object;
    const tail = if (rt.get("token_ids")) |t| t.array else &[_]Value{};
    const full = try a.alloc(Value, m.ids.items.len + tail.len);
    for (m.ids.items, full[0..m.ids.items.len]) |t, *slot| slot.* = try json.intValue(a, t);
    @memcpy(full[m.ids.items.len..], tail);
    if (rt.get("token_ids") != null) try rt.put(a, "token_ids", .{ .array = full });
    // the shas hash the same whole continuation the ids list, inserted results included
    const all = try std.mem.concat(a, u32, &.{ m.ids.items, last_ids });
    const tokens = @import("tokens.zig");
    try rt.put(a, "token_sha", .{ .string = try a.dupe(u8, &tokens.tokenSha(all)) });
    try rt.put(a, "token_sha256", .{ .string = try a.dupe(u8, &tokens.tokenSha256(all)) });
    const c = try json.newObject(a);
    try c.put(a, "calls", .{ .array = m.calls.items });
    try c.put(a, "inserted_tokens", try json.intValue(a, m.inserted));
    try c.put(a, "prefix_tokens", try json.intValue(a, m.ids.items.len));
    try counts(a, c, m.calls.items);
    try rt.put(a, "compute", .{ .object = c });
    return r;
}

/// S3's per-executor call counts (only when the reports name executors): in-model runner and code tool, never summed.
fn counts(a: Allocator, c: *json.Object, calls: []const Value) Allocator.Error!void {
    var by_circuit: i64 = 0;
    var by_tool: i64 = 0;
    for (calls) |r| {
        const e = r.strField("executor") orelse return;
        if (std.mem.eql(u8, e, s3_api.in_model)) by_circuit += 1 else by_tool += 1;
    }
    if (calls.len == 0) return;
    try c.put(a, "runner_calls", try json.intValue(a, by_circuit));
    try c.put(a, "tool_calls", try json.intValue(a, by_tool));
}
