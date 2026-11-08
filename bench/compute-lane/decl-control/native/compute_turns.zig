//! The compute lane's request side: the built-in `python` tool a request opts into, arming, and the client's view.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const chat = @import("chat.zig");
const tool_specs = @import("tool_specs.zig");
const Server = @import("server.zig").Server;
const lane = api.compute_lane;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// The built-in tool's name, the type a request lists to opt in, and its description (which marks it as ours).
pub const name = "python";
pub const opt_in = "code_interpreter";
pub const description = "Run Python 3 code in a sandbox with no network, no files and no installs, and get its stdout and " ++
    "stderr. print() what you need; the value of a final bare expression is printed too.";
const definition =
    \\{"type":"function","function":{"name":"python","description":"
++ description ++
    \\","parameters":{"type":"object","properties":{"code":{"type":"string","description":"Python 3 source to run"}},"required":["code"]}}}
;

/// The server's sandbox pool for an engine that serves the lane; boot fails when an opted-in sandbox cannot run.
pub fn open(srv: *Server, cfg: lane.Config) !lane.Pool {
    if (srv.engine.vtable.resume_held == null) {
        std.log.err("compute lane: this engine cannot hold a reply for a tool result", .{});
        return error.ComputeLaneUnsupported;
    }
    var pool = lane.Pool.init(cfg, &.{ "/proc/self/exe", "__compute_sandbox" });
    var arena: std.heap.ArenaAllocator = .init(srv.gpa);
    defer arena.deinit();
    pool.selfTest(arena.allocator()) catch |e| {
        std.log.err("compute lane: the sandbox self-test failed ({t}); unset the compute lane flag or fix the sandbox", .{e});
        return error.ComputeLaneUnavailable;
    };
    std.log.info("compute lane: {s} in a sandbox, {d} at a time, {d} ms, {d} MiB, results up to {d} tokens, {d} calls a reply", .{ cfg.python, cfg.workers, cfg.max_ms, cfg.mem_mb, cfg.max_result_tokens, cfg.max_calls });
    return pool;
}

/// The request's tools as tool_specs reads them, with `{"type": "code_interpreter"}` turned into the built-in tool.
pub fn tools(srv: *Server, cx: *Cx, raw: ?Value, choice: ?Value) errors.Refused![]Value {
    const list = (raw orelse return tool_specs.active(cx, raw, choice));
    if (list != .array) return tool_specs.active(cx, raw, choice);
    var rest: std.ArrayList(Value) = .empty;
    var asked = false;
    for (list.array) |t| {
        const kind = if (t == .object) t.strField("type") orelse "" else "";
        if (std.mem.eql(u8, kind, opt_in)) asked = true else try rest.append(cx.a, t);
    }
    if (!asked) return tool_specs.active(cx, raw, choice);
    if (srv.compute == null) return cx.other("code_interpreter is not enabled on this server");
    const specs = try tool_specs.active(cx, .{ .array = rest.items }, choice);
    for (specs) |t| if (std.mem.eql(u8, toolName(t), name)) return cx.other("a request with code_interpreter cannot also offer a function named python");
    // only an automatic choice lets the model call it: "none", "required" and a named function leave it out
    if (choice) |c| if (!(c == .null or (c == .string and std.mem.eql(u8, c.string, "auto")))) return specs;
    const built = switch (try json.parse(cx.a, definition)) {
        .ok => |v| v,
        .err => unreachable,
    };
    return std.mem.concat(cx.a, Value, &.{ specs, &.{built} });
}

fn toolName(t: Value) []const u8 {
    const f = t.get("function") orelse return "";
    return f.strField("name") orelse "";
}

/// Whether `t` is the built-in tool (its name and our description), as `tools` added it.
pub fn isBuiltIn(t: Value) bool {
    const f = t.get("function") orelse return false;
    return std.mem.eql(u8, f.strField("name") orelse "", name) and std.mem.eql(u8, f.strField("description") orelse "", description);
}

/// Whether a call names the built-in tool.
pub fn isBuiltInCall(call: Value) bool {
    return std.mem.eql(u8, toolName(call), name);
}

/// Whether a prepared request holds at its EOS for the lane: offered the tool, nothing that forced rows would break.
pub fn arms(srv: *const Server, input: chat.Input, r: *const api.Request) bool {
    if (srv.compute == null or r.eos.len == 0 or r.background or r.max_tokens == 0) return false;
    if (r.structure != null or r.call != null or r.logprobs != null or r.prompt_logprobs != null) return false;
    for (input.tools) |t| if (isBuiltIn(t)) return true;
    return false;
}

/// A lane reply as the client streams it: built-in calls stay hidden; client calls number on across segments.
pub const Filter = struct {
    a: Allocator,
    inner: chat.Sink,
    base: i64 = 0, // client calls earlier segments showed
    seen: i64 = 0, // client calls this segment showed
    map: std.AutoHashMapUnmanaged(i64, ?i64) = .empty, // this segment's call index -> the client's (null: built-in)

    pub fn sink(f: *Filter) chat.Sink {
        return .{ .ctx = f, .call = call, .begin = if (f.inner.begin != null) begin else null };
    }

    /// A resumed segment numbers its calls from 0 again.
    pub fn nextSegment(f: *Filter) void {
        f.base += f.seen;
        f.seen = 0;
        f.map.clearRetainingCapacity();
    }

    /// A delta straight to the client (the lane's own reports).
    pub fn direct(f: *Filter, delta: Value) error{Closed}!void {
        return f.inner.call(f.inner.ctx, delta);
    }

    fn begin(ctx: *anyopaque, prompt_len: usize) error{Closed}!void {
        const f: *Filter = @ptrCast(@alignCast(ctx));
        return f.inner.begin.?(f.inner.ctx, prompt_len);
    }

    fn call(ctx: *anyopaque, delta: Value) error{Closed}!void {
        const f: *Filter = @ptrCast(@alignCast(ctx));
        const list = if (delta == .object) delta.get("tool_calls") else null;
        if (list == null or list.? != .array) return f.inner.call(f.inner.ctx, delta);
        var kept: std.ArrayList(Value) = .empty;
        for (list.?.array) |one| {
            const at = (one.get("index") orelse continue).int64() orelse continue;
            if (one.get("function")) |fun| if (fun.strField("name")) |n| if (!f.map.contains(at)) {
                const client: ?i64 = if (std.mem.eql(u8, n, name)) null else f.base + f.seen;
                if (client != null) f.seen += 1;
                f.map.put(f.a, at, client) catch return error.Closed;
            };
            const to = (f.map.get(at) orelse continue) orelse continue;
            if (one != .object) continue;
            one.object.put(f.a, "index", json.intValue(f.a, to) catch return error.Closed) catch return error.Closed;
            kept.append(f.a, one) catch return error.Closed;
        }
        if (kept.items.len == 0) return;
        delta.object.put(f.a, "tool_calls", .{ .array = kept.items }) catch return error.Closed;
        return f.inner.call(f.inner.ctx, delta);
    }
};

test "the filter drops built-in call deltas and numbers client calls on across segments" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Seen = struct {
        out: std.ArrayList([]const u8) = .empty,
        a: Allocator,
        fn call(ctx: *anyopaque, delta: Value) error{Closed}!void {
            const s: *@This() = @ptrCast(@alignCast(ctx));
            s.out.append(s.a, json.stringify(s.a, delta, .{}) catch return error.Closed) catch return error.Closed;
        }
    };
    var seen: Seen = .{ .a = a };
    var f: Filter = .{ .a = a, .inner = .{ .ctx = &seen, .call = Seen.call } };
    const s = f.sink();
    const deltas = [_][]const u8{
        \\{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"python","arguments":""}}]}
        ,
        \\{"tool_calls":[{"index":0,"function":{"arguments":"print(1)"}}]}
        ,
        \\{"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"lookup","arguments":"{}"}}]}
        ,
    };
    for (deltas) |d| try s.call(s.ctx, (try json.parse(a, d)).ok);
    try s.call(s.ctx, .{ .string = "prose" });
    f.nextSegment();
    try s.call(s.ctx, (try json.parse(a, deltas[2])).ok);
    try std.testing.expectEqual(@as(usize, 3), seen.out.items.len);
    try std.testing.expect(std.mem.indexOf(u8, seen.out.items[0], "\"index\": 0") != null);
    try std.testing.expect(std.mem.indexOf(u8, seen.out.items[0], "lookup") != null);
    try std.testing.expectEqualStrings("\"prose\"", seen.out.items[1]);
    try std.testing.expect(std.mem.indexOf(u8, seen.out.items[2], "\"index\": 1") != null);
}

test "the built-in definition parses and is recognized; another function named python is not" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const built = (try json.parse(arena.allocator(), definition)).ok;
    try std.testing.expect(isBuiltIn(built));
    const other = (try json.parse(arena.allocator(), "{\"type\":\"function\",\"function\":{\"name\":\"python\"}}")).ok;
    try std.testing.expect(!isBuiltIn(other));
}
