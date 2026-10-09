//! The Spark wire against prod's own server: every exchange in ``fixtures/serve/wire.jsonl`` (recorded by
//! ``tools/zig/dsv41_ops/http_golden.py`` from the Python DeepSeek-V4.1 app at engine 8474f31 over a real socket)
//! is sent to this server with the same scripted engine, in the same order. Status, Content-Type, the body (JSON or
//! the whole SSE stream), what the engine was asked (prompt ids, max_tokens, sampling, drafts, EOS, background) and
//! the request log's lines must be equal, with ids, clocks and timestamps masked. Needs the release tokenizer
//! (TF_DSV41_MODEL: its directory); without it the test is skipped.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const serve_mod = @import("dsv41_serve");
const deepseek = @import("deepseek.zig");
const server_mod = @import("server.zig");
const listener_mod = @import("listener.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const testing = std.testing;

const eos: u32 = 1;
const window: u32 = 4096;
const D = "｜DSML｜";

/// http_golden.py's SCRIPTS, the last marker in the prompt's last user turn wins.
const scripts = [_]struct { []const u8, []const u8 }{
    .{ "@think", "Let me think about the greeting.</think>Hello there!" },
    .{ "@tool", "I should look up the weather.</think>Checking.\n\n<" ++ D ++ "calls>\n<" ++ D ++ "invoke name=\"get_weather\">\n<" ++ D ++ "parameter name=\"city\" string=\"true\">Paris</" ++ D ++ "parameter>\n<" ++ D ++ "parameter name=\"days\" string=\"false\">3</" ++ D ++ "parameter>\n</" ++ D ++ "invoke>\n<" ++ D ++ "invoke name=\"get_weather\">\n<" ++ D ++ "parameter name=\"city\" string=\"true\">東京</" ++ D ++ "parameter>\n</" ++ D ++ "invoke>\n</" ++ D ++ "calls>" },
    .{ "@stop", "ok</think>abc END def" },
    .{ "@long", "Plenty to say here, more than the limit lets through." },
    .{ "@plain", "Plain answer." },
};

/// The scripted engine: pieces of ``1 + i % 3`` ids, each sent once the server has read the one before (so a stop
/// string or a client stop ends the run where Python's synchronous callback ended it).
const Stub = struct {
    gpa: Allocator,
    io: std.Io,
    tok: *serve_mod.tokenizer.Tokenizer,
    srv: ?*server_mod.Server = null,
    lock: std.Io.Mutex = .init,
    seen: ?Seen = null,
    cancelled: std.atomic.Value(bool) = .init(false),

    const Seen = struct { prompt: []u32, max_tokens: u32, sampling: ?api.Sampling, drafts: bool, stop_eos: bool, background: bool };

    fn engine(s: *Stub) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn self(ctx: *anyopaque) *Stub {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = window, .name = "stub" };
    }
    fn cancel(ctx: *anyopaque, _: api.Id) void {
        self(ctx).cancelled.store(true, .release);
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s = self(ctx);
        s.lock.lockUncancelable(s.io);
        if (s.seen) |x| s.gpa.free(x.prompt);
        s.seen = .{ .prompt = s.gpa.dupe(u32, r.prompt) catch &.{}, .max_tokens = r.max_tokens, .sampling = r.sampling, .drafts = r.drafts, .stop_eos = r.eos.len > 0, .background = r.background };
        s.lock.unlock(s.io);
        s.cancelled.store(false, .release);
        const t = std.Thread.spawn(.{}, run, .{ s, id, r.prompt, r.max_tokens, sink }) catch return error.Busy;
        t.detach();
    }

    /// The tokens the server has read so far (its /health bookkeeping counts them as it reads them).
    fn read(s: *Stub) f64 {
        const h = s.srv.?.health.?;
        h.mutex.lockUncancelable(s.io);
        defer h.mutex.unlock(s.io);
        var n: f64 = 0;
        for (h.inflight.values()) |e| n += e.tokens;
        return n;
    }

    fn run(s: *Stub, id: api.Id, prompt_ids: []const u32, max_tokens: u32, sink: api.Sink) void {
        var arena = std.heap.ArenaAllocator.init(s.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const prompt = s.tok.decodeAlloc(a, prompt_ids) catch "";
        const last = if (std.mem.lastIndexOf(u8, prompt, "<｜User｜>")) |at| prompt[at..] else prompt;
        var reply: []const u8 = "?";
        for (scripts) |sc| if (std.mem.indexOf(u8, last, sc[0]) != null) {
            reply = sc[1];
        };
        var ids = s.tok.encodeAlloc(a, reply) catch &.{};
        ids = std.mem.concat(a, u32, &.{ ids, &.{eos} }) catch ids;
        sink.event(sink.ctx, id, &.{ .prefilled = 3 });
        var i: usize = 0;
        var n: usize = 0;
        var reason: api.Reason = .stop;
        while (i < ids.len and n < max_tokens) {
            const step = @min(1 + i % 3, ids.len - i, max_tokens - n);
            sink.event(sink.ctx, id, &.{ .tokens = ids[i .. i + step] });
            i += step;
            n += step;
            var waited: u32 = 0;
            while (s.read() < @as(f64, @floatFromInt(n)) and !s.cancelled.load(.acquire) and waited < 400) : (waited += 1) std.Io.sleep(s.io, .fromMilliseconds(5), .awake) catch {};
            if (s.cancelled.load(.acquire)) {
                reason = .cancelled;
                break;
            }
        }
        if (reason != .cancelled and i < ids.len) reason = .length;
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = reason, .stats = .{ .rounds = 4, .drafted = 6, .accepted = 5, .prefill_seconds = 0.25, .decode_seconds = 0.5 } } });
    }
};

/// Ids, clocks and timestamps masked: an id keeps its prefix and length, a time becomes 0 (null stays null).
fn mask(a: Allocator, v: Value) Allocator.Error!Value {
    switch (v) {
        .array => |items| {
            const out = try a.alloc(Value, items.len);
            for (items, out) |x, *slot| slot.* = try mask(a, x);
            return .{ .array = out };
        },
        .object => |o| {
            const out = try json.newObject(a);
            for (o.keys(), o.values()) |k, x| {
                const clock = for ([_][]const u8{ "created", "ttft_s", "uptime_s", "oldest_s", "idle_s", "ts", "start", "first_s" }) |c| {
                    if (std.mem.eql(u8, k, c)) break true;
                } else false;
                if (clock and x != .null) {
                    try out.put(a, k, .{ .int = "0" });
                } else if (std.mem.eql(u8, k, "id") and x == .string) {
                    const s = x.string;
                    const cut = (std.mem.indexOfScalar(u8, s, '-') orelse std.mem.indexOfScalar(u8, s, '_') orelse s.len) + 1;
                    try out.put(a, k, .{ .string = try std.fmt.allocPrint(a, "{s}<{d}>", .{ s[0..@min(cut, s.len)], s.len }) });
                } else try out.put(a, k, try mask(a, x));
            }
            return .{ .object = out };
        },
        else => return v,
    }
}

/// A response body as compared: JSON masked and re-dumped, an SSE stream event by event, /metrics with its clock.
fn canonical(a: Allocator, kind: []const u8, body: []const u8) ![]const u8 {
    if (std.mem.startsWith(u8, kind, "text/event-stream")) {
        var out: std.ArrayList(u8) = .empty;
        var it = std.mem.splitSequence(u8, body, "\n\n");
        while (it.next()) |ev| {
            if (ev.len == 0) continue;
            if (!std.mem.startsWith(u8, ev, "data: ")) {
                try out.appendSlice(a, ev);
                continue;
            }
            const data = ev[6..];
            try out.appendSlice(a, "data: ");
            if (std.mem.eql(u8, data, "[DONE]")) try out.appendSlice(a, data) else try out.appendSlice(a, try dump(a, data));
            try out.appendSlice(a, "\n\n");
        }
        return out.items;
    }
    if (std.mem.startsWith(u8, kind, "text/plain")) {
        var out: std.ArrayList(u8) = .empty;
        var it = std.mem.splitScalar(u8, body, '\n');
        while (it.next()) |line| {
            try out.appendSlice(a, if (std.mem.startsWith(u8, line, "tensorfold_uptime_seconds{")) line[0 .. std.mem.lastIndexOfScalar(u8, line, ' ') orelse line.len] else line);
            try out.append(a, '\n');
        }
        return out.items;
    }
    return dump(a, body);
}

fn dump(a: Allocator, text: []const u8) ![]const u8 {
    const parsed = try json.parse(a, text);
    if (parsed != .ok) return text;
    return json.stringify(a, try mask(a, parsed.ok), .{});
}

const Harness = struct {
    gpa: Allocator,
    io: std.Io,
    stub: Stub,
    ds: *deepseek.DeepSeek,
    srv: *server_mod.Server,
    stop: std.atomic.Value(bool) = .init(false),
    thread: std.Thread = undefined,
    port: u16 = 0,
    lis: listener_mod.Listener = undefined,

    fn start(h: *Harness, tok: *serve_mod.tokenizer.Tokenizer, log_path: []const u8, a: Allocator) !void {
        h.ds = try deepseek.DeepSeek.init(h.gpa, tok, .{ .eos = &.{eos} });
        h.stub = .{ .gpa = h.gpa, .io = h.io, .tok = tok };
        const sampling = (try json.parse(a, "{\"temperature\": 1.0, \"top_k\": 20, \"top_p\": 0.95}")).ok;
        h.srv = try server_mod.Server.init(h.gpa, h.io, h.stub.engine(), h.ds.text(), .{
            .served_name = "deepseek-v41",
            .model_ids = &.{ "deepseek-v41", "alias-a" },
            .default_sampling = sampling,
            .family = h.ds.family_(),
            .wire = .spark,
            .spark = .{ .log = .{ .path = log_path } },
        }, null);
        h.stub.srv = h.srv;
        h.lis = try listener_mod.Listener.open(.{ .ip4 = .loopback(0) });
        h.port = h.lis.port();
        h.thread = try std.Thread.spawn(.{}, server_mod.Server.serve, .{ h.srv, h.lis, &h.stop });
    }

    fn finish(h: *Harness) void {
        h.stop.store(true, .release);
        h.thread.join();
        h.lis.close();
        while (h.srv.open_connections.load(.acquire) > 0) std.Io.sleep(h.io, .fromMilliseconds(5), .awake) catch {};
        h.srv.deinit();
        h.ds.deinit(false);
        if (h.stub.seen) |x| h.gpa.free(x.prompt);
    }

    const Got = struct { status: u16, kind: []const u8, body: []const u8 };

    fn send(h: *Harness, a: Allocator, method: []const u8, path: []const u8, body: []const u8) !Got {
        const addr: std.Io.net.IpAddress = .{ .ip4 = .loopback(h.port) };
        const stream = try addr.connect(h.io, .{ .mode = .stream });
        defer stream.close(h.io);
        const head = if (body.len > 0)
            try std.fmt.allocPrint(a, "{s} {s} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ method, path, body.len, body })
        else
            try std.fmt.allocPrint(a, "{s} {s} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", .{ method, path });
        var wbuf: [4096]u8 = undefined;
        var w = std.Io.net.Stream.Writer.init(stream, h.io, &wbuf);
        try w.interface.writeAll(head);
        try w.interface.flush();
        var rbuf: [16384]u8 = undefined;
        var r = std.Io.net.Stream.Reader.init(stream, h.io, &rbuf);
        const text = try r.interface.allocRemaining(a, .unlimited);
        const status = try std.fmt.parseInt(u16, text[9..12], 10);
        const split = std.mem.indexOf(u8, text, "\r\n\r\n") orelse return error.Head;
        var kind: []const u8 = "";
        var lines = std.mem.splitSequence(u8, text[0..split], "\r\n");
        while (lines.next()) |line| if (std.ascii.startsWithIgnoreCase(line, "content-type: ")) {
            kind = line["content-type: ".len..];
        };
        return .{ .status = status, .kind = kind, .body = text[split + 4 ..] };
    }
};

/// What the engine got, as http_golden.py records it.
fn engineSeen(a: Allocator, s: ?Stub.Seen) !Value {
    const x = s orelse return .null;
    const o = try json.newObject(a);
    const ids = try a.alloc(Value, x.prompt.len);
    for (x.prompt, ids) |t, *slot| slot.* = try json.intValue(a, t);
    try o.put(a, "prompt", .{ .array = ids });
    try o.put(a, "max_tokens", try json.intValue(a, x.max_tokens));
    if (x.sampling) |sm| {
        const list = try a.alloc(Value, 4);
        list[0] = .{ .string = try std.fmt.allocPrint(a, "{d}", .{sm.seed}) };
        list[1] = .{ .float = round6(sm.temperature) };
        list[2] = try json.intValue(a, sm.top_k);
        list[3] = .{ .float = round6(sm.top_p) };
        try o.put(a, "sampling", .{ .array = list });
    } else try o.put(a, "sampling", .null);
    try o.put(a, "draft", .{ .bool = x.drafts });
    try o.put(a, "stop_eos", .{ .bool = x.stop_eos });
    try o.put(a, "background", .{ .bool = x.background });
    return .{ .object = o };
}

fn round6(x: anytype) f64 {
    return @round(@as(f64, @floatCast(x)) * 1e6) / 1e6;
}

fn roundedEngine(a: Allocator, v: Value) !Value {
    if (v != .object) return v;
    const s = v.get("sampling") orelse return v;
    if (s != .array) return v;
    const o = try json.copyObject(a, v.object);
    const list = try a.dupe(Value, s.array);
    for ([_]usize{ 1, 3 }) |i| list[i] = .{ .float = round6(list[i].float64() orelse 0) };
    try o.put(a, "sampling", .{ .array = list });
    return .{ .object = o };
}

test "the Spark wire equals prod's Python server on the recorded exchanges (TF_DSV41_MODEL)" {
    const dir = testing.environ.getPosix("TF_DSV41_MODEL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const tok = try serve_mod.tokenizer.Tokenizer.load(gpa, io, dir);
    defer tok.deinit();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const log_path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/requests.jsonl", .{&tmp.sub_path});
    var h: Harness = .{ .gpa = gpa, .io = io, .stub = undefined, .ds = undefined, .srv = undefined };
    try h.start(tok, log_path, a);
    var finished = false;
    defer if (!finished) h.finish();
    var bad: usize = 0;
    var total: usize = 0;
    var want_log: Value = .null;
    var lines = std.mem.splitScalar(u8, serve_mod.fixtures.wire, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const ex = (try json.parse(a, line)).ok;
        const name = ex.strField("name").?;
        if (std.mem.eql(u8, name, "_reqlog")) {
            want_log = ex.get("lines").?;
            continue;
        }
        total += 1;
        if (h.stub.seen) |x| gpa.free(x.prompt);
        h.stub.seen = null;
        const got = try h.send(a, ex.strField("method").?, ex.strField("path").?, ex.strField("body").?);
        const want_status: u16 = @intCast(ex.get("status").?.int64().?);
        const want_kind = ex.strField("type") orelse "";
        const want_body = try canonical(a, want_kind, ex.strField("response").?);
        const got_body = try canonical(a, got.kind, got.body);
        const want_engine = try json.stringify(a, try roundedEngine(a, ex.get("engine").?), .{});
        const got_engine = try json.stringify(a, try engineSeen(a, h.stub.seen), .{});
        const same = got.status == want_status and std.mem.eql(u8, got.kind, want_kind) and std.mem.eql(u8, got_body, want_body) and std.mem.eql(u8, got_engine, want_engine);
        if (!same) {
            bad += 1;
            std.debug.print("\n== MISMATCH {s}: status {d} vs {d}, type '{s}' vs '{s}'\n  zig: {s}\n  py:  {s}\n", .{ name, got.status, want_status, got.kind, want_kind, got_body[0..@min(got_body.len, 1500)], want_body[0..@min(want_body.len, 1500)] });
            if (!std.mem.eql(u8, got_engine, want_engine)) std.debug.print("  engine zig: {s}\n  engine py:  {s}\n", .{ got_engine[0..@min(got_engine.len, 400)], want_engine[0..@min(want_engine.len, 400)] });
        }
    }
    h.finish();
    finished = true;
    // the request log: one line a run, the same records
    const written = try std.Io.Dir.cwd().readFileAlloc(io, log_path, a, .limited(1 << 20));
    var got_lines = std.mem.splitScalar(u8, written, '\n');
    var i: usize = 0;
    var log_bad: usize = 0;
    for (want_log.array) |want| {
        const got_line = got_lines.next() orelse "";
        const got_v = if (got_line.len > 0) try dump(a, got_line) else "(none)";
        const want_v = try json.stringify(a, try mask(a, want), .{});
        if (!std.mem.eql(u8, got_v, want_v)) {
            log_bad += 1;
            std.debug.print("\n== REQLOG {d}\n  zig: {s}\n  py:  {s}\n", .{ i, got_v, want_v });
        }
        i += 1;
    }
    if (bad + log_bad > 0) std.debug.print("\nwire: {d}/{d} exchanges equal; request log {d}/{d} lines equal\n", .{ total - bad, total, want_log.array.len - log_bad, want_log.array.len });
    try testing.expectEqual(@as(usize, 49), total);
    try testing.expectEqual(@as(usize, 28), want_log.array.len);
    try testing.expectEqual(@as(usize, 0), bad);
    try testing.expectEqual(@as(usize, 0), log_bad);
}
