//! The compute lane through the chat route: a held python call runs, its result turn is forced in, the reply goes on.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const HfText = @import("hf_text.zig").HfText;
const f = @import("hf_eos_fixture.zig");
const tf = @import("control_tokens_fixture.zig");
const lane = api.compute_lane;
const t = std.testing;
const Allocator = std.mem.Allocator;

/// An engine whose reply calls python, holds at the EOS, and after the result answers and holds again.
const Script = struct {
    a: Allocator,
    call: []const u32, // the call segment, its EOS last
    answer: []const u32, // the answer segment, its EOS last
    sink: ?api.Sink = null,
    id: api.Id = 0,
    held_asked: bool = false,
    resumes: std.ArrayList([]const u32) = .empty,

    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .resume_held = resumeHeld } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 32768 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        s.held_asked = request.hold_eos;
        s.sink = sink;
        s.id = id;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = s.call });
        if (!request.hold_eos) sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn resumeHeld(ctx: *anyopaque, id: api.Id, r: api.Resume) void {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        const sink = s.sink.?;
        s.resumes.append(s.a, s.a.dupe(u32, r.tokens) catch unreachable) catch unreachable;
        if (r.tokens.len == 0) return sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
        // the forced rows come back as committed tokens, then the model's own
        sink.event(sink.ctx, id, &.{ .tokens = r.tokens });
        sink.event(sink.ctx, id, &.{ .tokens = s.answer });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Reply = struct {
    a: Allocator,
    code: u16 = 0,
    body: json.Value = .null,
    fn out(r: *@This()) @import("openai.zig").Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}
    fn reply(ctx: *anyopaque, code: u16, v: json.Value) void {
        const r: *@This() = @ptrCast(@alignCast(ctx));
        r.code = code;
        r.body = v;
    }
};

pub fn obj(a: Allocator, text: []const u8) !json.Value {
    return (try json.parseText(a, text)).ok;
}

pub fn appendTo(a: Allocator, list: json.Value, more: []const json.Value) !json.Value {
    return .{ .array = try std.mem.concat(a, json.Value, &.{ list.array, more }) };
}

/// The mini DeepSeek tokenizer, byte-level both ways so a reply decodes back to its text, and its EOS.
pub fn fixture(a: Allocator) !f.Fixture {
    var out = try f.Fixture.init(a, t.io, null);
    errdefer out.deinit();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(t.io, "zig/src/families/deepseek_v41/fixtures/serve/mini_tokenizer.json", a, .limited(1 << 20));
    const tok = try obj(a, bytes);
    try tok.object.put(a, "pre_tokenizer", try obj(a, "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}"));
    try tok.object.put(a, "decoder", try obj(a, "{\"type\":\"ByteLevel\"}"));
    try out.tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer.json", .data = try json.stringify(a, tok, .{}) });
    try out.tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"deepseek_v41\"}" });
    try out.tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer_config.json", .data = "{\"eos_token\":\"<｜end▁of▁sentence｜>\"}" });
    return out;
}

test "compute lane: a held python call's result is forced in as the template's tool turn and the reply goes on" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const python = "/usr/bin/python3";
    std.Io.Dir.accessAbsolute(t.io, python, .{}) catch return error.SkipZigTest;
    var tmp = try fixture(a);
    defer tmp.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, tmp.path, null, a, &problem);
    defer hf.deinit();
    const text = hf.text();
    // the model's segments, as the template spells an assistant call turn and a plain answer
    const history = try tf.messages(a, "What is 6*7?");
    const call_turn = try obj(a,
        \\{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"python","arguments":"{\"code\":\"print(6*7)\"}"}}]}
    );
    const asked = try text.renderIds(a, history, .{ .add_generation_prompt = true }, &problem);
    const called = try text.renderIds(a, try appendTo(a, history, &.{call_turn}), .{ .add_generation_prompt = false }, &problem);
    try t.expect(std.mem.startsWith(u32, called, asked));
    const eos = called[called.len - 1];
    const answer = try std.mem.concat(a, u32, &.{ try text.encodePlain(a, "It is 42."), &.{eos} });
    var script: Script = .{ .a = a, .call = called[asked.len..], .answer = answer };
    const srv = try @import("server.zig").Server.init(a, t.io, script.engine(), text, .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    // the pool without the trampoline: the test runner is not the server binary (the sandbox has its own tests)
    const cfg: lane.Config = .{ .python = python };
    srv.info.compute = cfg;
    srv.compute = lane.Pool.init(cfg, &.{ "/bin/sh", "-c", "exec \"$0\" -I -B -u -c \"$1\"", python, lane.runner.runner });
    var reply: Reply = .{ .a = a };
    var conn = try @import("http_conn.zig").Conn.init(a, -1, "fixture");
    defer conn.deinit();
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    try body.put(a, "max_tokens", try json.intValue(a, 400));
    try body.put(a, "messages", history);
    try body.put(a, "tools", try obj(a, "[{\"type\":\"code_interpreter\"}]"));
    @import("openai.zig").run(srv, a, reply.out(), .{ .conn = &conn }, true, .{ .object = body });
    const said = try json.stringify(a, reply.body, .{});
    errdefer std.debug.print("COMPUTE_LANE_REPLY {s}\n", .{said});
    try t.expectEqual(@as(u16, 200), reply.code);
    try t.expect(script.held_asked);
    // one result inserted, then the answer's EOS ends the reply with an empty resume
    try t.expectEqual(@as(usize, 2), script.resumes.items.len);
    try t.expectEqual(@as(usize, 0), script.resumes.items[1].len);
    const result = try obj(a, "{\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":\"42\"}");
    const full = try text.renderIds(a, try appendTo(a, history, &.{ call_turn, result }), .{ .add_generation_prompt = true }, &problem);
    try t.expect(std.mem.startsWith(u32, full, called));
    try t.expectEqualSlices(u32, full[called.len..], script.resumes.items[0]);
    // the client sees the answer, not the built-in call, and the lane's report of it
    const choice = reply.body.get("choices").?.array[0];
    try t.expectEqualStrings("stop", choice.get("finish_reason").?.string);
    const message = choice.get("message").?;
    try t.expect(std.mem.indexOf(u8, message.strField("content").?, "It is 42.") != null);
    try t.expect(message.get("tool_calls") == null or message.get("tool_calls").? == .null);
    const compute = reply.body.get("tensorfold").?.get("compute").?;
    try t.expectEqual(@as(i64, @intCast(script.resumes.items[0].len)), compute.get("inserted_tokens").?.int64().?);
    const report = compute.get("calls").?.array[0];
    try t.expectEqualStrings("42", report.strField("output").?);
    try t.expectEqualStrings("ok", report.strField("status").?);
    try t.expect(report.get("inserted").?.bool);
    // S3 off: the reply names no executor and counts none (the lane's reply as before)
    try t.expect(report.get("executor") == null and report.get("call_ms") == null and compute.get("runner_calls") == null);
}

test "compute lane: code_interpreter on a server without the lane is refused before the engine sees it" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try fixture(a);
    defer tmp.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, tmp.path, null, a, &problem);
    defer hf.deinit();
    var script: Script = .{ .a = a, .call = &.{}, .answer = &.{} };
    const srv = try @import("server.zig").Server.init(a, t.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var reply: Reply = .{ .a = a };
    var conn = try @import("http_conn.zig").Conn.init(a, -1, "fixture");
    defer conn.deinit();
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    try body.put(a, "max_tokens", try json.intValue(a, 4));
    try body.put(a, "messages", try tf.messages(a, "H"));
    try body.put(a, "tools", try obj(a, "[{\"type\":\"code_interpreter\"}]"));
    @import("openai.zig").run(srv, a, reply.out(), .{ .conn = &conn }, true, .{ .object = body });
    try t.expect(reply.code >= 400);
    try t.expect(script.sink == null);
}

/// S3's scripted circuit: code with "abstain" abstains (unsupported), else it runs and prints `out`.
const Circuit = struct {
    out: []const u8,
    fn frontend(c: *Circuit) api.runner_api.Frontend {
        return .{ .ctx = c, .translate = translate };
    }
    fn translate(_: ?*anyopaque, _: Allocator, cell: api.runner_api.Cell) Allocator.Error!api.runner_api.Translation {
        if (std.mem.indexOf(u8, cell.code, "abstain") != null) return .{ .abstain = .unsupported };
        return .{ .program = .{ .kind = .calc, .bytes = cell.code } };
    }
    fn backend(c: *Circuit) api.runner_api.Backend {
        return .{ .ctx = c, .vtable = &.{ .start = start, .poll = poll, .cancel = noop, .release = noop } };
    }
    fn start(_: *anyopaque, _: u64, _: api.runner_api.Program, _: u64) error{ OutOfMemory, Busy }!api.runner_api.JobId {
        return 1;
    }
    fn poll(ctx: *anyopaque, _: api.runner_api.JobId) api.runner_api.Poll {
        const c: *Circuit = @ptrCast(@alignCast(ctx));
        return .{ .done = .{ .status = .ok, .out = c.out, .steps = 3 } };
    }
    fn noop(_: *anyopaque, _: u64) void {}
};

/// A report's millisecond field: a float, or an int when it printed without a fraction.
fn ms(v: json.Value) f64 {
    return switch (v) {
        .float => |x| x,
        .int => |text| std.fmt.parseFloat(f64, text) catch -1,
        else => -1,
    };
}

/// One held call of `code` through the chat route with S3 on: the forced tokens and the result turn they must equal.
const S3Run = struct { reply: Reply, inserted: []const u32, want: []const u32, report: json.Value, compute: json.Value };

fn s3Run(a: Allocator, code: []const u8, circuit: *Circuit, sandbox_ok: bool, verify: bool, result: []const u8) !S3Run {
    const python = "/usr/bin/python3";
    std.Io.Dir.accessAbsolute(t.io, python, .{}) catch return error.SkipZigTest;
    var tmp = try fixture(a);
    defer tmp.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, tmp.path, null, a, &problem);
    defer hf.deinit();
    const text = hf.text();
    const history = try tf.messages(a, "What is 6*7?");
    const call_turn = try obj(a,
        \\{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"python","arguments":""}}]}
    );
    const args = try json.newObject(a);
    try args.put(a, "code", .{ .string = code });
    const function = call_turn.get("tool_calls").?.array[0].get("function").?;
    try function.object.put(a, "arguments", .{ .string = try json.stringify(a, .{ .object = args }, .{}) });
    const asked = try text.renderIds(a, history, .{ .add_generation_prompt = true }, &problem);
    const called = try text.renderIds(a, try appendTo(a, history, &.{call_turn}), .{ .add_generation_prompt = false }, &problem);
    const eos = called[called.len - 1];
    const answer = try std.mem.concat(a, u32, &.{ try text.encodePlain(a, "It is 42."), &.{eos} });
    const script = try a.create(Script);
    script.* = .{ .a = a, .call = called[asked.len..], .answer = answer };
    const srv = try @import("server.zig").Server.init(a, t.io, script.engine(), text, .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    const cfg: lane.Config = .{ .python = python };
    srv.info.compute = cfg;
    // a sandbox that cannot run python would print something else: a circuit-owned call never reaches it
    const launcher: []const []const u8 = if (sandbox_ok) &.{ "/bin/sh", "-c", "exec \"$0\" -I -B -u -c \"$1\"", python, lane.runner.runner } else &.{"/nonexistent"};
    srv.compute = lane.Pool.init(cfg, launcher);
    srv.compute_exec = .{ .frontend = circuit.frontend(), .backend = circuit.backend(), .max_steps = 1000, .cfg = .{ .verify = verify } };
    var reply: Reply = .{ .a = a };
    var conn = try @import("http_conn.zig").Conn.init(a, -1, "fixture");
    defer conn.deinit();
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    try body.put(a, "max_tokens", try json.intValue(a, 400));
    try body.put(a, "messages", history);
    try body.put(a, "tools", try obj(a, "[{\"type\":\"code_interpreter\"}]"));
    @import("openai.zig").run(srv, a, reply.out(), .{ .conn = &conn }, true, .{ .object = body });
    errdefer std.debug.print("S3_REPLY {s}\n", .{json.stringify(a, reply.body, .{}) catch ""});
    try t.expectEqual(@as(u16, 200), reply.code);
    const turn = try obj(a, "{\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":\"\"}");
    try turn.object.put(a, "content", .{ .string = result });
    const full = try text.renderIds(a, try appendTo(a, history, &.{ call_turn, turn }), .{ .add_generation_prompt = true }, &problem);
    const compute = reply.body.get("tensorfold").?.get("compute").?;
    return .{ .reply = reply, .inserted = script.resumes.items[0], .want = full[called.len..], .report = compute.get("calls").?.array[0], .compute = compute };
}

test "compute runner: a held call the circuit runs inserts the lane's own result turn, named in-model runner" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var c: Circuit = .{ .out = "42\n" };
    const r = try s3Run(arena.allocator(), "print(6*7)", &c, false, false, "42");
    try t.expectEqualSlices(u32, r.want, r.inserted);
    try t.expectEqualStrings("in-model runner", r.report.strField("executor").?);
    try t.expectEqualStrings("calc", r.report.strField("program").?);
    // the server's time on the call covers the run's own
    try t.expect(ms(r.report.get("call_ms").?) >= ms(r.report.get("exec_ms").?));
    try t.expectEqual(@as(i64, 1), r.compute.get("runner_calls").?.int64().?);
    try t.expectEqual(@as(i64, 0), r.compute.get("tool_calls").?.int64().?);
}

test "compute runner: a call the circuit declines runs in the sandbox, named code tool with the reason" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var c: Circuit = .{ .out = "wrong\n" };
    const r = try s3Run(arena.allocator(), "print(42)  # abstain", &c, true, false, "42");
    try t.expectEqualSlices(u32, r.want, r.inserted);
    try t.expectEqualStrings("code tool", r.report.strField("executor").?);
    try t.expectEqualStrings("unsupported", r.report.strField("runner_abstain").?);
    try t.expect(r.report.get("program") == null);
    try t.expectEqual(@as(i64, 0), r.compute.get("runner_calls").?.int64().?);
    try t.expectEqual(@as(i64, 1), r.compute.get("tool_calls").?.int64().?);
}

test "compute runner: verify inserts the sandbox's result and reports a circuit that printed otherwise" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var c: Circuit = .{ .out = "41\n" };
    const r = try s3Run(arena.allocator(), "print(6*7)", &c, true, true, "42");
    try t.expectEqualSlices(u32, r.want, r.inserted);
    try t.expectEqualStrings("code tool", r.report.strField("executor").?);
    try t.expectEqualStrings("differ", r.report.strField("verify").?);
    var same: Circuit = .{ .out = "42\n" };
    const s = try s3Run(arena.allocator(), "print(6*7)", &same, true, true, "42");
    try t.expectEqualStrings("same", s.report.strField("verify").?);
}
