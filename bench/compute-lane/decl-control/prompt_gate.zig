//! CPU-only: unchanged native tools/template/tokenizer functions; inert arming context.
const std = @import("std");
const json = @import("json");
const template = @import("template");
const tok = @import("native_tokenizer");
const turns = @import("turns");
const Server = @typeInfo(@typeInfo(@TypeOf(turns.tools)).@"fn".param_types[0].?).pointer.child;
const Cx = @typeInfo(@typeInfo(@TypeOf(turns.tools)).@"fn".param_types[1].?).pointer.child;
const Request = @typeInfo(@typeInfo(@TypeOf(turns.arms)).@"fn".param_types[2].?).pointer.child;
fn idsValue(a: std.mem.Allocator, ids: []const u32) !json.Value {
    const v = try a.alloc(json.Value, ids.len);
    for (ids, v) |id, *slot| slot.* = try json.intValue(a, id);
    return .{ .array = v };
}
fn encode(a: std.mem.Allocator, t: *tok.Tokenizer, messages: json.Value, tools: []json.Value) ![]u32 {
    var problem: template.Problem = .{};
    const text = try template.render(a, messages, .{ .tools = tools, .thinking = true, .budget = 75, .add_generation_prompt = true }, &problem);
    return t.encodeAlloc(a, text);
}
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) return error.Usage;
    const t = try tok.Tokenizer.load(init.gpa, init.io, args[2]);
    defer t.deinit();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(16 << 20));
    const records = (try json.parse(a, bytes)).ok;
    var on: Server = .{ .compute = 1 };
    var off: Server = .{ .compute = null };
    var cx: Cx = .{ .a = a };
    const optin = (try json.parse(a, "[{\"type\":\"code_interpreter\"}]")).ok;
    const built = try turns.tools(&on, &cx, optin, null);
    // Serialize and parse an actual native expansion, then use the same explicit declaration in D/L.
    const wire = try json.stringify(a, .{ .array = built }, .{ .ascii = false });
    const explicit = (try json.parse(a, wire)).ok;
    const dtools = try turns.tools(&off, &cx, explicit, null);
    const ltools = try turns.tools(&on, &cx, explicit, null);
    const ptools = try turns.tools(&off, &cx, null, null);
    var outbuf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(init.io, &outbuf);
    for (records.array) |record| {
        const messages = record.get("messages").?;
        const pids = try encode(a, t, messages, ptools);
        const dids = try encode(a, t, messages, dtools);
        const lids = try encode(a, t, messages, ltools);
        const old = try encode(a, t, messages, built);
        if (!std.mem.eql(u32, dids, lids) or !std.mem.eql(u32, lids, old)) return error.PromptMismatch;
        const r: Request = .{ .eos = &.{0}, .max_tokens = 8192 };
        const input_d = .{ .tools = dtools };
        const input_l = .{ .tools = ltools };
        // arms has a concrete chat.Input type; construct it from its function signature.
        const Input = @typeInfo(@TypeOf(turns.arms)).@"fn".param_types[1].?;
        const held_d = turns.arms(&off, Input{ .tools = input_d.tools }, &r);
        const held_l = turns.arms(&on, Input{ .tools = input_l.tools }, &r);
        if (held_d or !held_l) return error.ArmingMismatch;
        const result = try json.newObject(a);
        try result.put(a, "key", record.get("key").?);
        try result.put(a, "plain_ids", try idsValue(a, pids));
        try result.put(a, "declared_off_ids", try idsValue(a, dids));
        try result.put(a, "declared_lane_ids", try idsValue(a, lids));
        try result.put(a, "original_builtin_lane_ids", try idsValue(a, old));
        try result.put(a, "explicit_tools", explicit);
        try result.put(a, "held_off", .{ .bool = held_d });
        try result.put(a, "held_lane", .{ .bool = held_l });
        try out.interface.print("{s}\n", .{try json.stringify(a, .{ .object = result }, .{})});
    }
    try out.interface.flush();
}
