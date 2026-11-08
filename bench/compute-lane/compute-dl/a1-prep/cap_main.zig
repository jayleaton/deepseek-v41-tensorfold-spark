const std = @import("std");
const json = @import("json");
const tok = @import("tokenizer");
const Allocator = std.mem.Allocator;
const Text = struct {
    t: *tok.Tokenizer,
    pub fn encodePlain(s: Text, a: Allocator, body: []const u8) ![]u32 { return s.t.encodePlain(a, body); }
    pub fn decode(s: Text, a: Allocator, ids: []const u32) Allocator.Error![]u8 { return s.t.decode(a, ids, false) catch |e| if (e == error.OutOfMemory) error.OutOfMemory else a.dupe(u8, "\u{fffd}"); }
};
const Server = struct { text: Text };
// cap_function.zig is generated from the pinned production source, unchanged.
const policy = @import("cap_function.zig");
const body_policy = @import("body_function.zig");
pub const CapAllocator = Allocator;
pub const CapServer = Server;
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) return error.Usage;
    var t = try tok.loadTokenizer(init.io, init.gpa, args[2]);
    defer t.deinit();
    var srv = Server{ .text = .{ .t = &t } };
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(16 << 20));
    const input = (try json.parse(a, bytes)).ok;
    var buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(init.io, &buf);
    for (input.array) |r| {
        const body = if (r.strField("status")) |status| blk: {
            const code = if (r.get("exit_code")) |value| try std.fmt.parseInt(i32, value.int, 10) else 0;
            const outcome = body_policy.Outcome{
                .status = std.meta.stringToEnum(body_policy.Status, status) orelse return error.BadStatus,
                .stdout = r.strField("stdout") orelse "", .stderr = r.strField("stderr") orelse "",
                .exit_code = if (code >= 0) @intCast(code) else null,
                .signal = if (code < 0) @intCast(-code) else null,
            };
            break :blk try body_policy.body(a, outcome, 3000, 65536);
        } else r.strField("body") orelse return error.BodyRequired;
        // Fail closed on encoding failure before production's fallback can hide it.
        const ids = try srv.text.encodePlain(a, body);
        const clipped = try policy.capTokens(a, &srv, body, 512);
        const obj = try json.newObject(a);
        try obj.put(a, "key", r.get("key") orelse .null);
        try obj.put(a, "token_count", try json.intValue(a, ids.len));
        var retained: usize = @min(ids.len, 512);
        if (ids.len > 512) {
            while (retained > 0) : (retained -= 1) {
                const prefix = try srv.text.decode(a, ids[0..retained]);
                if (std.unicode.utf8ValidateSlice(prefix)) break;
            }
        }
        try obj.put(a, "retained_prefix_tokens", try json.intValue(a, retained));
        try obj.put(a, "output", .{ .string = clipped });
        try obj.put(a, "normalized_body", .{ .string = body });
        try out.interface.print("{s}\n", .{try json.stringify(a, .{ .object = obj }, .{ .ascii = false })});
    }
    try out.interface.flush();
}
