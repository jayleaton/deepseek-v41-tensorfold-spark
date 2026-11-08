const std = @import("std");
const Allocator = std.mem.Allocator;
pub const Status = enum { ok, exit_error, timeout, output_limit, cancelled, unavailable };
pub const Outcome = struct {
    status: Status,
    exit_code: ?u8 = null,
    signal: ?u8 = null,
    stdout: []const u8 = "",
    stderr: []const u8 = "",
    exec_ns: u64 = 0,
};
pub fn body(a: Allocator, o: Outcome, max_ms: u32, max_output_bytes: u32) Allocator.Error![]u8 {
    var t: std.ArrayList(u8) = .empty;
    try t.appendSlice(a, o.stdout);
    if (o.stderr.len > 0) {
        try t.appendSlice(a, if (t.items.len == 0) "[stderr]\n" else "\n[stderr]\n");
        try t.appendSlice(a, o.stderr);
    }
    if (o.status != .ok) {
        if (t.items.len > 0 and t.items[t.items.len - 1] != '\n') try t.append(a, '\n');
        switch (o.status) {
            .ok => unreachable,
            .exit_error => if (o.signal) |s|
                try t.print(a, "[killed by signal {d}]", .{s})
            else
                try t.print(a, "[exit status {d}]", .{o.exit_code orelse 0}),
            .timeout => try t.print(a, "[timed out after {d} ms]", .{max_ms}),
            .output_limit => try t.print(a, "[output truncated at {d} bytes]", .{max_output_bytes}),
            .cancelled => try t.appendSlice(a, "[cancelled]"),
            .unavailable => try t.appendSlice(a, "[sandbox unavailable]"),
        }
    }
    if (t.items.len > 0 and t.items[t.items.len - 1] == '\n') t.items.len -= 1;
    if (t.items.len == 0) try t.appendSlice(a, "[no output]");
    return t.toOwnedSlice(a);
}
