const std = @import("std");
const Allocator = @import("root").CapAllocator;
const Server = @import("root").CapServer;
pub fn capTokens(a: Allocator, srv: *Server, body: []const u8, max: u32) Allocator.Error![]const u8 {
    const ids = srv.text.encodePlain(a, body) catch return body;
    if (ids.len <= max) return body;
    var n: usize = max;
    while (n > 0) : (n -= 1) {
        const text = try srv.text.decode(a, ids[0..n]);
        if (std.unicode.utf8ValidateSlice(text)) return std.fmt.allocPrint(a, "{s}\n[output truncated at {d} tokens]", .{ text, max });
    }
    return std.fmt.allocPrint(a, "[output truncated at {d} tokens]", .{max});
}
