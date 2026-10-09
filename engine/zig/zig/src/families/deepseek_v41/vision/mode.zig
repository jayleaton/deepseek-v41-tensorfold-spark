//! ``TF_DSV41_IMAGES`` (``vision.mode``): placeholder (the default: a notice the model cannot see the image), reject
//! (HTTP 400), native (the tower on rank 0). Every rank reads it; only native loads the tower and the image bias.
const std = @import("std");

pub const Mode = enum { placeholder, reject, native };

pub fn parse(raw: ?[]const u8) ?Mode {
    const t = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (t.len == 0) return .placeholder;
    var buf: [16]u8 = undefined;
    if (t.len > buf.len) return null;
    return std.meta.stringToEnum(Mode, std.ascii.lowerString(&buf, t));
}

test "TF_DSV41_IMAGES reads as Python's mode()" {
    try std.testing.expectEqual(Mode.placeholder, parse(null).?);
    try std.testing.expectEqual(Mode.native, parse(" Native ").?);
    try std.testing.expect(parse("on") == null);
}
