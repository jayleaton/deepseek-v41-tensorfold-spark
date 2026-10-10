//! Python's ``json.dumps(v, sort_keys=True)`` (separators ", " / ": ", ensure_ascii, float ``repr``) of a parsed
//! ``std.json.Value``: the bytes the image digests hash.
const std = @import("std");
const Writer = std.Io.Writer;

pub fn dump(w: *Writer, v: std.json.Value) Writer.Error!void {
    switch (v) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .integer => |i| try w.print("{d}", .{i}),
        .number_string => |s| try w.writeAll(s),
        .float => |f| try float(w, f),
        .string => |s| try string(w, s),
        .array => |a| {
            try w.writeByte('[');
            for (a.items, 0..) |x, i| {
                if (i > 0) try w.writeAll(", ");
                try dump(w, x);
            }
            try w.writeByte(']');
        },
        .object => |o| {
            var keys_buf: [256][]const u8 = undefined;
            const n = @min(o.count(), keys_buf.len);
            for (o.keys()[0..n], 0..) |k, i| keys_buf[i] = k;
            const keys = keys_buf[0..n];
            std.mem.sort([]const u8, keys, {}, struct {
                fn lt(_: void, a: []const u8, b: []const u8) bool {
                    return lessUtf16(a, b);
                }
            }.lt);
            try w.writeByte('{');
            for (keys, 0..) |k, i| {
                if (i > 0) try w.writeAll(", ");
                try string(w, k);
                try w.writeAll(": ");
                try dump(w, o.get(k).?);
            }
            try w.writeByte('}');
        },
    }
}

/// Python sorts str keys by code point; for UTF-8 that is byte order.
fn lessUtf16(a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

pub fn string(w: *Writer, s: []const u8) Writer.Error!void {
    try w.writeByte('"');
    var it = std.unicode.Utf8View.initUnchecked(s).iterator();
    while (it.nextCodepoint()) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        0x08 => try w.writeAll("\\b"),
        0x0c => try w.writeAll("\\f"),
        0x20...0x21, 0x23...0x5b, 0x5d...0x7e => try w.writeByte(@intCast(c)),
        else => if (c >= 0x10000) {
            const v = c - 0x10000;
            try w.print("\\u{x:0>4}\\u{x:0>4}", .{ 0xD800 + (v >> 10), 0xDC00 + (v & 0x3ff) });
        } else try w.print("\\u{x:0>4}", .{c}),
    };
    try w.writeByte('"');
}

/// ``float.__repr__``: the shortest round-trip digits, scientific when the exponent is < -4 or >= 16.
pub fn float(w: *Writer, f: f64) Writer.Error!void {
    if (std.math.isNan(f)) return w.writeAll("NaN");
    if (std.math.isInf(f)) return w.writeAll(if (f > 0) "Infinity" else "-Infinity");
    var buf: [64]u8 = undefined;
    const e = std.fmt.bufPrint(&buf, "{e}", .{f}) catch unreachable; // d.ddde[-]x, shortest
    var neg = false;
    var s = e;
    if (s[0] == '-') {
        neg = true;
        s = s[1..];
    }
    const epos = std.mem.indexOfScalar(u8, s, 'e').?;
    const exp = std.fmt.parseInt(i32, s[epos + 1 ..], 10) catch unreachable;
    var digits: [32]u8 = undefined;
    var nd: usize = 0;
    for (s[0..epos]) |c| if (c != '.') {
        digits[nd] = c;
        nd += 1;
    };
    while (nd > 1 and digits[nd - 1] == '0') nd -= 1;
    if (neg) try w.writeByte('-');
    if (exp < -4 or exp >= 16) {
        try w.writeByte(digits[0]);
        if (nd > 1) {
            try w.writeByte('.');
            try w.writeAll(digits[1..nd]);
        }
        try w.print("e{c}{d:0>2}", .{ @as(u8, if (exp < 0) '-' else '+'), @abs(exp) });
    } else if (exp < 0) {
        try w.writeAll("0.");
        for (0..@intCast(-exp - 1)) |_| try w.writeByte('0');
        try w.writeAll(digits[0..nd]);
    } else {
        const ip: usize = @intCast(exp + 1);
        for (0..ip) |i| try w.writeByte(if (i < nd) digits[i] else '0');
        try w.writeByte('.');
        if (nd > ip) try w.writeAll(digits[ip..nd]) else try w.writeByte('0');
    }
}

test "floats and objects as Python's json.dumps" {
    var buf: [256]u8 = undefined;
    var w: Writer = .fixed(&buf);
    for ([_]f64{ 1.0, 0.1, 1e-5, 1e16, 123.456, 0.0001, 2.5e20, -3.0 }) |f| {
        try float(&w, f);
        try w.writeByte(' ');
    }
    try std.testing.expectEqualStrings("1.0 0.1 1e-05 1e+16 123.456 0.0001 2.5e+20 -3.0 ", w.buffered());
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), "{\"b\": [1, null], \"a\": \"\\u00e9x\", \"c\": true}", .{});
    var w2: Writer = .fixed(&buf);
    try dump(&w2, v);
    try std.testing.expectEqualStrings("{\"a\": \"\\u00e9x\", \"b\": [1, null], \"c\": true}", w2.buffered());
}
