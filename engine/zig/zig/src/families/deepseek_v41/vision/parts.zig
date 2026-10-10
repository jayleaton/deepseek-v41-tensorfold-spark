//! A request's image parts (``vision_prep.part_url`` / ``Marker`` / ``expand`` and GLM's ``load_bytes`` for data:
//! URLs): each part renders as a per-request sentinel, the rendered text's sentinels become ``<｜deepseek_image｜>``
//! with the URLs in prompt order, and each placeholder id becomes its image's span of virtual ids.
const std = @import("std");
const json = @import("json");
const prep = @import("prep.zig");
const vids = @import("vids.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const IMAGE_PLACEHOLDER = "<｜deepseek_image｜>";

/// A request's problem with its images: HTTP 400, or 503 for ``busy``.
pub const Problem = struct { message: []const u8 = "", busy: bool = false };

/// ``part_url``: OpenAI ``image_url`` (object or string), ``input_image`` / ``image``, or Messages API ``source``.
pub fn partUrl(a: Allocator, part: Value) error{ Refused, OutOfMemory }![]const u8 {
    const kind = part.strField("type") orelse "";
    if (part.get("source")) |src| if (src == .object) {
        const data = src.get("data");
        const is_b64 = if (src.strField("type")) |t| std.mem.eql(u8, t, "base64") else false;
        if (is_b64 or (data != null and data.? != .null)) {
            const media = if (src.get("media_type")) |m| (if (m.truthy()) (m.str() orelse "image/png") else "image/png") else "image/png";
            const d = if (data) |x| (if (x.truthy()) (x.str() orelse "") else "") else "";
            return std.fmt.allocPrint(a, "data:{s};base64,{s}", .{ media, d });
        }
        if (src.get("url")) |u| if (u.truthy()) if (u.str()) |s| return s;
    };
    var val: ?Value = null;
    if (std.mem.eql(u8, kind, "image_url")) val = part.get("image_url") else {
        for ([_][]const u8{ "image_url", "url", "image" }) |k| if (part.get(k)) |x| if (x.truthy()) {
            val = x;
            break;
        };
    }
    if (val) |v| if (v == .object) {
        val = v.get("url");
    };
    if (val) |v| if (v == .string and v.string.len > 0) return v.string;
    return error.Refused;
}

/// The image parts as sentinels (``Marker``): ``hook`` is ``template.ImageHook.text``; ``splice`` after rendering.
pub const Marker = struct {
    a: Allocator,
    nonce: [12]u8,
    urls: std.ArrayList([]const u8) = .empty,
    problem: ?[]const u8 = null,

    pub fn init(a: Allocator, io: std.Io) Marker {
        var raw: [6]u8 = undefined;
        io.random(&raw);
        var m: Marker = .{ .a = a, .nonce = undefined };
        _ = std.fmt.bufPrint(&m.nonce, "{x}", .{raw}) catch unreachable;
        return m;
    }

    fn tag(m: *const Marker, buf: []u8, i: usize) []const u8 {
        return std.fmt.bufPrint(buf, "\x00tf-image-{s}-{d}\x00", .{ m.nonce, i }) catch unreachable;
    }

    /// One image part -> its sentinel (a malformed part is remembered and refused after the render).
    pub fn hook(ctx: ?*anyopaque, part: Value) []const u8 {
        const m: *Marker = @ptrCast(@alignCast(ctx.?));
        const url = partUrl(m.a, part) catch |e| {
            if (m.problem == null) m.problem = if (e == error.OutOfMemory) "out of memory" else std.fmt.allocPrint(m.a, "an {s} part needs a URL (image_url.url)", .{part.strField("type") orelse "image"}) catch "an image part needs a URL";
            return "";
        };
        m.urls.append(m.a, url) catch {
            m.problem = "out of memory";
            return "";
        };
        var buf: [64]u8 = undefined;
        return m.a.dupe(u8, m.tag(&buf, m.urls.items.len - 1)) catch "";
    }

    /// The rendered text with each sentinel as the placeholder, and the URLs in prompt order.
    pub fn splice(m: *Marker, text: []const u8, problem: *Problem) error{ Refused, OutOfMemory }!struct { text: []u8, urls: [][]const u8 } {
        if (m.problem) |p| return refuse(problem, p);
        if (std.mem.indexOf(u8, text, IMAGE_PLACEHOLDER) != null) return refuse(problem, "the messages hold the image placeholder " ++ IMAGE_PLACEHOLDER ++ " as text: send images as image parts");
        const n = m.urls.items.len;
        const At = struct { at: usize, i: usize };
        const order = try m.a.alloc(At, n);
        var buf: [64]u8 = undefined;
        for (0..n) |i| order[i] = .{ .at = std.mem.indexOf(u8, text, m.tag(&buf, i)) orelse return refuse(problem, "an image part was dropped by the prompt encoding"), .i = i };
        std.mem.sort(At, order, {}, struct {
            fn lt(_: void, x: At, y: At) bool {
                return x.at < y.at;
            }
        }.lt);
        var out: std.ArrayList(u8) = .empty;
        var rest = text;
        while (std.mem.indexOf(u8, rest, "\x00tf-image-" ++ "")) |at| {
            const end = std.mem.indexOfScalarPos(u8, rest, at + 1, 0) orelse return refuse(problem, "an image sentinel survived the prompt encoding");
            const t = rest[at .. end + 1];
            const known = for (0..n) |i| {
                if (std.mem.eql(u8, t, m.tag(&buf, i))) break true;
            } else false;
            if (!known) return refuse(problem, "an image sentinel survived the prompt encoding");
            try out.appendSlice(m.a, rest[0..at]);
            try out.appendSlice(m.a, IMAGE_PLACEHOLDER);
            rest = rest[end + 1 ..];
        }
        try out.appendSlice(m.a, rest);
        const urls = try m.a.alloc([]const u8, n);
        for (order, urls) |o, *u| u.* = m.urls.items[o.i];
        return .{ .text = out.items, .urls = urls };
    }
};

fn refuse(problem: *Problem, msg: []const u8) error{Refused} {
    problem.* = .{ .message = msg };
    return error.Refused;
}

/// ``expand``: each image token id -> its image's virtual ids (``virtual``) or the image token at every position.
pub fn expand(a: Allocator, ids: []const u32, images: []const prep.Prepared, image_token: u32, virtual: bool, problem: *Problem) error{ Refused, OutOfMemory }![]u32 {
    var out: std.ArrayList(u32) = .empty;
    var k: usize = 0;
    for (ids) |t| {
        if (t != image_token) {
            if (t >= vids.VBASE) return refuse(problem, "a prompt id past the vocabulary");
            try out.append(a, t);
            continue;
        }
        if (k >= images.len) return refuse(problem, "the prompt holds an image placeholder that is not an image part");
        const img = &images[k];
        if (virtual) try out.appendSlice(a, img.vids) else try out.appendNTimes(a, image_token, img.tokens());
        k += 1;
    }
    if (k != images.len) return refuse(problem, try std.fmt.allocPrint(a, "{d} image(s) sent, {d} placed in the prompt", .{ images.len, k }));
    return out.items;
}

/// ``load_bytes`` for a ``data:`` URL: base64 (``b64decode(validate=False)``) or percent-encoded.
pub fn dataUrl(a: Allocator, url: []const u8, max_bytes: usize, problem: *Problem) error{ Refused, OutOfMemory }![]u8 {
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse return refuse(problem, "a data: URL without a comma");
    const head = url[0..comma];
    const data = url[comma + 1 ..];
    const too_big = try std.fmt.allocPrint(a, "image larger than {d} bytes (TF_DSV41_VISION_MAX_BYTES)", .{max_bytes});
    var raw: []u8 = undefined;
    if (std.mem.indexOf(u8, head, ";base64") == null) {
        raw = try unquote(a, data);
    } else {
        if (data.len > (max_bytes * 4) / 3 + 8) return refuse(problem, too_big);
        raw = b64decode(a, std.mem.trim(u8, data, " \t\n\r\x0b\x0c")) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(problem, "the data: URL is not valid base64"),
        };
    }
    if (raw.len > max_bytes) return refuse(problem, too_big);
    return raw;
}

fn unquote(a: Allocator, s: []const u8) Allocator.Error![]u8 {
    var out = try a.alloc(u8, s.len);
    var n: usize = 0;
    var i: usize = 0;
    while (i < s.len) : (n += 1) {
        if (s[i] == '%' and i + 2 < s.len) {
            if (std.fmt.parseInt(u8, s[i + 1 .. i + 3], 16)) |v| {
                out[n] = v;
                i += 3;
                continue;
            } else |_| {}
        }
        out[n] = s[i];
        i += 1;
    }
    return out[0..n];
}

/// CPython's ``binascii.a2b_base64`` in non-strict mode: characters outside the alphabet skipped, decoding ends at a
/// completing pad, a dangling quad refused.
pub fn b64decode(a: Allocator, s: []const u8) error{ OutOfMemory, Invalid }![]u8 {
    const out = try a.alloc(u8, s.len / 4 * 3 + 3);
    var n: usize = 0;
    var quad: u3 = 0;
    var pads: u3 = 0;
    var left: u32 = 0;
    for (s) |c| {
        if (c == '=') {
            if (quad >= 2) {
                pads += 1;
                if (@as(u4, quad) + pads >= 4) return out[0..n];
            }
            continue;
        }
        const v: u32 = switch (c) {
            'A'...'Z' => c - 'A',
            'a'...'z' => c - 'a' + 26,
            '0'...'9' => c - '0' + 52,
            '+' => 62,
            '/' => 63,
            else => continue,
        };
        switch (quad) {
            0 => {
                left = v;
                quad = 1;
            },
            1 => {
                out[n] = @intCast((left << 2) | (v >> 4));
                n += 1;
                left = v & 0x0f;
                quad = 2;
            },
            2 => {
                out[n] = @intCast((left << 4) | (v >> 2));
                n += 1;
                left = v & 0x03;
                quad = 3;
            },
            else => {
                out[n] = @intCast((left << 6) | v);
                n += 1;
                quad = 0;
            },
        }
    }
    if (quad != 0) return error.Invalid;
    return out[0..n];
}

test "base64 as binascii and the image parts' URLs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("hello", try b64decode(a, "aGVs\nbG8="));
    try std.testing.expectEqualStrings("hi", try b64decode(a, "aGk=junk"));
    try std.testing.expectError(error.Invalid, b64decode(a, "aGVsb"));
    try std.testing.expectEqualStrings("a b", try unquote(a, "a%20b"));
    const part = (try json.parseText(a, "{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,AA==\"}}")).ok;
    try std.testing.expectEqualStrings("data:image/png;base64,AA==", try partUrl(a, part));
    const anth = (try json.parseText(a, "{\"type\": \"image\", \"source\": {\"type\": \"base64\", \"data\": \"QQ==\"}}")).ok;
    try std.testing.expectEqualStrings("data:image/png;base64,QQ==", try partUrl(a, anth));
    try std.testing.expectError(error.Refused, partUrl(a, (try json.parseText(a, "{\"type\": \"image_url\"}")).ok));
}
