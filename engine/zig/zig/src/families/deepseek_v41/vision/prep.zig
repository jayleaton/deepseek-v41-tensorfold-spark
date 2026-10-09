//! DeepSeek-V4.1's image processor (``vision_prep.py``: ``Settings``, ``plan_grid``, ``preprocess``, the span) on
//! an RGB image: ``ImageOps.pad`` to the planned size (grey 127), ``(x / 255 - 0.5) / 0.5`` in float32 then bf16,
//! 14 x 14 patches in reading order, and the digest every virtual id derives from.
const std = @import("std");
const pixels = @import("pixels.zig");
const resample = @import("resample.zig");
const pyjson = @import("pyjson.zig");
const Allocator = std.mem.Allocator;

pub const VERSION = "tensorfold-dsv41-vision/1";
pub const IMAGE_TOKEN: u32 = 129264;

pub const Type = enum(i8) { text = -1, start = 0, image = 1, newline = 2, end = 3 };

pub const Error = error{ NoGrid, BadSettings } || Allocator.Error;

extern "c" fn pow(x: f64, y: f64) f64; // Python's float ** (libm), not a Zig reimplementation

/// ``config.json``'s ``vision_config`` / ``image_token_id`` and the server's limits (``TF_DSV41_VISION_*``).
pub const Settings = struct {
    patch: u32 = 14,
    downsample: u32 = 3,
    max_tokens: u32 = 1024,
    min_pixels: u64 = 544 * 544,
    max_wh_ratio: ?f64 = null,
    image_token: u32 = IMAGE_TOKEN,
    vision_config: std.json.Value = .{ .object = .empty },
    bias_vl: bool = false,
    max_images: u32 = 8,
    cap_tokens: u32 = 0,
    fetch: bool = true,
    fetch_timeout_s: f64 = 10,
    fetch_total_s: f64 = 30,
    max_bytes: usize = 20 * 1024 * 1024,
    max_pixels: u64 = 64 * 1024 * 1024,
    fetch_http: bool = false,
    fetch_private: bool = false,
    prep_slots: u32 = 8,
    prep_mb: u64 = 256,

    /// ``Settings.read``: ``config`` is config.json's text (null: the defaults), ``env`` the process environment.
    pub fn read(arena: Allocator, config: ?[]const u8, env: ?*const std.process.Environ.Map, problem: *[]const u8) Error!Settings {
        var s: Settings = .{};
        if (config) |text| {
            const root = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch return bad(problem, "config.json is not JSON");
            if (root == .object) {
                if (root.object.get("vision_config")) |vc| if (vc == .object) {
                    s.vision_config = vc;
                    s.patch = intOf(vc.object.get("patch_size"), s.patch);
                    s.downsample = intOf(vc.object.get("downsample_ratio"), s.downsample);
                    s.max_tokens = intOf(vc.object.get("max_image_tokens"), s.max_tokens);
                    s.min_pixels = intOf(vc.object.get("min_pixels"), s.min_pixels);
                    if (vc.object.get("max_wh_ratio")) |r| s.max_wh_ratio = switch (r) {
                        .integer => |i| @floatFromInt(i),
                        .float => |f| f,
                        else => null,
                    };
                };
                s.image_token = intOf(root.object.get("image_token_id"), s.image_token);
            }
        }
        const m = env orelse return s;
        s.bias_vl = std.mem.trim(u8, m.get("TF_DSV41_BIAS_VL") orelse "", " \t\r\n").len > 0;
        s.max_images = try envInt(m, "TF_DSV41_VISION_MAX_IMAGES", s.max_images, problem);
        s.cap_tokens = try envInt(m, "TF_DSV41_VISION_MAX_TOKENS", @as(u32, 0), problem);
        s.fetch = try envFlag(m, "TF_DSV41_VISION_FETCH", true, problem);
        s.fetch_timeout_s = try envFloat(m, "TF_DSV41_VISION_FETCH_TIMEOUT", 10, problem);
        s.fetch_total_s = try envFloat(m, "TF_DSV41_VISION_FETCH_TOTAL_S", 30, problem);
        s.max_bytes = try envInt(m, "TF_DSV41_VISION_MAX_BYTES", s.max_bytes, problem);
        s.max_pixels = try envInt(m, "TF_DSV41_VISION_MAX_PIXELS", s.max_pixels, problem);
        s.fetch_http = try envFlag(m, "TF_DSV41_VISION_FETCH_HTTP", false, problem);
        s.fetch_private = try envFlag(m, "TF_DSV41_VISION_FETCH_PRIVATE", false, problem);
        s.prep_slots = try envInt(m, "TF_DSV41_VISION_PREP_SLOTS", s.prep_slots, problem);
        s.prep_mb = try envInt(m, "TF_DSV41_VISION_PREP_MB", s.prep_mb, problem);
        if (s.max_images < 1 or s.prep_slots < 1) return bad(problem, "TF_DSV41_VISION_MAX_IMAGES >= 1, _MAX_TOKENS >= 0, _PREP_SLOTS >= 1");
        if (s.cap_tokens != 0 and s.cap_tokens < 5) return bad(problem, "TF_DSV41_VISION_MAX_TOKENS: at least 5 (one row of one cell)");
        return s;
    }

    pub fn budget(s: *const Settings) u32 {
        return if (s.cap_tokens != 0) @min(s.max_tokens, s.cap_tokens) else s.max_tokens;
    }

    /// ``Settings.key``: what an image's rows depend on besides its patches.
    pub fn key(s: *const Settings, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.writeAll("{\"bias_vl\": ");
        try w.writeAll(if (s.bias_vl) "true" else "false");
        try w.print(", \"downsample\": {d}, \"max_wh_ratio\": ", .{s.downsample});
        if (s.max_wh_ratio) |r| try pyjson.float(w, r) else try w.writeAll("null");
        try w.print(", \"min_pixels\": {d}, \"patch\": {d}, \"vision\": ", .{ s.min_pixels, s.patch });
        try pyjson.dump(w, s.vision_config);
        try w.writeByte('}');
    }
};

fn bad(problem: *[]const u8, msg: []const u8) error{BadSettings} {
    problem.* = msg;
    return error.BadSettings;
}

fn intOf(v: ?std.json.Value, default: anytype) @TypeOf(default) {
    const x = v orelse return default;
    return switch (x) {
        .integer => |i| std.math.cast(@TypeOf(default), i) orelse default,
        .float => |f| if (f >= 0 and f < 1e15) @intFromFloat(f) else default,
        else => default,
    };
}

/// ``_env_int``: int(float(raw)).
fn envInt(m: *const std.process.Environ.Map, name: []const u8, default: anytype, problem: *[]const u8) error{BadSettings}!@TypeOf(default) {
    const raw = std.mem.trim(u8, m.get(name) orelse "", " \t\r\n");
    if (raw.len == 0) return default;
    const f = std.fmt.parseFloat(f64, raw) catch return bad(problem, name);
    if (!(f >= 0) or f > 1e15) return bad(problem, name);
    return std.math.cast(@TypeOf(default), @as(u64, @intFromFloat(@trunc(f)))) orelse bad(problem, name);
}

fn envFloat(m: *const std.process.Environ.Map, name: []const u8, default: f64, problem: *[]const u8) error{BadSettings}!f64 {
    const raw = std.mem.trim(u8, m.get(name) orelse "", " \t\r\n");
    if (raw.len == 0) return default;
    return std.fmt.parseFloat(f64, raw) catch bad(problem, name);
}

fn envFlag(m: *const std.process.Environ.Map, name: []const u8, default: bool, problem: *[]const u8) error{BadSettings}!bool {
    const raw = std.mem.trim(u8, m.get(name) orelse "", " \t\r\n");
    if (raw.len == 0) return default;
    if (std.mem.eql(u8, raw, "1")) return true;
    if (std.mem.eql(u8, raw, "0")) return false;
    return bad(problem, name);
}

pub fn spanLen(h: u32, w: u32) u32 {
    return h * (w + 1) + 2;
}

/// ``llm_grid``: the aligner's grid of a pixel size.
fn llmGrid(height: f64, width: f64, patch: u32, ratio: u32) [2]u32 {
    const p: f64 = @floatFromInt(patch);
    const r: f64 = @floatFromInt(ratio);
    return .{ @intFromFloat(@ceil(@floor(height / p) / r)), @intFromFloat(@ceil(@floor(width / p) / r)) };
}

fn solve(height: f64, width: f64, patch: u32, ratio: u32, most: u32) [2]f64 {
    const r = height / width;
    const mostf: f64 = @floatFromInt(most);
    const max_w = @sqrt((mostf - 2) / r + 0.25) - 0.5;
    const max_h = max_w * r;
    const cell: f64 = @floatFromInt(patch * ratio);
    const p: f64 = @floatFromInt(patch);
    if (max_w < 1.0) return .{ @as(f64, @floatFromInt((most - 2) / 2)) * cell, cell };
    if (max_h < 1.0) return .{ cell, @as(f64, @floatFromInt(most - 3)) * cell };
    const beta = @min(@floor(max_w) * cell / width, @floor(max_h) * cell / height);
    return .{ @floor(height * beta / p) * p, @floor(width * beta / p) * p };
}

/// ``plan_grid``: (n_llm_h, n_llm_w, pixel height, pixel width) for an image of this size.
pub fn plan(width0: u32, height0: u32, s: *const Settings) Error![4]u32 {
    const most = s.budget();
    var width: f64 = @floatFromInt(width0);
    var height: f64 = @floatFromInt(height0);
    const p: f64 = @floatFromInt(s.patch);
    if (s.max_wh_ratio) |ratio| if (width > height * ratio) {
        width = height * ratio;
    };
    const area = width * height;
    const minp: f64 = @floatFromInt(s.min_pixels);
    if (area > 0 and area < minp) {
        const k = pow(minp / area, 0.5);
        width = @trunc(width * k);
        height = @trunc(height * k);
    }
    var bw = @ceil(width / p) * p;
    var bh = @ceil(height / p) * p;
    var g = llmGrid(bh, bw, s.patch, s.downsample);
    if (spanLen(g[0], g[1]) > most) {
        const hw = solve(height, width, s.patch, s.downsample, most);
        bh = hw[0];
        bw = hw[1];
        g = llmGrid(bh, bw, s.patch, s.downsample);
        if (spanLen(g[0], g[1]) > most) return error.NoGrid;
    }
    return .{ g[0], g[1], @intFromFloat(bh), @intFromFloat(bw) };
}

/// One preprocessed image: patches bf16 [vh vw, 3, 14, 14], the grids, its digest and virtual ids.
pub const Prepared = struct {
    patches: []u16,
    vit_h: u32,
    vit_w: u32,
    llm_h: u32,
    llm_w: u32,
    digest: [32]u8,
    size: [2]u32, // the original (width, height)
    vids: []u32 = &.{},

    pub fn tokens(p: *const Prepared) u32 {
        return spanLen(p.llm_h, p.llm_w);
    }

    /// Span position ``i``'s type (``span_types``).
    pub fn typeAt(p: *const Prepared, i: u32) Type {
        if (i == 0) return .start;
        if (i == p.tokens() - 1) return .end;
        return if ((i - 1) % (p.llm_w + 1) == p.llm_w) .newline else .image;
    }

    pub fn deinit(p: *Prepared, gpa: Allocator) void {
        gpa.free(p.patches);
        gpa.free(p.vids);
    }
};

/// ``(x / 255 - 0.5) / 0.5`` in float32, to bf16 round-to-nearest-even, for every byte value.
pub const norm_table: [256]u16 = blk: {
    var t: [256]u16 = undefined;
    for (0..256) |i| {
        const x: f32 = @as(f32, @floatFromInt(i)) / 255.0;
        t[i] = bf16((x - 0.5) / 0.5);
    }
    break :blk t;
};

pub fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// ``preprocess``: an RGB image -> its patches and digest (the virtual ids come from the registry).
pub fn preprocess(gpa: Allocator, image: pixels.Rgb, s: *const Settings) Error!Prepared {
    const g = try plan(image.w, image.h, s);
    const bh = g[2];
    const bw = g[3];
    const p = s.patch;
    const vh = bh / p;
    const vw = bw / p;
    var canvas = if (s.max_wh_ratio != null and @as(f64, @floatFromInt(image.w)) >= s.max_wh_ratio.? * @as(f64, @floatFromInt(image.h)))
        try resample.resize(gpa, image, bw, bh)
    else
        try resample.pad(gpa, image, bw, bh, 127);
    defer canvas.deinit(gpa);
    const patches = try gpa.alloc(u16, @as(usize, vh) * vw * 3 * p * p);
    errdefer gpa.free(patches);
    // [vh, vw, 3, p, p] from the [h, w, 3] canvas: one patch row (p bytes) at a time per channel
    const pp = @as(usize, p) * p;
    for (0..vh) |py| for (0..vw) |px| {
        const base = (py * vw + px) * 3 * pp;
        for (0..p) |r| {
            const src = canvas.data[((py * p + r) * @as(usize, bw) + px * p) * 3 ..][0 .. @as(usize, p) * 3];
            for (0..p) |c| inline for (0..3) |ch| {
                patches[base + ch * pp + r * p + c] = norm_table[src[c * 3 + ch]];
            };
        }
    };
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(VERSION);
    var kbuf: [4096]u8 = undefined;
    var kw: std.Io.Writer = .fixed(&kbuf);
    s.key(&kw) catch return error.BadSettings;
    h.update(kw.buffered());
    var gbuf: [64]u8 = undefined;
    h.update(std.fmt.bufPrint(&gbuf, "[{d}, {d}, {d}, {d}]", .{ vh, vw, g[0], g[1] }) catch unreachable);
    h.update(std.mem.sliceAsBytes(patches)); // little-endian int16 view, as torch's numpy bytes
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return .{ .patches = patches, .vit_h = vh, .vit_w = vw, .llm_h = g[0], .llm_w = g[1], .digest = digest, .size = .{ image.w, image.h } };
}

test "span types and the planned grid of a small image" {
    const s: Settings = .{};
    const g = try plan(100, 50, &s); // scaled up to 544^2 area: 769 x 384 -> 770 x 392 -> grid 10 x 19 (h 392/14=28)
    try std.testing.expect(spanLen(g[0], g[1]) <= 1024);
    var pr: Prepared = .{ .patches = &.{}, .vit_h = 0, .vit_w = 0, .llm_h = 2, .llm_w = 3, .digest = undefined, .size = .{ 0, 0 } };
    const want = [_]Type{ .start, .image, .image, .image, .newline, .image, .image, .image, .newline, .end };
    try std.testing.expectEqual(@as(u32, 10), pr.tokens());
    for (want, 0..) |t, i| try std.testing.expectEqual(t, pr.typeAt(@intCast(i)));
    try std.testing.expectEqual(@as(u16, 0xbf80), norm_table[0]); // -1.0
    try std.testing.expectEqual(@as(u16, 0x3f80), norm_table[255]); // 1.0
}
