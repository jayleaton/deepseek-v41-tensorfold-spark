//! Decoded images and Pillow's conversions to RGB (``glm5_next/spark/vision_prep.to_rgb``: transparency composited
//! on white with ``Image.alpha_composite``, anything else ``convert("RGB")``), byte for byte as Pillow 12.3 does them.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Pillow's mode of a decoded image. ``bilevel`` ("1") is stored as 0 / 255 bytes, ``i16`` ("I;16") as native u16.
pub const Mode = enum { bilevel, l, la, rgb, rgba, p, i16, cmyk };

pub fn channels(m: Mode) usize {
    return switch (m) {
        .bilevel, .l, .p => 1,
        .la, .i16 => 2,
        .rgb => 3,
        .rgba, .cmyk => 4,
    };
}

/// One decoded frame. ``palette`` is RGBA x 256 (Pillow's: unset entries (0, 0, 0, 255)); ``transparent`` is
/// Pillow's ``"transparency" in info`` for a palette image (alpha from tRNS already in ``palette``).
pub const Image = struct {
    w: u32,
    h: u32,
    mode: Mode,
    data: []u8,
    palette: [1024]u8 = defaultPalette(),
    transparent: bool = false,

    pub fn deinit(im: *Image, gpa: Allocator) void {
        gpa.free(im.data);
        im.* = undefined;
    }

    pub fn pixels(im: *const Image) usize {
        return @as(usize, im.w) * im.h;
    }
};

pub fn defaultPalette() [1024]u8 {
    var p: [1024]u8 = @splat(0);
    for (0..256) |i| p[i * 4 + 3] = 255;
    return p;
}

/// An RGB image, 3 bytes a pixel, rows top first.
pub const Rgb = struct {
    w: u32,
    h: u32,
    data: []u8,

    pub fn deinit(im: *Rgb, gpa: Allocator) void {
        gpa.free(im.data);
        im.* = undefined;
    }
};

/// ``to_rgb``: the image as Pillow's RGB, consuming ``im``'s buffer when it already is RGB.
pub fn toRgb(gpa: Allocator, im: *Image) Allocator.Error!Rgb {
    const n = im.pixels();
    if (im.mode == .rgb) {
        const out: Rgb = .{ .w = im.w, .h = im.h, .data = im.data };
        im.data = &.{};
        return out;
    }
    const out = try gpa.alloc(u8, n * 3);
    errdefer gpa.free(out);
    const src = im.data;
    switch (im.mode) {
        .rgb => unreachable,
        .bilevel, .l => for (0..n) |i| {
            @memset(out[i * 3 ..][0..3], src[i]);
        },
        .i16 => {
            const v: []align(1) const u16 = std.mem.bytesAsSlice(u16, src[0 .. n * 2]);
            for (0..n) |i| @memset(out[i * 3 ..][0..3], if (v[i] < 256) @intCast(v[i]) else 255);
        },
        .p => for (0..n) |i| {
            const e = im.palette[@as(usize, src[i]) * 4 ..][0..4];
            if (im.transparent) composite(out[i * 3 ..][0..3], e[0], e[1], e[2], e[3]) else @memcpy(out[i * 3 ..][0..3], e[0..3]);
        },
        .la => for (0..n) |i| composite(out[i * 3 ..][0..3], src[i * 2], src[i * 2], src[i * 2], src[i * 2 + 1]),
        .rgba => for (0..n) |i| {
            const s = src[i * 4 ..][0..4];
            composite(out[i * 3 ..][0..3], s[0], s[1], s[2], s[3]);
        },
        .cmyk => for (0..n) |i| {
            const s = src[i * 4 ..][0..4];
            const nk: i32 = 255 - @as(i32, s[3]);
            for (0..3) |c| out[i * 3 + c] = clip8(nk - mulDiv255(s[c], nk));
        },
    }
    return .{ .w = im.w, .h = im.h, .data = out };
}

/// Pillow's ``MULDIV255``: round(a * b / 255) the integer way.
fn mulDiv255(a: u8, b: i32) i32 {
    const tmp = @as(i32, a) * b + 128;
    return (tmp + (tmp >> 8)) >> 8;
}

fn clip8(v: i32) u8 {
    return @intCast(std.math.clamp(v, 0, 255));
}

/// ``ImagingAlphaComposite`` of (r, g, b, a) over opaque white, then ``convert("RGB")`` (alpha dropped).
fn composite(out: *[3]u8, r: u8, g: u8, b: u8, a: u8) void {
    if (a == 0) {
        out.* = .{ 255, 255, 255 };
        return;
    }
    const prec = 7;
    const blend: u32 = 255 * (255 - @as(u32, a));
    const outa255: u32 = @as(u32, a) * 255 + blend;
    const coef1: u32 = @as(u32, a) * 255 * 255 * (1 << prec) / outa255;
    const coef2: u32 = 255 * (1 << prec) - coef1;
    const src = [3]u32{ r, g, b };
    for (0..3) |c| {
        const t = src[c] * coef1 + 255 * coef2 + (0x80 << prec);
        out[c] = @intCast((((t >> 8) + t) >> 8) >> prec);
    }
}

test "alpha over white and the palette defaults read as Pillow's" {
    var o: [3]u8 = undefined;
    composite(&o, 0, 0, 0, 255);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0 }, &o);
    composite(&o, 10, 20, 30, 0);
    try std.testing.expectEqualSlices(u8, &.{ 255, 255, 255 }, &o);
    composite(&o, 0, 0, 0, 128);
    try std.testing.expectEqual(@as(u8, 127), o[0]); // Image.alpha_composite((255,)*4, (0,0,0,128)) -> 127
    try std.testing.expectEqual(@as(u8, 255), defaultPalette()[3]);
}
