//! Pillow 12.3's ``Image.resize(size, BICUBIC)`` of an RGB image (``Resample.c``: double coefficients per output
//! pixel, normalized to 22-bit fixed point, horizontal pass over the rows the vertical pass reads, then vertical,
//! clipped through ``clip8``), and ``ImageOps.pad`` / ``contain`` (banker's rounding, centred paste on a fill).
const std = @import("std");
const pixels = @import("pixels.zig");
const Allocator = std.mem.Allocator;
const Rgb = pixels.Rgb;

const PRECISION_BITS = 32 - 8 - 2;

fn bicubic(x0: f64) f64 {
    const a = -0.5;
    const x = @abs(x0);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

/// One axis's coefficients (``precompute_coeffs`` + ``normalize_coeffs_8bpc``): bounds (xmin, count) and ksize
/// fixed-point weights an output pixel.
const Coeffs = struct {
    ksize: usize,
    bounds: []i32,
    k: []i32,

    fn init(gpa: Allocator, in_size: usize, out_size: usize) Allocator.Error!Coeffs {
        const in0: f32 = 0;
        const in1: f32 = @floatFromInt(in_size);
        const scale: f64 = @as(f64, in1 - in0) / @as(f64, @floatFromInt(out_size));
        const filterscale: f64 = @max(scale, 1.0);
        const support = 2.0 * filterscale;
        const ksize: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
        const bounds = try gpa.alloc(i32, out_size * 2);
        errdefer gpa.free(bounds);
        const k = try gpa.alloc(i32, out_size * ksize);
        errdefer gpa.free(k);
        const pre = try gpa.alloc(f64, ksize);
        defer gpa.free(pre);
        const inv = 1.0 / filterscale;
        for (0..out_size) |xx| {
            const center = @as(f64, in0) + (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
            var ww: f64 = 0.0;
            var xmin: i32 = @intFromFloat(center - support + 0.5);
            if (xmin < 0) xmin = 0;
            var xmax: i32 = @intFromFloat(center + support + 0.5);
            if (xmax > @as(i32, @intCast(in_size))) xmax = @intCast(in_size);
            xmax -= xmin;
            const n: usize = @intCast(@max(xmax, 0));
            for (0..n) |x| {
                const w = bicubic((@as(f64, @floatFromInt(@as(i32, @intCast(x)) + xmin)) - center + 0.5) * inv);
                pre[x] = w;
                ww += w;
            }
            if (ww != 0.0) for (pre[0..n]) |*w| {
                w.* /= ww;
            };
            const row = k[xx * ksize ..][0..ksize];
            for (row, 0..) |*q, x| {
                const v = if (x < n) pre[x] else 0.0;
                const s = v * @as(f64, 1 << PRECISION_BITS);
                q.* = if (v < 0) @intFromFloat(-0.5 + s) else @intFromFloat(0.5 + s);
            }
            bounds[xx * 2] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }
        return .{ .ksize = ksize, .bounds = bounds, .k = k };
    }

    fn deinit(c: *Coeffs, gpa: Allocator) void {
        gpa.free(c.bounds);
        gpa.free(c.k);
    }
};

/// ``clip8``: the sum (with its half added) >> 22, clamped to 0..255 (the lookup's range is -640..639).
fn clip8(ss: i32) u8 {
    return @intCast(std.math.clamp(ss >> PRECISION_BITS, 0, 255));
}

/// ``ImagingResample(im, w, h, BICUBIC, (0, 0, W, H))`` of an RGB image.
pub fn resize(gpa: Allocator, src: Rgb, w: u32, h: u32) Allocator.Error!Rgb {
    const need_h = w != src.w;
    const need_v = h != src.h;
    var vert = try Coeffs.init(gpa, src.h, h);
    defer vert.deinit(gpa);
    const first: usize = @intCast(vert.bounds[0]);
    const last: usize = @intCast(vert.bounds[(h - 1) * 2] + vert.bounds[(h - 1) * 2 + 1]);
    var cur = src;
    var tmp: ?[]u8 = null;
    defer if (tmp) |t| gpa.free(t);
    if (need_h) {
        var hor = try Coeffs.init(gpa, src.w, w);
        defer hor.deinit(gpa);
        const rows = last - first;
        const out = try gpa.alloc(u8, rows * @as(usize, w) * 3);
        tmp = out;
        for (0..rows) |yy| {
            const in = src.data[(yy + first) * @as(usize, src.w) * 3 ..][0 .. @as(usize, src.w) * 3];
            const o = out[yy * @as(usize, w) * 3 ..][0 .. @as(usize, w) * 3];
            for (0..w) |xx| {
                const xmin: usize = @intCast(hor.bounds[xx * 2]);
                const n: usize = @intCast(hor.bounds[xx * 2 + 1]);
                const k = hor.k[xx * hor.ksize ..][0..n];
                var ss = [3]i32{ 1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1) };
                for (k, 0..) |kk, x| {
                    const p = in[(x + xmin) * 3 ..][0..3];
                    inline for (0..3) |c| ss[c] +%= @as(i32, p[c]) * kk;
                }
                inline for (0..3) |c| o[xx * 3 + c] = clip8(ss[c]);
            }
        }
        cur = .{ .w = w, .h = @intCast(rows), .data = out };
    }
    const out = try gpa.alloc(u8, @as(usize, w) * h * 3);
    errdefer gpa.free(out);
    if (!need_v) {
        @memcpy(out, cur.data[0..out.len]);
        return .{ .w = w, .h = h, .data = out };
    }
    const off: usize = if (need_h) first else 0;
    const stride = @as(usize, cur.w) * 3;
    for (0..h) |yy| {
        const ymin: usize = @as(usize, @intCast(vert.bounds[yy * 2])) - off;
        const n: usize = @intCast(vert.bounds[yy * 2 + 1]);
        const k = vert.k[yy * vert.ksize ..][0..n];
        const o = out[yy * stride ..][0..stride];
        for (0..stride) |xc| {
            var ss: i32 = 1 << (PRECISION_BITS - 1);
            for (k, 0..) |kk, y| ss +%= @as(i32, cur.data[(y + ymin) * stride + xc]) * kk;
            o[xc] = clip8(ss);
        }
    }
    return .{ .w = w, .h = h, .data = out };
}

/// Python's ``round`` of a non-negative double to an int (half to even).
pub fn roundHalfEven(x: f64) i64 {
    const f = @floor(x);
    const d = x - f;
    var r: i64 = @intFromFloat(f);
    if (d > 0.5 or (d == 0.5 and @mod(r, 2) == 1)) r += 1;
    return r;
}

/// ``ImageOps.pad(image, (bw, bh), color=(127, 127, 127))``: ``contain`` (aspect kept, BICUBIC), centred.
pub fn pad(gpa: Allocator, src: Rgb, bw: u32, bh: u32, fill: u8) Allocator.Error!Rgb {
    const im_ratio = @as(f64, @floatFromInt(src.w)) / @as(f64, @floatFromInt(src.h));
    const dest_ratio = @as(f64, @floatFromInt(bw)) / @as(f64, @floatFromInt(bh));
    var sw = bw;
    var sh = bh;
    if (im_ratio != dest_ratio) {
        if (im_ratio > dest_ratio) {
            const nh = roundHalfEven(@as(f64, @floatFromInt(src.h)) / @as(f64, @floatFromInt(src.w)) * @as(f64, @floatFromInt(bw)));
            if (nh != bh) sh = @intCast(nh);
        } else {
            const nw = roundHalfEven(@as(f64, @floatFromInt(src.w)) / @as(f64, @floatFromInt(src.h)) * @as(f64, @floatFromInt(bh)));
            if (nw != bw) sw = @intCast(nw);
        }
    }
    var resized: Rgb = if (sw == src.w and sh == src.h) .{ .w = sw, .h = sh, .data = try gpa.dupe(u8, src.data) } else try resize(gpa, src, sw, sh);
    if (sw == bw and sh == bh) return resized;
    defer resized.deinit(gpa);
    const out = try gpa.alloc(u8, @as(usize, bw) * bh * 3);
    @memset(out, fill);
    var x0: usize = 0;
    var y0: usize = 0;
    if (sw != bw) x0 = @intCast(roundHalfEven(@as(f64, @floatFromInt(@as(i64, bw) - sw)) * 0.5)) else y0 = @intCast(roundHalfEven(@as(f64, @floatFromInt(@as(i64, bh) - sh)) * 0.5));
    for (0..sh) |y| @memcpy(out[((y + y0) * bw + x0) * 3 ..][0 .. @as(usize, sw) * 3], resized.data[y * @as(usize, sw) * 3 ..][0 .. @as(usize, sw) * 3]);
    return .{ .w = bw, .h = bh, .data = out };
}

test "round half to even, as Python's round" {
    try std.testing.expectEqual(@as(i64, 2), roundHalfEven(2.5));
    try std.testing.expectEqual(@as(i64, 4), roundHalfEven(3.5));
    try std.testing.expectEqual(@as(i64, 3), roundHalfEven(2.51));
}
