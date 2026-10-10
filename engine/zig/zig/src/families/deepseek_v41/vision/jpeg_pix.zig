//! libjpeg-turbo 3.1's sample side as Pillow decodes with it (JDCT_ISLOW, fancy upsampling, no merged upsampling):
//! ``jidctint.c`` jpeg_idct_islow, ``jdsample.c`` h2v1 / h1v2 / h2v2 fancy and integral upsampling with the main
//! controller's context rows (edge rows repeated), ``jdcolor.c``'s YCbCr / YCCK tables.
const std = @import("std");

const CONST_BITS = 13;
const PASS1_BITS = 2;
const F0_298631336 = 2446;
const F0_390180644 = 3196;
const F0_541196100 = 4433;
const F0_765366865 = 6270;
const F0_899976223 = 7373;
const F1_175875602 = 9633;
const F1_501321110 = 12299;
const F1_847759065 = 15137;
const F1_961570560 = 16069;
const F2_053119869 = 16819;
const F2_562915447 = 20995;
const F3_072711026 = 25172;

fn descale(x: i64, comptime n: u6) i64 {
    return (x + (@as(i64, 1) << (n - 1))) >> n;
}

/// The post-IDCT range limit (``IDCT_range_limit`` [x & RANGE_MASK]): the 10-bit wrapped value + 128, clamped.
fn limit(x: i64) u8 {
    const v: i64 = @as(i64, @as(i10, @truncate(x))) + 128;
    return @intCast(std.math.clamp(v, 0, 255));
}

/// One 8x8 block: coefficients in natural order, the quantization table, 8 output rows of ``stride``.
pub fn idctIslow(coef: *const [64]i16, quant: *const [64]u16, out: []u8, stride: usize) void {
    var ws: [64]i64 = undefined;
    for (0..8) |c| {
        const in = struct {
            fn at(cf: *const [64]i16, q: *const [64]u16, col: usize, r: usize) i64 {
                return @as(i64, cf[r * 8 + col]) * q[r * 8 + col];
            }
        }.at;
        if (coef[8 + c] == 0 and coef[16 + c] == 0 and coef[24 + c] == 0 and coef[32 + c] == 0 and coef[40 + c] == 0 and coef[48 + c] == 0 and coef[56 + c] == 0) {
            const dc = in(coef, quant, c, 0) << PASS1_BITS;
            for (0..8) |r| ws[r * 8 + c] = dc;
            continue;
        }
        var z2 = in(coef, quant, c, 2);
        var z3 = in(coef, quant, c, 6);
        const z1 = (z2 + z3) * F0_541196100;
        var tmp2 = z1 + z3 * -F1_847759065;
        var tmp3 = z1 + z2 * F0_765366865;
        z2 = in(coef, quant, c, 0);
        z3 = in(coef, quant, c, 4);
        var tmp0 = (z2 + z3) << CONST_BITS;
        var tmp1 = (z2 - z3) << CONST_BITS;
        const tmp10 = tmp0 + tmp3;
        const tmp13 = tmp0 - tmp3;
        const tmp11 = tmp1 + tmp2;
        const tmp12 = tmp1 - tmp2;
        tmp0 = in(coef, quant, c, 7);
        tmp1 = in(coef, quant, c, 5);
        tmp2 = in(coef, quant, c, 3);
        tmp3 = in(coef, quant, c, 1);
        odd(&tmp0, &tmp1, &tmp2, &tmp3);
        const n = CONST_BITS - PASS1_BITS;
        ws[0 * 8 + c] = descale(tmp10 + tmp3, n);
        ws[7 * 8 + c] = descale(tmp10 - tmp3, n);
        ws[1 * 8 + c] = descale(tmp11 + tmp2, n);
        ws[6 * 8 + c] = descale(tmp11 - tmp2, n);
        ws[2 * 8 + c] = descale(tmp12 + tmp1, n);
        ws[5 * 8 + c] = descale(tmp12 - tmp1, n);
        ws[3 * 8 + c] = descale(tmp13 + tmp0, n);
        ws[4 * 8 + c] = descale(tmp13 - tmp0, n);
    }
    for (0..8) |r| {
        const w = ws[r * 8 ..][0..8];
        const o = out[r * stride ..][0..8];
        const n = CONST_BITS + PASS1_BITS + 3;
        if (w[1] == 0 and w[2] == 0 and w[3] == 0 and w[4] == 0 and w[5] == 0 and w[6] == 0 and w[7] == 0) {
            @memset(o, limit(descale(w[0], PASS1_BITS + 3)));
            continue;
        }
        const z1 = (w[2] + w[6]) * F0_541196100;
        var tmp2 = z1 + w[6] * -F1_847759065;
        var tmp3 = z1 + w[2] * F0_765366865;
        var tmp0 = (w[0] + w[4]) << CONST_BITS;
        var tmp1 = (w[0] - w[4]) << CONST_BITS;
        const tmp10 = tmp0 + tmp3;
        const tmp13 = tmp0 - tmp3;
        const tmp11 = tmp1 + tmp2;
        const tmp12 = tmp1 - tmp2;
        tmp0 = w[7];
        tmp1 = w[5];
        tmp2 = w[3];
        tmp3 = w[1];
        odd(&tmp0, &tmp1, &tmp2, &tmp3);
        o[0] = limit(descale(tmp10 + tmp3, n));
        o[7] = limit(descale(tmp10 - tmp3, n));
        o[1] = limit(descale(tmp11 + tmp2, n));
        o[6] = limit(descale(tmp11 - tmp2, n));
        o[2] = limit(descale(tmp12 + tmp1, n));
        o[5] = limit(descale(tmp12 - tmp1, n));
        o[3] = limit(descale(tmp13 + tmp0, n));
        o[4] = limit(descale(tmp13 - tmp0, n));
    }
}

/// The odd part (figure 8), shared by both passes: in y7, y5, y3, y1, out the four terms.
fn odd(t0: *i64, t1: *i64, t2: *i64, t3: *i64) void {
    const z1 = t0.* + t3.*;
    const z2 = t1.* + t2.*;
    var z3 = t0.* + t2.*;
    var z4 = t1.* + t3.*;
    const z5 = (z3 + z4) * F1_175875602;
    const a = t0.* * F0_298631336;
    const b = t1.* * F2_053119869;
    const c = t2.* * F3_072711026;
    const d = t3.* * F1_501321110;
    const m1 = z1 * -F0_899976223;
    const m2 = z2 * -F2_562915447;
    z3 = z3 * -F1_961570560 + z5;
    z4 = z4 * -F0_390180644 + z5;
    t0.* = a + m1 + z3;
    t1.* = b + m2 + z4;
    t2.* = c + m2 + z3;
    t3.* = d + m1 + z4;
}

/// A component plane after the IDCT: ``w`` x ``h`` real samples (``downsampled_width`` / ``_height``) in rows of
/// ``stride``.
pub const Plane = struct { data: []const u8, stride: usize, w: usize, h: usize };

/// One output row ``y`` of a component upsampled by (hx, vy) into ``out`` (at least w * hx + 1 bytes), libjpeg's
/// method for that factor pair. ``fancy``: ``do_fancy`` and the component wider than 2 (2h methods only).
pub fn upsampleRow(p: Plane, hx: usize, vy: usize, fancy: bool, y: usize, out: []u8) void {
    const r = y / vy;
    const row = p.data[@min(r, p.h - 1) * p.stride ..][0..p.w];
    if (hx == 1 and vy == 1) return @memcpy(out[0..p.w], row);
    if (vy == 2 and fancy and (hx == 1 or (hx == 2 and p.w > 2))) {
        const below = (y % 2) == 1; // v == 1: next nearest is the row below
        const nr = if (below) @min(r + 1, p.h - 1) else (if (r == 0) 0 else r - 1);
        const near = p.data[@min(r, p.h - 1) * p.stride ..][0..p.w];
        const far = p.data[nr * p.stride ..][0..p.w];
        if (hx == 1) {
            const bias: i32 = if (below) 2 else 1;
            for (0..p.w) |x| out[x] = @intCast((@as(i32, near[x]) * 3 + far[x] + bias) >> 2);
            return;
        }
        var this: i32 = @as(i32, near[0]) * 3 + far[0];
        var next: i32 = @as(i32, near[1]) * 3 + far[1];
        out[0] = @intCast((this * 4 + 8) >> 4);
        out[1] = @intCast((this * 3 + next + 7) >> 4);
        var last = this;
        this = next;
        for (2..p.w) |x| {
            next = @as(i32, near[x]) * 3 + far[x];
            out[2 * x - 2] = @intCast((this * 3 + last + 8) >> 4);
            out[2 * x - 1] = @intCast((this * 3 + next + 7) >> 4);
            last = this;
            this = next;
        }
        out[2 * p.w - 2] = @intCast((this * 3 + last + 8) >> 4);
        out[2 * p.w - 1] = @intCast((this * 4 + 7) >> 4);
        return;
    }
    if (hx == 2 and vy == 1 and fancy and p.w > 2) {
        out[0] = row[0];
        out[1] = @intCast((@as(i32, row[0]) * 3 + row[1] + 2) >> 2);
        for (1..p.w - 1) |x| {
            const v = @as(i32, row[x]) * 3;
            out[2 * x] = @intCast((v + row[x - 1] + 1) >> 2);
            out[2 * x + 1] = @intCast((v + row[x + 1] + 2) >> 2);
        }
        const v = @as(i32, row[p.w - 1]);
        out[2 * p.w - 2] = @intCast((v * 3 + row[p.w - 2] + 1) >> 2);
        out[2 * p.w - 1] = @intCast(v);
        return;
    }
    for (0..p.w) |x| @memset(out[x * hx ..][0..hx], row[x]); // h2v1 / h2v2 plain, int_upsample: replicate
}

const SCALEBITS = 16;
const ONE_HALF: i32 = 1 << (SCALEBITS - 1);
fn fix(comptime x: f64) i32 {
    return @intFromFloat(x * 65536.0 + 0.5);
}

/// ``build_ycc_rgb_table``, as one table of 4 x 256.
pub const Ycc = struct {
    cr_r: [256]i32,
    cb_b: [256]i32,
    cr_g: [256]i32,
    cb_g: [256]i32,

    pub fn init() Ycc {
        var t: Ycc = undefined;
        for (0..256) |i| {
            const x: i32 = @as(i32, @intCast(i)) - 128;
            t.cr_r[i] = (fix(1.40200) * x + ONE_HALF) >> SCALEBITS;
            t.cb_b[i] = (fix(1.77200) * x + ONE_HALF) >> SCALEBITS;
            t.cr_g[i] = -fix(0.71414) * x;
            t.cb_g[i] = -fix(0.34414) * x + ONE_HALF;
        }
        return t;
    }

    /// One pixel: Y, Cb, Cr -> R, G, B.
    pub fn rgb(t: *const Ycc, y: u8, cb: u8, cr: u8, out: *[3]u8) void {
        const yy: i32 = y;
        out[0] = clamp(yy + t.cr_r[cr]);
        out[1] = clamp(yy + ((t.cb_g[cb] + t.cr_g[cr]) >> SCALEBITS));
        out[2] = clamp(yy + t.cb_b[cb]);
    }
};

fn clamp(v: i32) u8 {
    return @intCast(std.math.clamp(v, 0, 255));
}

test "the islow IDCT of a DC-only block is the level shift of the DC" {
    var coef: [64]i16 = @splat(0);
    var q: [64]u16 = @splat(1);
    coef[0] = 80; // 80 * 1 << 2, >> 5 ... = 10 -> 138
    q[0] = 1;
    var out: [64]u8 = undefined;
    idctIslow(&coef, &q, &out, 8);
    for (out) |v| try std.testing.expectEqual(@as(u8, 138), v);
    const t = Ycc.init();
    var px: [3]u8 = undefined;
    t.rgb(128, 128, 128, &px);
    try std.testing.expectEqualSlices(u8, &.{ 128, 128, 128 }, &px);
}
