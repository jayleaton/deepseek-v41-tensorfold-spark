//! JPEG as Pillow 12.3 decodes it through libjpeg-turbo (``JpegImagePlugin`` + ``JpegDecode.c``): 8-bit baseline,
//! extended and progressive Huffman, 1 / 3 / 4 components. The colour space is libjpeg's guess
//! (``default_decompress_parms``: JFIF, Adobe transform, component ids); Pillow asks RGB for 3 components, L for 1,
//! CMYK for 4 (read "CMYK;I", inverted). Every scan's coefficients are kept (no buffered output, no block
//! smoothing: a complete file has none), then each component is IDCT'd and upsampled (``jpeg_pix``).
//! Arithmetic coding, 12-bit and lossless files are refused, as a truncated file is.
const std = @import("std");
const pixels = @import("pixels.zig");
const pix = @import("jpeg_pix.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ BadJpeg, UnsupportedJpeg, TooManyPixels } || Allocator.Error;

const zigzag = [64]u8{ 0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };

const Huff = struct {
    look: [1 << 9]u16 = @splat(0), // (length << 8) | symbol for codes of <= 9 bits, 0: longer
    maxcode: [18]i32 = undefined,
    valoff: [18]i32 = undefined,
    vals: [256]u8 = undefined,
    ok: bool = false,

    fn build(h: *Huff, counts: []const u8, vals: []const u8) Error!void {
        h.* = .{};
        var code: i32 = 0;
        var k: usize = 0;
        for (1..17) |len| {
            const n = counts[len - 1];
            h.valoff[len] = @as(i32, @intCast(k)) - code;
            for (0..n) |_| {
                if (len <= 9) {
                    const shift: u4 = @intCast(9 - len);
                    const base: usize = @as(usize, @intCast(code)) << shift;
                    for (0..(@as(usize, 1) << shift)) |j| h.look[base + j] = @intCast((len << 8) | vals[k]);
                }
                k += 1;
                code += 1;
            }
            h.maxcode[len] = if (n > 0) code - 1 else -1;
            if (code > (@as(i32, 1) << @intCast(len))) return error.BadJpeg;
            code <<= 1;
        }
        h.maxcode[17] = std.math.maxInt(i32);
        @memcpy(h.vals[0..k], vals[0..k]);
        h.ok = true;
    }
};

/// The entropy-coded segment's bits (0xFF00 stuffing removed; at a marker, zeros as libjpeg inserts them).
const Bits = struct {
    data: []const u8,
    at: usize,
    acc: u64 = 0,
    n: u7 = 0,
    hit_marker: bool = false,

    fn fill(b: *Bits) void {
        while (b.n <= 56) {
            var byte: u8 = 0;
            if (!b.hit_marker and b.at < b.data.len) {
                byte = b.data[b.at];
                if (byte == 0xFF) {
                    const next = if (b.at + 1 < b.data.len) b.data[b.at + 1] else 0xD9;
                    if (next == 0) {
                        b.at += 2;
                    } else {
                        b.hit_marker = true;
                        byte = 0;
                    }
                } else b.at += 1;
            } else b.hit_marker = true;
            b.acc |= @as(u64, byte) << @intCast(56 - b.n);
            b.n += 8;
        }
    }
    fn peek(b: *Bits, comptime k: u6) u32 {
        if (b.n < k) b.fill();
        return @intCast(b.acc >> (64 - @as(u7, k)));
    }
    fn skip(b: *Bits, k: u6) void {
        b.acc <<= k;
        b.n -= @as(u7, k);
    }
    fn get(b: *Bits, k: u6) u32 {
        if (k == 0) return 0;
        if (b.n < k) b.fill();
        const v: u32 = @intCast(b.acc >> @intCast(@as(u7, 64) - k));
        b.skip(k);
        return v;
    }
    fn decode(b: *Bits, h: *const Huff) Error!u8 {
        const e = h.look[b.peek(9)];
        if (e != 0) {
            b.skip(@intCast(e >> 8));
            return @truncate(e);
        }
        var code: i32 = @intCast(b.get(9));
        var len: usize = 9;
        while (true) {
            len += 1;
            if (len > 16) return error.BadJpeg;
            code = (code << 1) | @as(i32, @intCast(b.get(1)));
            if (code <= h.maxcode[len]) return h.vals[@intCast(code + h.valoff[len])];
        }
    }
    fn extend(v: u32, s: u5) i32 {
        if (s == 0) return 0;
        const vi: i32 = @intCast(v);
        return if (vi < (@as(i32, 1) << (s - 1))) vi - (@as(i32, 1) << s) + 1 else vi;
    }
    fn receive(b: *Bits, s: u8) Error!i32 {
        if (s > 16) return error.BadJpeg;
        return extend(b.get(@intCast(s)), @intCast(s));
    }
    /// After a restart interval: drop the bit buffer, then the RSTn marker.
    fn restart(b: *Bits) Error!void {
        b.acc = 0;
        b.n = 0;
        b.hit_marker = false;
        while (b.at + 1 < b.data.len and !(b.data[b.at] == 0xFF and b.data[b.at + 1] != 0 and b.data[b.at + 1] != 0xFF)) b.at += 1;
        if (b.at + 1 < b.data.len and b.data[b.at + 1] >= 0xD0 and b.data[b.at + 1] <= 0xD7) {
            b.at += 2;
        } else return error.BadJpeg;
    }
};

const Comp = struct {
    id: u8,
    h: u8,
    v: u8,
    tq: u8,
    bw: usize = 0, // blocks across the component (MCU-padded)
    bh: usize = 0,
    real_bw: usize = 0, // ceil(downsampled width / 8): what a non-interleaved scan covers
    real_bh: usize = 0,
    dw: usize = 0, // downsampled_width / _height
    dh: usize = 0,
    coef: []i16 = &.{},
    dc_pred: i32 = 0,
    td: u8 = 0,
    ta: u8 = 0,
};

const Frame = struct {
    w: usize,
    h: usize,
    comps: [4]Comp,
    nc: usize,
    progressive: bool,
    hmax: u8,
    vmax: u8,
    mcux: usize,
    mcuy: usize,
};

/// Width and height from the first SOF marker (for the pixel cap), null if none.
pub fn size(raw: []const u8) ?[2]u32 {
    var at: usize = 2;
    while (at + 4 <= raw.len) {
        if (raw[at] != 0xFF) return null;
        const m = raw[at + 1];
        if (m == 0xFF) {
            at += 1;
            continue;
        }
        const len = std.mem.readInt(u16, raw[at + 2 ..][0..2], .big);
        if ((m >= 0xC0 and m <= 0xCF) and m != 0xC4 and m != 0xC8 and m != 0xCC) {
            if (at + 9 > raw.len) return null;
            return .{ std.mem.readInt(u16, raw[at + 7 ..][0..2], .big), std.mem.readInt(u16, raw[at + 5 ..][0..2], .big) };
        }
        at += 2 + len;
    }
    return null;
}

pub fn decode(gpa: Allocator, raw: []const u8, max_pixels: u64) Error!pixels.Image {
    if (raw.len < 4 or raw[0] != 0xFF or raw[1] != 0xD8) return error.BadJpeg;
    var quant: [4][64]u16 = undefined;
    var qset: [4]bool = @splat(false);
    var dc: [4]Huff = .{ .{}, .{}, .{}, .{} };
    var ac: [4]Huff = .{ .{}, .{}, .{}, .{} };
    var frame: ?Frame = null;
    defer if (frame) |*f| for (f.comps[0..f.nc]) |c| gpa.free(c.coef);
    var restart: usize = 0;
    var jfif = false;
    var adobe: ?u8 = null;
    var at: usize = 2;
    var eoi = false;
    var scans: usize = 0;
    while (at < raw.len and !eoi) {
        if (raw[at] != 0xFF) { // libjpeg skips junk before a marker with a warning
            at += 1;
            continue;
        }
        while (at < raw.len and raw[at] == 0xFF) at += 1;
        if (at >= raw.len) break;
        const m = raw[at];
        at += 1;
        if (m == 0xD9) {
            eoi = true;
            break;
        }
        if (m >= 0xD0 and m <= 0xD7 or m == 0x01) continue;
        if (at + 2 > raw.len) return error.BadJpeg;
        const len = std.mem.readInt(u16, raw[at..][0..2], .big);
        if (len < 2 or at + len > raw.len) return error.BadJpeg;
        const seg = raw[at + 2 .. at + len];
        at += len;
        switch (m) {
            0xC0, 0xC1, 0xC2 => frame = try sof(gpa, seg, m == 0xC2, max_pixels),
            0xC3, 0xC5...0xC7, 0xC9...0xCB, 0xCD...0xCF => return error.UnsupportedJpeg,
            0xC4 => try dht(seg, &dc, &ac),
            0xDB => try dqt(seg, &quant, &qset),
            0xDD => {
                if (seg.len < 2) return error.BadJpeg;
                restart = std.mem.readInt(u16, seg[0..2], .big);
            },
            0xE0 => if (seg.len >= 14 and std.mem.eql(u8, seg[0..5], "JFIF\x00")) {
                jfif = true;
            },
            0xEE => if (seg.len >= 12 and std.mem.eql(u8, seg[0..5], "Adobe")) {
                adobe = seg[11];
            },
            0xDA => {
                const f = if (frame) |*fr| fr else return error.BadJpeg;
                at = try scan(f, seg, raw, at, &dc, &ac, restart);
                scans += 1;
            },
            else => {},
        }
    }
    const f = if (frame) |*fr| fr else return error.BadJpeg;
    if (scans == 0 or !eoi) return error.BadJpeg; // Pillow: "image file is truncated"
    for (f.comps[0..f.nc]) |c| if (!qset[c.tq]) return error.BadJpeg;
    return finish(gpa, f, &quant, jfif, adobe);
}

fn sof(gpa: Allocator, s: []const u8, progressive: bool, max_pixels: u64) Error!Frame {
    if (s.len < 6) return error.BadJpeg;
    if (s[0] != 8) return error.UnsupportedJpeg;
    const h = std.mem.readInt(u16, s[1..3], .big);
    const w = std.mem.readInt(u16, s[3..5], .big);
    const nc = s[5];
    if (w == 0 or h == 0) return error.BadJpeg;
    if (nc != 1 and nc != 3 and nc != 4) return error.UnsupportedJpeg;
    if (s.len < 6 + 3 * @as(usize, nc)) return error.BadJpeg;
    if (@as(u64, w) * h > max_pixels) return error.TooManyPixels;
    var f: Frame = .{ .w = w, .h = h, .comps = undefined, .nc = nc, .progressive = progressive, .hmax = 1, .vmax = 1, .mcux = 0, .mcuy = 0 };
    for (0..nc) |i| {
        const b = s[6 + 3 * i ..][0..3];
        const hs = b[1] >> 4;
        const vs = b[1] & 15;
        if (hs < 1 or hs > 4 or vs < 1 or vs > 4 or b[2] > 3) return error.BadJpeg;
        f.comps[i] = .{ .id = b[0], .h = hs, .v = vs, .tq = b[2] };
        f.hmax = @max(f.hmax, hs);
        f.vmax = @max(f.vmax, vs);
    }
    f.mcux = (w + 8 * @as(usize, f.hmax) - 1) / (8 * @as(usize, f.hmax));
    f.mcuy = (h + 8 * @as(usize, f.vmax) - 1) / (8 * @as(usize, f.vmax));
    for (f.comps[0..nc]) |*c| {
        c.bw = f.mcux * c.h;
        c.bh = f.mcuy * c.v;
        c.dw = (w * c.h + f.hmax - 1) / f.hmax;
        c.dh = (h * c.v + f.vmax - 1) / f.vmax;
        c.real_bw = (c.dw + 7) / 8;
        c.real_bh = (c.dh + 7) / 8;
        c.coef = try gpa.alloc(i16, c.bw * c.bh * 64);
        @memset(c.coef, 0);
    }
    return f;
}

fn dht(s: []const u8, dc: *[4]Huff, ac: *[4]Huff) Error!void {
    var at: usize = 0;
    while (at < s.len) {
        if (at + 17 > s.len) return error.BadJpeg;
        const tc = s[at] >> 4;
        const th = s[at] & 15;
        if (tc > 1 or th > 3) return error.BadJpeg;
        const counts = s[at + 1 .. at + 17];
        var n: usize = 0;
        for (counts) |c| n += c;
        if (n > 256 or at + 17 + n > s.len) return error.BadJpeg;
        try (if (tc == 0) &dc[th] else &ac[th]).build(counts, s[at + 17 .. at + 17 + n]);
        at += 17 + n;
    }
}

fn dqt(s: []const u8, q: *[4][64]u16, set: *[4]bool) Error!void {
    var at: usize = 0;
    while (at < s.len) {
        const p = s[at] >> 4;
        const t = s[at] & 15;
        if (t > 3 or p > 1) return error.BadJpeg;
        const n: usize = if (p == 0) 64 else 128;
        if (at + 1 + n > s.len) return error.BadJpeg;
        for (0..64) |k| q[t][zigzag[k]] = if (p == 0) s[at + 1 + k] else std.mem.readInt(u16, s[at + 1 + 2 * k ..][0..2], .big);
        set[t] = true;
        at += 1 + n;
    }
}

const Scan = struct {
    comps: [4]*Comp,
    n: usize,
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
    eobrun: u32 = 0,
};

/// One scan from SOS: its entropy-coded data at ``raw[at..]``; returns where the next marker starts.
fn scan(f: *Frame, s: []const u8, raw: []const u8, at: usize, dc: *[4]Huff, ac: *[4]Huff, restart: usize) Error!usize {
    if (s.len < 1) return error.BadJpeg;
    const n = s[0];
    if (n < 1 or n > 4 or s.len < 1 + 2 * @as(usize, n) + 3) return error.BadJpeg;
    var sc: Scan = .{ .comps = undefined, .n = n, .ss = 0, .se = 0, .ah = 0, .al = 0 };
    for (0..n) |i| {
        const id = s[1 + 2 * i];
        const c = for (f.comps[0..f.nc]) |*c| {
            if (c.id == id) break c;
        } else return error.BadJpeg;
        c.td = s[2 + 2 * i] >> 4;
        c.ta = s[2 + 2 * i] & 15;
        if (c.td > 3 or c.ta > 3) return error.BadJpeg;
        sc.comps[i] = c;
    }
    const tail = s[1 + 2 * @as(usize, n) ..];
    sc.ss = tail[0];
    sc.se = tail[1];
    sc.ah = tail[2] >> 4;
    sc.al = tail[2] & 15;
    if (f.progressive) {
        if (sc.se > 63 or sc.ss > sc.se or (sc.ss == 0 and sc.se != 0) or (sc.ss > 0 and n != 1) or sc.al > 13) return error.BadJpeg;
    } else if (sc.ss != 0 or sc.se != 63 or sc.ah != 0 or sc.al != 0) return error.BadJpeg;
    for (sc.comps[0..n]) |c| {
        c.dc_pred = 0;
        if ((!f.progressive or sc.ss == 0) and sc.ah == 0 and !dc[c.td].ok) return error.BadJpeg;
        if ((!f.progressive or sc.ss > 0) and !ac[c.ta].ok) return error.BadJpeg;
    }
    var b: Bits = .{ .data = raw, .at = at };
    const single = n == 1;
    const units_x = if (single) sc.comps[0].real_bw else f.mcux;
    const units_y = if (single) sc.comps[0].real_bh else f.mcuy;
    var count: usize = 0;
    for (0..units_y) |my| for (0..units_x) |mx| {
        if (restart > 0 and count > 0 and count % restart == 0) {
            try b.restart();
            for (sc.comps[0..n]) |c| c.dc_pred = 0;
            sc.eobrun = 0;
        }
        count += 1;
        if (single) {
            const c = sc.comps[0];
            try block(f, &sc, &b, c, c.coef[(my * c.bw + mx) * 64 ..][0..64], dc, ac);
        } else for (sc.comps[0..n]) |c| for (0..c.v) |by| for (0..c.h) |bx| {
            const row = my * c.v + by;
            const col = mx * c.h + bx;
            try block(f, &sc, &b, c, c.coef[(row * c.bw + col) * 64 ..][0..64], dc, ac);
        };
    };
    // the next marker: past what the bit reader consumed
    var e = b.at;
    while (e + 1 < raw.len and !(raw[e] == 0xFF and raw[e + 1] != 0 and !(raw[e + 1] >= 0xD0 and raw[e + 1] <= 0xD7))) e += 1;
    if (e + 1 >= raw.len) return error.BadJpeg;
    return e;
}

fn block(f: *const Frame, sc: *Scan, b: *Bits, c: *Comp, coef: *[64]i16, dc: *[4]Huff, ac: *[4]Huff) Error!void {
    if (!f.progressive) {
        const t = try b.decode(&dc[c.td]);
        c.dc_pred += try b.receive(t);
        coef[0] = @truncate(c.dc_pred);
        var k: usize = 1;
        while (k < 64) {
            const rs = try b.decode(&ac[c.ta]);
            const r = rs >> 4;
            const sz = rs & 15;
            if (sz == 0) {
                if (r != 15) break;
                k += 16;
                continue;
            }
            k += r;
            if (k > 63) return error.BadJpeg;
            coef[zigzag[k]] = @truncate(try b.receive(sz));
            k += 1;
        }
        return;
    }
    const al: u5 = @intCast(sc.al);
    if (sc.ss == 0) {
        if (sc.ah == 0) {
            const t = try b.decode(&dc[c.td]);
            c.dc_pred += try b.receive(t);
            coef[0] = @truncate(c.dc_pred << al);
        } else if (b.get(1) != 0) coef[0] |= @truncate(@as(i32, 1) << al);
        return;
    }
    if (sc.ah == 0) return acFirst(sc, b, coef, &ac[c.ta]);
    return acRefine(sc, b, coef, &ac[c.ta]);
}

fn acFirst(sc: *Scan, b: *Bits, coef: *[64]i16, h: *const Huff) Error!void {
    if (sc.eobrun > 0) {
        sc.eobrun -= 1;
        return;
    }
    const al: u5 = @intCast(sc.al);
    var k: usize = sc.ss;
    while (k <= sc.se) {
        const rs = try b.decode(h);
        const r = rs >> 4;
        const sz = rs & 15;
        if (sz == 0) {
            if (r < 15) {
                sc.eobrun = (@as(u32, 1) << @intCast(r)) - 1;
                if (r > 0) sc.eobrun += b.get(@intCast(r));
                return;
            }
            k += 16;
            continue;
        }
        k += r;
        if (k > 63) return error.BadJpeg;
        coef[zigzag[k]] = @truncate((try b.receive(sz)) << al);
        k += 1;
    }
}

fn acRefine(sc: *Scan, b: *Bits, coef: *[64]i16, h: *const Huff) Error!void {
    const p1: i32 = @as(i32, 1) << @intCast(sc.al);
    const m1: i32 = -p1;
    var k: usize = sc.ss;
    if (sc.eobrun == 0) {
        while (k <= sc.se) {
            const rs = try b.decode(h);
            var r: i32 = rs >> 4;
            const sz = rs & 15;
            var v: i32 = 0;
            if (sz != 0) {
                if (sz != 1) return error.BadJpeg;
                v = if (b.get(1) != 0) p1 else m1;
            } else if (r != 15) {
                sc.eobrun = @as(u32, 1) << @intCast(r);
                if (r > 0) sc.eobrun += b.get(@intCast(r));
                break;
            }
            while (k <= sc.se) : (k += 1) {
                const z = &coef[zigzag[k]];
                if (z.* != 0) {
                    try refineBit(b, z, p1, m1);
                } else {
                    if (r == 0) {
                        if (v != 0) z.* = @truncate(v);
                        k += 1;
                        break;
                    }
                    r -= 1;
                }
            }
        }
    }
    if (sc.eobrun > 0) {
        while (k <= sc.se) : (k += 1) {
            const z = &coef[zigzag[k]];
            if (z.* != 0) try refineBit(b, z, p1, m1);
        }
        sc.eobrun -= 1;
    }
}

fn refineBit(b: *Bits, z: *i16, p1: i32, m1: i32) Error!void {
    if (b.get(1) != 0 and (@as(i32, z.*) & p1) == 0) z.* = @truncate(@as(i32, z.*) + (if (z.* >= 0) p1 else m1));
}

/// IDCT, upsampling and the colour conversion libjpeg's guess and Pillow's rawmode call for.
fn finish(gpa: Allocator, f: *Frame, quant: *const [4][64]u16, jfif: bool, adobe: ?u8) Error!pixels.Image {
    var planes: [4][]u8 = .{ &.{}, &.{}, &.{}, &.{} };
    defer for (planes[0..f.nc]) |p| gpa.free(p);
    for (f.comps[0..f.nc], 0..) |*c, i| {
        const stride = c.bw * 8;
        planes[i] = try gpa.alloc(u8, stride * c.bh * 8);
        for (0..c.bh) |by| for (0..c.bw) |bx| {
            pix.idctIslow(c.coef[(by * c.bw + bx) * 64 ..][0..64], &quant[c.tq], planes[i][by * 8 * stride + bx * 8 ..], stride);
        };
    }
    const Space = enum { gray, ycc, rgb, cmyk, ycck };
    const space: Space = switch (f.nc) {
        1 => .gray,
        3 => if (jfif) .ycc else if (adobe) |t| (if (t == 0) .rgb else .ycc) else if (f.comps[0].id == 82 and f.comps[1].id == 71 and f.comps[2].id == 66) .rgb else .ycc,
        else => if (adobe) |t| (if (t == 0) .cmyk else .ycck) else .cmyk,
    };
    const mode: pixels.Mode = switch (f.nc) {
        1 => .l,
        3 => .rgb,
        else => .cmyk,
    };
    var im: pixels.Image = .{ .w = @intCast(f.w), .h = @intCast(f.h), .mode = mode, .data = try gpa.alloc(u8, f.w * f.h * f.nc) };
    errdefer im.deinit(gpa);
    const rowbuf = try gpa.alloc(u8, 4 * (f.w + 8 * @as(usize, f.hmax)));
    defer gpa.free(rowbuf);
    const span = f.w + 8 * @as(usize, f.hmax);
    const ycc = pix.Ycc.init();
    for (0..f.h) |y| {
        for (f.comps[0..f.nc], 0..) |c, i| {
            const p: pix.Plane = .{ .data = planes[i], .stride = c.bw * 8, .w = c.dw, .h = c.dh };
            pix.upsampleRow(p, f.hmax / c.h, f.vmax / c.v, true, y, rowbuf[i * span ..][0..span]);
        }
        const out = im.data[y * f.w * f.nc ..][0 .. f.w * f.nc];
        for (0..f.w) |x| {
            const a = rowbuf[x];
            switch (space) {
                .gray => out[x] = a,
                .rgb => {
                    out[x * 3] = a;
                    out[x * 3 + 1] = rowbuf[span + x];
                    out[x * 3 + 2] = rowbuf[2 * span + x];
                },
                .ycc => ycc.rgb(a, rowbuf[span + x], rowbuf[2 * span + x], out[x * 3 ..][0..3]),
                .cmyk, .ycck => {
                    var px: [4]u8 = .{ a, rowbuf[span + x], rowbuf[2 * span + x], rowbuf[3 * span + x] };
                    if (space == .ycck) { // ycck_cmyk_convert: C M Y = 255 - RGB of the YCC
                        var rgb: [3]u8 = undefined;
                        ycc.rgb(px[0], px[1], px[2], &rgb);
                        px = .{ 255 - rgb[0], 255 - rgb[1], 255 - rgb[2], px[3] };
                    }
                    for (0..4) |k| out[x * 4 + k] = 255 - px[k]; // "CMYK;I"
                },
            }
        }
    }
    return im;
}
