//! GIF's first frame as Pillow 12.3 loads it (``GifImagePlugin`` frame 0 + ``GifDecode.c``): the canvas is the
//! screen (grown to the frame's extent), mode P with the frame's palette (local, else global), or L when the palette
//! is the identity grey ramp (``_is_palette_needed``) or absent; the canvas starts at the transparency index (else 0)
//! and the frame's LZW codes are written into its rectangle, interlaced or not (transparent pixels written too:
//! frame 0 decodes with transparency -1). A P frame with a transparency index is "transparent" (``to_rgb``
//! composites it on white). The LZW stream ending before the rectangle is full is refused, as Pillow's load is
//! ("image file is truncated").
const std = @import("std");
const pixels = @import("pixels.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ BadGif, TooManyPixels } || Allocator.Error;

pub fn isGif(raw: []const u8) bool {
    return std.mem.startsWith(u8, raw, "GIF87a") or std.mem.startsWith(u8, raw, "GIF89a");
}

fn le16(b: []const u8) u32 {
    return std.mem.readInt(u16, b[0..2], .little);
}

/// Whether a palette is not the identity grey ramp (entry i == (i, i, i) for every entry read).
fn needed(p: []const u8) bool {
    var i: usize = 0;
    while (i + 2 < p.len) : (i += 3) if (!(i / 3 == p[i] and p[i] == p[i + 1] and p[i] == p[i + 2])) return true;
    return false;
}

pub fn decode(gpa: Allocator, raw: []const u8, max_pixels: u64) Error!pixels.Image {
    if (raw.len < 13 or !isGif(raw)) return error.BadGif;
    var w = le16(raw[6..]);
    var h = le16(raw[8..]);
    const flags = raw[10];
    var at: usize = 13;
    var global: ?[]const u8 = null;
    if (flags & 128 != 0) {
        const n = @as(usize, 3) << @intCast((flags & 7) + 1);
        if (at + n > raw.len) return error.BadGif;
        const p = raw[at .. at + n];
        if (needed(p)) global = p;
        at += n;
    }
    var transparency: ?u8 = null;
    // the blocks before the first image descriptor: extensions (graphic control: the transparency index)
    while (true) {
        if (at >= raw.len or raw[at] == ';') return error.BadGif; // "image not found in GIF frame"
        const s = raw[at];
        at += 1;
        if (s == '!') {
            if (at >= raw.len) return error.BadGif;
            const label = raw[at];
            at += 1;
            var first = true;
            while (at < raw.len and raw[at] != 0) {
                const len = raw[at];
                if (at + 1 + len > raw.len) return error.BadGif;
                if (first and label == 249 and len >= 4 and raw[at + 1] & 1 != 0) transparency = raw[at + 1 + 3];
                first = false;
                at += 1 + len;
            }
            at += 1;
        } else if (s == ',') break;
    }
    if (at + 9 > raw.len) return error.BadGif;
    const x0 = le16(raw[at..]);
    const y0 = le16(raw[at + 2 ..]);
    const fw = le16(raw[at + 4 ..]);
    const fh = le16(raw[at + 6 ..]);
    const fflags = raw[at + 8];
    at += 9;
    w = @max(w, x0 + fw);
    h = @max(h, y0 + fh);
    if (@as(u64, w) * h > max_pixels) return error.TooManyPixels;
    var palette: ?[]const u8 = global;
    if (fflags & 128 != 0) {
        const n = @as(usize, 3) << @intCast((fflags & 7) + 1);
        if (at + n > raw.len) return error.BadGif;
        const p = raw[at .. at + n];
        palette = if (needed(p)) p else null; // a local identity palette makes the frame L, even with a global one
        at += n;
    }
    if (at >= raw.len) return error.BadGif;
    const bits = raw[at];
    at += 1;
    if (bits > 12) return error.BadGif;
    var im: pixels.Image = .{ .w = w, .h = h, .mode = if (palette != null) .p else .l, .data = try gpa.alloc(u8, @as(usize, w) * h) };
    errdefer im.deinit(gpa);
    @memset(im.data, transparency orelse 0);
    if (palette) |p| {
        for (0..p.len / 3) |i| @memcpy(im.palette[i * 4 ..][0..3], p[i * 3 ..][0..3]);
        if (transparency) |t| {
            im.palette[@as(usize, t) * 4 + 3] = 0;
            im.transparent = true;
        }
    }
    try lzw(raw[at..], bits, fflags & 64 != 0, &im, x0, y0, fw, fh);
    return im;
}

const table = 4096;

/// ``ImagingGifDecode`` into the frame's rectangle; done when the last row is written.
fn lzw(src: []const u8, bits: u8, interlace: bool, im: *pixels.Image, x0: u32, y0: u32, xs: u32, ys: u32) Error!void {
    if (xs == 0 or ys == 0) return;
    const clear: u32 = @as(u32, 1) << @intCast(bits);
    const end = clear + 1;
    var data: [table]u8 = undefined;
    var link: [table]u16 = undefined;
    var stack: [table]u8 = undefined;
    var next: u32 = clear + 2;
    var codesize: u5 = @intCast(bits + 1);
    var codemask: u32 = (@as(u32, 1) << codesize) - 1;
    var state: u2 = 2; // 2: the first code after a clear, 3: the rest
    var lastdata: u8 = 0;
    var lastcode: u32 = 0;
    var bitbuf: u32 = 0;
    var bitcount: u5 = 0;
    var block: usize = 0;
    var at: usize = 0;
    var x: u32 = 0;
    var y: u32 = 0;
    var step: u32 = if (interlace) 8 else 1;
    var pass: u2 = if (interlace) 1 else 0;
    while (true) {
        while (bitcount < codesize) {
            if (block > 0) {
                if (at >= src.len) return error.BadGif;
                bitbuf |= @as(u32, src[at]) << bitcount;
                at += 1;
                block -= 1;
                bitcount += 8;
            } else {
                if (at >= src.len) return error.BadGif;
                block = src[at];
                if (at + 1 + block > src.len) return error.BadGif;
                at += 1;
            }
        }
        var c: u32 = bitbuf & codemask;
        bitbuf >>= codesize;
        bitcount -= codesize;
        if (c == clear) {
            next = clear + 2;
            codesize = @intCast(bits + 1);
            codemask = (@as(u32, 1) << codesize) - 1;
            state = 2;
            continue;
        }
        if (c == end) return error.BadGif; // the stream ended before the rectangle was full
        var n: usize = 0; // symbols on the stack (top at stack[table - n])
        if (state == 2) {
            if (c > clear) return error.BadGif;
            lastdata = @intCast(c);
            lastcode = c;
            state = 3;
            stack[table - 1] = lastdata;
            n = 1;
        } else {
            const this = c;
            if (c > next) return error.BadGif;
            if (c == next) {
                stack[table - 1] = lastdata;
                n = 1;
                c = lastcode;
            }
            while (c >= clear) {
                if (n >= table or c >= table) return error.BadGif;
                n += 1;
                stack[table - n] = data[c];
                c = link[c];
            }
            lastdata = @intCast(c);
            n += 1;
            stack[table - n] = lastdata;
            if (next < table) {
                data[next] = @intCast(c);
                link[next] = @intCast(lastcode);
                if (next == codemask and codesize < 12) {
                    codesize += 1;
                    codemask = (@as(u32, 1) << codesize) - 1;
                }
                next += 1;
            }
            lastcode = this;
        }
        for (stack[table - n ..]) |v| {
            im.data[(@as(usize, y + y0)) * im.w + x0 + x] = v;
            x += 1;
            if (x < xs) continue;
            x = 0;
            y += step;
            while (y >= ys) switch (pass) {
                1 => {
                    y = 4;
                    pass = 2;
                },
                2 => {
                    step = 4;
                    y = 2;
                    pass = 3;
                },
                3 => {
                    step = 2;
                    y = 1;
                    pass = 0;
                },
                0 => return, // the last row is in
            };
        }
    }
}
