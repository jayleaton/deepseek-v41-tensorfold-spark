//! BMP as Pillow 12.3 opens it (``BmpImagePlugin`` + the raw decoder / ``BmpRleDecoder``): core (12) and info (40 to
//! 124 byte) headers, 1 / 4 / 8-bit palettes (a 2-entry black / white or identity grey palette makes "1" / "L"),
//! 16-bit 5-5-5 and the bitfield layouts Pillow knows (5-6-5, 24, the 32-bit orders, with or without alpha), RLE8 /
//! RLE4 (Pillow's own decoder, quirks included), bottom-up and top-down rows. Pillow's refusals are refused alike.
const std = @import("std");
const pixels = @import("pixels.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ BadBmp, UnsupportedBmp, TooManyPixels } || Allocator.Error;

pub fn isBmp(raw: []const u8) bool {
    return std.mem.startsWith(u8, raw, "BM");
}

fn le32(b: []const u8) u32 {
    return std.mem.readInt(u32, b[0..4], .little);
}
fn le16(b: []const u8) u32 {
    return std.mem.readInt(u16, b[0..2], .little);
}

/// Pillow's rawmodes of an uncompressed row.
const Raw = enum { bit1, p1, p4, p8, l8, bgr15, bgr16, bgr, bgrx, xbgr, bgxr, abgr, rgba, bgra, bgar };

fn bitsOf(r: Raw) u32 {
    return switch (r) {
        .bit1, .p1 => 1,
        .p4 => 4,
        .p8, .l8 => 8,
        .bgr15, .bgr16 => 16,
        .bgr => 24,
        else => 32,
    };
}

pub fn decode(gpa: Allocator, raw: []const u8, max_pixels: u64) Error!pixels.Image {
    if (raw.len < 18 or !isBmp(raw)) return error.BadBmp;
    var offset: usize = le32(raw[10..]);
    const hs = le32(raw[14..]);
    if (raw.len < 14 + @as(usize, hs) or hs < 4) return error.BadBmp;
    const hd = raw[18 .. 14 + hs];
    var at: usize = 14 + hs; // the stream after the header (bitfield masks, then the palette)
    var w: u32 = 0;
    var h: u32 = 0;
    var bits: u32 = 0;
    var compression: u32 = 0;
    var colors: u32 = 0;
    var pad: usize = 4;
    var down = false; // rows top first (a negative height)
    var masks = [4]u32{ 0, 0, 0, 0 };
    if (hs == 12) {
        if (hd.len < 8) return error.BadBmp;
        w = le16(hd[0..]);
        h = le16(hd[2..]);
        bits = le16(hd[6..]);
        pad = 3;
    } else if (hs == 40 or hs == 52 or hs == 56 or hs == 64 or hs == 108 or hs == 124) {
        if (hd.len < 32) return error.BadBmp;
        down = hd[7] == 0xFF;
        w = le32(hd[0..]);
        h = if (down) @truncate((@as(u64, 1) << 32) - le32(hd[4..])) else le32(hd[4..]);
        bits = le16(hd[10..]);
        compression = le32(hd[12..]);
        colors = le32(hd[28..]);
        if (compression == 3) {
            if (hd.len >= 48) {
                for (0..@as(usize, if (hd.len >= 52) 4 else 3)) |i| masks[i] = le32(hd[36 + i * 4 ..]);
            } else {
                if (at + 12 > raw.len) return error.BadBmp;
                for (0..3) |i| masks[i] = le32(raw[at + i * 4 ..]);
                at += 12;
            }
        }
    } else return error.UnsupportedBmp;
    if (colors == 0) colors = if (bits < 32) @as(u32, 1) << @intCast(bits) else 0;
    if (offset == 14 + hs and bits <= 8) offset += pad * colors;
    var mode: pixels.Mode = switch (bits) {
        1, 4, 8 => .p,
        16, 24, 32 => .rgb,
        else => return error.UnsupportedBmp,
    };
    var rawmode: Raw = switch (bits) {
        1 => .p1,
        4 => .p4,
        8 => .p8,
        16 => .bgr15,
        24 => .bgr,
        else => .bgrx,
    };
    var rle = false;
    switch (compression) {
        0 => {},
        1, 2 => rle = true,
        3 => {
            const m = masks;
            if (bits == 32) {
                const known = [_]struct { m: [4]u32, r: Raw }{
                    .{ .m = .{ 0xFF0000, 0xFF00, 0xFF, 0 }, .r = .bgrx },        .{ .m = .{ 0xFF000000, 0xFF0000, 0xFF00, 0 }, .r = .xbgr },
                    .{ .m = .{ 0xFF000000, 0xFF00, 0xFF, 0 }, .r = .bgxr },      .{ .m = .{ 0xFF000000, 0xFF0000, 0xFF00, 0xFF }, .r = .abgr },
                    .{ .m = .{ 0xFF, 0xFF00, 0xFF0000, 0xFF000000 }, .r = .rgba }, .{ .m = .{ 0xFF0000, 0xFF00, 0xFF, 0xFF000000 }, .r = .bgra },
                    .{ .m = .{ 0xFF000000, 0xFF00, 0xFF, 0xFF0000 }, .r = .bgar }, .{ .m = .{ 0, 0, 0, 0 }, .r = .bgra },
                };
                rawmode = for (known) |k| {
                    if (std.mem.eql(u32, &k.m, &m)) break k.r;
                } else return error.UnsupportedBmp;
                if (rawmode == .abgr or rawmode == .rgba or rawmode == .bgra or rawmode == .bgar) mode = .rgba;
            } else if (bits == 24 and m[0] == 0xFF0000 and m[1] == 0xFF00 and m[2] == 0xFF) {
                rawmode = .bgr;
            } else if (bits == 16 and m[0] == 0xF800 and m[1] == 0x7E0 and m[2] == 0x1F) {
                rawmode = .bgr16;
            } else if (bits == 16 and m[0] == 0x7C00 and m[1] == 0x3E0 and m[2] == 0x1F) {
                rawmode = .bgr15;
            } else return error.UnsupportedBmp;
        },
        else => return error.UnsupportedBmp,
    }
    if (w == 0 or h == 0) return error.BadBmp;
    if (@as(u64, w) * h > max_pixels) return error.TooManyPixels;
    var im: pixels.Image = .{ .w = w, .h = h, .mode = mode, .data = &.{} };
    if (mode == .p) {
        if (colors == 0 or colors > 65536) return error.UnsupportedBmp;
        if (at + pad * colors > raw.len) return error.BadBmp;
        const pal = raw[at .. at + pad * colors];
        at += pad * colors; // the stream after the palette (the data when the header's offset is 0)
        var grey = true;
        for (0..colors) |i| {
            const v: u32 = if (colors == 2) (if (i == 0) 0 else 255) else @intCast(i);
            const e = pal[i * pad ..][0..3];
            if (e[0] != v or e[1] != v or e[2] != v) grey = false;
        }
        if (grey) {
            im.mode = if (colors == 2) .bilevel else .l;
            rawmode = if (colors == 2) .bit1 else .l8;
        } else for (0..@min(colors, 256)) |i| {
            const e = pal[i * pad ..][0..3];
            im.palette[i * 4 ..][0..3].* = .{ e[2], e[1], e[0] }; // BGR(X)
        }
    }
    if (offset == 0) offset = at;
    if (offset > raw.len) return error.BadBmp;
    const n = @as(usize, w) * h;
    im.data = try gpa.alloc(u8, n * pixels.channels(im.mode));
    errdefer im.deinit(gpa);
    if (rle) {
        if (im.mode == .bilevel) return error.UnsupportedBmp; // Pillow unpacks RLE rows as P into a "1" image
        try decodeRle(gpa, raw, offset, compression == 2, w, h, down, &im);
        return im;
    }
    const stride = ((@as(usize, w) * bits + 31) >> 3) & ~@as(usize, 3);
    const need = (@as(usize, w) * bitsOf(rawmode) + 7) / 8; // the rawmode's bytes a row (Pillow's state->bytes)
    if (stride < need) return error.BadBmp; // RawDecode: IMAGING_CODEC_CONFIG
    if (offset + stride * (h - 1) + need > raw.len) return error.BadBmp; // "image file is truncated"
    const ch = pixels.channels(im.mode);
    for (0..h) |r| {
        const y = if (down) r else h - 1 - r;
        unpack(rawmode, raw[offset + r * stride ..][0..need], im.data[y * w * ch ..][0 .. w * ch], w);
    }
    return im;
}

fn unpack(r: Raw, in: []const u8, out: []u8, w: usize) void {
    for (0..w) |x| switch (r) {
        .bit1 => out[x] = if ((in[x / 8] >> @intCast(7 - x % 8)) & 1 != 0) 255 else 0,
        .p1 => out[x] = (in[x / 8] >> @intCast(7 - x % 8)) & 1,
        .p4 => out[x] = if (x % 2 == 0) in[x / 2] >> 4 else in[x / 2] & 15,
        .p8, .l8 => out[x] = in[x],
        .bgr15, .bgr16 => {
            const v: u32 = @as(u32, in[2 * x]) | (@as(u32, in[2 * x + 1]) << 8);
            const o = out[3 * x ..][0..3];
            o[2] = @intCast((v & 31) * 255 / 31);
            if (r == .bgr15) {
                o[1] = @intCast(((v >> 5) & 31) * 255 / 31);
                o[0] = @intCast(((v >> 10) & 31) * 255 / 31);
            } else {
                o[1] = @intCast(((v >> 5) & 63) * 255 / 63);
                o[0] = @intCast(((v >> 11) & 31) * 255 / 31);
            }
        },
        .bgr => out[3 * x ..][0..3].* = .{ in[3 * x + 2], in[3 * x + 1], in[3 * x] },
        else => {
            const p = in[4 * x ..][0..4];
            const rgba: [4]u8 = switch (r) {
                .bgrx => .{ p[2], p[1], p[0], 255 },
                .xbgr => .{ p[3], p[2], p[1], 255 },
                .bgxr => .{ p[3], p[1], p[0], 255 },
                .abgr => .{ p[3], p[2], p[1], p[0] },
                .rgba => p.*,
                .bgra => .{ p[2], p[1], p[0], p[3] },
                .bgar => .{ p[3], p[1], p[0], p[2] },
                else => unreachable,
            };
            const ch: usize = out.len / w;
            @memcpy(out[ch * x ..][0..ch], rgba[0..ch]);
        },
    };
}

/// ``BmpRleDecoder.decode`` then ``set_as_raw`` (L or P, the header's row order): Pillow's loop as written, the
/// absolute runs' word alignment by file offset, RLE4 absolute runs of n // 2 bytes.
fn decodeRle(gpa: Allocator, raw: []const u8, offset: usize, rle4: bool, w: u32, h: u32, down: bool, im: *pixels.Image) Error!void {
    const dest = @as(usize, w) * h;
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(gpa);
    var at = offset;
    var x: usize = 0;
    while (data.items.len < dest) {
        if (at + 2 > raw.len) break;
        var count: usize = raw[at];
        const byte = raw[at + 1];
        at += 2;
        if (count != 0) {
            if (x + count > w) count = if (w > x) w - x else 0;
            for (0..count) |i| try data.append(gpa, if (!rle4) byte else if (i % 2 == 0) byte >> 4 else byte & 15);
            x += count;
        } else if (byte == 0) {
            while (data.items.len % w != 0) try data.append(gpa, 0);
            x = 0;
        } else if (byte == 1) {
            break;
        } else if (byte == 2) {
            if (at + 2 > raw.len) break;
            const right: usize = raw[at];
            const up: usize = raw[at + 1];
            at += 2;
            try data.appendNTimes(gpa, 0, right + up * w);
            x = data.items.len % w;
        } else {
            const want: usize = if (rle4) byte / 2 else byte;
            const got = @min(want, raw.len - at);
            const run = raw[at .. at + got];
            at += got;
            if (rle4) {
                for (run) |b| try data.appendSlice(gpa, &.{ b >> 4, b & 15 });
            } else try data.appendSlice(gpa, run);
            if (got < want) break;
            x += byte;
            if (at % 2 != 0) at += 1;
        }
    }
    if (data.items.len < dest) return error.BadBmp; // set_as_raw: "not enough image data"
    for (0..h) |r| {
        const y = if (down) r else h - 1 - r;
        @memcpy(im.data[y * w ..][0..w], data.items[r * w ..][0..w]);
    }
}
