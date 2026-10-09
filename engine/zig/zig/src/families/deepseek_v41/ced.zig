//! CED bounded replay prefill (`TF_DSV41_PREFILL=replay`): Python's replay.py at 8474f31 on the Zig forward. The rules
//! and the stash's bookkeeping live here (host only); forward_prefill.zig runs the passes, block_prefill.emitReplay
//! emits them.
//!
//! A served prompt of n tokens prefills [0, n - 1) and its last token is the reply's first verify row
//! (TF_DSV41_PROMPT_TAIL=verify, the only shape Python's server has). In replay mode:
//! - **encoder pass**: every prefill segment runs layers 0 .. D - 1 and, of layer D (the decoder's first: the first
//!   ratio-1 layer, 20), only its attention site and its compressor. Layer D's compressed rows and index keys are the
//!   whole decoder's global KV (layers D .. 39 read kv source D), so they are stored for every prompt position. Layers
//!   D .. 39 do not run. The commit is a full segment's (carries, Engram tail, position);
//! - **stash**: the rows entering layer D (the streams, layer D's normed input and its site's pre / post / comb) of the
//!   slot's last `keep` positions, in a ring of `ring` rows indexed by position (`Stash` here, the roles "s.ced.*");
//! - **decoder replay** (`Forward.finishPrompt`): layers D .. 39 over the prompt's prefilled tail [R0, n - 1),
//!   R0 = max(0, n - 128), from the stash, every row's SWA window starting at R0 (`w.lo`), layer D's compressor skipped
//!   (the encoder stored its rows), the boundaries in place (replay.finish's Streams has no second buffer), no final
//!   norm or head (`_finish(logits=False)` keeps nothing). It commits nothing: the position stays at n - 1;
//! - the last token then runs as a full 40-layer verify row (window [R0, n - 1]), as every reply row.
//! For n <= 128, R0 = 0 and replay equals the full prefill bit for bit (Python's test_dsv41_replay.py).
//!
//! Sessions (sessions.py, batch.py `save_at`, rounds.py `_finish`): replay mode keeps no turn snapshots (a turn's end
//! holds decoder rows a fresh prompt replays differently); a prompt is snapshotted at its replay point
//! S = `snapshotPoint(n)` (when S > the resumed position and S >= 64), the snapshot carrying the stash; entries carry
//! their own tag (`tag`), so full and replay entries never resume each other.

const std = @import("std");
const Config = @import("config.zig").Config;

/// protocol.REPLAY: the decoder's rows over a prompt's end, its last token (the verify row) included
pub const replay_rows: u32 = 128;
/// replay.KEEP: the stash rows a slot needs (the prefilled part of the tail)
pub const keep: u32 = replay_rows - 1;
/// the stash ring's rows (position p at row p % ring): at least `keep`, a power of two
pub const ring: u32 = 128;
/// protocol.GRID: snapshot and piece alignment
pub const grid: u64 = 16;
/// rounds.py `session_min`: a prompt snapshot needs this many tokens
pub const session_min: u64 = 64;
/// sessions.Tag.code()'s prefill field: Zig's prefill is Python's fast prefill path (prefill_mm), so its entries are
/// the fast tag's (prompt snapshots only); the old exact-tag (0) turn entries never resume them
pub const tag_fast: u32 = 1 << 8;
/// with the CED field (MODES.index("replay") << 4): replay entries never resume full ones
pub const tag: u32 = tag_fast | 1 << 4;

pub const Error = error{ NoDecoder, EngramInDecoder, StashGap };

/// replay.decoder_start: the first ratio-1 layer; it must be a kv source (its compressor is the decoder's global KV),
/// and no decoder layer may be an Engram layer (replay.finish runs the decoder without Engram rows).
pub fn decoderStart(cfg: *const Config) Error!u32 {
    var d: u32 = cfg.layers;
    for (0..cfg.layers) |i| if (cfg.compressRatio(@intCast(i)) == 1) {
        d = @intCast(i);
        break;
    };
    if (d >= cfg.layers or !cfg.isKvSource(d)) return error.NoDecoder;
    for (cfg.engram_layers.items()) |L| if (L >= d) return error.EngramInDecoder;
    return d;
}

/// The decoder replay of a prompt whose prefilled part ends at `end` (= n - 1): rows [r0, end), m = end - r0 of them
/// (rounds.py: tail = prompt[max(0, n - REPLAY) : n - 1]).
pub const Tail = struct { r0: u64, m: u32 };

pub fn tailOf(end: u64) Tail {
    const m: u32 = @intCast(@min(end, keep));
    return .{ .r0 = end - m, .m = m };
}

/// sessions.snapshot_point: the last grid point strictly before a prompt's end (GLM 0540).
pub fn snapshotPoint(n: u64) u64 {
    return if (n > 0) (n - 1) / grid * grid else 0;
}

/// batch.py's `save_at`: where a prompt of `n` tokens resumed at `cached` is snapshotted, or null.
pub fn savePoint(n: u64, cached: u64) ?u64 {
    const s = snapshotPoint(n);
    return if (s > cached and s >= session_min) s else null;
}

/// A slot's stash bookkeeping (the rows themselves are on the device, "s.ced.*" row p % ring): the stash holds
/// positions [max(from, hi - keep), hi) and their ids (Engram's lookback is not needed: no decoder layer is Engram's,
/// but the ids name image positions and check the tail, as replay.finish does).
pub const Stash = struct {
    from: u64 = 0,
    hi: u64 = 0,
    ids: [ring]u32 = @splat(0),

    /// replay.encode: a segment at `start` (a gap starts the stash over there, as `empty(fw, sg.start)`).
    pub fn extend(s: *Stash, start: u64, ids: []const u32) void {
        if (s.hi != start) s.from = start;
        for (ids, 0..) |t, i| s.ids[@intCast((start + i) % ring)] = t;
        s.hi = start + ids.len;
    }

    /// The first position held.
    pub fn lo(s: *const Stash) u64 {
        return @max(s.from, s.hi -| keep);
    }

    /// replay.finish's check: the stash ends at `end` and holds [r0, end).
    pub fn covers(s: *const Stash, r0: u64, end: u64) bool {
        return s.hi == end and s.lo() <= r0 and end - r0 <= keep;
    }

    /// The ids of positions [r0, end) (`covers` first).
    pub fn tail(s: *const Stash, r0: u64, end: u64, out: []u32) void {
        for (out[0..@intCast(end - r0)], 0..) |*o, i| o.* = s.ids[@intCast((r0 + i) % ring)];
    }
};

/// The DSpark hand-off after a served prompt's prefill, before its first verify row: "w.taps" rows [0, rows) hold
/// the taps of positions [at, at + rows) (replay: the decoder pass over the tail, `replayed(slot, R0, taps)`; full: the
/// last segment, the last piece's keep). One ingest of their last `window` rows (drafter.ingest's skip): `n` rows from
/// taps row `row0`, landing at `start`. Null: no rows.
pub const Handoff = struct { start: u64, row0: u32, n: u32 };

pub fn handoff(at: u64, rows: u32, window: u32) ?Handoff {
    if (rows == 0 or window == 0) return null;
    const skip = rows -| window;
    return .{ .start = at + skip, .row0 = skip, .n = rows - skip };
}

/// One contiguous run of positions in the ring: rows [row, row + n) hold positions [at, at + n) of a range.
pub const Span = struct { row: u32, n: u32, at: u32 };

/// Positions [a, b) (at most `ring`) as at most two contiguous ring runs; `at` counts from a.
pub fn spans(a: u64, b: u64, out: *[2]Span) []const Span {
    std.debug.assert(b >= a and b - a <= ring);
    const total: u32 = @intCast(b - a);
    if (total == 0) return out[0..0];
    const r: u32 = @intCast(a % ring);
    const first = @min(total, ring - r);
    out[0] = .{ .row = r, .n = first, .at = 0 };
    if (first == total) return out[0..1];
    out[1] = .{ .row = 0, .n = total - first, .at = first };
    return out[0..2];
}

/// The stash roles (bf16 streams [ring, 4 D], bf16 normed input [ring, D], fp32 pre / post [ring, 4], comb [ring, 16])
/// in the order the stash / load glue steps name them after the segment's own tensors.
pub const roles = [_][]const u8{ "s.ced.x", "s.ced.out", "s.ced.pre", "s.ced.post", "s.ced.comb" };

test "replay rules as replay.py / sessions.py / batch.py" {
    const cfg: Config = .{};
    try std.testing.expectEqual(@as(u32, 20), try decoderStart(&cfg));
    // n <= 128: R0 = 0, the decoder sees the whole prefilled part
    try std.testing.expectEqual(Tail{ .r0 = 0, .m = 0 }, tailOf(0));
    try std.testing.expectEqual(Tail{ .r0 = 0, .m = 44 }, tailOf(44));
    try std.testing.expectEqual(Tail{ .r0 = 0, .m = 127 }, tailOf(127));
    // n = 129 (end 128): R0 = 1 = n - 128
    try std.testing.expectEqual(Tail{ .r0 = 1, .m = 127 }, tailOf(128));
    try std.testing.expectEqual(Tail{ .r0 = 32768 - 128, .m = 127 }, tailOf(32767));
    try std.testing.expectEqual(@as(u64, 32), snapshotPoint(33));
    try std.testing.expectEqual(@as(u64, 32), snapshotPoint(48));
    try std.testing.expectEqual(@as(u64, 48), snapshotPoint(49));
    try std.testing.expectEqual(@as(?u64, null), savePoint(60, 0)); // S = 48 < session_min
    try std.testing.expectEqual(@as(?u64, 64), savePoint(65, 0));
    try std.testing.expectEqual(@as(?u64, null), savePoint(65, 64)); // nothing past the resumed position
    try std.testing.expectEqual(@as(?u64, 4096), savePoint(4100, 1024));
    // a decoder layer with Engram, or no ratio-1 layer: refused
    var bad = cfg;
    bad.engram_layers = @TypeOf(cfg.engram_layers).of(&.{ 1, 24 });
    try std.testing.expectError(error.EngramInDecoder, decoderStart(&bad));
}

test "the DSpark hand-off: the decoder pass's rows whole, a longer run's last window" {
    // replay, n = 1,536: the tail [1408, 1535), 127 rows, one ingest at R0 (replay.finish -> replayed)
    try std.testing.expectEqual(Handoff{ .start = 1408, .row0 = 0, .n = 127 }, handoff(tailOf(1535).r0, tailOf(1535).m, 128).?);
    // a prompt of 45 tokens: its 44 prefilled rows from 0
    try std.testing.expectEqual(Handoff{ .start = 0, .row0 = 0, .n = 44 }, handoff(tailOf(44).r0, tailOf(44).m, 128).?);
    // full mode, a last segment of 2,048 rows at 4,096: its last 128
    try std.testing.expectEqual(Handoff{ .start = 4096 + 1920, .row0 = 1920, .n = 128 }, handoff(4096, 2048, 128).?);
    try std.testing.expectEqual(@as(?Handoff, null), handoff(0, 0, 128));
}

test "the stash: the last rows of every segment, a gap restarts it, two ring runs at most" {
    var s: Stash = .{};
    var ids: [300]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i);
    s.extend(0, ids[0..100]);
    try std.testing.expect(s.covers(0, 100));
    s.extend(100, ids[100..300]);
    try std.testing.expectEqual(@as(u64, 300 - keep), s.lo());
    try std.testing.expect(s.covers(173, 300));
    try std.testing.expect(!s.covers(172, 300));
    try std.testing.expect(!s.covers(173, 299));
    var got: [keep]u32 = undefined;
    s.tail(173, 300, &got);
    try std.testing.expectEqualSlices(u32, ids[173..300], &got);
    // a segment that does not start at hi: the stash starts over there (a restored slot whose stash ended elsewhere)
    s.extend(500, ids[0..10]);
    try std.testing.expectEqual(@as(u64, 500), s.lo());
    try std.testing.expect(!s.covers(495, 510));
    var sp: [2]Span = undefined;
    try std.testing.expectEqualSlices(Span, &.{.{ .row = 44, .n = 84, .at = 0 }, .{ .row = 0, .n = 43, .at = 84 }}, spans(300, 427, &sp));
    try std.testing.expectEqualSlices(Span, &.{.{ .row = 0, .n = 127, .at = 0 }}, spans(256, 383, &sp));
    try std.testing.expectEqual(@as(usize, 0), spans(7, 7, &sp).len);
}
