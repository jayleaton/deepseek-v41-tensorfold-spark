//! Prod's session rules over several live slots (Python 8474f31), shared by the GPU side (sessions_gpu.zig) and the host
//! gate (sess4_test.zig, against prod's own store, pool and NVMe tier: tools/zig/dsv41_sess4/golden.py):
//! - `admit`: a request's pool admission (batch.py `_admit`): its pages (prompt + reply + slack, at most the slot's
//!   capacity), less a RAM hit's shared full pages; when the pool's unreserved free pages fall short, the coldest RAM
//!   entries to let go of (`Store.plan_spill`: never the hit), or a wait when even all of them are not enough;
//! - `Measure`: the RAM tier's measure of a snapshot (`Bounded.nbytes` of prod's arrays: slots.py `snapshot`,
//!   replay.Stash.arrays, drafter.snapshot), so the tier's budget (TF_DSV41_SESSION_RAM_MIB) trims at the same saves
//!   as prod's whatever the Zig snapshot holds;
//! - `dsRows` / `stashRows`: the drafter's and the CED stash's rows in such a snapshot.

const std = @import("std");
const sessions = @import("sessions");

/// kvpool.DEFAULT_SLACK (pool.py SLACK): positions past prompt + max_tokens a request may write (verify / draft windows)
pub const slack: u64 = 64;

pub const Error = error{RequestTooLarge};

/// A request's admission: `need` pages reserved for its slot (`Slot.reserve`), `want` of them new; `wait`: the pool
/// cannot hold them even with every RAM entry gone (prod leaves the request queued; nothing was decided).
pub const Plan = struct { need: u32, want: u32, wait: bool = false };

/// batch.py `_admit` for one request of `prompt` tokens and `max_new` reply tokens on a slot of `capacity` positions,
/// `hit` the store's find for its prompt: its pages, and in `spills` the entries to evict first (both ranks evict the
/// leader's list, in its order). `held`: entries never spilled (prod's `used + [key]`: the hit and the entries the
/// round's earlier admissions resume); `extra`: pages the round's earlier admissions take (prod's `new_pages`: a round's
/// admissions are decided before any of them runs), so `want` counts them too. Refuses a request larger than the whole
/// pool (prod: a ValueError to the request).
pub fn admit(store: *sessions.Store, hit: ?sessions.store.Hit, prompt: u64, max_new: u64, capacity: u64, held: []const u32, extra: u32, spills: *std.ArrayList(u32)) !Plan {
    spills.clearRetainingCapacity();
    const pool = store.pool;
    const need = pool.needPages(prompt, max_new, capacity, slack);
    if (need > pool.npages) return error.RequestTooLarge;
    // a RAM hit's full pages are shared, not new (`cached // PAGE if tier == "ram"`)
    const shared: u32 = if (hit) |h| (if (h.ram) @intCast(h.pos / pool.page) else 0) else 0;
    const want = need - @min(need, shared) + extra;
    if (pool.available() >= want) return .{ .need = need, .want = want };
    if (!try store.planSpill(want, held, spills)) {
        spills.clearRetainingCapacity();
        return .{ .need = need, .want = want, .wait = true };
    }
    return .{ .need = need, .want = want };
}

/// batch.py `_admit`'s `save_at`: a prompt of n tokens resumed at `cached` is snapshotted at its replay point
/// (sessions.snapshot_point: the last 16-token grid point strictly before its end) when that lies past `cached` and holds
/// at least rounds.py's `session_min` (64) tokens. (ced.savePoint on the GPU side: the same rule.)
pub fn savePoint(n: u64, cached: u64) ?u64 {
    const s: u64 = if (n > 0) (n - 1) / 16 * 16 else 0;
    return if (s > cached and s >= 64) s else null;
}

/// csa2.rows.ROW_BYTES: an SWA / DSpark ring row as prod's snapshot holds it (Records raw: 576 values + 8 scales)
pub const row_bytes: u64 = 584;
/// slots.py LOOKBACK: Engram's lookback ids (int64)
pub const lookback: u64 = 3;
/// replay.KEEP: the stash rows a slot keeps (REPLAY - 1)
pub const keep: u64 = 127;

/// What prod's snapshot of one slot holds, as sizes (`Bounded.nbytes` = the RAM tier's measure of an entry).
pub const Measure = struct {
    /// SWA rings in the snapshot: replay mode the layers below the decoder (`replay.decoder_start`), full mode every one
    rings: u64,
    /// the SWA window (cfg.sliding_window: the rings' rows and the drafter's)
    window: u64,
    /// ratio-2 kv sources (their float32 [2 x head_dim] carries)
    carries: u64,
    head_dim: u64,
    hidden: u64,
    hc: u64,
    /// the drafter's blocks (0: no drafter, nothing of DSpark in the snapshot)
    ds_blocks: u64 = 0,
    /// CED replay: the stash's rows ride along
    replay: bool,

    /// A replay stash row: ced_x [hc x D] and ced_out [D] bf16, ced_pre / ced_post [4] and ced_comb [16] float32, its id
    fn stashRow(m: Measure) u64 {
        return 2 * m.hc * m.hidden + 2 * m.hidden + 4 * (4 + 4 + 16) + 8;
    }

    /// prod's `Bounded.nbytes` at `pos`, with `stash_rows` (`stashRows`) and `ds_rows` (`dsRows`)
    pub fn bytes(m: Measure, pos: u64, stash_rows: u64, ds_rows: u64) u64 {
        var n: u64 = 0;
        // swa_values / swa_scales: [rings, rows, 576 | 8] uint8 (none below one row)
        n += m.rings * @min(pos, m.window) * row_bytes;
        // the stash (replay: ced_lo int64, then its rows)
        if (m.replay) n += 8 + stash_rows * m.stashRow();
        // carry: [carries, 2 x head_dim] float32 (prod: `if s.carry`)
        if (m.carries > 0) n += m.carries * 2 * m.head_dim * 4;
        // lookback [3] and rings (the layer ids) int64
        n += 8 * lookback + 8 * m.rings;
        // dspark_values / dspark_scales: [blocks, rows, 576 | 8] uint8
        if (m.ds_blocks > 0) n += m.ds_blocks * ds_rows * row_bytes;
        return n;
    }
};

/// drafter.snapshot's rows: positions [max(0, pos - window, valid), pos); `valid` is 0 after a reset, and pos - n after
/// a restore of n rows at pos (drafter.restore: n = 0 restores none and the context restarts at pos).
pub fn dsRows(pos: u64, valid: u64, window: u64) u64 {
    return pos -| @max(pos -| window, valid);
}

/// replay.Stash's rows at `pos` for a stash started at `from` (a slot's start, or a restored stash's first row): the
/// last KEEP of them.
pub fn stashRows(pos: u64, from: u64) u64 {
    return pos -| @max(from, pos -| keep);
}

test "the measure is prod's Bounded.nbytes (golden.py: 96 and >= 128 positions)" {
    const m: Measure = .{ .rings = 20, .window = 128, .carries = 3, .head_dim = 512, .hidden = 5120, .hc = 4, .ds_blocks = 3, .replay = true };
    try std.testing.expectEqual(@as(u64, 6_227_136), m.bytes(96, stashRows(96, 0), dsRows(96, 0, 128)));
    try std.testing.expectEqual(@as(u64, 8_247_384), m.bytes(1488, stashRows(1488, 0), dsRows(1488, 0, 128)));
    // a restored slot: the drafter's context from the restored rows on, the stash from its restored first row
    try std.testing.expectEqual(@as(u64, 128), dsRows(1696, 1488 - 128, 128));
    try std.testing.expectEqual(@as(u64, 0), dsRows(1488, 1488, 128));
    try std.testing.expectEqual(@as(u64, 40), dsRows(1528, 1488, 128));
    try std.testing.expectEqual(@as(u64, 127), stashRows(1696, 1488 - 127));
    try std.testing.expectEqual(@as(u64, 10), stashRows(10, 0));
}
