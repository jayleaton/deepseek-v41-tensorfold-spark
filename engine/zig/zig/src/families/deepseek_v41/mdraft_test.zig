//! Host tests of DSpark over several slots (`zig build test-dsv41-mdraft`): the real emitter's passes at 1-3 slots
//! against the one-slot pass (the same launches, every Triton launch in the one-slot pass's specialization, the rings
//! stacked), the default emission unchanged, and dspark_rows.zig's statics, grouping, messages and taps placement;
//! TF_DSV41_DRAFT_OVERLAP's side ingest (the serial ingest on its own scratch, disjoint from a prod row window past its
//! taps mark) and TF_DSV41_DRAFT_CHAIN's inputs (Python's captured `_chain` launch, the gather's layout, the staging).

const std = @import("std");
const testing = std.testing;
const calls = @import("calls.zig");
const block = @import("block.zig");
const buffers = @import("buffers.zig");
const check = @import("m1_check.zig");
const emit = @import("dspark_emit.zig");
const rows = @import("dspark_rows.zig");
const Config = @import("config.zig").Config;
const graphs = @import("graphs.zig");
const rowmode = @import("rowmode.zig");
const run = @import("run.zig");
const dsd = @import("dspark_dev.zig");
const dsk = @import("draft/dspark.zig");

test {
    _ = rows;
}

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    // the ingest's main_proj (not in the backbone fixture): a width of the DSpark blocks' own
    try w.dense.put(a, "dspark.main_proj", try w.k2("L40.attn.wkv"));
    return w;
}

const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config = .{},
    w: block.Widths = undefined,
    o: block.Options = .{ .taps = true },

    fn init(f: *Fixture) !void {
        f.arena = std.heap.ArenaAllocator.init(testing.allocator);
        f.w = try widths(f.arena.allocator());
    }
};

fn roleOf(t: calls.Tensor) []const u8 {
    return switch (t.role) {
        .buf => |b| b,
        .weight => |w| w,
        .empty => "",
    };
}

/// An argument's Triton specialization class: a tensor's role and address residue mod 16 (roles are 256-aligned),
/// an int's (== 1, multiple of 16); anything else its tag.
fn class(x: calls.Arg) [3]i64 {
    return switch (x) {
        .t => |t| .{ 1, @mod(t.offset, 16), @intFromEnum(t.dt) },
        .i => |v| .{ 2, @intFromBool(v == 1), @intFromBool(@mod(v, 16) == 0) },
        .f => .{ 3, 0, 0 },
        .b => |v| .{ 4, @intFromBool(v), 0 },
        .none => .{ 5, 0, 0 },
        else => .{ 6, 0, 0 },
    };
}

fn sameCall(a: calls.Call, b: calls.Call) !void {
    try testing.expectEqualStrings(a.name, b.name);
    try testing.expectEqual(a.args.len, b.args.len);
    for (a.args, b.args) |x, y| try testing.expectEqual(std.meta.activeTag(x.arg), std.meta.activeTag(y.arg));
}

fn equalArgs(x: calls.Arg, y: calls.Arg) bool {
    return switch (x) {
        .t, .opaque_table => |t| blk: {
            const u = switch (y) {
                .t, .opaque_table => |u| u,
                else => break :blk false,
            };
            break :blk std.mem.eql(u8, roleOf(t), roleOf(u)) and std.meta.activeTag(t.role) == std.meta.activeTag(u.role) and t.dt == u.dt and t.offset == u.offset and std.mem.eql(i64, t.shape, u.shape) and std.mem.eql(i64, t.stride, u.stride);
        },
        .list => |l| y == .list and l.len == y.list.len and for (l, y.list) |p, q| {
            if (!equalArgs(p, q)) break false;
        } else true,
        else => std.meta.eql(x, y),
    };
}

fn identical(a: []const calls.Call, b: []const calls.Call) !void {
    try testing.expectEqual(a.len, b.len);
    for (a, b) |x, y| {
        try sameCall(x, y);
        try testing.expectEqual(x.triton, y.triton);
        try testing.expectEqual(x.glue, y.glue);
        try testing.expectEqual(x.begin, y.begin);
        try testing.expectEqual(x.grid, y.grid);
        for (x.args, y.args) |p, q| {
            try testing.expectEqualStrings(p.name, q.name);
            try testing.expect(equalArgs(p.arg, q.arg));
        }
    }
}

test "one slot: Layout.of(1) is the one-slot statics, and the default emission is the one-slot pass / ingest" {
    var f: Fixture = .{ .arena = undefined };
    try f.init();
    defer f.arena.deinit();
    const a = f.arena.allocator();
    const n: i64 = f.cfg.dspark_block;
    const one: emit.Statics = .{ .n = n, .window = f.cfg.window };
    const l = emit.Layout.of(1, n, f.cfg.window);
    const lay: emit.Statics = .{ .n = n, .window = f.cfg.window, .lay = l };
    inline for (.{ "ids", "positions", "readRows", "writeRows", "len64", "tokens", "counts", "lo", "hi", "len32" }) |m|
        try testing.expectEqual(@field(emit.Statics, m)(one), @field(emit.Statics, m)(lay));
    try identical(try emit.emitPass(a, &f.cfg, &f.w, f.o, n), try emit.emitPassSlots(a, &f.cfg, &f.w, f.o, n, 1, 1));
    try identical(try emit.emitIngest(a, &f.cfg, &f.w, f.o, 7, 3), try emit.emitIngestAt(a, &f.cfg, &f.w, f.o, 7, 3, .{}));
}

test "several slots: the same launches in the one-slot pass's Triton specializations, rows x slots, rings stacked" {
    var f: Fixture = .{ .arena = undefined };
    try f.init();
    defer f.arena.deinit();
    const a = f.arena.allocator();
    const n: i64 = f.cfg.dspark_block;
    const S: i64 = 4;
    const ring = emit.ringRows(&f.cfg);
    const base = try emit.emitPass(a, &f.cfg, &f.w, f.o, n);
    for (1..rows.maxGroup(f.cfg.dspark_block) + 1) |k| {
        const cs = try emit.emitPassSlots(a, &f.cfg, &f.w, f.o, n, @intCast(k), S);
        try testing.expect(@as(i64, @intCast(k)) * n < 16);
        try testing.expectEqual(base.len, cs.len);
        var rings: usize = 0;
        for (base, cs) |x, y| {
            try sameCall(x, y);
            if (x.triton) for (x.args, y.args) |p, q| {
                // a Triton launch's ints that are rows here are never 1 or a multiple of 16 at k x n rows
                try testing.expectEqual(class(p.arg), class(q.arg));
            };
            for (y.args) |q| if (q.arg == .t) {
                const r = roleOf(q.arg.t);
                if (std.mem.startsWith(u8, r, "s.L") and std.mem.indexOf(u8, r, ".ring.") != null) {
                    rings += 1;
                    try testing.expectEqual(S * ring, q.arg.t.shape[0]);
                }
            };
        }
        try testing.expect(rings > 0);
    }
    // the plan sizes the rings for every slot and the statics for the largest pass
    var p: buffers.Plan = .{ .a = a };
    const g = rows.maxGroup(f.cfg.dspark_block);
    try p.add(try emit.emitPassSlots(a, &f.cfg, &f.w, f.o, n, g, S));
    try testing.expectEqual(@as(u64, @intCast(S * ring * 576)), p.sizes.get("s.L40.ring.v").?);
    const l = emit.Layout.of(g, n, f.cfg.window);
    try testing.expect(p.sizes.get("w.ds.i64").? >= 8 * @as(u64, @intCast(l.len64)));
    try testing.expect(p.sizes.get("w.ds.i32").? >= 4 * @as(u64, @intCast(l.end32)));
}

test "an ingest writes its slot's ring (a view at slot x ring rows) from the chosen taps role" {
    var f: Fixture = .{ .arena = undefined };
    try f.init();
    defer f.arena.deinit();
    const a = f.arena.allocator();
    const ring = emit.ringRows(&f.cfg);
    const base = try emit.emitIngest(a, &f.cfg, &f.w, f.o, 6, 2);
    const cs = try emit.emitIngestAt(a, &f.cfg, &f.w, f.o, 6, 2, .{ .ring_slots = 4, .slot = 3, .taps_role = "s.rows.taps" });
    try testing.expectEqual(base.len, cs.len);
    var stores: usize = 0;
    var taps: usize = 0;
    for (base, cs) |x, y| {
        try sameCall(x, y);
        if (x.triton) for (x.args, y.args) |p, q| try testing.expectEqual(class(p.arg), class(q.arg));
        for (x.args, y.args) |p, q| if (p.arg == .t) {
            const r = roleOf(p.arg.t);
            if (std.mem.indexOf(u8, r, ".ring.v") != null) {
                stores += 1;
                try testing.expectEqual(3 * ring * 576, q.arg.t.offset);
                try testing.expectEqual(ring, q.arg.t.shape[0]);
            } else if (std.mem.indexOf(u8, r, ".ring.s") != null) {
                try testing.expectEqual(3 * ring * 8, q.arg.t.offset);
            } else if (std.mem.eql(u8, r, "w.taps")) {
                taps += 1;
                try testing.expectEqualStrings("s.rows.taps", roleOf(q.arg.t));
                try testing.expectEqual(p.arg.t.offset, q.arg.t.offset);
            }
        };
    }
    try testing.expectEqual(@as(usize, f.cfg.mtp_layers), stores);
    try testing.expect(taps > 0);
}

test "stage: Python's row-mode DraftInputs (ids, positions, ring slots, the context in the stacked ring)" {
    const n: u32 = 5;
    const W: u32 = 128;
    const ring: u32 = 256;
    const lay = emit.Layout.of(2, n, W);
    var h64: [256]i64 = undefined;
    var h32: [4096]i32 = undefined;
    const ms = [_]rows.Member{ .{ .slot = 3, .anchor = 77, .start = 300, .valid = 0 }, .{ .slot = 1, .anchor = 9, .start = 40, .valid = 35 } };
    rows.stage(lay, n, W, ring, 2, &ms, &h64, &h32);
    const at = struct {
        fn i64s(h: []const i64, o: i64, i: usize) i64 {
            return h[@as(usize, @intCast(o)) + i];
        }
        fn i32s(h: []const i32, o: i64, i: usize) i32 {
            return h[@as(usize, @intCast(o)) + i];
        }
    };
    try testing.expectEqual(77, at.i64s(&h64, lay.ids, 0));
    try testing.expectEqual(2, at.i64s(&h64, lay.ids, 1));
    try testing.expectEqual(9, at.i64s(&h64, lay.ids, 5));
    try testing.expectEqual(304, at.i64s(&h64, lay.positions, 4));
    try testing.expectEqual(40, at.i64s(&h64, lay.positions, 5));
    try testing.expectEqual(3, at.i64s(&h64, lay.read, 4));
    try testing.expectEqual(1, at.i64s(&h64, lay.write, 9));
    // slot 3 at 300: positions 172 .. 299 (the window), rows 3 x 256 + p % 256, every row of the slot alike
    try testing.expectEqual(W, @as(u32, @intCast(at.i32s(&h32, lay.counts, 0))));
    try testing.expectEqual(3 * 256 + 172, at.i32s(&h32, lay.tokens, 0));
    try testing.expectEqual(3 * 256 + 299 % 256, at.i32s(&h32, lay.tokens, 4 * W + W - 1));
    // slot 1 at 40 with valid 35: positions 35 .. 39 only
    try testing.expectEqual(5, at.i32s(&h32, lay.counts, 5));
    try testing.expectEqual(256 + 35, at.i32s(&h32, lay.tokens, 5 * W));
    try testing.expectEqual(-1, at.i32s(&h32, lay.tokens, 5 * W + 5));
    try testing.expectEqual(40, at.i32s(&h32, lay.lo, 9));
    try testing.expectEqual(44, at.i32s(&h32, lay.hi, 9));
    // no array overlaps the next one
    try testing.expect(lay.positions >= lay.ids + 10 and lay.read >= lay.positions + 10 and lay.write >= lay.read + 10);
    try testing.expect(lay.counts >= lay.tokens + 10 * W and lay.lo >= lay.counts + 10 and lay.hi >= lay.lo + 10);
}

test "groups, messages, taps placement, the gather's order" {
    var gs: [16][2]usize = undefined;
    try testing.expectEqual(@as(usize, 2), rows.groups(4, 3, &gs));
    try testing.expectEqualSlices(usize, &.{ 0, 2 }, &gs[0]);
    try testing.expectEqualSlices(usize, &.{ 2, 4 }, &gs[1]);
    try testing.expectEqual(@as(usize, 1), rows.groups(3, 3, &gs));
    try testing.expectEqual(@as(usize, 2), rows.groups(5, 3, &gs));
    try testing.expectEqualSlices(usize, &.{ 3, 5 }, &gs[1]);
    try testing.expectEqual(@as(usize, 4), rows.groups(4, 1, &gs));
    try testing.expectEqual(@as(u32, 3), rows.maxGroup(5));
    try testing.expectEqual(@as(u32, 1), rows.maxGroup(16));

    const ms = [_]rows.Member{ .{ .slot = 0, .anchor = 5, .start = 10, .valid = 0 }, .{ .slot = 2, .anchor = 6, .start = 70, .valid = 0 } };
    var buf: [50]i64 = undefined;
    const msg = try rows.proposeMsg(&ms, &buf);
    const valid = [_]u64{ 4, 0, 61, 0 };
    var out: [3]rows.Member = undefined;
    const got = try rows.proposeOf(msg, &valid, &out);
    try testing.expectEqual(@as(usize, 2), got.len);
    try testing.expectEqual(@as(u64, 4), got[0].valid);
    try testing.expectEqual(@as(u64, 61), got[1].valid);
    try testing.expectEqual(@as(u32, 6), got[1].anchor);
    try testing.expectError(error.BadPlan, rows.proposeOf(msg, &valid, out[0..1]));

    const taps: u64 = 0x10000;
    const stride: u64 = 2 * 3 * 4096;
    const stash = [2]u64{ 0x900000, 16 * 4 * stride };
    // in "w.taps": the slot's rows from row 4 of the window, rows 1 .. 3 of them
    try testing.expectEqual(rows.TapsAt{ .src = 0, .skip = 5 }, try rows.locate(taps + 4 * stride, 6, stride, 1, 3, 6, taps, stash));
    // stashed: slot 2's block of the stash
    try testing.expectEqual(rows.TapsAt{ .src = 1, .skip = 32 }, try rows.locate(stash[0] + 32 * stride, 4, stride, 0, 4, 6, taps, stash));
    // past the taps (a resumed prompt) or past the planned rows: not on the device
    try testing.expectEqual(@as(i64, -1), (try rows.locate(taps, 6, stride, 0, 7, 16, taps, stash)).src);
    try testing.expectEqual(@as(i64, -1), (try rows.locate(taps, 128, stride, 0, 100, 16, taps, stash)).src);
    try testing.expectError(error.Rows, rows.locate(taps + 3, 6, stride, 0, 1, 6, taps, stash));

    // [W][R][2k] -> rows of [W k]
    const all = [_]f32{ 1, 2, @bitCast(@as(i32, 10)), @bitCast(@as(i32, 11)), 3, 4, @bitCast(@as(i32, 12)), @bitCast(@as(i32, 13)), 5, 6, @bitCast(@as(i32, 20)), @bitCast(@as(i32, 21)), 7, 8, @bitCast(@as(i32, 22)), @bitCast(@as(i32, 23)) };
    var cand: [8]i32 = undefined;
    var cval: [8]f32 = undefined;
    rows.ungather(&all, 2, 2, 2, &cand, &cval);
    try testing.expectEqualSlices(i32, &.{ 10, 11, 20, 21, 12, 13, 22, 23 }, &cand);
    try testing.expectEqualSlices(f32, &.{ 1, 2, 5, 6, 3, 4, 7, 8 }, &cval);
}

test "the speculative pass's device start: ds_stage's arithmetic over the placeholder staging == the host's statics" {
    // dspark_gpu.afterPick stages stageOne(anchor 0, the window's start), then glue.cu ds_stage (mirrored by
    // stageMirror) writes every position-dependent element from the pick: the result must be stageOne(bonus, next
    // position), the statics `begin` would stage on the host. stageOne itself against Context.of (draft/dspark.zig).
    const dspark = @import("draft/dspark.zig");
    const cfg: Config = .{};
    const st: emit.Statics = .{ .n = cfg.dspark_block, .window = cfg.window };
    const ring: u64 = @intCast(emit.ringRows(&cfg));
    const shape: dspark.Shape = .{ .block = cfg.dspark_block, .window = cfg.window };
    var want64: [64]i64 = undefined;
    var want32: [1024]i32 = undefined;
    var got64: [64]i64 = undefined;
    var got32: [1024]i32 = undefined;
    const l64: usize = @intCast(st.len64());
    const l32: usize = @intCast(st.len32());
    for ([_]u64{ 1, 2, 5, 127, 128, 129, 200, 255, 256, 257, 4000, 1 << 20 }) |P| {
        for ([_]u64{ 0, 1, 64, 150, 3999 }) |valid| {
            if (valid > P) continue;
            for ([_]u32{ 1, 3, 6 }) |nw| {
                const window_start = P - @min(P, nw); // the window the pick came from (any start below P)
                var pick: [16]i64 = @splat(0);
                pick[nw] = 2; // accepted (unused by the statics)
                pick[nw + 1] = 77_123; // bonus
                pick[nw + 2] = @intCast(P);
                rows.stageOne(st, ring, cfg.dspark_noise_token, valid, 77_123, P, want64[0..l64], want32[0..l32]);
                rows.stageOne(st, ring, cfg.dspark_noise_token, valid, 0, window_start, got64[0..l64], got32[0..l32]);
                rows.stageMirror(st, ring, cfg.dspark_noise_token, valid, &pick, nw, got64[0..l64], got32[0..l32]);
                try testing.expectEqualSlices(i64, want64[0..l64], got64[0..l64]);
                try testing.expectEqualSlices(i32, want32[0..l32], got32[0..l32]);
                // stageOne's context == Context.of's (the old fillStatics' rule)
                const c = dspark.Context.of(shape, P, valid);
                try testing.expectEqual(@as(i32, @intCast(c.count)), want32[@intCast(st.counts())]);
                if (c.count > 0) try testing.expectEqual(@as(i32, @intCast(c.first % ring)), want32[@intCast(st.tokens())]);
            }
        }
    }
}

test "the slot passes' device start: ds_accept_rows / ds_stage_rows over the placeholder staging == the host's statics" {
    // dspark_slots.afterPick: each armed slot's (accepted, bonus, next position) from the row window's picks (lanes'
    // accept, a slot's own rows), then a group's statics staged as stage(anchor 0, the window's start) and rewritten
    // from those values: the result must be stage() at (bonus, next position), the round's `begin` statics.
    const cfg: Config = .{};
    const n: u32 = cfg.dspark_block;
    const W: u32 = cfg.window;
    const ring: u32 = @intCast(emit.ringRows(&cfg));
    const noise: u32 = cfg.dspark_noise_token;
    // a 4-slot row window: slot 2 (rows 0-5), slot 0 (6-8), slot 3 (9-14), slot 1 (15)
    const segs = [_][3]i64{ .{ 0, 6, 300 }, .{ 6, 3, 129 }, .{ 9, 6, 5000 }, .{ 15, 1, 40 } };
    const seg_slot = [_]u32{ 2, 0, 3, 1 };
    var ids: [16]i64 = undefined;
    var picks: [16]i64 = undefined;
    for (&ids, &picks, 0..) |*t, *q, i| {
        t.* = @intCast(1000 + i);
        q.* = @intCast(5000 + i);
    }
    // accepted 3 for slot 2 (picks 0..2 equal the next rows' ids), 0 for slot 0, 5 (all) for slot 3, 0 for slot 1
    for (0..3) |i| picks[i] = ids[i + 1];
    for (9..14) |i| picks[i] = ids[i + 1];
    var acc: [4][3]i64 = undefined;
    rows.acceptMirror(&picks, &ids, &segs, &acc);
    try testing.expectEqual([3]i64{ 3, picks[3], 304 }, acc[0]);
    try testing.expectEqual([3]i64{ 0, picks[6], 130 }, acc[1]);
    try testing.expectEqual([3]i64{ 5, picks[14], 5006 }, acc[2]);
    try testing.expectEqual([3]i64{ 0, picks[15], 41 }, acc[3]);
    const valids = [4]u64{ 0, 100, 4990, 0 };
    // groups of 1-4 members in sorted-slot order, as afterPick forms them
    for ([_][2]usize{ .{ 0, 4 }, .{ 0, 2 }, .{ 2, 4 }, .{ 1, 2 } }) |g| {
        const gk = g[1] - g[0];
        const lay = emit.Layout.of(@intCast(gk), n, W);
        var want64: [256]i64 = undefined;
        var want32: [8192]i32 = undefined;
        var got64: [256]i64 = undefined;
        var got32: [8192]i32 = undefined;
        var ms_want: [4]rows.Member = undefined;
        var ms_ph: [4]rows.Member = undefined;
        var mem: [4][3]i64 = undefined;
        for (g[0]..g[1], 0..) |j, i| {
            const sl = seg_slot[j];
            ms_want[i] = .{ .slot = sl, .anchor = @intCast(acc[j][1]), .start = @intCast(acc[j][2]), .valid = valids[j] };
            ms_ph[i] = .{ .slot = sl, .anchor = 0, .start = @intCast(segs[j][2]), .valid = valids[j] };
            mem[i] = .{ @intCast(j), sl, @intCast(valids[j]) };
        }
        const l64: usize = @intCast(lay.len64);
        const l32: usize = @intCast(lay.len32);
        rows.stage(lay, n, W, ring, noise, ms_want[0..gk], want64[0..l64], want32[0..l32]);
        rows.stage(lay, n, W, ring, noise, ms_ph[0..gk], got64[0..l64], got32[0..l32]);
        rows.stageRowsMirror(lay, n, W, ring, noise, &acc, mem[0..gk], got64[0..l64], got32[0..l32]);
        try testing.expectEqualSlices(i64, want64[0..l64], got64[0..l64]);
        try testing.expectEqualSlices(i32, want32[0..l32], got32[0..l32]);
    }
}

// -------------------------------------------------------------------------------------------------------------------
// TF_DSV41_DRAFT_OVERLAP / TF_DSV41_DRAFT_CHAIN

fn sameArgSide(x: calls.Arg, y: calls.Arg) !void {
    switch (x) {
        .t, .opaque_table => |t| {
            const u = switch (y) {
                .t, .opaque_table => |v| v,
                else => return error.TestUnexpectedResult,
            };
            try testing.expectEqual(t.dt, u.dt);
            try testing.expectEqualSlices(i64, t.shape, u.shape);
            try testing.expectEqualSlices(i64, t.stride, u.stride);
            try testing.expectEqual(t.offset, u.offset);
            switch (t.role) {
                .buf => |r| if (emit.sideRole(r)) {
                    try testing.expect(std.mem.startsWith(u8, u.role.buf, "w.ovl."));
                    try testing.expectEqualStrings(r, u.role.buf["w.ovl.".len..]);
                } else try testing.expectEqualStrings(r, u.role.buf),
                .weight => |w| try testing.expectEqualStrings(w, u.role.weight),
                .empty => try testing.expect(u.role == .empty),
            }
        },
        .list => |items| {
            try testing.expectEqual(items.len, y.list.len);
            for (items, y.list) |p, q| try sameArgSide(p, q);
        },
        .i => |v| try testing.expectEqual(v, y.i),
        .f => |v| try testing.expectEqual(v, y.f),
        .b => |v| try testing.expectEqual(v, y.b),
        .none => try testing.expect(y == .none),
    }
}

test "overlapped ingest: the serial ingest's launches, its scratch roles its own" {
    var fx: Fixture = .{ .arena = undefined };
    try fx.init();
    defer fx.arena.deinit();
    const a = fx.arena.allocator();
    for ([_]i64{ 1, 5, 16 }) |n| for ([_]i64{ 0, 3 }) |skip| {
        const serial = try emit.emitIngestAt(a, &fx.cfg, &fx.w, fx.o, n, skip, .{ .ring_slots = 4, .slot = 2 });
        const side = try emit.sideRoles(a, serial);
        try testing.expectEqual(serial.len, side.len);
        var renamed: usize = 0;
        for (serial, side) |c, d| {
            try testing.expectEqualStrings(c.name, d.name);
            try testing.expectEqual(c.triton, d.triton);
            try testing.expectEqual(c.grid, d.grid);
            // Runner.windowOn's contract: no branch, deferral or glue marks in a drafter program
            try testing.expect(!c.side and !c.fork and !c.join and !c.defer_side and !c.defer_join and !c.glue);
            try testing.expectEqual(c.args.len, d.args.len);
            for (c.args, d.args) |x, y| {
                try testing.expectEqualStrings(x.name, y.name);
                try sameArgSide(x.arg, y.arg);
                if (x.arg == .t and x.arg.t.role == .buf and emit.sideRole(x.arg.t.role.buf)) renamed += 1;
            }
        }
        try testing.expect(renamed > 0);
    };
}

fn rolesInto(a: std.mem.Allocator, set: *std.StringHashMapUnmanaged(void), x: calls.Arg) !void {
    switch (x) {
        .t, .opaque_table => |t| if (t.role == .buf) try set.put(a, t.role.buf, {}),
        .list => |items| for (items) |y| try rolesInto(a, set, y),
        else => {},
    }
}

fn rolesOf(a: std.mem.Allocator, cs: []const calls.Call) !std.StringHashMapUnmanaged(void) {
    var set: std.StringHashMapUnmanaged(void) = .empty;
    for (cs) |c| for (c.args) |x| try rolesInto(a, &set, x.arg);
    return set;
}

test "overlapped ingest: a prod row window's taps mark, and nothing the window touches but the taps and the rope table" {
    var fx: Fixture = .{ .arena = undefined };
    try fx.init();
    defer fx.arena.deinit();
    const a = fx.arena.allocator();
    const cfg = &fx.cfg;
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    // prod-zig.env's decode shape: R1, row mode in the paged pool, taps, the branches and the deferred mHC coefficients
    var o: block.Options = .{ .limit = limit, .r1 = true, .expert_topp = 0.85, .index_budget = 64 << 20, .rows = true, .rope_rows = limit + 2048, .taps = true, .branches = true, .mhc_defer = true };
    const pages = @divExact(limit, block.Pool.page);
    o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages };
    const s: graphs.Settings = .{};
    const ing = try emit.sideRoles(a, try emit.emitIngestAt(a, cfg, &fx.w, fx.o, 16, 5, .{ .ring_slots = 4, .slot = 3 }));
    const side = try rolesOf(a, ing);
    for ([_]i64{ 5, 16, 20, 24 }) |n| {
        const ctx = graphs.bucketOf(300_000, s.bucket, s.grow);
        const one = try block.emit(a, cfg, &fx.w, o, &backbone, n, graphs.emitStart(@intCast(n), ctx, s, @intCast(limit)), true);
        const cs = try rowmode.transform(a, one, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = o.pool.?.pts });
        const m = run.lastTaps(cs) orelse return error.TestUnexpectedResult;
        // the mark sits past the last DSpark target's boundary (layer 39's), before the last layer's MoE and the head
        try testing.expect(m + 1 < cs.len);
        const after = try rolesOf(a, cs[m + 1 ..]);
        try testing.expect(!after.contains("w.taps"));
        const whole = try rolesOf(a, cs);
        var it = side.keyIterator();
        while (it.next()) |k| {
            const shared = std.mem.eql(u8, k.*, "w.taps") or std.mem.startsWith(u8, k.*, "s.rope.");
            if (whole.contains(k.*) and !shared) {
                std.debug.print("{d} rows: the side ingest's role {s} is the window's too\n", .{ n, k.* });
                return error.TestUnexpectedResult;
            }
            if (after.contains(k.*) and !std.mem.startsWith(u8, k.*, "s.rope.")) return error.TestUnexpectedResult;
        }
    }
}

test "device chain: emitChain is Python's captured _chain launch (names, dtypes, shapes, weights, constexprs)" {
    var fx: Fixture = .{ .arena = undefined };
    try fx.init();
    defer fx.arena.deinit();
    const a = fx.arena.allocator();
    const cap = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/chain-op-cap16.json"), .{});
    const want = cap.object.get("args").?.array.items;
    const cs = try emit.emitChain(a, &fx.cfg, &fx.w, fx.o, fx.cfg.dspark_block, 1, 128, true);
    try testing.expectEqual(@as(usize, 1), cs.len);
    const c = cs[0];
    try testing.expectEqualStrings("_chain", c.name);
    try testing.expect(c.triton);
    const g = cap.object.get("grid").?.array.items;
    for (c.grid, g) |x, y| try testing.expectEqual(y.integer, x);
    try testing.expectEqual(want.len, c.args.len);
    for (c.args, want) |x, w| {
        const wo = w.object;
        try testing.expectEqualStrings(wo.get("name").?.string, x.name);
        const kind = wo.get("t").?.string;
        if (std.mem.eql(u8, kind, "tensor")) {
            const t = x.arg.t;
            const dt = wo.get("dtype").?.string;
            const ours: []const u8 = switch (t.dt) {
                .i32 => "int32",
                .i64 => "int64",
                .f32 => "float32",
                .f64 => "float64",
                .bf16 => "bfloat16",
                else => "?",
            };
            try testing.expectEqualStrings(dt, ours);
            for (wo.get("shape").?.array.items, t.shape) |p, q| try testing.expectEqual(p.integer, q);
            for (wo.get("stride").?.array.items, t.stride) |p, q| try testing.expectEqual(p.integer, q);
            if (wo.get("weight")) |wn| try testing.expectEqualStrings(wn.string, t.role.weight) else try testing.expect(t.role == .buf);
        } else if (std.mem.eql(u8, kind, "int")) {
            try testing.expectEqual(wo.get("v").?.integer, x.arg.i);
        } else if (std.mem.eql(u8, kind, "bool")) {
            try testing.expectEqual(wo.get("v").?.bool, x.arg.b);
        } else return error.TestUnexpectedResult;
    }
    // without the confidence head HID / CW / CONF are cval's tensor (Python's chain), D 1024
    const nc = (try emit.emitChain(a, &fx.cfg, &fx.w, fx.o, fx.cfg.dspark_block, 3, 128, false))[0];
    for (nc.args) |x| if (std.mem.eql(u8, x.name, "HID") or std.mem.eql(u8, x.name, "CW") or std.mem.eql(u8, x.name, "CONF"))
        try testing.expectEqualStrings(emit.chain_roles.cval, x.arg.t.role.buf);
    try testing.expectEqual(@as(i64, 3), nc.grid[0]);
}

test "device chain: ds_out's strided copies lay the gathered candidates out as the host's ungather" {
    const W = 2;
    const k = 4;
    const at = 3;
    const R = 7;
    const rows_max = at + R + 2;
    const c = W * k;
    // the gathered buffer: regions [W][rows][2 k] at each round row; ids as int bits after the values
    var gathered: [rows_max * 2 * k * W]f32 = undefined;
    for (&gathered, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast(i)) * 3 - 500);
    const region = gathered[at * 2 * k * W ..][0 .. R * 2 * k * W];
    for (0..W) |w| for (0..R) |i| for (0..k) |j| {
        region[(w * R + i) * 2 * k + k + j] = @bitCast(@as(i32, @intCast(1000 * w + 10 * i + j)));
    };
    var cand_want: [R * c]i32 = undefined;
    var cval_want: [R * c]f32 = undefined;
    rows.ungather(region, W, R, k, &cand_want, &cval_want);
    var cand: [rows_max * c]u32 = @splat(0xAAAAAAAA);
    var cval: [rows_max * c]u32 = @splat(0xAAAAAAAA);
    const src = std.mem.sliceAsBytes(&gathered);
    var cs: [8]dsd.Copy2D = undefined;
    for (dsd.reorder(W, R, k, at, &cs)) |cp| {
        const dst = std.mem.sliceAsBytes(if (cp.ids) &cand else &cval);
        for (0..cp.height) |h| @memcpy(dst[cp.dst + h * cp.dst_pitch ..][0..cp.width], src[cp.src + h * cp.src_pitch ..][0..cp.width]);
    }
    for (cand[at * c ..][0 .. R * c], cand_want) |x, y| try testing.expectEqual(@as(u32, @bitCast(y)), x);
    for (cval[at * c ..][0 .. R * c], cval_want) |x, y| try testing.expectEqual(@as(u32, @bitCast(y)), x);
    // nothing outside the region's rows
    for (cand[0 .. at * c]) |x| try testing.expectEqual(@as(u32, 0xAAAAAAAA), x);
    for (cval[(at + R) * c ..]) |x| try testing.expectEqual(@as(u32, 0xAAAAAAAA), x);
}

test "device chain: the staged inputs are Python's (sampling_params: ipar, anchor, fpar)" {
    var host: [4 * 48 + 2 * 4 * 5 * 4]u8 align(8) = undefined;
    const greedy = dsk.Params.of(null, 99);
    const smp = dsk.Params.of(.{ .seed = (1 << 63) | 77, .temperature = 0.7, .top_k = 20, .top_p = 0.95, .min_p = 0.05 }, 1234);
    const got = dsd.Chain.stageArgs(&host, 4, &.{ 11, 22 }, &.{ greedy, smp });
    const ipar: []const i64 = @alignCast(std.mem.bytesAsSlice(i64, got.ipar));
    const anchor: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, got.anchor));
    const fpar: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, got.fpar));
    // greedy: [0, p + 1, 0, 0], [1.0, 1.0, 0.0]; sampled: [seed & (2^63 - 1), p + 1, top_k, 1], [T, top_p, min_p]
    try testing.expectEqualSlices(i64, &.{ 0, 100, 0, 0, 77, 1235, 20, 1 }, ipar);
    try testing.expectEqualSlices(i32, &.{ 11, 22 }, anchor);
    try testing.expectEqualSlices(f32, &.{ 1.0, 1.0, 0.0, 0.7, 0.95, 0.05 }, fpar);
}

// -------------------------------------------------------------------------------------------------------------------
// TF_DSV41_DRAFT_BATCH_PROJ

/// `y` is `x` but for "w.ds.mx" tensors (lists included: the fused wkv's inputs), read `by` bytes further.
fn sameMoved(x: calls.Arg, y: calls.Arg, by: i64, moved: *usize) !void {
    switch (x) {
        .t => |t| if (t.role == .buf and std.mem.eql(u8, t.role.buf, "w.ds.mx")) {
            try testing.expectEqual(t.offset + by, y.t.offset);
            try testing.expectEqualSlices(i64, t.shape, y.t.shape);
            try testing.expectEqualSlices(i64, t.stride, y.t.stride);
            try testing.expect(@mod(y.t.offset, 16) == 0); // the slot's rows keep 16-byte alignment
            moved.* += 1;
        } else try testing.expect(equalArgs(x, y)),
        .list => |items| {
            try testing.expectEqual(items.len, y.list.len);
            for (items, y.list) |p, q| try sameMoved(p, q, by, moved);
        },
        else => try testing.expect(equalArgs(x, y)),
    }
}

test "batched main_proj: the projection then the blocks is the whole ingest; a slot's blocks read its rows of w.ds.mx" {
    var fx: Fixture = .{ .arena = undefined };
    try fx.init();
    defer fx.arena.deinit();
    const a = fx.arena.allocator();
    const D: i64 = fx.cfg.hidden;
    for ([_]i64{ 1, 5, 16 }) |n| for ([_]i64{ 0, 7 }) |skip| {
        const sl: emit.Slots = .{ .ring_slots = 4, .slot = 1 };
        const whole = try emit.emitIngestAt(a, &fx.cfg, &fx.w, fx.o, n, skip, sl);
        const proj = try emit.emitIngestProj(a, &fx.cfg, &fx.w, fx.o, n, skip, "w.taps");
        const blocks0 = try emit.emitIngestBlocks(a, &fx.cfg, &fx.w, fx.o, n, 0, sl);
        try identical(whole, try std.mem.concat(a, calls.Call, &.{ proj, blocks0 }));
        // at row r of a batched projection: the same calls, "w.ds.mx" read r rows further, nothing else moved
        const r: i64 = 11;
        const at = try emit.emitIngestBlocks(a, &fx.cfg, &fx.w, fx.o, n, r, sl);
        try testing.expectEqual(blocks0.len, at.len);
        var moved: usize = 0;
        for (blocks0, at) |c, d| {
            try testing.expectEqualStrings(c.name, d.name);
            try testing.expectEqual(c.grid, d.grid);
            for (c.args, d.args) |x, y| try sameMoved(x.arg, y.arg, r * D * 2, &moved);
        }
        try testing.expect(moved > 0);
    };
}

test "batched main_proj: the projection's launches do not depend on its rows but for the row count" {
    // linear.cu's row bits depend on the row alone given (K, N, SK, WK): dense.plan(K, N) and the strides take no
    // row count, so the batched call runs the per-slot call's kernel, k ranges and sum order at more rows
    var fx: Fixture = .{ .arena = undefined };
    try fx.init();
    defer fx.arena.deinit();
    const a = fx.arena.allocator();
    const one = try emit.emitIngestProj(a, &fx.cfg, &fx.w, fx.o, 1, 0, "w.taps");
    for ([_]i64{ 2, 5, 16, 17, 20, 33, 64 }) |n| {
        const many = try emit.emitIngestProj(a, &fx.cfg, &fx.w, fx.o, n, 3, "w.taps");
        try testing.expectEqual(one.len, many.len);
        for (one, many) |c, d| {
            try testing.expectEqualStrings(c.name, d.name);
            try testing.expectEqual(c.triton, d.triton);
            for (c.args, d.args) |x, y| switch (x.arg) {
                .i, .f, .b, .none => try testing.expect(equalArgs(x.arg, y.arg)),
                .t => |t| {
                    try testing.expect(t.dt == y.arg.t.dt);
                    try testing.expectEqualSlices(i64, t.stride, y.arg.t.stride);
                    // rows are the leading dimension; the rest (K, N, the weights) are the same
                    if (t.shape.len > 1) try testing.expectEqualSlices(i64, t.shape[1..], y.arg.t.shape[1..]);
                },
                else => {},
            };
        }
    }
}

test "KV norm/store fuses DSpark ingest and multislot passes without touching other norms" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(a);
    const off: block.Options = .{ .taps = true };
    var on = off;
    on.kv_norm_store = true;
    for ([_]i64{ 1, 16, 24, 64 }) |n| {
        const plain = try emit.emitIngest(a, &cfg, &w, off, n, 0);
        const fused = try emit.emitIngest(a, &cfg, &w, on, n, 0);
        try @import("kv_glue_emit_test.zig").compare(plain, fused, cfg.mtp_layers);
    }
    for ([_]i64{ 1, 2, 3, 4 }) |slots| {
        const plain = try emit.emitPassSlots(a, &cfg, &w, off, cfg.dspark_block, slots, slots);
        const fused = try emit.emitPassSlots(a, &cfg, &w, on, cfg.dspark_block, slots, slots);
        try @import("kv_glue_emit_test.zig").compare(plain, fused, cfg.mtp_layers);
        for (fused) |c| if (std.mem.eql(u8, c.name, "tf_dsv41_kv_glue_v1.norm_store")) {
            try testing.expect(c.args[14].arg.b);
            try testing.expectEqual(@as(i64, slots * cfg.dspark_block), c.args[5].arg.t.shape[0]);
            try testing.expectEqual(@as(i64, slots * emit.ringRows(&cfg)), c.args[3].arg.t.shape[0]);
        };
    }
}
