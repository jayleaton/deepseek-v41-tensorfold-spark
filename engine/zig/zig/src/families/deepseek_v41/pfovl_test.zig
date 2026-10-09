//! Host tests of the prefill exchange overlap (TF_DSV41_PF_OVERLAP; `zig build test-dsv41-knobs`): the knob's reader,
//! and the real prefill emitter with it on: each layer's two exchanges become `k` row pieces (glue exchange_rows), the
//! first k - 1 on the side stream right after their producer's piece (wo_a's groups + wo_b, or the shared expert +
//! bf16(routed + shared)), the last on the main stream after a join; the pieces tile the segment's rows in order;
//! the only roles both streams touch inside a fork region are the partials themselves, at disjoint rows (the side
//! reads rows the main stream wrote before the fork, the main stream writes later rows); every other call is the
//! knob-off program's, in order.
const std = @import("std");
const testing = std.testing;
const calls = @import("calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const branches = @import("branches.zig");
const check = @import("m1_check.zig");
const pk = @import("prod_knobs.zig");
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "TF_DSV41_PF_OVERLAP: unset / 0 / 1 off, 2-8 row pieces, anything else refused" {
    defer Fake.pairs = &.{};
    Fake.pairs = &.{};
    try testing.expectEqual(@as(i64, 0), try pk.pfOverlap(&Fake.get));
    for ([_][2][]const u8{ .{ "0", "0" }, .{ "1", "0" }, .{ "2", "2" }, .{ "4", "4" }, .{ "8", "8" } }) |c| {
        Fake.pairs = &.{.{ "TF_DSV41_PF_OVERLAP", c[0] }};
        try testing.expectEqual(try std.fmt.parseInt(i64, c[1], 10), try pk.pfOverlap(&Fake.get));
    }
    for ([_][]const u8{ "9", "-1", "on" }) |v| {
        Fake.pairs = &.{.{ "TF_DSV41_PF_OVERLAP", v }};
        try testing.expectError(error.BadKnob, pk.pfOverlap(&Fake.get));
    }
}

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    return w;
}

fn roleOf(x: calls.Arg) []const u8 {
    return switch (x.t.role) {
        .buf => |b| b,
        .weight => |w| w,
        .empty => "",
    };
}

/// A call the overlap re-emits per piece: the producers' projections (rot_in + GEMM, `_gemm`'s unpacks) on wo_a / wo_b
/// / the shared expert, their glue, and the exchanges.
fn isPieced(c: calls.Call) bool {
    if (std.mem.startsWith(u8, c.name, "glue.exchange")) return true;
    if (std.mem.eql(u8, c.name, "glue.swiglu") or std.mem.eql(u8, c.name, "glue.moe_sum")) return true;
    // a projection's rot_in, unpack and GEMM read the producer's weights (suh / svh / trellis) or its words ("s.<w>.lanes")
    for (c.args) |x| if (x.arg == .t and x.arg.t.role != .empty) {
        const r = roleOf(x.arg);
        for ([_][]const u8{ ".attn.wo_a.", ".attn.wo_b", ".shared.0." }) |w| if (std.mem.indexOf(u8, r, w) != null) return true;
    };
    return false;
}

fn count(cs: []const calls.Call, name: []const u8) usize {
    var n: usize = 0;
    for (cs) |c| n += @intFromBool(std.mem.eql(u8, c.name, name));
    return n;
}

const Mode = struct { n: i64, o: block.Options };

test "TF_DSV41_PF_OVERLAP on the prefill emitter: k pieces an exchange behind their producers, disjoint rows across the streams, every other call unchanged" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const layers = [_]u32{ 1, 2, 3 };
    const D: i64 = cfg.hidden;
    const base: block.Options = .{ .gm_v2 = .one, .branches = true, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048 };
    var big = base;
    big.pf4k = true;
    big.prefill_rows = 4096;
    for ([_]Mode{ .{ .n = 2048, .o = base }, .{ .n = 1024, .o = base }, .{ .n = 4096, .o = big } }) |md| {
        for ([_]i64{ 2, 4 }) |k| {
            var on_o = md.o;
            on_o.pf_overlap = k;
            const off = try bp.emitPrefill(aa, &cfg, &w, md.o, &layers, md.n, 0, false);
            const on = try bp.emitPrefill(aa, &cfg, &w, on_o, &layers, md.n, 0, false);
            // every call outside the pieced producers and exchanges: the same, in order
            var i: usize = 0;
            var j: usize = 0;
            while (true) {
                while (i < off.len and isPieced(off[i])) i += 1;
                while (j < on.len and isPieced(on[j])) j += 1;
                if (i == off.len or j == on.len) break;
                try testing.expectEqualStrings(off[i].name, on[j].name);
                try testing.expectEqual(off[i].args.len, on[j].args.len);
                i += 1;
                j += 1;
            }
            // the pieced projections' rot_in: k per projection in place of one
            const xs = count(off, "glue.exchange");
            try testing.expectEqual(@as(usize, 2 * layers.len), xs);
            try testing.expectEqual(@as(usize, 0), count(on, "glue.exchange"));
            try testing.expectEqual(xs * @as(usize, @intCast(k)), count(on, "glue.exchange_rows"));
            try testing.expectEqual(count(off, "tensorfold_exl3_linear_v4.rot_in") + (@as(usize, @intCast(k)) - 1) * layers.len * (4 + 1 + 3), count(on, "tensorfold_exl3_linear_v4.rot_in"));
            try testing.expectEqual(@as(usize, @intCast(k)) * count(off, "glue.moe_sum"), count(on, "glue.moe_sum"));
            // the streams: only the partials are shared inside a fork region
            const sh = try branches.shared(aa, on);
            for (sh) |r| try testing.expect(std.mem.eql(u8, r, "L.part") or std.mem.eql(u8, r, "L.moe16"));
            try testing.expect(sh.len == 2);
            try testing.expectEqual(@as(usize, 0), (try branches.shared(aa, off)).len);
            try pieces(on, md.n, k, D);
        }
        // fewer than 256 rows a piece, or no side stream: the knob-off program
        for ([_]block.Options{ blk: {
            var o = md.o;
            o.pf_overlap = 8;
            break :blk o;
        }, blk: {
            var o = md.o;
            o.pf_overlap = 4;
            o.branches = false;
            break :blk o;
        } }) |o| {
            if (o.pf_overlap == 8 and md.n >= 8 * 256) continue;
            const off = try bp.emitPrefill(aa, &cfg, &w, md.o, &layers, md.n, 0, false);
            const same = try bp.emitPrefill(aa, &cfg, &w, o, &layers, md.n, 0, false);
            try testing.expectEqual(off.len, same.len);
            for (off, same) |x, y| try testing.expectEqualStrings(x.name, y.name);
        }
    }
}

/// Each exchange's pieces: rows [r0, r0 + m) tiling [0, n) in order, the 16-row grid, the first k - 1 forked onto the
/// side stream, the last joined on the main stream; each piece right after the producer that wrote its rows (wo_b's
/// output or moe_sum's), and into the whole gathered buffer [2, n, D].
fn pieces(cs: []const calls.Call, n: i64, k: i64, D: i64) !void {
    var next: i64 = 0;
    var seen: i64 = 0;
    for (cs, 0..) |c, idx| {
        if (!std.mem.eql(u8, c.name, "glue.exchange_rows")) continue;
        const send = c.args[0].arg.t;
        const into = c.args[1].arg.t;
        const r0 = c.args[2].arg.i;
        const m = send.shape[0];
        try testing.expectEqual(next, r0);
        try testing.expectEqual(@as(i64, 0), @mod(r0, 16));
        try testing.expectEqual(r0 * D * 2, send.offset);
        try testing.expectEqualSlices(i64, &.{ 2, n, D }, into.shape);
        seen += 1;
        const last = @mod(seen, k) == 0;
        try testing.expectEqual(!last, c.side);
        try testing.expectEqual(!last, c.fork);
        try testing.expectEqual(last, c.join);
        // its producer: the call before it wrote these rows of the same role
        const prev = cs[idx - 1];
        const out = for (prev.args) |x| {
            if (x.arg == .t and x.arg.t.role == .buf and std.mem.eql(u8, x.arg.t.role.buf, roleOf(c.args[0].arg))) break x.arg.t;
        } else return error.TestUnexpectedResult;
        try testing.expectEqual(send.offset, out.offset);
        try testing.expectEqual(m, out.shape[0]);
        next = if (last) 0 else r0 + m;
        if (last) try testing.expectEqual(n, r0 + m);
    }
}
