//! TF_DSV41_MHC_PFDEC (block_wide.zig) on the real emitters, host only (`zig build test-dsv41-wide`): with the knob,
//! each mixing mHC site of a decode window, a DSpark pass and a short prefill segment is mhc_pf's `site_kernel`, the
//! normed input (`_finish_k` without COEF) and mhc_cuda's `coef_kernel`, reading and writing exactly the tensors the
//! site it replaces did (mhc_cuda's boundary at <= 16 rows, Triton's `_site` + `_finish_k` above); every other call is
//! unchanged, the deferred coefficients keep MHC_DEFER's stream rules, row mode takes the programs. The bits are the
//! GPU gate's (oracle case `mhcdec`).

const std = @import("std");
const testing = std.testing;
const calls = @import("calls.zig");
const block = @import("block.zig");
const check = @import("m1_check.zig");
const rowmode = @import("rowmode.zig");
const branches = @import("branches.zig");
const dspark_emit = @import("dspark_emit.zig");
const pk = @import("prod_knobs.zig");
const Config = @import("config.zig").Config;
const Arg = calls.Arg;
const Call = calls.Call;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    return w;
}

fn prodOptions(limit: i64, pf: i64, deferred: bool) block.Options {
    var o: block.Options = .{ .limit = limit, .r1 = true, .mhc_defer = deferred, .expert_topp = 0.85, .index_budget = 64 << 20, .rows = true, .rope_rows = limit + 2048, .taps = true, .mhc_pf_rows = pf };
    const pages = @divExact(limit, block.Pool.page);
    o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages };
    return o;
}

fn same(x: Arg, y: Arg) bool {
    if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
    return switch (x) {
        .t, .opaque_table => |t| blk: {
            const u = if (y == .t) y.t else y.opaque_table;
            if (std.meta.activeTag(t.role) != std.meta.activeTag(u.role)) break :blk false;
            const rs = switch (t.role) {
                .weight => |r| std.mem.eql(u8, r, u.role.weight),
                .buf => |r| std.mem.eql(u8, r, u.role.buf),
                .empty => true,
            };
            break :blk rs and t.dt == u.dt and t.offset == u.offset and std.mem.eql(i64, t.shape, u.shape) and std.mem.eql(i64, t.stride, u.stride);
        },
        .i => |v| v == y.i,
        .f => |v| v == y.f,
        .b => |v| v == y.b,
        .none => true,
        .list => |l| l.len == y.list.len and for (l, y.list) |p, q| {
            if (!same(p, q)) break false;
        } else true,
    };
}

fn sameCall(x: Call, y: Call) bool {
    if (!std.mem.eql(u8, x.name, y.name) or x.triton != y.triton or x.glue != y.glue or x.begin != y.begin) return false;
    if (!std.mem.eql(i64, &x.grid, &y.grid) or x.args.len != y.args.len) return false;
    if (x.side != y.side or x.fork != y.fork or x.join != y.join) return false;
    for (x.args, y.args) |p, q| if (!std.mem.eql(u8, p.name, q.name) or !same(p.arg, q.arg)) return false;
    return true;
}

fn named(c: Call, name: []const u8) Arg {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return x.arg;
    std.debug.print("{s}: no argument {s}\n", .{ c.name, name });
    @panic("argument");
}

fn isEmpty(x: Arg) bool {
    return x == .t and x.t.role == .empty;
}

fn eq(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

/// What a mixing site reads and writes, whichever kernels run it.
const Site = struct {
    mode: i64,
    n: i64,
    x: Arg,
    xout: ?Arg = null,
    g: ?Arg = null,
    post: ?Arg = null,
    comb: ?Arg = null,
    pre: ?Arg = null,
    fnw: Arg,
    part: Arg,
    c: Arg,
    tap: ?Arg = null,
    nw: Arg,
    out: Arg,
    base: Arg,
    scale: Arg,
    next: [3]Arg,
    consts: [4]Arg, // eps, hc_eps, post_alpha, iters
    begin: calls.Begin,
    calls: usize,
};

/// The site the off path runs at cs[i] (mhc_cuda's boundary [+ R1's coefficient launch], or `_site` + `_finish_k`),
/// or null when cs[i] is not a mixing site.
fn offSite(cs: []const Call, i: usize) ?Site {
    const c = cs[i];
    if (eq(c.name, "tf_dsv41_mhc_cuda_v1.run")) {
        const a = c.args;
        const mode = a[19].arg.i;
        if (mode == 3) return null;
        const has_coef = i + 1 < cs.len and eq(cs[i + 1].name, "tf_dsv41_mhc_cuda_v1.coef");
        return .{
            .mode = mode,                                         .n = a[18].arg.i,
            .x = a[0].arg,                                        .xout = if (mode == 0) a[1].arg else null,
            .g = if (mode == 0) a[2].arg else null,               .post = if (mode == 0) a[3].arg else null,
            .comb = if (mode == 0) a[4].arg else null,            .pre = if (mode != 1) a[5].arg else null,
            .fnw = a[6].arg,                                      .part = a[9].arg,
            .c = a[10].arg,                                       .tap = if (isEmpty(a[11].arg)) null else a[11].arg,
            .nw = a[12].arg,                                      .out = a[13].arg,
            .base = a[7].arg,                                     .scale = a[8].arg,
            .next = .{ a[14].arg, a[15].arg, a[16].arg },         .consts = .{ a[20].arg, a[21].arg, a[22].arg, a[23].arg },
            .begin = c.begin,                                     .calls = if (has_coef) 2 else 1,
        };
    }
    if (eq(c.name, "_site") and named(c, "MIX").b) {
        const f = cs[i + 1];
        std.debug.assert(eq(f.name, "_finish_k") and named(f, "COEF").b);
        const post = named(c, "POST_ON").b;
        const coll = named(c, "COLLAPSE").i;
        return .{
            .mode = if (post) 0 else if (coll == 1) 1 else 2,     .n = named(c, "R").i,
            .x = named(c, "X"),                                   .xout = if (post) named(c, "XOUT") else null,
            .g = if (post) named(c, "G") else null,               .post = if (post) named(c, "POST") else null,
            .comb = if (post) named(c, "COMB") else null,         .pre = if (coll == 2) named(c, "PRE") else null,
            .fnw = named(c, "FN"),                                .part = named(c, "PART"),
            .c = named(c, "C"),                                   .tap = if (named(c, "TAP_ON").b) named(c, "TAP") else null,
            .nw = named(f, "NW"),                                 .out = named(f, "OUT"),
            .base = named(f, "BASE"),                             .scale = named(f, "SCALE"),
            .next = .{ named(f, "PRE"), named(f, "POST"), named(f, "COMB") },
            .consts = .{ named(f, "eps"), named(f, "hc_eps"), named(f, "post_alpha"), named(f, "ITERS") },
            .begin = c.begin,                                     .calls = 2,
        };
    }
    return null;
}

/// PFDEC's three calls at cs[j], checked against the off path's `s`.
fn expectPf(cs: []const Call, j: usize, s: Site, deferred: bool) !void {
    const p = cs[j];
    const f = cs[j + 1];
    const k = cs[j + 2];
    try testing.expectEqualStrings("tf_dsv41_mhc_pf_v1.run", p.name);
    try testing.expectEqualStrings("_finish_k", f.name);
    try testing.expectEqualStrings("tf_dsv41_mhc_cuda_v1.coef", k.name);
    const a = p.args;
    try testing.expect(s.begin == p.begin and f.begin == .none and k.begin == .none);
    try testing.expectEqual(s.mode, a[11].arg.i);
    try testing.expectEqual(s.n, a[10].arg.i);
    try testing.expect(same(s.x, a[0].arg));
    // mhc_pf: the post's buffers with a boundary, empty tensors otherwise (its binding's numel-0 checks)
    inline for (.{ .{ "xout", 1 }, .{ "g", 2 }, .{ "post", 3 }, .{ "comb", 4 }, .{ "pre", 5 }, .{ "tap", 9 } }) |m| {
        if (@field(s, m[0])) |v| try testing.expect(same(v, a[m[1]].arg)) else try testing.expect(isEmpty(a[m[1]].arg));
    }
    try testing.expect(same(s.fnw, a[6].arg) and same(s.part, a[7].arg) and same(s.c, a[8].arg));
    // the normed input alone: the final's `_finish_k` instance
    try testing.expect(!named(f, "COEF").b and named(f, "NORM_FIRST").b and named(f, "UNROLL").b);
    try testing.expectEqual(@as(i64, 1024), named(f, "BLOCK").i);
    try testing.expectEqual([3]i64{ s.n, 1, 1 }, f.grid);
    try testing.expect(same(s.part, named(f, "PART")) and same(s.c, named(f, "C")) and same(s.nw, named(f, "NW")) and same(s.out, named(f, "OUT")));
    try testing.expect(same(s.consts[0], named(f, "eps")));
    // coef_kernel: the next coefficients from the same partials, base and scale
    const b = k.args;
    try testing.expect(same(s.part, b[0].arg) and same(s.base, b[1].arg) and same(s.scale, b[2].arg));
    for (0..3) |q| try testing.expect(same(s.next[q], b[3 + q].arg));
    try testing.expectEqual(s.n, b[6].arg.i);
    for (0..4) |q| try testing.expect(same(s.consts[q], b[7 + q].arg));
    try testing.expectEqual(deferred, k.defer_side);
    try testing.expect(!p.defer_side and !f.defer_side and !f.defer_join and !k.defer_join);
}

/// `on` is `off` with each mixing site at or past `rows` rows as PFDEC's calls; returns the sites converted.
fn expectSwap(off: []const Call, on: []const Call, pf: bool, deferred: bool) !usize {
    var i: usize = 0;
    var j: usize = 0;
    var sites: usize = 0;
    while (i < off.len) {
        errdefer std.debug.print("off call {d} ({s}) / on call {d} ({s})\n", .{ i, off[i].name, j, if (j < on.len) on[j].name else "-" });
        if (pf) if (offSite(off, i)) |s| {
            try expectPf(on, j, s, deferred);
            i += s.calls;
            j += 3;
            sites += 1;
            continue;
        };
        try testing.expect(j < on.len);
        try testing.expect(sameCall(off[i], on[j]));
        // the deferred stream's marks: the off path's coefficient launch moves into PFDEC's triple, a join may move
        // to an mHC call that reads the coefficients (any site, the final, Engram's post)
        try testing.expectEqual(off[i].defer_side, on[j].defer_side);
        i += 1;
        j += 1;
    }
    try testing.expectEqual(on.len, j);
    return sites;
}

fn mixingSites(cs: []const Call) usize {
    var m: usize = 0;
    for (cs, 0..) |_, i| m += @intFromBool(offSite(cs, i) != null);
    return m;
}

/// MHC_DEFER's rules on a program: one pending coefficient launch at a time (sharedDeferred), each joined before the
/// next mHC call reads the coefficients (a site, a final, Engram's post-only: mhc.await_coefs) and before the program
/// ends.
fn expectJoined(a: std.mem.Allocator, cs: []const Call) ![]const []const u8 {
    var pending = false;
    for (cs, 0..) |c, i| {
        errdefer std.debug.print("call {d} ({s}) reads coefficients still on the deferred stream\n", .{ i, c.name });
        if (c.defer_join) {
            try testing.expect(pending);
            pending = false;
        }
        const reads = eq(c.name, "tf_dsv41_mhc_pf_v1.run") or eq(c.name, "tf_dsv41_mhc_cuda_v1.run") or eq(c.name, "_site");
        if (reads) try testing.expect(!pending);
        if (c.defer_side) pending = true;
    }
    try testing.expect(!pending);
    return branches.sharedDeferred(a, cs);
}

/// With PFDEC the deferred launches share no role with main's calls that the off path's did not.
fn expectDeferred(a: std.mem.Allocator, off: []const Call, on: []const Call) !void {
    const so = try expectJoined(a, off);
    const sn = try expectJoined(a, on);
    for (sn) |r| {
        errdefer std.debug.print("new shared role {s}\n", .{r});
        for (so) |q| {
            if (eq(q, r)) break;
        } else return error.TestUnexpectedResult;
    }
}

test "TF_DSV41_MHC_PFDEC reader: unset / 0 off, 1..64 the fewest rows, anything else refused" {
    const Fake = struct {
        var v: ?[]const u8 = null;
        fn get(name: []const u8) ?[]const u8 {
            return if (eq(name, "TF_DSV41_MHC_PFDEC")) v else null;
        }
    };
    Fake.v = null;
    try testing.expectEqual(@as(i64, 0), try pk.mhcPfRows(&Fake.get));
    for ([_][]const u8{ "0", "1", "17", " 64 " }, [_]i64{ 0, 1, 17, 64 }) |s, want| {
        Fake.v = s;
        try testing.expectEqual(want, try pk.mhcPfRows(&Fake.get));
    }
    for ([_][]const u8{ "65", "-1", "on", "" }) |s| {
        Fake.v = s;
        try testing.expectError(error.BadKnob, pk.mhcPfRows(&Fake.get));
    }
}

test "TF_DSV41_MHC_PFDEC: decode windows of 1-64 rows, each mixing site as mhc_pf + norm + coef_kernel on the same tensors" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    for ([_]bool{ true, false }) |deferred| for ([_]i64{ 1, 17 }) |rows| for ([_]i64{ 1, 2, 5, 16, 17, 20, 24, 32, 33, 48, 64 }) |n| {
        errdefer std.debug.print("pfdec {d}, {d} rows, defer {}\n", .{ rows, n, deferred });
        const start: i64 = 300_000;
        const off = try block.emit(a, &cfg, &w, prodOptions(limit, 0, deferred), &backbone, n, start, true);
        const on = try block.emit(a, &cfg, &w, prodOptions(limit, rows, deferred), &backbone, n, start, true);
        const pf = n >= rows;
        const sites = try expectSwap(off, on, pf, deferred);
        // 40 layers' two sites (an Engram layer's site after its post-only included); none left for the old path
        try testing.expectEqual(if (pf) @as(usize, 2 * cfg.layers) else 0, sites);
        try testing.expectEqual(@as(usize, 0), if (pf) mixingSites(on) else 0);
        if (deferred) try expectDeferred(a, off, on);
        if (n > 1) {
            const rw = try rowmode.transform(a, on, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = prodOptions(limit, 0, false).pool.?.pts });
            try testing.expectEqual(on.len, rw.len);
        }
    };
}

test "TF_DSV41_MHC_PFDEC: DSpark passes (1 and 4 slots) take it at their in-place boundaries too; MHC_DEFER joins there" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    for ([_]bool{ true, false }) |deferred| for ([_]i64{ 1, 17 }) |rows| for ([_]i64{ 1, 4 }) |slots| {
        errdefer std.debug.print("pfdec {d}, {d} slots, defer {}\n", .{ rows, slots, deferred });
        const off = try dspark_emit.emitPassSlots(a, &cfg, &w, prodOptions(1 << 20, 0, deferred), cfg.dspark_block, slots, 4);
        const on = try dspark_emit.emitPassSlots(a, &cfg, &w, prodOptions(1 << 20, rows, deferred), cfg.dspark_block, slots, 4);
        const pf = slots * cfg.dspark_block >= rows;
        const sites = try expectSwap(off, on, pf, deferred);
        try testing.expectEqual(if (pf) @as(usize, 2 * cfg.mtp_layers) else 0, sites);
        // the off path too: block 0's site leaves its coefficients deferred, the next boundary and the final join them
        if (deferred) try expectDeferred(a, off, on);
    };
}
