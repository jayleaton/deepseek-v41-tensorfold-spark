//! Host tests of the buffer plan with native images (TF_DSV41_IMAGES=native with TF_DSV41_BIAS_VL; `zig build
//! test-dsv41-knobs`): forward_prefill.plan after vision_rows.flags, as model.zig boots, names every role an image
//! prompt's segments read (Engram's keep, the image MoE call's split) at their rows, in full and CED replay mode,
//! and still plans the multi-segment runs and a TF_DSV41_PF_TBO pair; a plan made before the flags leaves the image
//! roles unbound (a served image request's first segment failed with error.Unbound, rank 1 with it).
const std = @import("std");
const testing = std.testing;
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const calls = @import("calls.zig");
const ced = @import("ced.zig");
const check = @import("m1_check.zig");
const fwd = @import("forward.zig");
const fp = @import("forward_prefill.zig");
const pk = @import("prod_knobs.zig");
const vision_rows = @import("vision_rows.zig");
const Config = @import("config.zig").Config;

const limit: i64 = 8192;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    for (0..40) |L| try w.gm.put(a, @intCast(L), .{ (1 << 3) | (1 << 4), (1 << 3) | (1 << 4) });
    return w;
}

fn options() block.Options {
    const pages = @divExact(limit, block.Pool.page);
    var o: block.Options = .{ .limit = limit, .rope_rows = limit + 2048, .branches = true };
    o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages, .split = false };
    return o;
}

/// A host forward over layers 0-24 (the gates' prefix); nothing here reaches its runner or collective.
fn forward(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, layers: []const u32, mode: pk.PrefillMode) fwd.Forward {
    var f = fwd.Forward.init(a, cfg, w, options(), undefined, undefined);
    f.layers = layers;
    f.prefill_state.mode = mode;
    return f;
}

/// The roles of `cs` the plan does not hold at their bytes, as a sorted list.
fn unplanned(a: std.mem.Allocator, plan: *const buffers.Plan, cs: []const calls.Call, out: *std.ArrayList([]const u8)) !void {
    var need: buffers.Plan = .{ .a = a };
    try need.add(cs);
    for (need.sizes.keys(), need.sizes.values()) |role, bytes| {
        if ((plan.sizes.get(role) orelse 0) >= bytes) continue;
        const seen = for (out.items) |x| {
            if (std.mem.eql(u8, x, role)) break true;
        } else false;
        if (!seen) try out.append(a, role);
    }
}

/// The served prompt (Spark: 224 tokens, a 64 x 64 image as its 184-position span) and its pieces: [0, 208) to the
/// snapshot point, [208, 223) a 16-row short segment, the last token the first verify window's row.
fn promptIds(a: std.mem.Allocator) ![]u32 {
    const ids = try a.alloc(u32, 224);
    for (ids, 0..) |*t, i| t.* = if (i >= 30 and i < 214) vision_rows.VBASE + @as(u32, @intCast(i - 30)) else 100 + @as(u32, @intCast(i));
    return ids;
}

/// Every unplanned role of the prompt's segments as forward_prefill emits them (`segment`'s options, CED's decoder
/// replay over the tail).
fn imageGaps(a: std.mem.Allocator, f: *fwd.Forward, plan: *const buffers.Plan, ids: []const u32, out: *std.ArrayList([]const u8)) !void {
    const layers = try f.backbone(a);
    const d = std.mem.indexOfScalar(u32, layers, try ced.decoderStart(f.cfg)).?;
    const replay = f.prefill_state.mode == .replay;
    for ([_][2]usize{ .{ 0, 208 }, .{ 208, 223 } }) |piece| {
        const seg = ids[piece[0]..piece[1]];
        const o = vision_rows.segmentOptions(f, seg);
        const n: i64 = @intCast(seg.len);
        const cs = if (replay) try bp.emitReplay(a, f.cfg, f.widths, o, layers[0 .. d + 1], n, @intCast(piece[0]), .encoder) else try bp.emitPrefill(a, f.cfg, f.widths, o, layers, n, @intCast(piece[0]), true);
        try unplanned(a, plan, cs, out);
    }
    if (replay) {
        const t = ced.tailOf(223);
        const tail = ids[@intCast(t.r0)..223];
        try unplanned(a, plan, try bp.emitReplay(a, f.cfg, f.widths, vision_rows.segmentOptions(f, tail), layers[d..], t.m, @intCast(t.r0), .decoder), out);
    }
}

test "native images: the plan after vision_rows.flags holds every role an image prompt's segments read, full and replay" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    var layers: [25]u32 = undefined;
    for (&layers, 0..) |*x, i| x.* = @intCast(i);
    const ids = try promptIds(a);
    for ([_]pk.PrefillMode{ .full, .replay }) |mode| {
        var f = forward(a, &cfg, &w, &layers, mode);
        try testing.expectEqualStrings("/cache/dsv41-bias-vl", vision_rows.flags(&f, true, " /cache/dsv41-bias-vl ").?);
        try testing.expect(f.images and f.image_bias);
        var plan: buffers.Plan = .{ .a = a };
        try fp.plan(&f, &plan, 2048);
        var gaps: std.ArrayList([]const u8) = .empty;
        try imageGaps(a, &f, &plan, ids, &gaps);
        // a text prompt (Spark's 44 tokens: 43 rows prefilled) on the same plan
        const text = try a.alloc(u32, 43);
        for (text, 0..) |*t, i| t.* = 100 + @as(u32, @intCast(i));
        try unplanned(a, &plan, try bp.emitPrefill(a, &cfg, &w, vision_rows.segmentOptions(&f, text), &layers, 43, 0, true), &gaps);
        for (gaps.items) |role| std.debug.print("{t}: role {s} is in no planned program\n", .{ mode, role });
        try testing.expectEqual(@as(usize, 0), gaps.items.len);
        // the multi-segment runs stay planned (planRuns at runOptions: their emitter refuses image rows)
        try testing.expect(plan.sizes.get("w.mpos64") != null);
    }
}

test "native images: a plan made before vision_rows.flags leaves the image roles unbound (the served boot's order)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    var layers: [25]u32 = undefined;
    for (&layers, 0..) |*x, i| x.* = @intCast(i);
    var f = forward(a, &cfg, &w, &layers, .full);
    var plan: buffers.Plan = .{ .a = a };
    try fp.plan(&f, &plan, 2048);
    _ = vision_rows.flags(&f, true, "/cache/dsv41-bias-vl");
    var gaps: std.ArrayList([]const u8) = .empty;
    try imageGaps(a, &f, &plan, try promptIds(a), &gaps);
    std.mem.sort([]const u8, gaps.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    try testing.expectEqual(@as(usize, 5), gaps.items.len);
    for ([_][]const u8{ "L.moe16.img", "L.moe16.txt", "w.engram.keep", "w.out.img", "w.out.txt" }, gaps.items) |want, got| try testing.expectEqualStrings(want, got);
}

test "native images: TF_DSV41_PF_TBO's workspace and the runs still plan with the image flags (runOptions)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    var layers: [25]u32 = undefined;
    for (&layers, 0..) |*x, i| x.* = @intCast(i);
    var f = forward(a, &cfg, &w, &layers, .replay);
    f.opts.pf_tbo = true;
    _ = vision_rows.flags(&f, true, "/cache/dsv41-bias-vl");
    try testing.expect(try fp.workspaceBytes(&f, 2048) > 0);
    var plan: buffers.Plan = .{ .a = a };
    try fp.plan(&f, &plan, 2048);
    try testing.expect(plan.sizes.get("w.mpos64") != null);
    // not native: no flags; native without TF_DSV41_BIAS_VL: Engram's keep alone
    var g = forward(a, &cfg, &w, &layers, .full);
    try testing.expectEqual(@as(?[]const u8, null), vision_rows.flags(&g, false, "/cache/dsv41-bias-vl"));
    try testing.expect(!g.images and !g.image_bias);
    try testing.expectEqualStrings("", vision_rows.flags(&g, true, " ").?);
    try testing.expect(g.images and !g.image_bias);
}
